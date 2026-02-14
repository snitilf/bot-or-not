"""french bot detector for bot or not challenge. hybrid approach: high-confidence rules + multi-seed rf/gbm/xgboost ensemble. uses cross-language training (en+fr) for more training data. tuned for lower bot prevalence (~16%) and fr-specific patterns."""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features, load_ground_truth, load_dataset, group_posts_by_author
from dna_cluster import build_dna_sequences, cluster_sequences, expand_from_clusters

# feature toggles (env-driven)
USE_DNA_CLUSTER = os.getenv('BOTORN0T_USE_DNA_CLUSTER', '0') == '1'
USE_COMPRESSION_FEATURES = os.getenv('BOTORN0T_USE_COMPRESSION', '1') != '0'


# feature set: behavioral features with auc > 0.65 on both fr datasets,
# plus batch injection and stylometric features.
COMPRESSION_FEATURES = [
    'compression_ratio',
    'char_3gram_entropy',
    'char_3gram_unique_ratio',
]

FR_FEATURES = [
    # tier 1: auc > 0.80 on both fr datasets
    'excl_rate',           # 0.91/0.87
    'hour_entropy',        # 0.85/0.84
    'hashtag_rate',        # 0.87/0.80
    'question_rate',       # 0.82/0.85
    'active_hours',        # 0.84/0.83
    'night_ratio',         # 0.81/0.85
    'peak_hour_frac',      # 0.82/0.83
    'within_window_ratio',
    'window_coverage',
    'edge_activity_ratio',
    'mid_hashtag_rate',    # 0.84/0.78
    'comma_rate',          # 0.857/0.865 — stylometric
    # tier 2: auc > 0.70 on both fr datasets
    'std_interval',        # 0.80/0.75
    'max_interval',        # 0.818/0.778 — was excluded by oversight
    'burstiness',          # 0.73/0.78
    'cv_intervals',        # 0.73/0.76
    'topic_diversity',     # 0.76/0.71
    'ellipsis_rate',       # 0.612/0.705 — stylometric
    # tier 3: moderate but consistent (auc > 0.65)
    'self_similarity',     # 0.74/0.67
    'url_rate',            # 0.69/0.71
    'tweet_count',         # 0.69/0.72
    'z_score',             # 0.69/0.72
    'tweet_count_ratio',
    'num_tweets_ratio',
    'z_score_delta',
    'avg_tweet_len',       # 0.65/0.72
    # batch injection features (for ml model, also caught by rule)
    'max_same_second',
    'unique_second_ratio',
    'timestamp_collision_rate',
]

if USE_COMPRESSION_FEATURES:
    FR_FEATURES.extend(COMPRESSION_FEATURES)

# multi-seed rf ensemble (like en's 8-seed approach)
FR_RF_SEEDS = [0, 7, 42, 99, 123, 456]

# ensemble weights: rf + xgb + gbm
ENSEMBLE_WEIGHTS = {'rf': 0.45, 'xgb': 0.30, 'gbm': 0.25}

# phase 4: calibration + risk-aware decision (env-tunable)
CALIBRATE_PROBA = os.getenv('BOTORN0T_CALIBRATE', '0') != '0'
MIN_COST_THRESHOLD = float(os.getenv('BOTORN0T_MIN_COST_THRESHOLD', str(2.0 / 7.0)))
ROBUST_THRESHOLD_FR = os.getenv('BOTORN0T_ROBUST_THRESHOLD_FR', '1') != '0'
ROBUST_SCENARIOS_FR = int(os.getenv('BOTORN0T_ROBUST_SCENARIOS_FR', '400'))
ROBUST_SEED_FR = int(os.getenv('BOTORN0T_ROBUST_SEED_FR', '42'))
ROBUST_MAX_DROP_FR = int(os.getenv('BOTORN0T_ROBUST_MAX_DROP_FR', '1'))

# all practice datasets (en + fr) for cross-language training
ALL_PRACTICE = {
    'dataset.posts&users.30.json': 'dataset.bots.30.txt',  # en
    'dataset.posts&users.31.json': 'dataset.bots.31.txt',  # fr
    'dataset.posts&users.32.json': 'dataset.bots.32.txt',  # en
    'dataset.posts&users.33.json': 'dataset.bots.33.txt',  # fr
}

# phase 1: stabilization + human veto
USE_RATE_CLAMP = True
USE_HUMAN_VETO = True
HUMAN_VETO_QUANTILE = 0.99
VETO_MAX_PROBA = 0.40
FR_BOT_RATE_BUFFER = 0.08

# phase 2: digital dna clustering (conservative)
DNA_EPS = 0.30
DNA_MIN_SAMPLES = 3
DNA_K = 3
DNA_MIN_SEED = 2
DNA_MIN_SEED_RATIO = 0.6
DNA_MAX_CLUSTER_SIZE = 15
DNA_SEED_PROBA_MARGIN = 0.20


def apply_rules(df: pd.DataFrame) -> pd.Series:
    """layer 1: high-confidence rules for fr bots. each rule verified on both ds31 and ds33 for near-zero fp. no out-of-range date signal in fr, so rules use behavioral combinations."""
    flagged = pd.Series(False, index=df.index)

    # rule 1: very high hour entropy + any night posting (100% precision on both ds)
    flagged |= (df['hour_entropy'] >= 4.0) & (df['night_ratio'] >= 0.05)

    # rule 2: 24/7 posting + metronomic timing (100% precision)
    flagged |= (df['active_hours'] >= 18) & (df['burstiness'] < 0.0)

    # rule 3: high hour entropy + moderate excl_rate (100% precision)
    flagged |= (df['hour_entropy'] >= 3.8) & (df['excl_rate'] >= 0.3)

    # rule 4: very flat posting distribution (tightened to avoid fp)
    flagged |= (df['peak_hour_frac'] <= 0.10) & (df['active_hours'] >= 14)

    # rule 5: moderate excl + wide active hours + meaningful night signal
    # fixed: night_ratio >= 0.10 (was > 0.0) to avoid timezone fps
    flagged |= (df['excl_rate'] >= 0.4) & (df['active_hours'] >= 14) & (df['night_ratio'] >= 0.10)

    # rule 6: batch injection — multiple tweets at the exact same second
    # threshold >=4: catches 6 bots across ds31+ds33 with 0 fp
    if 'max_same_second' in df.columns:
        flagged |= (df['max_same_second'] >= 4)

    return flagged


def _parse_dataset_id(dataset_path: str) -> str:
    name = os.path.basename(dataset_path)
    # dataset.posts&users.30.json -> 30
    parts = name.split('.')
    if len(parts) >= 3:
        return parts[-2]
    return name


def _dataset_lang_from_id(ds_id: str) -> str:
    if ds_id in ('31', '33'):
        return 'fr'
    if ds_id in ('30', '32'):
        return 'en'
    return 'unknown'


def _compute_fr_bot_rate(train_meta: list) -> float:
    """compute average bot rate on fr datasets only (from training data)."""
    fr_rates = [m['bot_rate'] for m in train_meta if m['lang'] == 'fr']
    if fr_rates:
        return float(np.mean(fr_rates))
    # fallback to overall
    all_rates = [m['bot_rate'] for m in train_meta]
    return float(np.mean(all_rates)) if all_rates else 0.0


def _apply_rate_clamp(proba: np.ndarray, rule_flags: np.ndarray, max_rate: float) -> tuple:
    """adjust ml threshold upward so total predicted bot rate stays <= max_rate. returns (adjusted_threshold, adjusted_ml_flags)."""
    n = len(proba)
    if n == 0 or max_rate <= 0:
        return 1.0, np.zeros_like(proba, dtype=bool)

    n_rules = int(rule_flags.sum())
    max_total = int(np.floor(max_rate * n))
    max_ml = max(max_total - n_rules, 0)

    # if current ml predictions already under cap, keep as-is.
    sorted_probs = np.sort(proba)[::-1]
    if max_ml == 0:
        return 1.0, np.zeros_like(proba, dtype=bool)
    if len(sorted_probs) < max_ml:
        return 0.0, np.ones_like(proba, dtype=bool)

    threshold = float(sorted_probs[max_ml - 1])
    ml_flags = proba >= threshold
    return threshold, ml_flags


def _score_from_predictions(preds: np.ndarray, y_true: np.ndarray) -> int:
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    return 4 * tp - 1 * fn - 2 * fp


def _score_fold_at_threshold(proba: np.ndarray, y_true: np.ndarray,
                             rules: np.ndarray, thresh: float,
                             max_rate: float,
                             human_probs: np.ndarray = None,
                             veto_threshold: float = None) -> int:
    effective = max(float(thresh), MIN_COST_THRESHOLD)
    ml_flags = (proba >= effective)
    rules_bool = rules.astype(bool)

    if USE_RATE_CLAMP:
        clamp_thresh, clamp_flags = _apply_rate_clamp(proba, rules_bool, max_rate)
        if clamp_thresh > effective:
            ml_flags = clamp_flags

    if USE_HUMAN_VETO and human_probs is not None and veto_threshold is not None:
        veto_mask = (
            (human_probs >= veto_threshold)
            & ml_flags
            & (proba < VETO_MAX_PROBA)
            & (~rules_bool)
        )
        if np.any(veto_mask):
            ml_flags = ml_flags & (~veto_mask)

    combined = np.maximum(ml_flags.astype(int), rules.astype(int))
    return _score_from_predictions(combined, y_true)


def _estimate_fr_max_rate(train_paths: list, train_bots_paths: list) -> float:
    """estimate fr max-rate clamp from training datasets only."""
    train_meta = []
    for dp, bp in zip(train_paths, train_bots_paths):
        df = extract_features(dp, use_cache=True)
        bots = load_ground_truth(bp)
        ds_id = _parse_dataset_id(dp)
        ds_lang = _dataset_lang_from_id(ds_id)
        bot_rate = float(df.index.isin(bots).astype(int).mean()) if len(df) else 0.0
        train_meta.append({'id': ds_id, 'lang': ds_lang, 'bot_rate': bot_rate})

    fr_bot_rate = _compute_fr_bot_rate(train_meta)
    return max(0.25, fr_bot_rate + FR_BOT_RATE_BUFFER)


def _prepare_fr_fold(dataset_path: str, bots_path: str,
                     probabilities: dict, max_rate: float,
                     human_probabilities: dict = None,
                     veto_threshold: float = None) -> dict:
    """collect arrays for fr threshold selection and stress scenarios."""
    df = extract_features(dataset_path, use_cache=True)
    bots = load_ground_truth(bots_path)

    y_true = np.array(df.index.isin(bots).astype(int))
    proba = np.array([probabilities.get(uid, 0.0) for uid in df.index])
    rules = apply_rules(df).values.astype(int)
    if human_probabilities is not None:
        human_probs = np.array([human_probabilities.get(uid, 0.0) for uid in df.index])
    else:
        human_probs = None

    botlike_human = np.zeros(len(df), dtype=int)
    human_idx = np.where(y_true == 0)[0]
    if len(human_idx) > 0:
        cutoff = float(np.quantile(proba[human_idx], 0.75))
        botlike_human[human_idx] = (proba[human_idx] >= cutoff).astype(int)

    return {
        'dataset_path': dataset_path,
        'y_true': y_true,
        'proba': proba,
        'rules': rules,
        'max_rate': float(max_rate),
        'human_probs': human_probs,
        'veto_threshold': veto_threshold,
        'botlike_human': botlike_human,
    }


def _build_stress_samples(fold: dict, n_scenarios: int, seed: int) -> list:
    """build pseudo-final fr samples with prevalence, rule reliability, and hard-negative shifts."""
    rng = np.random.RandomState(seed)
    y_true = fold['y_true']
    n = len(y_true)

    bot_idx = np.where(y_true == 1)[0]
    human_idx = np.where(y_true == 0)[0]
    if len(bot_idx) == 0 or len(human_idx) == 0:
        return []

    botlike_human = fold['botlike_human']

    samples = []
    for _ in range(n_scenarios):
        target_bot_rate = float(rng.uniform(0.10, 0.24))
        rule_reliability = float(rng.uniform(0.75, 1.00))
        hard_negative_boost = float(rng.uniform(1.00, 3.00))

        n_bot = int(round(target_bot_rate * n))
        n_bot = max(1, min(n - 1, n_bot))
        n_human = n - n_bot

        sampled_bots = rng.choice(bot_idx, size=n_bot, replace=True)

        human_weights = np.ones(len(human_idx), dtype=float)
        human_weights += (hard_negative_boost - 1.0) * botlike_human[human_idx].astype(float)
        human_weights /= human_weights.sum()
        sampled_humans = rng.choice(human_idx, size=n_human, replace=True, p=human_weights)

        idx = np.concatenate([sampled_bots, sampled_humans])
        rng.shuffle(idx)

        rules_sample = fold['rules'][idx].copy()
        rule_pos = np.where(rules_sample == 1)[0]
        if len(rule_pos) > 0:
            keep = rng.rand(len(rule_pos)) < rule_reliability
            drop_pos = rule_pos[~keep]
            if len(drop_pos) > 0:
                rules_sample[drop_pos] = 0

        samples.append({
            'idx': idx,
            'rules': rules_sample.astype(int),
        })

    return samples


def _select_robust_threshold(folds: list, sweep: np.ndarray,
                             n_scenarios: int, seed: int) -> dict:
    """select fr threshold by worst-case-aware pseudo-final stress objective."""
    scenario_sets = []
    for i, fold in enumerate(folds):
        scenario_sets.append(_build_stress_samples(fold, n_scenarios=n_scenarios, seed=seed + i * 997))

    best = {
        'threshold': float(sweep[0]),
        'objective': -1e18,
        'baseline_score': -999999,
        'stress_p10': -999999,
        'stress_median': -999999,
        'stress_min': -999999,
        'stress_scores': [],
    }

    for thresh in sweep:
        baseline_total = 0
        for fold in folds:
            baseline_total += _score_fold_at_threshold(
                fold['proba'], fold['y_true'], fold['rules'],
                float(thresh), fold['max_rate'],
                human_probs=fold.get('human_probs'),
                veto_threshold=fold.get('veto_threshold'),
            )

        stress_scores = []
        for scenario_i in range(n_scenarios):
            total = 0
            for fold, samples in zip(folds, scenario_sets):
                if not samples:
                    continue
                sample = samples[scenario_i]
                idx = sample['idx']
                y_s = fold['y_true'][idx]
                p_s = fold['proba'][idx]
                r_s = sample['rules']
                h_s = None
                if fold.get('human_probs') is not None:
                    h_s = fold['human_probs'][idx]
                total += _score_fold_at_threshold(
                    p_s, y_s, r_s, float(thresh), fold['max_rate'],
                    human_probs=h_s,
                    veto_threshold=fold.get('veto_threshold'),
                )
            stress_scores.append(total)

        if stress_scores:
            stress_arr = np.array(stress_scores, dtype=float)
            p10 = float(np.quantile(stress_arr, 0.10))
            median = float(np.median(stress_arr))
            minimum = float(stress_arr.min())
        else:
            p10 = float(baseline_total)
            median = float(baseline_total)
            minimum = float(baseline_total)

        objective = 0.60 * p10 + 0.30 * median + 0.10 * float(baseline_total)

        if objective > best['objective']:
            best = {
                'threshold': float(thresh),
                'objective': float(objective),
                'baseline_score': int(baseline_total),
                'stress_p10': p10,
                'stress_median': median,
                'stress_min': minimum,
                'stress_scores': stress_scores,
            }

    return best


def _fit_human_veto(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """train a conservative human-likeness veto model. returns (model, veto_threshold) using training-only cv to set threshold."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_bot = np.zeros(len(y_train))

    for train_idx, val_idx in skf.split(X_train, y_train):
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
        clf.fit(X_train[train_idx], y_train[train_idx])
        oof_bot[val_idx] = clf.predict_proba(X_train[val_idx])[:, 1]

    bot_human_probs = 1.0 - oof_bot[y_train == 1]
    if len(bot_human_probs) == 0:
        veto_thresh = 1.0
    else:
        veto_thresh = float(np.quantile(bot_human_probs, HUMAN_VETO_QUANTILE))

    final_clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
    final_clf.fit(X_train, y_train)

    return final_clf, veto_thresh


def _build_fr_ensemble(X_train, y_train, X_test):
    """train fr ensemble and return probabilities and rf importances."""
    # multi-seed random forest ensemble
    rf_probas = []
    first_rf = None
    for seed in FR_RF_SEEDS:
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=4,
            min_samples_split=6,
            class_weight={0: 1, 1: 3},
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        rf_probas.append(rf.predict_proba(X_test)[:, 1])
        if first_rf is None:
            first_rf = rf
    rf_proba = np.mean(rf_probas, axis=0)

    # xgboost
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=n_neg / max(n_pos, 1),
        reg_alpha=0.5,
        reg_lambda=2.0,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]

    # gradientboosting
    gbm = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        min_samples_leaf=4,
        min_samples_split=6,
        random_state=42,
    )
    gbm.fit(X_train, y_train)
    gbm_proba = gbm.predict_proba(X_test)[:, 1]

    # 3-model weighted ensemble
    test_proba = (ENSEMBLE_WEIGHTS['rf'] * rf_proba +
                  ENSEMBLE_WEIGHTS['xgb'] * xgb_proba +
                  ENSEMBLE_WEIGHTS['gbm'] * gbm_proba)

    importances = first_rf.feature_importances_ if first_rf is not None else np.zeros(X_train.shape[1])
    return test_proba, importances


def _fit_isotonic_calibrator(X_train, y_train, n_splits=3):
    """fit isotonic calibrator using out-of-fold predictions."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y_train))

    for train_idx, val_idx in skf.split(X_train, y_train):
        proba_val, _ = _build_fr_ensemble(X_train[train_idx], y_train[train_idx], X_train[val_idx])
        oof[val_idx] = proba_val

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(oof, y_train)
    return iso


def optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                       already_flagged: np.ndarray = None) -> float:
    """find the probability threshold that maximizes the competition score. score = 4*tp - 1*fn - 2*fp"""
    best_score = -999999
    best_thresh = 0.5

    for thresh_int in range(15, 90):
        thresh = thresh_int / 100.0
        preds = (y_proba >= thresh).astype(int)

        if already_flagged is not None:
            combined = np.maximum(preds, already_flagged)
        else:
            combined = preds

        tp = int(((combined == 1) & (y_true == 1)).sum())
        fp = int(((combined == 1) & (y_true == 0)).sum())
        fn = int(((combined == 0) & (y_true == 1)).sum())
        score = 4 * tp - 1 * fn - 2 * fp

        if score > best_score:
            best_score = score
            best_thresh = thresh

    return best_thresh


def train_and_detect(train_paths: list, train_bots_paths: list,
                     test_path: str, test_bots_path: str = None,
                     fixed_threshold: float = None) -> dict:
    """train on given datasets and detect bots in test dataset. uses multi-seed rf + xgboost + gbm ensemble."""
    # extract features for all training datasets
    train_dfs = []
    train_meta = []
    for dp, bp in zip(train_paths, train_bots_paths):
        df = extract_features(dp, use_cache=True)
        bots = load_ground_truth(bp)
        df['is_bot'] = df.index.isin(bots).astype(int)
        ds_id = _parse_dataset_id(dp)
        ds_lang = _dataset_lang_from_id(ds_id)
        bot_rate = float(df['is_bot'].mean()) if len(df) else 0.0
        train_meta.append({'id': ds_id, 'lang': ds_lang, 'bot_rate': bot_rate})
        train_dfs.append(df)

    train_df = pd.concat(train_dfs)

    # extract features for test dataset
    test_df = extract_features(test_path, use_cache=True)

    # layer 1: apply rules
    test_rules = apply_rules(test_df)

    # get feature matrix
    feature_cols = [c for c in FR_FEATURES if c in train_df.columns]
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['is_bot'].values
    X_test = test_df[feature_cols].fillna(0).values

    # train ensemble
    test_proba, rf_importances = _build_fr_ensemble(X_train, y_train, X_test)

    # phase 4: calibrate probabilities with isotonic regression
    if CALIBRATE_PROBA:
        calibrator = _fit_isotonic_calibrator(X_train, y_train)
        test_proba = calibrator.transform(test_proba)

    # determine threshold
    if fixed_threshold is not None:
        thresh = fixed_threshold
    elif test_bots_path:
        test_bots = load_ground_truth(test_bots_path)
        test_df_temp = test_df.copy()
        test_df_temp['is_bot'] = test_df_temp.index.isin(test_bots).astype(int)
        y_test = test_df_temp['is_bot'].values
        test_rules_arr = test_rules.values.astype(int)
        thresh = optimize_threshold(y_test, test_proba, test_rules_arr)
    else:
        thresh = 0.40  # conservative default for fr

    # phase 4: risk-aware minimum threshold
    if thresh < MIN_COST_THRESHOLD:
        thresh = MIN_COST_THRESHOLD

    # phase 1: rate clamp (fr only) to prevent fp explosions
    ml_flags = (test_proba >= thresh).astype(int)
    max_rate = None
    if USE_RATE_CLAMP:
        fr_bot_rate = _compute_fr_bot_rate(train_meta)
        max_rate = max(0.25, fr_bot_rate + FR_BOT_RATE_BUFFER)
        clamp_thresh, clamp_flags = _apply_rate_clamp(
            test_proba, test_rules.values.astype(bool), max_rate
        )
        if clamp_thresh > thresh:
            thresh = clamp_thresh
            ml_flags = clamp_flags.astype(int)

    # apply to test
    test_preds = ml_flags

    # phase 1: human-likeness veto on ml-only predictions
    human_probs = None
    veto_thresh = None
    if USE_HUMAN_VETO:
        veto_model, veto_thresh = _fit_human_veto(X_train, y_train)
        human_probs = 1.0 - veto_model.predict_proba(X_test)[:, 1]
        veto_mask = (
            (human_probs >= veto_thresh)
            & (test_preds == 1)
            & (test_proba < VETO_MAX_PROBA)
            & (~test_rules.values.astype(bool))
        )
        test_preds[veto_mask] = 0

    test_combined = np.maximum(test_preds, test_rules.values.astype(int))

    # phase 2: digital dna cluster expansion (seeded by high-confidence bots)
    n_cluster_added = 0
    if USE_DNA_CLUSTER:
        try:
            data = load_dataset(test_path)
            posts_by_author = group_posts_by_author(data['posts'])
            sequences = build_dna_sequences(posts_by_author)
            users, labels = cluster_sequences(
                sequences, eps=DNA_EPS, min_samples=DNA_MIN_SAMPLES, k=DNA_K
            )
            proba_series = pd.Series(test_proba, index=test_df.index)
            seed_thresh = max(0.6, thresh + DNA_SEED_PROBA_MARGIN)
            seed_mask = {
                uid: bool(test_rules.loc[uid]) or (proba_series.loc[uid] >= seed_thresh)
                for uid in test_df.index
            }
            expanded = expand_from_clusters(
                users, labels, seed_mask,
                min_seed=DNA_MIN_SEED,
                min_seed_ratio=DNA_MIN_SEED_RATIO,
                max_cluster_size=DNA_MAX_CLUSTER_SIZE,
            )
            if expanded:
                cluster_flags = test_df.index.isin(expanded).astype(int)
                test_combined = np.maximum(test_combined, cluster_flags)
                n_cluster_added = int(cluster_flags.sum())
        except Exception:
            n_cluster_added = 0

    detections = list(test_df.index[test_combined == 1])

    result = {
        'detections': detections,
        'threshold': thresh,
        'n_rule_flagged': int(test_rules.sum()),
        'n_ml_flagged': int(test_preds.sum()),
        'n_cluster_flagged': int(n_cluster_added),
        'n_total_flagged': len(detections),
        'probabilities': dict(zip(test_df.index.tolist(), test_proba.tolist())),
        'human_probabilities': (
            dict(zip(test_df.index.tolist(), human_probs.tolist()))
            if human_probs is not None else None
        ),
        'human_veto_threshold': veto_thresh,
        'max_rate': max_rate,
        'feature_importances': dict(zip(feature_cols,
                                        rf_importances.tolist())),
    }

    # if ground truth available, compute metrics
    if test_bots_path:
        test_bots = load_ground_truth(test_bots_path)
        tp = sum(1 for uid in detections if uid in test_bots)
        fp = sum(1 for uid in detections if uid not in test_bots)
        fn = len(test_bots) - tp
        score = 4 * tp - 1 * fn - 2 * fp
        max_score = 4 * len(test_bots)

        result.update({
            'tp': tp, 'fp': fp, 'fn': fn,
            'score': score, 'max_score': max_score,
            'efficiency': score / max_score if max_score > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        })

    return result


def cross_validate_fr(use_robust: bool = ROBUST_THRESHOLD_FR,
                      n_scenarios: int = ROBUST_SCENARIOS_FR,
                      seed: int = ROBUST_SEED_FR) -> dict:
    """cross-validate using cross-language training: - test on ds33: train on ds31 + ds30 + ds32 - test on ds31: train on ds33 + ds30 + ds32 sweeps thresholds to find the best fixed threshold for combined score."""
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    # for each fr test set, train on all other datasets (including en)
    # test on ds33: train on ds31, ds30, ds32
    r1 = train_and_detect(
        [os.path.join(base, 'dataset.posts&users.31.json'),
         os.path.join(base, 'dataset.posts&users.30.json'),
         os.path.join(base, 'dataset.posts&users.32.json')],
        [os.path.join(base, 'dataset.bots.31.txt'),
         os.path.join(base, 'dataset.bots.30.txt'),
         os.path.join(base, 'dataset.bots.32.txt')],
        os.path.join(base, 'dataset.posts&users.33.json'),
        os.path.join(base, 'dataset.bots.33.txt'),
    )

    # test on ds31: train on ds33, ds30, ds32
    r2 = train_and_detect(
        [os.path.join(base, 'dataset.posts&users.33.json'),
         os.path.join(base, 'dataset.posts&users.30.json'),
         os.path.join(base, 'dataset.posts&users.32.json')],
        [os.path.join(base, 'dataset.bots.33.txt'),
         os.path.join(base, 'dataset.bots.30.txt'),
         os.path.join(base, 'dataset.bots.32.txt')],
        os.path.join(base, 'dataset.posts&users.31.json'),
        os.path.join(base, 'dataset.bots.31.txt'),
    )

    max_rate_33 = r1.get('max_rate')
    if max_rate_33 is None:
        max_rate_33 = _estimate_fr_max_rate(
            [os.path.join(base, 'dataset.posts&users.31.json'),
             os.path.join(base, 'dataset.posts&users.30.json'),
             os.path.join(base, 'dataset.posts&users.32.json')],
            [os.path.join(base, 'dataset.bots.31.txt'),
             os.path.join(base, 'dataset.bots.30.txt'),
             os.path.join(base, 'dataset.bots.32.txt')],
        )

    max_rate_31 = r2.get('max_rate')
    if max_rate_31 is None:
        max_rate_31 = _estimate_fr_max_rate(
            [os.path.join(base, 'dataset.posts&users.33.json'),
             os.path.join(base, 'dataset.posts&users.30.json'),
             os.path.join(base, 'dataset.posts&users.32.json')],
            [os.path.join(base, 'dataset.bots.33.txt'),
             os.path.join(base, 'dataset.bots.30.txt'),
             os.path.join(base, 'dataset.bots.32.txt')],
        )

    fold_33 = _prepare_fr_fold(
        os.path.join(base, 'dataset.posts&users.33.json'),
        os.path.join(base, 'dataset.bots.33.txt'),
        r1['probabilities'],
        max_rate=max_rate_33,
        human_probabilities=r1.get('human_probabilities'),
        veto_threshold=r1.get('human_veto_threshold'),
    )
    fold_31 = _prepare_fr_fold(
        os.path.join(base, 'dataset.posts&users.31.json'),
        os.path.join(base, 'dataset.bots.31.txt'),
        r2['probabilities'],
        max_rate=max_rate_31,
        human_probabilities=r2.get('human_probabilities'),
        veto_threshold=r2.get('human_veto_threshold'),
    )
    folds = [fold_33, fold_31]

    sweep = np.arange(MIN_COST_THRESHOLD, 0.701, 0.005)

    baseline_best = {
        'threshold': float(sweep[0]),
        'combined_score': -999999,
    }
    for thresh in sweep:
        total = 0
        for fold in folds:
            total += _score_fold_at_threshold(
                fold['proba'], fold['y_true'], fold['rules'],
                float(thresh), fold['max_rate'],
                human_probs=fold.get('human_probs'),
                veto_threshold=fold.get('veto_threshold'),
            )
        if total > baseline_best['combined_score']:
            baseline_best = {'threshold': float(thresh), 'combined_score': int(total)}

    robust_summary = None
    selected_threshold = baseline_best['threshold']
    selected_score = baseline_best['combined_score']
    mode = 'baseline'

    if use_robust:
        robust_summary = _select_robust_threshold(
            folds, sweep=sweep, n_scenarios=n_scenarios, seed=seed
        )
        robust_score = int(robust_summary['baseline_score'])
        if robust_score >= baseline_best['combined_score'] - ROBUST_MAX_DROP_FR:
            selected_threshold = robust_summary['threshold']
            selected_score = robust_score
            mode = 'robust'
        else:
            mode = 'baseline_fallback'

    return {
        'train_all_test33': r1,
        'train_all_test31': r2,
        'avg_threshold': selected_threshold,
        'combined_score_at_fixed': selected_score,
        'threshold_mode': mode,
        'baseline_threshold': baseline_best['threshold'],
        'baseline_combined_score': baseline_best['combined_score'],
        'robust_summary': robust_summary,
    }


def _evaluate_at_threshold(dataset_path: str, bots_path: str,
                           probabilities: dict, threshold: float) -> int:
    """evaluate score at a given threshold, accounting for rules."""
    test_df = extract_features(dataset_path, use_cache=True)
    test_bots = load_ground_truth(bots_path)

    rules = apply_rules(test_df)

    proba = np.array([probabilities.get(uid, 0.0) for uid in test_df.index])
    preds = (proba >= threshold).astype(int)
    combined = np.maximum(preds, rules.values.astype(int))

    y_true = np.array(test_df.index.isin(test_bots).astype(int))
    tp = int(((combined == 1) & (y_true == 1)).sum())
    fp = int(((combined == 1) & (y_true == 0)).sum())
    fn = int(((combined == 0) & (y_true == 1)).sum())
    return 4 * tp - 1 * fn - 2 * fp


def detect_bots(dataset_path: str) -> list:
    """main entry point: detect bots in a given fr dataset. uses cross-validated threshold with cross-language training."""
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    cv = cross_validate_fr()
    avg_thresh = cv['avg_threshold']

    practice_fr = {
        'dataset.posts&users.31.json': 'dataset.bots.31.txt',
        'dataset.posts&users.33.json': 'dataset.bots.33.txt',
    }

    ds_name = os.path.basename(dataset_path)

    # build training set with fr-first ordering for stable behavior.
    if ds_name == 'dataset.posts&users.31.json':
        ordered = [
            'dataset.posts&users.33.json',
            'dataset.posts&users.30.json',
            'dataset.posts&users.32.json',
        ]
    elif ds_name == 'dataset.posts&users.33.json':
        ordered = [
            'dataset.posts&users.31.json',
            'dataset.posts&users.30.json',
            'dataset.posts&users.32.json',
        ]
    else:
        ordered = [
            'dataset.posts&users.31.json',
            'dataset.posts&users.33.json',
            'dataset.posts&users.30.json',
            'dataset.posts&users.32.json',
        ]

    train_paths = [os.path.join(base, k) for k in ordered if k != ds_name]
    train_bots = [os.path.join(base, ALL_PRACTICE[k]) for k in ordered if k != ds_name]

    test_bots = None
    if ds_name in practice_fr:
        test_bots = os.path.join(base, practice_fr[ds_name])

    result = train_and_detect(
        train_paths, train_bots, dataset_path, test_bots,
        fixed_threshold=avg_thresh,
    )

    if test_bots:
        print(f"FR Detection: {result['n_total_flagged']} flagged | "
              f"TP={result['tp']} FP={result['fp']} FN={result['fn']}")
        print(f"Score: {result['score']}/{result['max_score']} "
              f"({result['efficiency']:.1%} efficiency)")
        print(f"Threshold: {avg_thresh:.3f}")

    return result['detections']


def detect_bots_final(dataset_path: str) -> list:
    """for final submission: train on all practice data with cross-validated threshold."""
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    cv = cross_validate_fr()
    avg_thresh = cv['avg_threshold']

    ordered = [
        'dataset.posts&users.31.json',
        'dataset.posts&users.33.json',
        'dataset.posts&users.30.json',
        'dataset.posts&users.32.json',
    ]
    train_paths = [os.path.join(base, k) for k in ordered]
    train_bots = [os.path.join(base, ALL_PRACTICE[k]) for k in ordered]

    result = train_and_detect(
        train_paths, train_bots, dataset_path,
        fixed_threshold=avg_thresh,
    )

    return result['detections']


if __name__ == '__main__':
    print("=== FR Detector Cross-Validation (Cross-Language Training) ===\n")
    cv = cross_validate_fr()

    for name, r in [('Train DS31+EN -> Test DS33', cv['train_all_test33']),
                     ('Train DS33+EN -> Test DS31', cv['train_all_test31'])]:
        print(f"{name}:")
        print(f"  Optimal threshold: {r['threshold']:.3f}")
        print(f"  Flagged: {r['n_total_flagged']} (rules: {r['n_rule_flagged']}, ML: {r['n_ml_flagged']})")
        if 'tp' in r:
            print(f"  TP={r['tp']} FP={r['fp']} FN={r['fn']}")
            print(f"  Score: {r['score']}/{r['max_score']} ({r['efficiency']:.1%})")
            print(f"  Precision: {r['precision']:.3f} Recall: {r['recall']:.3f}")
        top_feats = sorted(r['feature_importances'].items(), key=lambda x: -x[1])[:10]
        print(f"  Top features: {[(f, round(v, 3)) for f, v in top_feats]}")
        print()

    print(f"Threshold mode: {cv.get('threshold_mode', 'baseline')}")
    print(f"Selected threshold: {cv['avg_threshold']:.3f}")
    print(f"Combined score at selected threshold: {cv['combined_score_at_fixed']}")
    print(f"Baseline best threshold: {cv.get('baseline_threshold', cv['avg_threshold']):.3f}")
    print(f"Baseline best combined score: {cv.get('baseline_combined_score', cv['combined_score_at_fixed'])}")

    robust = cv.get('robust_summary')
    if robust:
        print(f"Stress p10: {robust['stress_p10']:.1f} | "
              f"median: {robust['stress_median']:.1f} | "
              f"min: {robust['stress_min']:.1f} | "
              f"objective: {robust['objective']:.2f}")

    # evaluate at fixed threshold
    print("\n=== Evaluation at fixed threshold ===")
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    fixed = cv['avg_threshold']

    for ds in ['31', '33']:
        ds_path = os.path.join(base, f'dataset.posts&users.{ds}.json')
        bots_path = os.path.join(base, f'dataset.bots.{ds}.txt')

        # use the same fr-first ordering used by detect_bots().
        if ds == '31':
            ordered = [
                'dataset.posts&users.33.json',
                'dataset.posts&users.30.json',
                'dataset.posts&users.32.json',
            ]
        else:
            ordered = [
                'dataset.posts&users.31.json',
                'dataset.posts&users.30.json',
                'dataset.posts&users.32.json',
            ]
        train_p = [os.path.join(base, k) for k in ordered]
        train_b = [os.path.join(base, ALL_PRACTICE[k]) for k in ordered]

        r = train_and_detect(train_p, train_b, ds_path, bots_path,
                             fixed_threshold=fixed)
        print(f"DS{ds} at thresh={fixed:.3f}: TP={r['tp']} FP={r['fp']} FN={r['fn']} "
              f"Score={r['score']}/{r['max_score']} ({r['efficiency']:.1%})")
