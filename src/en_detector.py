"""english bot detector for bot or not challenge. hybrid approach: high-confidence rules + multi-model ensemble classifier."""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features, load_ground_truth, get_feature_columns, load_dataset, group_posts_by_author
from dna_cluster import build_dna_sequences, cluster_sequences, expand_from_clusters

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ensemble config
RF_SEEDS = [0, 1, 7, 13, 42, 99, 123, 456]
ENSEMBLE_WEIGHTS = {'rf': 0.6, 'gbm': 0.2, 'xgb': 0.2}

# phase 4: calibration + risk-aware decision (env-tunable)
CALIBRATE_PROBA = os.getenv('BOTORN0T_CALIBRATE', '0') != '0'
MIN_COST_THRESHOLD = float(os.getenv('BOTORN0T_MIN_COST_THRESHOLD', str(2.0 / 7.0)))
ROBUST_THRESHOLD = os.getenv('BOTORN0T_ROBUST_THRESHOLD', '1') != '0'
ROBUST_SCENARIOS = int(os.getenv('BOTORN0T_ROBUST_SCENARIOS', '400'))
ROBUST_SEED = int(os.getenv('BOTORN0T_ROBUST_SEED', '42'))

# feature toggles (env-driven)
USE_DNA_CLUSTER = os.getenv('BOTORN0T_USE_DNA_CLUSTER', '0') == '1'
USE_COMPRESSION_FEATURES = os.getenv('BOTORN0T_USE_COMPRESSION', '1') != '0'
DNA_EPS = 0.30
DNA_MIN_SAMPLES = 3
DNA_K = 3
DNA_MIN_SEED = 3
DNA_MIN_SEED_RATIO = 0.7
DNA_MAX_CLUSTER_SIZE = 15
DNA_SEED_PROBA_MARGIN = 0.20

# all practice datasets (en + fr) for cross-language training
ALL_PRACTICE = {
    'dataset.posts&users.30.json': 'dataset.bots.30.txt',  # en
    'dataset.posts&users.31.json': 'dataset.bots.31.txt',  # fr
    'dataset.posts&users.32.json': 'dataset.bots.32.txt',  # en
    'dataset.posts&users.33.json': 'dataset.bots.33.txt',  # fr
}


# features used for en model
# pruned weak features (auc < 0.56): night_ratio, has_description, username_len, username_entropy, mean_interval
# phase 1b: added template features (bigram_repetition_rate, unique_opening_ratio) + punctuation_consistency (0.78 auc)
COMPRESSION_FEATURES = [
    'compression_ratio',
    'char_3gram_entropy',
    'char_3gram_unique_ratio',
]

EN_FEATURES = [
    # temporal features
    'hour_entropy', 'active_hours', 'peak_hour_frac',
    'cv_intervals', 'burstiness', 'std_interval',
    'max_interval', 'min_interval',
    'within_window_ratio', 'window_coverage', 'edge_activity_ratio',
    # text features
    'excl_rate', 'hashtag_rate', 'question_rate',
    'avg_tweet_len', 'tweet_len_std', 'emoji_rate',
    'ttr', 'mid_hashtag_rate', 'url_rate', 'mention_rate',
    'self_similarity',
    'comma_rate',
    # template & stylometric features (phase 1b)
    'punctuation_consistency',  # 0.78 auc - strong llm uniformity detector
    'bigram_repetition_rate',   # 0.61 auc - template phrase detection
    'unique_opening_ratio',     # 0.63 auc - template opening detection
    # account features
    'username_digit_ratio', 'description_length',
    'tweet_count_ratio', 'num_tweets_ratio', 'z_score_delta',
    # topic & activity
    'topic_diversity', 'tweet_count', 'z_score', 'num_tweets',
    # batch injection features
    'max_same_second', 'unique_second_ratio', 'timestamp_collision_rate',
]

if USE_COMPRESSION_FEATURES:
    EN_FEATURES.extend(COMPRESSION_FEATURES)


def apply_rules(df: pd.DataFrame) -> pd.Series:
    """layer 1: high-confidence rules that catch obvious bots with near-zero fp. returns a boolean series: true = flagged as bot by rules."""
    flagged = pd.Series(False, index=df.index)

    # rule 1: out-of-range tweets (perfect precision for en)
    if 'out_of_range' in df.columns:
        flagged |= (df['out_of_range'] == 1)

    # rule 2: very high hashtag rate + posting around the clock
    flagged |= (df['hashtag_rate'] > 1.5) & (df['active_hours'] >= 20)

    # rule 3: batch injection -- multiple tweets at exact same second
    # bots inject 4-9 tweets at identical timestamps; humans rarely exceed 2.
    if 'max_same_second' in df.columns:
        flagged |= (df['max_same_second'] >= 4)

    # rule 4: extreme question rate (every tweet is a question)
    # catches character bots that exclusively ask questions.
    flagged |= (df['question_rate'] >= 0.95) & (df['num_tweets'] >= 8)

    return flagged


def apply_rules_no_out_of_range(df: pd.DataFrame) -> pd.Series:
    """rules 2-4 only, used for stress-testing robustness when out_of_range weakens."""
    flagged = pd.Series(False, index=df.index)

    # rule 2: very high hashtag rate + posting around the clock
    flagged |= (df['hashtag_rate'] > 1.5) & (df['active_hours'] >= 20)

    # rule 3: batch injection -- multiple tweets at exact same second
    if 'max_same_second' in df.columns:
        flagged |= (df['max_same_second'] >= 4)

    # rule 4: extreme question rate (every tweet is a question)
    flagged |= (df['question_rate'] >= 0.95) & (df['num_tweets'] >= 8)

    return flagged


def _score_from_predictions(preds: np.ndarray, y_true: np.ndarray) -> int:
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    return 4 * tp - 1 * fn - 2 * fp


def _score_fold_at_threshold(proba: np.ndarray, y_true: np.ndarray,
                             rules: np.ndarray, thresh: float) -> int:
    effective = max(float(thresh), MIN_COST_THRESHOLD)
    ml_preds = (proba >= effective).astype(int)
    combined = np.maximum(ml_preds, rules.astype(int))
    return _score_from_predictions(combined, y_true)


def _prepare_en_fold(dataset_path: str, bots_path: str, probabilities: dict) -> dict:
    """collect fold arrays for threshold selection and stress scenarios."""
    df = extract_features(dataset_path, use_cache=True)
    bots = load_ground_truth(bots_path)

    y_true = np.array(df.index.isin(bots).astype(int))
    proba = np.array([probabilities.get(uid, 0.0) for uid in df.index])
    rules_all = apply_rules(df).values.astype(int)
    rules_no_oor = apply_rules_no_out_of_range(df).values.astype(int)

    if 'out_of_range' in df.columns:
        out_of_range = (df['out_of_range'] == 1).values.astype(int)
    else:
        out_of_range = np.zeros(len(df), dtype=int)

    botlike_human = np.zeros(len(df), dtype=int)
    human_idx = np.where(y_true == 0)[0]
    if len(human_idx) > 0:
        cutoff = float(np.quantile(proba[human_idx], 0.75))
        botlike_human[human_idx] = (proba[human_idx] >= cutoff).astype(int)

    return {
        'dataset_path': dataset_path,
        'y_true': y_true,
        'proba': proba,
        'rules_all': rules_all,
        'rules_no_oor': rules_no_oor,
        'out_of_range': out_of_range,
        'botlike_human': botlike_human,
    }


def _build_stress_samples(fold: dict, n_scenarios: int, seed: int) -> list:
    """build pseudo-final samples that perturb prevalence, out_of_range reliability, and hard-negative density."""
    rng = np.random.RandomState(seed)
    y_true = fold['y_true']
    n = len(y_true)

    bot_idx = np.where(y_true == 1)[0]
    human_idx = np.where(y_true == 0)[0]
    if len(bot_idx) == 0 or len(human_idx) == 0:
        return []

    out_of_range = fold['out_of_range']
    botlike_human = fold['botlike_human']

    samples = []
    for _ in range(n_scenarios):
        target_bot_rate = float(rng.uniform(0.16, 0.30))
        oor_reliability = float(rng.uniform(0.35, 1.00))
        hard_negative_boost = float(rng.uniform(1.00, 3.00))

        n_bot = int(round(target_bot_rate * n))
        n_bot = max(1, min(n - 1, n_bot))
        n_human = n - n_bot

        bot_weights = np.ones(len(bot_idx), dtype=float)
        bot_weights += (1.0 - oor_reliability) * out_of_range[bot_idx].astype(float) * 2.0
        bot_weights /= bot_weights.sum()

        human_weights = np.ones(len(human_idx), dtype=float)
        human_weights += (hard_negative_boost - 1.0) * botlike_human[human_idx].astype(float)
        human_weights /= human_weights.sum()

        sampled_bots = rng.choice(bot_idx, size=n_bot, replace=True, p=bot_weights)
        sampled_humans = rng.choice(human_idx, size=n_human, replace=True, p=human_weights)
        idx = np.concatenate([sampled_bots, sampled_humans])
        rng.shuffle(idx)

        # degrade out_of_range reliability for sampled users
        rules_sample = fold['rules_all'][idx].copy()
        rules_no_oor_sample = fold['rules_no_oor'][idx]
        oor_mask = fold['out_of_range'][idx].astype(bool)
        if oor_mask.any():
            keep_mask = rng.rand(int(oor_mask.sum())) < oor_reliability
            drop_pos = np.where(oor_mask)[0][~keep_mask]
            if len(drop_pos) > 0:
                rules_sample[drop_pos] = rules_no_oor_sample[drop_pos]

        samples.append({
            'idx': idx,
            'rules': rules_sample.astype(int),
        })

    return samples


def _select_robust_threshold(folds: list, sweep: np.ndarray,
                             n_scenarios: int, seed: int) -> dict:
    """choose threshold using stress scenarios. objective prioritizes worst-case resilience while keeping baseline score."""
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
                fold['proba'], fold['y_true'], fold['rules_all'], float(thresh)
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
                total += _score_fold_at_threshold(p_s, y_s, r_s, float(thresh))
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


def optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                       already_flagged: np.ndarray = None) -> float:
    """find the probability threshold that maximizes the competition score. score = 4*tp - 1*fn - 2*fp if already_flagged is provided, those users are counted as tp/fp already and excluded from threshold optimization."""
    best_score = -999999
    best_thresh = 0.5

    for thresh_int in range(10, 96):
        thresh = thresh_int / 100.0
        preds = (y_proba >= thresh).astype(int)

        # if we have rule-flagged users, combine predictions
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


def _build_ensemble(X_train, y_train, X_test):
    """build multi-model ensemble: multi-seed rf + gbm + xgboost. returns weighted average of predicted probabilities."""
    # multi-seed random forest ensemble
    rf_proba = np.zeros(X_test.shape[0])
    rf_importances = np.zeros(X_train.shape[1])
    for seed in RF_SEEDS:
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=3,
            class_weight={0: 1, 1: 1},
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        rf_proba += rf.predict_proba(X_test)[:, 1]
        rf_importances += rf.feature_importances_
    rf_proba /= len(RF_SEEDS)
    rf_importances /= len(RF_SEEDS)

    # gradient boosting
    gbm = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42,
    )
    gbm.fit(X_train, y_train)
    gbm_proba = gbm.predict_proba(X_test)[:, 1]

    # xgboost (if available, otherwise increase rf weight)
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            scale_pos_weight=2, random_state=42, eval_metric='logloss',
        )
        xgb.fit(X_train, y_train)
        xgb_proba = xgb.predict_proba(X_test)[:, 1]
        ensemble_proba = (ENSEMBLE_WEIGHTS['rf'] * rf_proba +
                          ENSEMBLE_WEIGHTS['gbm'] * gbm_proba +
                          ENSEMBLE_WEIGHTS['xgb'] * xgb_proba)
    else:
        ensemble_proba = 0.9 * rf_proba + 0.1 * gbm_proba

    return ensemble_proba, rf_importances


def _fit_isotonic_calibrator(X_train, y_train, n_splits=3):
    """fit isotonic calibrator using out-of-fold predictions."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y_train))

    for train_idx, val_idx in skf.split(X_train, y_train):
        proba_val, _ = _build_ensemble(X_train[train_idx], y_train[train_idx], X_train[val_idx])
        oof[val_idx] = proba_val

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(oof, y_train)
    return iso


def train_and_detect(train_paths: list, train_bots_paths: list,
                     test_path: str, test_bots_path: str = None,
                     fixed_threshold: float = None) -> dict:
    """train on given datasets and detect bots in test dataset. if test_bots_path is provided and no fixed_threshold, optimizes threshold on test set."""
    # extract features for all training datasets
    train_dfs = []
    for dp, bp in zip(train_paths, train_bots_paths):
        df = extract_features(dp, use_cache=True)
        bots = load_ground_truth(bp)
        df['is_bot'] = df.index.isin(bots).astype(int)
        train_dfs.append(df)

    train_df = pd.concat(train_dfs)

    # extract features for test dataset
    test_df = extract_features(test_path, use_cache=True)

    # layer 1: apply rules
    test_rules = apply_rules(test_df)

    # get feature matrix
    feature_cols = [c for c in EN_FEATURES if c in train_df.columns]
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['is_bot'].values
    X_test = test_df[feature_cols].fillna(0).values

    # train ensemble
    test_proba, rf_importances = _build_ensemble(X_train, y_train, X_test)

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
        thresh = 0.30  # cross-validated optimal for combined score with cross-language training

    # phase 4: risk-aware minimum threshold
    if thresh < MIN_COST_THRESHOLD:
        thresh = MIN_COST_THRESHOLD

    # apply to test
    test_preds = (test_proba >= thresh).astype(int)
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


def cross_validate_en(use_robust: bool = ROBUST_THRESHOLD,
                      n_scenarios: int = ROBUST_SCENARIOS,
                      seed: int = ROBUST_SEED) -> dict:
    """cross-validate using cross-language training: - test on ds32: train on ds30 + ds31 + ds33 (en + fr) - test on ds30: train on ds32 + ds31 + ds33 (en + fr) finds the best fixed threshold that maximizes combined score."""
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    # test on ds32: train on ds30, ds31, ds33
    r1 = train_and_detect(
        [os.path.join(base, 'dataset.posts&users.30.json'),
         os.path.join(base, 'dataset.posts&users.31.json'),
         os.path.join(base, 'dataset.posts&users.33.json')],
        [os.path.join(base, 'dataset.bots.30.txt'),
         os.path.join(base, 'dataset.bots.31.txt'),
         os.path.join(base, 'dataset.bots.33.txt')],
        os.path.join(base, 'dataset.posts&users.32.json'),
        os.path.join(base, 'dataset.bots.32.txt'),
    )

    # test on ds30: train on ds32, ds31, ds33
    r2 = train_and_detect(
        [os.path.join(base, 'dataset.posts&users.32.json'),
         os.path.join(base, 'dataset.posts&users.31.json'),
         os.path.join(base, 'dataset.posts&users.33.json')],
        [os.path.join(base, 'dataset.bots.32.txt'),
         os.path.join(base, 'dataset.bots.31.txt'),
         os.path.join(base, 'dataset.bots.33.txt')],
        os.path.join(base, 'dataset.posts&users.30.json'),
        os.path.join(base, 'dataset.bots.30.txt'),
    )

    fold_32 = _prepare_en_fold(
        os.path.join(base, 'dataset.posts&users.32.json'),
        os.path.join(base, 'dataset.bots.32.txt'),
        r1['probabilities'],
    )
    fold_30 = _prepare_en_fold(
        os.path.join(base, 'dataset.posts&users.30.json'),
        os.path.join(base, 'dataset.bots.30.txt'),
        r2['probabilities'],
    )
    folds = [fold_32, fold_30]

    sweep = np.arange(MIN_COST_THRESHOLD, 0.551, 0.005)

    # baseline threshold: maximize combined en score on fold predictions.
    baseline_best = {
        'threshold': float(sweep[0]),
        'combined_score': -999999,
    }
    for thresh in sweep:
        total = 0
        for fold in folds:
            total += _score_fold_at_threshold(
                fold['proba'], fold['y_true'], fold['rules_all'], float(thresh)
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
        selected_threshold = robust_summary['threshold']
        selected_score = robust_summary['baseline_score']
        mode = 'robust'

    return {
        'train30_31_33_test32': r1,
        'train32_31_33_test30': r2,
        'best_threshold': selected_threshold,
        'combined_score': selected_score,
        'threshold_mode': mode,
        'baseline_threshold': baseline_best['threshold'],
        'baseline_combined_score': baseline_best['combined_score'],
        'robust_summary': robust_summary,
    }


def detect_bots(dataset_path: str) -> list:
    """main entry point: detect bots in a given en dataset. uses cross-validated threshold, trains on available practice data."""
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    # cross-validate to get optimal threshold
    cv = cross_validate_en()
    best_thresh = cv['best_threshold']

    # determine if dataset is one of the practice datasets
    ds_name = os.path.basename(dataset_path)

    if ds_name in ALL_PRACTICE:
        # for practice data: train on all other datasets (cross-language)
        other_datasets = [k for k in ALL_PRACTICE if k != ds_name]
        train_paths = [os.path.join(base, k) for k in other_datasets]
        train_bots = [os.path.join(base, ALL_PRACTICE[k]) for k in other_datasets]
        test_bots = os.path.join(base, ALL_PRACTICE[ds_name])
    else:
        # new data: train on all practice data (en + fr)
        train_paths = [os.path.join(base, k) for k in ALL_PRACTICE]
        train_bots = [os.path.join(base, ALL_PRACTICE[k]) for k in ALL_PRACTICE]
        test_bots = None

    result = train_and_detect(
        train_paths, train_bots, dataset_path, test_bots,
        fixed_threshold=best_thresh,
    )

    if test_bots:
        print(f"EN Detection: {result['n_total_flagged']} flagged | "
              f"TP={result['tp']} FP={result['fp']} FN={result['fn']}")
        print(f"Score: {result['score']}/{result['max_score']} "
              f"({result['efficiency']:.1%} efficiency)")
        print(f"Threshold: {best_thresh:.3f}")

    return result['detections']


def detect_bots_final(dataset_path: str) -> list:
    """for final submission: train on all practice data (en + fr) with cross-validated threshold."""
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    # cross-validate to get threshold
    cv = cross_validate_en()
    best_thresh = cv['best_threshold']

    # train on all practice data (en + fr)
    train_paths = [os.path.join(base, k) for k in ALL_PRACTICE]
    train_bots = [os.path.join(base, ALL_PRACTICE[k]) for k in ALL_PRACTICE]

    result = train_and_detect(
        train_paths, train_bots, dataset_path,
        fixed_threshold=best_thresh,
    )

    return result['detections']


if __name__ == '__main__':
    print("=== EN Detector Cross-Validation (Cross-Language Training) ===\n")
    cv = cross_validate_en()

    for name, r in [('Train DS30+31+33 -> Test DS32', cv['train30_31_33_test32']),
                     ('Train DS32+31+33 -> Test DS30', cv['train32_31_33_test30'])]:
        print(f"{name}:")
        print(f"  Per-fold threshold: {r['threshold']:.3f}")
        print(f"  Flagged: {r['n_total_flagged']} (rules: {r['n_rule_flagged']}, ML: {r['n_ml_flagged']})")
        if 'tp' in r:
            print(f"  TP={r['tp']} FP={r['fp']} FN={r['fn']}")
            print(f"  Score: {r['score']}/{r['max_score']} ({r['efficiency']:.1%})")
            print(f"  Precision: {r['precision']:.3f} Recall: {r['recall']:.3f}")
        top_feats = sorted(r['feature_importances'].items(), key=lambda x: -x[1])[:10]
        print(f"  Top features: {[(f, round(v, 3)) for f, v in top_feats]}")
        print()

    print(f"Threshold mode: {cv.get('threshold_mode', 'baseline')}")
    print(f"Selected threshold: {cv['best_threshold']:.3f}")
    print(f"Combined EN score at selected threshold: {cv['combined_score']}")
    print(f"Baseline best threshold: {cv.get('baseline_threshold', cv['best_threshold']):.3f}")
    print(f"Baseline best combined EN score: {cv.get('baseline_combined_score', cv['combined_score'])}")

    robust = cv.get('robust_summary')
    if robust:
        print(f"Stress p10: {robust['stress_p10']:.1f} | "
              f"median: {robust['stress_median']:.1f} | "
              f"min: {robust['stress_min']:.1f} | "
              f"objective: {robust['objective']:.2f}")
