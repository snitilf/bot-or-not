"""phase 2: tune asymmetric loss parameters to reduce false positives. tests different xgboost scale_pos_weight and ensemble weight combinations."""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features, load_ground_truth
from en_detector import apply_rules, EN_FEATURES, RF_SEEDS

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print('warning: xgboost not available')

def evaluate_config(xgb_scale_pos_weight, ensemble_weights):
    """test a configuration and return combined score."""
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    # fold 1: train on ds30+31+33, test on ds32
    train_dfs = []
    for dp, bp in [
        ('dataset.posts&users.30.json', 'dataset.bots.30.txt'),
        ('dataset.posts&users.31.json', 'dataset.bots.31.txt'),
        ('dataset.posts&users.33.json', 'dataset.bots.33.txt'),
    ]:
        df = extract_features(os.path.join(base, dp), use_cache=True)
        bots = load_ground_truth(os.path.join(base, bp))
        df['is_bot'] = df.index.isin(bots).astype(int)
        train_dfs.append(df)

    train_df = pd.concat(train_dfs)
    test_df = extract_features(os.path.join(base, 'dataset.posts&users.32.json'), use_cache=True)
    test_bots = load_ground_truth(os.path.join(base, 'dataset.bots.32.txt'))
    test_df['is_bot'] = test_df.index.isin(test_bots).astype(int)

    # apply rules
    test_rules = apply_rules(test_df)

    # feature matrix
    feature_cols = [c for c in EN_FEATURES if c in train_df.columns]
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['is_bot'].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df['is_bot'].values

    # multi-seed rf
    rf_proba = np.zeros(X_test.shape[0])
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
    rf_proba /= len(RF_SEEDS)

    # gbm
    gbm = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42,
    )
    gbm.fit(X_train, y_train)
    gbm_proba = gbm.predict_proba(X_test)[:, 1]

    # xgboost with tuned scale_pos_weight
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            scale_pos_weight=xgb_scale_pos_weight, random_state=42, eval_metric='logloss',
        )
        xgb.fit(X_train, y_train)
        xgb_proba = xgb.predict_proba(X_test)[:, 1]
        ensemble_proba = (ensemble_weights[0] * rf_proba +
                          ensemble_weights[1] * gbm_proba +
                          ensemble_weights[2] * xgb_proba)
    else:
        ensemble_proba = 0.9 * rf_proba + 0.1 * gbm_proba

    # find best threshold for this fold
    best_score_32 = -999
    best_thresh_32 = 0.30
    test_rules_arr = test_rules.values.astype(int)

    for thresh_int in range(20, 55):
        thresh = thresh_int / 100.0
        preds = (ensemble_proba >= thresh).astype(int)
        combined = np.maximum(preds, test_rules_arr)

        tp = int(((combined == 1) & (y_test == 1)).sum())
        fp = int(((combined == 1) & (y_test == 0)).sum())
        fn = int(((combined == 0) & (y_test == 1)).sum())
        score = 4 * tp - 1 * fn - 2 * fp

        if score > best_score_32:
            best_score_32 = score
            best_thresh_32 = thresh

    # fold 2: train on ds32+31+33, test on ds30 (abbreviated - use same logic)
    train_dfs = []
    for dp, bp in [
        ('dataset.posts&users.32.json', 'dataset.bots.32.txt'),
        ('dataset.posts&users.31.json', 'dataset.bots.31.txt'),
        ('dataset.posts&users.33.json', 'dataset.bots.33.txt'),
    ]:
        df = extract_features(os.path.join(base, dp), use_cache=True)
        bots = load_ground_truth(os.path.join(base, bp))
        df['is_bot'] = df.index.isin(bots).astype(int)
        train_dfs.append(df)

    train_df = pd.concat(train_dfs)
    test_df = extract_features(os.path.join(base, 'dataset.posts&users.30.json'), use_cache=True)
    test_bots = load_ground_truth(os.path.join(base, 'dataset.bots.30.txt'))
    test_df['is_bot'] = test_df.index.isin(test_bots).astype(int)

    test_rules = apply_rules(test_df)

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['is_bot'].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df['is_bot'].values

    rf_proba = np.zeros(X_test.shape[0])
    for seed in RF_SEEDS:
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=7, min_samples_leaf=3,
            class_weight={0: 1, 1: 1}, random_state=seed, n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        rf_proba += rf.predict_proba(X_test)[:, 1]
    rf_proba /= len(RF_SEEDS)

    gbm = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42,
    )
    gbm.fit(X_train, y_train)
    gbm_proba = gbm.predict_proba(X_test)[:, 1]

    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            scale_pos_weight=xgb_scale_pos_weight, random_state=42, eval_metric='logloss',
        )
        xgb.fit(X_train, y_train)
        xgb_proba = xgb.predict_proba(X_test)[:, 1]
        ensemble_proba = (ensemble_weights[0] * rf_proba +
                          ensemble_weights[1] * gbm_proba +
                          ensemble_weights[2] * xgb_proba)
    else:
        ensemble_proba = 0.9 * rf_proba + 0.1 * gbm_proba

    best_score_30 = -999
    test_rules_arr = test_rules.values.astype(int)

    for thresh_int in range(20, 55):
        thresh = thresh_int / 100.0
        preds = (ensemble_proba >= thresh).astype(int)
        combined = np.maximum(preds, test_rules_arr)

        tp = int(((combined == 1) & (y_test == 1)).sum())
        fp = int(((combined == 1) & (y_test == 0)).sum())
        fn = int(((combined == 0) & (y_test == 1)).sum())
        score = 4 * tp - 1 * fn - 2 * fp

        if score > best_score_30:
            best_score_30 = score

    combined_score = best_score_32 + best_score_30
    return combined_score, best_score_32, best_score_30


if __name__ == '__main__':
    print('asymmetric loss hyperparameter tuning')

    # test configurations
    configs = [
        # (xgb_scale_pos_weight, (rf_weight, gbm_weight, xgb_weight), name)
        (2.0, (0.6, 0.2, 0.2), 'baseline (current)'),
        (1.0, (0.6, 0.2, 0.2), 'xgb scale=1.0'),
        (1.5, (0.6, 0.2, 0.2), 'xgb scale=1.5'),
        (0.8, (0.6, 0.2, 0.2), 'xgb scale=0.8'),
        (1.0, (0.5, 0.25, 0.25), 'xgb scale=1.0, ensemble 50/25/25'),
        (1.0, (0.4, 0.3, 0.3), 'xgb scale=1.0, ensemble 40/30/30'),
    ]

    results = []
    for xgb_spw, ens_weights, name in configs:
        print(f'testing: {name}')
        combined, s32, s30 = evaluate_config(xgb_spw, ens_weights)
        results.append((name, combined, s32, s30))
        print(f'  combined: {combined} (ds32: {s32}, ds30: {s30})')

    print('\nresults summary:')
    print(f"{'configuration':<40} {'combined':>10} {'ds32':>8} {'ds30':>8}")

    results.sort(key=lambda x: -x[1])
    for name, combined, s32, s30 in results:
        print(f"{name:<40} {combined:>10} {s32:>8} {s30:>8}")

    print(f"\nbest configuration: {results[0][0]}")
    print(f"combined score: {results[0][1]}")
