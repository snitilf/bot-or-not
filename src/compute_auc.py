"""compute auc for all features to evaluate discriminative power."""
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features, load_ground_truth

def compute_aucs(dataset_path, bots_path):
    """compute auc for all features in a dataset."""
    df = extract_features(dataset_path, use_cache=True)
    bots = load_ground_truth(bots_path)
    df['is_bot'] = df.index.isin(bots).astype(int)

    aucs = {}
    for col in df.columns:
        if col == 'is_bot':
            continue

        # skip non-numeric columns
        if df[col].dtype not in [np.float64, np.int64, np.float32, np.int32]:
            continue

        # skip if all same value
        if df[col].nunique() <= 1:
            continue

        try:
            auc = roc_auc_score(df['is_bot'], df[col])
            # convert to bidirectional auc (max of auc and 1-auc)
            auc = max(auc, 1 - auc)
            aucs[col] = auc
        except Exception:
            pass

    return aucs

if __name__ == '__main__':
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    # compute aucs for ds30 and ds32
    ds30_aucs = compute_aucs(
        os.path.join(base, 'dataset.posts&users.30.json'),
        os.path.join(base, 'dataset.bots.30.txt')
    )

    ds32_aucs = compute_aucs(
        os.path.join(base, 'dataset.posts&users.32.json'),
        os.path.join(base, 'dataset.bots.32.txt')
    )

    # find new template features
    new_features = [
        'bigram_repetition_rate', 'unique_opening_ratio', 'vocab_richness_ratio',
        'self_similarity_std', 'self_similarity_max'
    ]


    print('template feature aucs:')
    print(f"{'feature':<30} {'ds30':>8} {'ds32':>8} {'min':>8} {'pass':>6}")
    for feat in new_features:
        auc30 = ds30_aucs.get(feat, 0.0)
        auc32 = ds32_aucs.get(feat, 0.0)
        min_auc = min(auc30, auc32)
        passes = 'yes' if min_auc > 0.60 else 'no'
        print(f"{feat:<30} {auc30:>8.3f} {auc32:>8.3f} {min_auc:>8.3f} {passes:>6}")

    # show top existing features for comparison
    print('\ntop baseline features:')
    print(f"{'feature':<30} {'ds30':>8} {'ds32':>8}")

    all_features = set(ds30_aucs.keys()) | set(ds32_aucs.keys())
    all_features -= set(new_features)

    feature_scores = []
    for feat in all_features:
        avg_auc = (ds30_aucs.get(feat, 0.0) + ds32_aucs.get(feat, 0.0)) / 2
        feature_scores.append((feat, avg_auc))

    feature_scores.sort(key=lambda x: -x[1])
    for feat, _ in feature_scores[:10]:
        auc30 = ds30_aucs.get(feat, 0.0)
        auc32 = ds32_aucs.get(feat, 0.0)
        print(f"{feat:<30} {auc30:>8.3f} {auc32:>8.3f}")
