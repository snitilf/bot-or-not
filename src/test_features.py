"""test suite for feature extraction pipeline. validates feature extraction across all 4 practice datasets."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features, load_ground_truth, load_dataset

import numpy as np
from sklearn.metrics import roc_auc_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

DATASETS = {
    30: {'lang': 'en', 'expected_users': 275, 'expected_bots': 66},
    31: {'lang': 'fr', 'expected_users': 171, 'expected_bots': 27},
    32: {'lang': 'en', 'expected_users': 271, 'expected_bots': 63},
    33: {'lang': 'fr', 'expected_users': 172, 'expected_bots': 28},
}

EXPECTED_FEATURE_COUNT = 56  # all features extracted by features.py


def dataset_path(ds_id):
    return os.path.join(DATA_DIR, f'dataset.posts&users.{ds_id}.json')


def bots_path(ds_id):
    return os.path.join(DATA_DIR, f'dataset.bots.{ds_id}.txt')


def test_user_counts():
    """verify correct user counts for each dataset."""
    passed = True
    for ds_id, info in DATASETS.items():
        df = extract_features(dataset_path(ds_id), use_cache=True)
        actual = len(df)
        expected = info['expected_users']
        status = 'pass' if actual == expected else 'fail'
        if actual != expected:
            passed = False
        print(f'  ds{ds_id} user count: {actual} (expected {expected}) [{status}]')
    return passed


def test_feature_count():
    """verify feature count matches expected."""
    passed = True
    for ds_id in DATASETS:
        df = extract_features(dataset_path(ds_id), use_cache=True)
        actual = len(df.columns)
        status = 'pass' if actual == EXPECTED_FEATURE_COUNT else 'fail'
        if actual != EXPECTED_FEATURE_COUNT:
            passed = False
        print(f'  ds{ds_id} feature count: {actual} (expected {EXPECTED_FEATURE_COUNT}) [{status}]')
    return passed


def test_no_nan():
    """check no nan values in features."""
    passed = True
    for ds_id in DATASETS:
        df = extract_features(dataset_path(ds_id), use_cache=True)
        nan_count = int(df.isna().sum().sum())
        status = 'pass' if nan_count == 0 else 'fail'
        if nan_count != 0:
            passed = False
            nan_cols = df.columns[df.isna().any()].tolist()
            print(f'  ds{ds_id} nan count: {nan_count} in columns {nan_cols} [{status}]')
        else:
            print(f'  ds{ds_id} nan count: 0 [{status}]')
    return passed


def test_out_of_range_en():
    """verify out_of_range catches 12 bots in en datasets with 0 fp."""
    passed = True
    for ds_id in [30, 32]:
        df = extract_features(dataset_path(ds_id), use_cache=True)
        bots = load_ground_truth(bots_path(ds_id))
        flagged = df[df['out_of_range'] == 1].index.tolist()
        flagged_bots = [u for u in flagged if u in bots]
        flagged_humans = [u for u in flagged if u not in bots]

        bot_ok = len(flagged_bots) == 12
        fp_ok = len(flagged_humans) == 0
        status = 'pass' if (bot_ok and fp_ok) else 'fail'
        if not (bot_ok and fp_ok):
            passed = False
        print(f"  ds{ds_id} out_of_range: {len(flagged_bots)} bots (expected 12), "
              f"{len(flagged_humans)} fp (expected 0) [{status}]")
    return passed


def test_out_of_range_fr():
    """verify out_of_range catches 0 bots in fr datasets."""
    passed = True
    for ds_id in [31, 33]:
        df = extract_features(dataset_path(ds_id), use_cache=True)
        flagged = df[df['out_of_range'] == 1].index.tolist()
        status = 'pass' if len(flagged) == 0 else 'fail'
        if len(flagged) != 0:
            passed = False
        print(f'  ds{ds_id} out_of_range: {len(flagged)} flagged (expected 0) [{status}]')
    return passed


def print_top_features_by_auc():
    """print top 5 features by auc for each dataset."""
    for ds_id, info in DATASETS.items():
        df = extract_features(dataset_path(ds_id), use_cache=True)
        bots = load_ground_truth(bots_path(ds_id))
        y = df.index.isin(bots).astype(int)

        aucs = {}
        for col in df.columns:
            vals = df[col].values.astype(float)
            if np.std(vals) == 0:
                aucs[col] = 0.5
            else:
                auc = roc_auc_score(y, vals)
                aucs[col] = max(auc, 1 - auc)

        top5 = sorted(aucs.items(), key=lambda x: -x[1])[:5]
        print(f"  ds{ds_id} ({info['lang']}):")
        for feat, auc in top5:
            print(f'    {feat:25s} auc={auc:.3f}')


def main():
    results = {}

    print('feature extraction test suite')

    print('\n[1] user counts')
    results['user_counts'] = test_user_counts()

    print('\n[2] feature count')
    results['feature_count'] = test_feature_count()

    print('\n[3] no nan values')
    results['no_nan'] = test_no_nan()

    print('\n[4] out-of-range (en datasets)')
    results['oor_en'] = test_out_of_range_en()

    print('\n[5] out-of-range (fr datasets)')
    results['oor_fr'] = test_out_of_range_fr()

    print('\n[6] top 5 features by auc')
    print_top_features_by_auc()

    # summary
    print('summary')
    all_pass = True
    for name, passed in results.items():
        status = 'pass' if passed else 'fail'
        if not passed:
            all_pass = False
        print(f"  {name:25s} [{status}]")

    if all_pass:
        print('all tests passed')
    else:
        print('some tests failed')
        sys.exit(1)


if __name__ == '__main__':
    main()
