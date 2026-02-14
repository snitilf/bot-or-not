"""strict, non-leaky evaluation harness. protocol: - for a chosen test dataset, use only the other datasets for training. - select the ml threshold using internal folds within the training set only. - evaluate on the held-out test dataset once, with a fixed threshold."""

import os
import argparse
import numpy as np

from features import extract_features, load_ground_truth
from en_detector import apply_rules as en_rules, train_and_detect as en_train_and_detect
from fr_detector import apply_rules as fr_rules, train_and_detect as fr_train_and_detect


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

DATASETS = {
    30: {'lang': 'en'},
    31: {'lang': 'fr'},
    32: {'lang': 'en'},
    33: {'lang': 'fr'},
}


def dataset_path(ds_id: int) -> str:
    return os.path.join(DATA_DIR, f'dataset.posts&users.{ds_id}.json')


def bots_path(ds_id: int) -> str:
    return os.path.join(DATA_DIR, f'dataset.bots.{ds_id}.txt')


def score_at_threshold(ds_id: int, probabilities: dict, threshold: float, rules_fn) -> int:
    df = extract_features(dataset_path(ds_id), use_cache=True)
    bots = load_ground_truth(bots_path(ds_id))

    rules = rules_fn(df)
    proba = np.array([probabilities.get(uid, 0.0) for uid in df.index])
    preds = (proba >= threshold).astype(int)
    combined = np.maximum(preds, rules.values.astype(int))

    y_true = np.array(df.index.isin(bots).astype(int))
    tp = int(((combined == 1) & (y_true == 1)).sum())
    fp = int(((combined == 1) & (y_true == 0)).sum())
    fn = int(((combined == 0) & (y_true == 1)).sum())

    return 4 * tp - 1 * fn - 2 * fp


def select_threshold(lang: str, train_ids: list) -> tuple:
    # choose the model family and sweep range by language
    if lang == 'en':
        rules_fn = en_rules
        train_and_detect = en_train_and_detect
        sweep = np.arange(0.20, 0.61, 0.005)
    else:
        rules_fn = fr_rules
        train_and_detect = fr_train_and_detect
        sweep = np.arange(0.25, 0.71, 0.005)

    fold_probs = []
    for val_id in train_ids:
        # hold out one training split at a time for internal cv
        train_fold_ids = [i for i in train_ids if i != val_id]
        train_paths = [dataset_path(i) for i in train_fold_ids]
        train_bots = [bots_path(i) for i in train_fold_ids]

        r = train_and_detect(
            train_paths,
            train_bots,
            dataset_path(val_id),
            bots_path(val_id),
            fixed_threshold=None,
        )
        fold_probs.append((val_id, r['probabilities']))

    best_thresh = 0.40
    best_score = -999999
    for thresh in sweep:
        total = 0
        for val_id, probs in fold_probs:
            total += score_at_threshold(val_id, probs, thresh, rules_fn)
        if total > best_score:
            best_score = total
            best_thresh = float(thresh)

    return best_thresh, best_score


def evaluate_strict(lang: str, test_id: int) -> dict:
    # strict mode uses every other dataset for training
    all_ids = sorted(DATASETS.keys())
    train_ids = [i for i in all_ids if i != test_id]

    threshold, cv_score = select_threshold(lang, train_ids)

    if lang == 'en':
        train_and_detect = en_train_and_detect
    else:
        train_and_detect = fr_train_and_detect

    train_paths = [dataset_path(i) for i in train_ids]
    train_bots = [bots_path(i) for i in train_ids]

    result = train_and_detect(
        train_paths,
        train_bots,
        dataset_path(test_id),
        bots_path(test_id),
        fixed_threshold=threshold,
    )

    result['strict_threshold'] = threshold
    result['strict_cv_score'] = cv_score
    return result


def run(lang: str, test_id: int = None) -> None:
    targets = []
    if lang == 'en':
        targets = [30, 32]
    elif lang == 'fr':
        targets = [31, 33]
    else:
        targets = [30, 32, 31, 33]

    if test_id is not None:
        targets = [test_id]

    for ds_id in targets:
        ds_lang = DATASETS[ds_id]['lang']
        print(f'strict eval | ds{ds_id} ({ds_lang})')
        result = evaluate_strict(ds_lang, ds_id)
        print(f"threshold (training-only cv): {result['strict_threshold']:.3f}")
        print(f"training-only cv score: {result['strict_cv_score']}")
        print(
            f"tp={result['tp']} fp={result['fp']} fn={result['fn']} | "
            f"score={result['score']}/{result['max_score']} "
            f"({result['efficiency']:.1%})"
        )
        print(f"precision={result['precision']:.3f} recall={result['recall']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='strict, non-leaky evaluation harness')
    parser.add_argument('--lang', choices=['en', 'fr', 'all'], default='all')
    parser.add_argument('--test', type=int, choices=[30, 31, 32, 33], default=None)
    args = parser.parse_args()

    run(args.lang, args.test)


if __name__ == '__main__':
    main()
