"""scoring and validation system for bot or not challenge. competition scoring: +4 tp, -1 fn, -2 fp"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import load_ground_truth


def score(detections_file: str, ground_truth_file: str) -> dict:
    """score detections against ground truth. returns dict with tp, fp, fn, competition score, precision, recall."""
    # load detections
    with open(detections_file, 'r') as f:
        detections = {line.strip() for line in f if line.strip()}

    # load ground truth
    bots = load_ground_truth(ground_truth_file)

    tp = len(detections & bots)
    fp = len(detections - bots)
    fn = len(bots - detections)
    tn = 0  # not directly available without total user count

    competition_score = 4 * tp - 1 * fn - 2 * fp
    max_score = 4 * len(bots)

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'competition_score': competition_score,
        'max_score': max_score,
        'efficiency': competition_score / max_score if max_score > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'n_detections': len(detections),
        'n_bots': len(bots),
    }


def detailed_report(detections_file: str, ground_truth_file: str,
                     dataset_path: str = None) -> str:
    """generate a detailed scoring report."""
    result = score(detections_file, ground_truth_file)

    lines = []
    lines.append('bot detection scoring report')
    lines.append(f'detections file: {detections_file}')
    lines.append(f'ground truth:    {ground_truth_file}')
    lines.append("")
    lines.append(f"total bots in ground truth: {result['n_bots']}")
    lines.append(f"total detections:           {result['n_detections']}")
    lines.append("")
    lines.append(f"true positives (tp):   {result['tp']:4d}  (+{4 * result['tp']} pts)")
    lines.append(f"false positives (fp):  {result['fp']:4d}  ({-2 * result['fp']} pts)")
    lines.append(f"false negatives (fn):  {result['fn']:4d}  ({-1 * result['fn']} pts)")
    lines.append("")
    lines.append(f"competition score: {result['competition_score']}/{result['max_score']}")
    lines.append(f"efficiency:        {result['efficiency']:.1%}")
    lines.append(f"precision:         {result['precision']:.3f}")
    lines.append(f"recall:            {result['recall']:.3f}")
    lines.append(f"f1:                {result['f1']:.3f}")

    if detections_file and ground_truth_file:
        with open(detections_file, 'r') as f:
            detections = {line.strip() for line in f if line.strip()}
        bots = load_ground_truth(ground_truth_file)

        missed = bots - detections
        false_alarms = detections - bots

        if false_alarms:
            lines.append("")
            lines.append(f"false positives ({len(false_alarms)}):")
            for uid in sorted(false_alarms):
                lines.append(f"  {uid}")

        if missed:
            lines.append("")
            lines.append(f"missed bots ({len(missed)}):")
            for uid in sorted(missed):
                lines.append(f"  {uid}")

    return "\n".join(lines)


def score_all() -> dict:
    """score all detection files against available ground truth."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, 'data')

    datasets = {
        30: {'lang': 'en', 'bots': os.path.join(data_dir, 'dataset.bots.30.txt')},
        31: {'lang': 'fr', 'bots': os.path.join(data_dir, 'dataset.bots.31.txt')},
        32: {'lang': 'en', 'bots': os.path.join(data_dir, 'dataset.bots.32.txt')},
        33: {'lang': 'fr', 'bots': os.path.join(data_dir, 'dataset.bots.33.txt')},
    }

    results = {}
    for ds_id, info in datasets.items():
        det_file = os.path.join(base, f'detections_{ds_id}.txt')
        if os.path.exists(det_file) and os.path.exists(info['bots']):
            results[ds_id] = score(det_file, info['bots'])
            results[ds_id]['lang'] = info['lang']

    return results


def print_comparison_table(results: dict):
    """print a comparison table of scores across datasets."""
    print(f"{'ds':>4} {'lang':>4} {'tp':>4} {'fp':>4} {'fn':>4} {'score':>7} {'max':>7} {'eff':>7}")
    total_score = 0
    total_max = 0

    for ds_id in sorted(results.keys()):
        r = results[ds_id]
        total_score += r['competition_score']
        total_max += r['max_score']
        print(
            f"{ds_id:>4} {r.get('lang', '??'):>4} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
            f"{r['competition_score']:>7} {r['max_score']:>7} {r['efficiency']:>7.1%}"
        )

    if results:
        eff = total_score / total_max if total_max > 0 else 0
        print(f"total score: {total_score}/{total_max} ({eff:.1%})")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage:')
        print('  python scorer.py <detections.txt> <ground_truth.txt>  # score single file')
        print('  python scorer.py --all                                 # score all detection files')
        sys.exit(1)

    if sys.argv[1] == '--all':
        results = score_all()
        if results:
            print_comparison_table(results)
        else:
            print('no detection files found. run detectors first.')
    else:
        det_file = sys.argv[1]
        gt_file = sys.argv[2] if len(sys.argv) > 2 else None

        if gt_file:
            report = detailed_report(det_file, gt_file)
            print(report)
        else:
            print(f'detections in {det_file}:')
            with open(det_file) as f:
                ids = [line.strip() for line in f if line.strip()]
            print(f'  {len(ids)} bot ids detected')
