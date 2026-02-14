"""stress evaluation harness for en threshold selection. usage: python3 src/stress_eval_en.py python3 src/stress_eval_en.py --scenarios 800 --seed 123"""

import argparse

from en_detector import cross_validate_en


def main():
    parser = argparse.ArgumentParser(description='en pseudo-final stress evaluation')
    parser.add_argument('--scenarios', type=int, default=400,
                        help='number of pseudo-final stress scenarios')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for stress scenario generation')
    args = parser.parse_args()

    # run both threshold modes off the same config knobs
    baseline = cross_validate_en(use_robust=False)
    robust = cross_validate_en(use_robust=True, n_scenarios=args.scenarios, seed=args.seed)

    print(f'scenarios: {args.scenarios} | seed: {args.seed}')

    print('baseline selection:')
    print(f"  threshold: {baseline['baseline_threshold']:.3f}")
    print(f"  combined en score (ds30+ds32): {baseline['baseline_combined_score']}")

    print('robust selection:')
    print(f"  threshold: {robust['best_threshold']:.3f}")
    print(f"  combined en score (at robust threshold): {robust['combined_score']}")

    summary = robust.get('robust_summary')
    if summary:
        print(f"  stress p10 score: {summary['stress_p10']:.1f}")
        print(f"  stress median score: {summary['stress_median']:.1f}")
        print(f"  stress min score: {summary['stress_min']:.1f}")
        print(f"  robust objective: {summary['objective']:.2f}")

    print('fold metrics at selected robust threshold:')
    for name, r in [
        ('train ds30+31+33 -> test ds32', robust['train30_31_33_test32']),
        ('train ds32+31+33 -> test ds30', robust['train32_31_33_test30']),
    ]:
        print(f"{name}:")
        print(f"  tp={r.get('tp', 0)} fp={r.get('fp', 0)} fn={r.get('fn', 0)} "
              f"score={r.get('score', 0)}/{r.get('max_score', 0)}")


if __name__ == '__main__':
    main()
