"""english detection entry point for bot or not challenge. usage: python src/run_en.py <dataset_path> [--output <output_path>] [--final]"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from en_detector import detect_bots, detect_bots_final


def main():
    parser = argparse.ArgumentParser(description='Detect English bots')
    parser.add_argument('dataset', help='Path to dataset JSON file')
    parser.add_argument('--output', '-o', help='Output file path',
                        default=None)
    parser.add_argument('--final', action='store_true',
                        help='Use final submission mode (train on all practice data)')
    parser.add_argument('--team', default='localhost',
                        help='Team name for output file')
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found: {args.dataset}")
        sys.exit(1)

    print(f"Running EN detector on {args.dataset}...")

    if args.final:
        detections = detect_bots_final(args.dataset)
    else:
        detections = detect_bots(args.dataset)

    # determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f'{args.team}.detections.en.txt'

    # write detections
    with open(output_path, 'w') as f:
        for uid in sorted(detections):
            f.write(uid + '\n')

    print(f"Wrote {len(detections)} detections to {output_path}")


if __name__ == '__main__':
    main()
