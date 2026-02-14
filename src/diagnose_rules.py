"""diagnostic script to check which rules fire on which users."""
import sys
import pandas as pd
from features import extract_features, load_ground_truth

def check_rules(dataset_path, ground_truth_path):
    """check which rules fire and their accuracy."""
    df = extract_features(dataset_path)
    gt = load_ground_truth(ground_truth_path)
    df['is_bot'] = df.index.isin(gt)

    # apply each rule
    rules = {}

    # rule 1: out of range
    if 'out_of_range' in df.columns:
        rules['out_of_range'] = df['out_of_range'] == 1

    # rule 2: high hashtag + active hours
    rules['high_hashtag'] = (df['hashtag_rate'] > 1.5) & (df['active_hours'] >= 20)

    # rule 3: batch injection
    if 'max_same_second' in df.columns:
        rules['batch_injection'] = df['max_same_second'] >= 4

    # rule 4: extreme questions
    rules['extreme_questions'] = (df['question_rate'] >= 0.95) & (df['num_tweets'] >= 8)

    # rule 5: exclamation + hour entropy
    rules['exclamation_entropy'] = (df['hour_entropy'] >= 3.7) & (df['excl_rate'] >= 0.3) & (df['active_hours'] >= 18)

    # rule 6: metronomic timing
    rules['metronomic'] = (df['active_hours'] >= 16) & (df['burstiness'] < -0.1)

    # analyze each rule
    for rule_name, mask in rules.items():
        caught = mask.sum()
        if caught == 0:
            continue
        tp = (mask & df['is_bot']).sum()
        fp = (mask & ~df['is_bot']).sum()
        precision = tp / caught if caught > 0 else 0
        print(f'\n{rule_name}: {caught} flagged | tp={tp} fp={fp} | precision={precision:.3f}')

        if fp > 0:
            print(f'  false positives:')
            fps = df[mask & ~df['is_bot']].head(5).index.tolist()
            for uid in fps:
                user_data = df.loc[uid]
                print(f"    {uid[:16]}: hour_ent={user_data['hour_entropy']:.2f} excl={user_data['excl_rate']:.2f} "
                      f"active_hrs={user_data['active_hours']:.0f} burst={user_data['burstiness']:.2f} "
                      f"cv_int={user_data['cv_intervals']:.2f}")

if __name__ == '__main__':
    check_rules('data/dataset.posts&users.30.json', 'data/dataset.bots.30.txt')
