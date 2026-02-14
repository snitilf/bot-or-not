"""analyze specific error cases."""
import sys
import json
import pandas as pd
from features import extract_features

def analyze_users(dataset_path, user_ids):
    """print detailed features for specific users."""
    df = extract_features(dataset_path)

    # load raw data for context
    with open(dataset_path) as f:
        data = json.load(f)

    users_map = {u['id']: u for u in data['users']}

    for uid in user_ids:
        if uid not in df.index:
            print(f'\n{uid}: not found')
            continue

        # pull the feature row and profile info for this uid
        features = df.loc[uid]
        user_info = users_map.get(uid, {})

        print(f'user: {uid}')
        print(f"username: {user_info.get('username', 'n/a')}")
        print(f"name: {user_info.get('name', 'n/a')}")
        print(f"description: {user_info.get('description', 'n/a')[:100]}")
        print(f"tweet count (metadata): {user_info.get('tweet_count', 'n/a')}")
        print(f'\nkey features:')
        print(f"  temporal: hour_ent={features['hour_entropy']:.2f} active_hrs={features['active_hours']:.0f} "
              f"peak_frac={features['peak_hour_frac']:.2f} burst={features['burstiness']:.2f}")
        print(f"  intervals: cv={features['cv_intervals']:.2f} max={features['max_interval']:.0f} "
              f"min={features['min_interval']:.0f}")
        print(f"  content: excl={features['excl_rate']:.2f} hash={features['hashtag_rate']:.2f} "
              f"ques={features['question_rate']:.2f} emoji={features['emoji_rate']:.2f}")
        print(f"  text: ttr={features['ttr']:.2f} avg_len={features['avg_tweet_len']:.1f} "
              f"comma={features['comma_rate']:.2f}")
        print(f"  style: punct_cons={features['punctuation_consistency']:.2f} "
              f"bigram_rep={features['bigram_repetition_rate']:.2f} "
              f"open_ratio={features['unique_opening_ratio']:.2f}")
        print(f"  batch: max_same_sec={features['max_same_second']:.0f} "
              f"unique_sec={features['unique_second_ratio']:.2f}")
        print(f"  other: num_tweets={features['num_tweets']:.0f} z_score={features['z_score']:.2f}")

if __name__ == '__main__':
    print('ds30 false positives:')
    analyze_users('data/dataset.posts&users.30.json', [
        '0b476e50-47ae-8f82-900a-6df64c517bce',
        '19e9dc51-5663-a9c0-b91e-dfd6bf3bd9a7',
        'fc31265a-8ed2-9573-8b18-3befbdd3f39a'
    ])

    print('\n\nds30 false negative:')
    analyze_users('data/dataset.posts&users.30.json', [
        'd9bf4aff-69e8-4f3a-9d54-e997658c3b1b'
    ])

    print('\n\nds32 false positive:')
    analyze_users('data/dataset.posts&users.32.json', [
        '5fef3628-ba50-881f-bffd-4c150bb1720d'
    ])

    print('\n\nds32 false negatives:')
    analyze_users('data/dataset.posts&users.32.json', [
        'a72605f1-5b9a-482b-a11f-fd5c5f3ca76e',
        'ef97d042-0ba1-4c45-aa67-c6c2d0a72689'
    ])
