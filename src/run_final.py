"""final submission script for bot or not challenge. runs both en and fr detectors on given datasets and produces submission files. usage: python src/run_final.py <en_dataset> <fr_dataset> [--team localhost]"""

import json
import os
import sys
import time
import argparse
import traceback
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def adapt_dataset(dataset_path: str) -> str:
    """if the dataset is in separate tweets+users format (as described in the
    competition pdf), convert it to the combined format our pipeline expects.
    if it's already in combined format, return the path unchanged."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        # combined format starts with '{' (single json object)
        if first_char == '{':
            data = json.load(f)
            # already combined format with 'posts' key
            if 'posts' in data:
                return dataset_path
            # could be a single-tweet json object (ndjson first line)
            # fall through to ndjson handling below

        # try ndjson format (one json object per line = separate tweets file)
        f.seek(0)
        lines = f.readlines()

    # check if it looks like ndjson tweets
    try:
        first_obj = json.loads(lines[0].strip())
    except (json.JSONDecodeError, IndexError):
        # not parseable, return as-is and let downstream handle the error
        return dataset_path

    if 'text' in first_obj and 'author_id' in first_obj:
        # this is a tweets-only file in ndjson format
        # look for a companion users file
        dir_path = os.path.dirname(dataset_path)
        base = os.path.basename(dataset_path)

        # try common naming patterns for the users file
        users_candidates = []
        if 'tweets' in base:
            users_candidates.append(os.path.join(dir_path, base.replace('tweets', 'users')))
        users_candidates.append(os.path.join(dir_path, 'dataset.users.json'))
        users_candidates.append(os.path.join(dir_path, base.replace('.json', '.users.json')))

        users_data = []
        for users_path in users_candidates:
            if os.path.exists(users_path):
                with open(users_path, 'r', encoding='utf-8') as uf:
                    content = uf.read().strip()
                    if content.startswith('['):
                        users_data = json.loads(content)
                    else:
                        # ndjson users
                        users_data = [json.loads(line.strip()) for line in content.split('\n') if line.strip()]
                break

        # parse all tweets
        posts = [json.loads(line.strip()) for line in lines if line.strip()]

        # detect language from tweets
        lang_counts = Counter(t.get('lang', 'en') for t in posts)
        lang = lang_counts.most_common(1)[0][0] if lang_counts else 'en'

        # compute metadata
        timestamps = []
        for t in posts:
            ts = t.get('created_at', '')
            if ts:
                timestamps.append(ts)

        author_ids = set(t.get('author_id', '') for t in posts)
        posts_per_author = Counter(t.get('author_id', '') for t in posts)

        start_time = min(timestamps) if timestamps else '2024-01-01T00:00:00Z'
        end_time = max(timestamps) if timestamps else '2024-01-03T00:00:00Z'
        avg_posts = len(posts) / max(len(author_ids), 1)

        # build combined format
        combined = {
            'id': 0,
            'lang': lang,
            'metadata': {
                'start_time': start_time,
                'end_time': end_time,
                'user_count': len(author_ids),
                'post_count': len(posts),
                'topics': [],
                'users_average_amount_posts': avg_posts,
                'users_average_z_score': 0.0,
            },
            'posts': posts,
            'users': users_data if users_data else [
                {'id': aid, 'username': '', 'name': '', 'description': '',
                 'location': '', 'tweet_count': posts_per_author.get(aid, 0),
                 'z_score': 0.0}
                for aid in author_ids
            ],
        }

        # write combined file
        combined_path = dataset_path + '.combined.json'
        with open(combined_path, 'w', encoding='utf-8') as cf:
            json.dump(combined, cf)
        print(f"  Converted {base} to combined format -> {os.path.basename(combined_path)}")
        return combined_path

    # not a recognized format, return as-is
    return dataset_path


def main():
    parser = argparse.ArgumentParser(description='Run bot detection for final submission')
    parser.add_argument('en_dataset', help='Path to English dataset JSON')
    parser.add_argument('fr_dataset', help='Path to French dataset JSON')
    parser.add_argument('--team', default='localhost', help='Team name')
    args = parser.parse_args()

    start = time.time()

    # adapt datasets if needed (handles separate tweets+users format)
    en_path = adapt_dataset(args.en_dataset)
    fr_path = adapt_dataset(args.fr_dataset)

    en_ok = False
    fr_ok = False

    # run en detector (isolated so fr still runs if en fails)
    try:
        print("=" * 50)
        print("Running English detector...")
        print("=" * 50)
        from en_detector import detect_bots_final as en_detect
        en_detections = en_detect(en_path)
        en_output = f'{args.team}.detections.en.txt'
        with open(en_output, 'w') as f:
            for uid in sorted(en_detections):
                f.write(str(uid) + '\n')
        print(f"EN: {len(en_detections)} bots detected -> {en_output}")
        en_ok = True
    except Exception as e:
        print(f"\nEN DETECTOR FAILED: {e}")
        traceback.print_exc()
        print("Continuing with FR detector...\n")

    # run fr detector (isolated so en output is preserved if fr fails)
    try:
        print()
        print("=" * 50)
        print("Running French detector...")
        print("=" * 50)
        from fr_detector import detect_bots_final as fr_detect
        fr_detections = fr_detect(fr_path)
        fr_output = f'{args.team}.detections.fr.txt'
        with open(fr_output, 'w') as f:
            for uid in sorted(fr_detections):
                f.write(str(uid) + '\n')
        print(f"FR: {len(fr_detections)} bots detected -> {fr_output}")
        fr_ok = True
    except Exception as e:
        print(f"\nFR DETECTOR FAILED: {e}")
        traceback.print_exc()

    elapsed = time.time() - start
    print()
    print("=" * 50)
    print(f"Done! Total time: {elapsed:.1f}s")
    if en_ok:
        print(f"  {args.team}.detections.en.txt ({len(en_detections)} detections)")
    else:
        print("  EN: FAILED (no output)")
    if fr_ok:
        print(f"  {args.team}.detections.fr.txt ({len(fr_detections)} detections)")
    else:
        print("  FR: FAILED (no output)")
    print()
    print("Submit to: bot.or.not.competition.adm@gmail.com")
    print("=" * 50)

    if not en_ok or not fr_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
