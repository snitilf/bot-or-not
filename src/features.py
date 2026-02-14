"""feature extraction pipeline for bot or not challenge. extracts 56 features per user from dataset json files."""

import json
import math
import os
import re
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def load_dataset(dataset_path: str) -> dict:
    """load and return a dataset json file."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_ground_truth(bots_path: str) -> set:
    """load bot ids from ground truth file."""
    with open(bots_path, 'r') as f:
        return {line.strip() for line in f if line.strip()}


def group_posts_by_author(posts: list) -> dict:
    """group posts by author_id."""
    by_author = defaultdict(list)
    for post in posts:
        by_author[post['author_id']].append(post)
    return dict(by_author)


def parse_timestamp(ts: str) -> datetime:
    """parse iso timestamp to datetime."""
    try:
        ts = str(ts).replace('Z', '+00:00')
        return datetime.fromisoformat(ts)
    except (ValueError, AttributeError):
        return datetime(2024, 1, 1, tzinfo=timezone.utc)


def shannon_entropy(counts: list) -> float:
    """compute shannon entropy of a distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def extract_temporal_features(tweets: list, metadata: dict) -> dict:
    """extract temporal/timing features from a user's tweets."""
    features = {}

    # parse timestamps and sort
    timestamps = sorted([parse_timestamp(t['created_at']) for t in tweets]) if tweets else []
    hours = [ts.hour for ts in timestamps]

    # hour distribution
    hour_counts = [0] * 24
    for h in hours:
        hour_counts[h] += 1

    features['hour_entropy'] = shannon_entropy(hour_counts) if tweets else 0.0
    features['active_hours'] = sum(1 for c in hour_counts if c > 0) if tweets else 0
    features['peak_hour_frac'] = max(hour_counts) / len(tweets) if tweets else 0.0

    # night ratio (2am-6am utc)
    night_tweets = sum(1 for h in hours if 2 <= h <= 5)
    features['night_ratio'] = night_tweets / len(tweets) if tweets else 0.0

    # inter-tweet intervals (in seconds)
    intervals = []
    for i in range(1, len(timestamps)):
        delta = (timestamps[i] - timestamps[i - 1]).total_seconds()
        intervals.append(delta)

    if intervals:
        mu = np.mean(intervals)
        sigma = np.std(intervals)
        features['mean_interval'] = mu
        features['std_interval'] = sigma
        features['cv_intervals'] = sigma / mu if mu > 0 else 0.0
        features['min_interval'] = min(intervals)
        features['max_interval'] = max(intervals)
        # burstiness: (sigma - mu) / (sigma + mu)
        denom = sigma + mu
        features['burstiness'] = (sigma - mu) / denom if denom > 0 else 0.0
    else:
        features['mean_interval'] = 0.0
        features['std_interval'] = 0.0
        features['cv_intervals'] = 0.0
        features['min_interval'] = 0.0
        features['max_interval'] = 0.0
        features['burstiness'] = 0.0

    # window-aware features based on dataset metadata
    start_time = parse_timestamp(metadata['start_time'])
    end_time = parse_timestamp(metadata['end_time'])
    window_seconds = max((end_time - start_time).total_seconds(), 1.0)

    if timestamps:
        within_count = sum(1 for ts in timestamps if start_time <= ts <= end_time)
        features['within_window_ratio'] = within_count / len(timestamps)

        span_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
        features['window_coverage'] = max(min(span_seconds / window_seconds, 1.0), 0.0)

        edge_seconds = window_seconds * 0.05
        edge_count = 0
        for ts in timestamps:
            if (ts - start_time).total_seconds() <= edge_seconds:
                edge_count += 1
            elif (end_time - ts).total_seconds() <= edge_seconds:
                edge_count += 1
        features['edge_activity_ratio'] = edge_count / len(timestamps)
    else:
        features['within_window_ratio'] = 0.0
        features['window_coverage'] = 0.0
        features['edge_activity_ratio'] = 0.0

    # out-of-range detection: tweets outside the metadata collection window
    out_count = sum(1 for ts in timestamps if ts < start_time or ts > end_time)
    features['out_of_range'] = 1 if out_count > 0 else 0

    return features


def extract_dataset_normalization_features(user: dict, tweets: list, metadata: dict) -> dict:
    """normalize user activity against dataset-level metadata."""
    features = {}

    avg_posts = metadata.get('users_average_amount_posts', 0.0) or 0.0
    avg_z = metadata.get('users_average_z_score', 0.0) or 0.0

    tweet_count = user.get('tweet_count', 0)
    num_tweets = len(tweets)
    z_score = user.get('z_score', 0.0)

    if avg_posts > 0:
        features['tweet_count_ratio'] = tweet_count / avg_posts
        features['num_tweets_ratio'] = num_tweets / avg_posts
    else:
        features['tweet_count_ratio'] = 0.0
        features['num_tweets_ratio'] = 0.0

    features['z_score_delta'] = z_score - avg_z

    return features


def extract_timestamp_collision_features(tweets: list) -> dict:
    """extract timestamp collision features to detect batch injection bots."""
    features = {}

    if len(tweets) < 2:
        return {
            'max_same_second': 0,
            'unique_second_ratio': 1.0,
            'timestamp_collision_rate': 0.0,
        }

    # truncate to second-level precision
    second_strs = []
    for t in tweets:
        ts = t['created_at']
        # truncate to second (remove fractional seconds)
        dot_idx = ts.find('.')
        if dot_idx != -1:
            # find the timezone part after fractional seconds
            for i in range(dot_idx + 1, len(ts)):
                if ts[i] in ('+', '-', 'Z'):
                    ts = ts[:dot_idx] + ts[i:]
                    break
        second_strs.append(ts)

    counts = Counter(second_strs)
    features['max_same_second'] = max(counts.values())
    features['unique_second_ratio'] = len(counts) / len(tweets)

    # fraction of tweets that share a second with at least one other tweet
    collisions = sum(c for c in counts.values() if c > 1)
    features['timestamp_collision_rate'] = collisions / len(tweets)

    return features


def extract_stylometric_features(tweets: list) -> dict:
    """extract stylometric features for writing style analysis."""
    features = {}

    if not tweets:
        return {
            'ellipsis_rate': 0.0,
            'comma_rate': 0.0,
            'hapax_ratio': 0.0,
        }

    texts = [t['text'] for t in tweets]
    n = len(texts)

    # ellipsis rate: fraction of tweets with ... or \u2026
    features['ellipsis_rate'] = sum(1 for t in texts if '...' in t or '\u2026' in t) / n

    # comma rate: commas per character across all tweets
    total_chars = sum(len(t) for t in texts)
    total_commas = sum(t.count(',') for t in texts)
    features['comma_rate'] = total_commas / max(total_chars, 1)

    # hapax ratio: words appearing exactly once / total unique words
    all_words = []
    for t in texts:
        clean = re.sub(r'https?://\S+|t\.co/\S+|@\w+|#\w+', '', t).lower()
        all_words.extend(clean.split())
    if all_words:
        word_counts = Counter(all_words)
        hapax = sum(1 for c in word_counts.values() if c == 1)
        features['hapax_ratio'] = hapax / max(len(word_counts), 1)
    else:
        features['hapax_ratio'] = 0.0

    return features


def extract_compression_features(tweets: list) -> dict:
    """extract compression and n-gram entropy features for template detection."""
    if not tweets:
        return {
            'compression_ratio': 0.0,
            'char_3gram_entropy': 0.0,
            'char_3gram_unique_ratio': 0.0,
        }

    text = ' '.join(t['text'] for t in tweets).strip()
    if not text:
        return {
            'compression_ratio': 0.0,
            'char_3gram_entropy': 0.0,
            'char_3gram_unique_ratio': 0.0,
        }

    compressed = zlib.compress(text.encode('utf-8'))
    compression_ratio = len(compressed) / max(len(text), 1)

    clean = re.sub(r'\\s+', ' ', text.lower())
    if len(clean) < 3:
        return {
            'compression_ratio': compression_ratio,
            'char_3gram_entropy': 0.0,
            'char_3gram_unique_ratio': 0.0,
        }

    grams = [clean[i:i + 3] for i in range(len(clean) - 2)]
    counts = Counter(grams)
    entropy = shannon_entropy(list(counts.values()))
    unique_ratio = len(counts) / max(len(grams), 1)

    return {
        'compression_ratio': compression_ratio,
        'char_3gram_entropy': entropy,
        'char_3gram_unique_ratio': unique_ratio,
    }


def extract_template_features(tweets: list) -> dict:
    """extract template and repetition features to detect template bots and tweet stealers."""
    features = {}

    if len(tweets) < 2:
        return {
            'bigram_repetition_rate': 0.0,
            'unique_opening_ratio': 1.0,
            'vocab_richness_ratio': 0.0,
        }

    texts = [t['text'] for t in tweets]

    # 1. bigram repetition rate: fraction of bigrams appearing in 2+ tweets
    bigram_tweets = defaultdict(set)  # bigram -> set of tweet indices

    for i, text in enumerate(texts):
        # remove urls, mentions, hashtags
        clean = re.sub(r'https?://\S+|t\.co/\S+|@\w+|#\w+', '', text).lower()
        words = clean.split()

        for j in range(len(words) - 1):
            bigram = (words[j], words[j+1])
            bigram_tweets[bigram].add(i)

    if bigram_tweets:
        repeated = sum(1 for tweet_set in bigram_tweets.values() if len(tweet_set) >= 2)
        features['bigram_repetition_rate'] = float(repeated / len(bigram_tweets))
    else:
        features['bigram_repetition_rate'] = 0.0

    # 2. unique opening ratio: fraction of tweets with unique first 3 words
    openings = []
    for text in texts:
        words = text.split()[:3]
        opening = ' '.join(words).lower()
        openings.append(opening)

    if openings:
        features['unique_opening_ratio'] = float(len(set(openings)) / len(openings))
    else:
        features['unique_opening_ratio'] = 1.0

    # 3. vocab richness ratio (herdan's c): log(v) / log(n)
    all_words = []
    for text in texts:
        clean = re.sub(r'https?://\S+|t\.co/\S+|@\w+|#\w+', '', text).lower()
        all_words.extend(clean.split())

    if len(all_words) >= 2:
        V = len(set(all_words))
        N = len(all_words)

        if N > 1 and V > 1:
            features['vocab_richness_ratio'] = float(math.log(V) / math.log(N))
        else:
            features['vocab_richness_ratio'] = 0.0
    else:
        features['vocab_richness_ratio'] = 0.0

    return features


def extract_uniformity_features(tweets: list) -> dict:
    """extract writing uniformity features to detect llm-generated text. llm bots show low variance in sentence length, punctuation, and vocabulary."""
    features = {}

    if len(tweets) < 2:
        return {
            'sentence_len_cv': 0.0,
            'punctuation_consistency': 0.0,
            'vocabulary_consistency': 0.0,
        }

    texts = [t['text'] for t in tweets]

    # 1. sentence length cv: coefficient of variation of sentence lengths
    sentences = []
    for text in texts:
        sents = re.split(r'[.!?]+', text)
        sents = [s.strip() for s in sents if s.strip()]
        sentences.extend(sents)

    if len(sentences) >= 2:
        lengths = [len(s.split()) for s in sentences]
        mean_len = np.mean(lengths)
        if mean_len > 0:
            features['sentence_len_cv'] = float(np.std(lengths) / mean_len)
        else:
            features['sentence_len_cv'] = 0.0
    else:
        features['sentence_len_cv'] = 0.0

    # 2. punctuation consistency: std dev of per-tweet punctuation density
    import string
    punct_set = set(string.punctuation)

    densities = []
    for text in texts:
        if len(text) == 0:
            densities.append(0.0)
        else:
            punct_count = sum(1 for c in text if c in punct_set)
            densities.append(punct_count / len(text))

    if len(densities) >= 2:
        features['punctuation_consistency'] = float(np.std(densities))
    else:
        features['punctuation_consistency'] = 0.0

    # 3. vocabulary consistency: std dev of per-tweet type-token ratio
    ttrs = []
    for text in texts:
        words = text.lower().split()
        if len(words) >= 3:  # need minimum words for meaningful ttr
            ttr = len(set(words)) / len(words)
            ttrs.append(ttr)

    if len(ttrs) >= 2:
        features['vocabulary_consistency'] = float(np.std(ttrs))
    else:
        features['vocabulary_consistency'] = 0.0

    return features


def extract_text_features(tweets: list) -> dict:
    """extract text/content features from a user's tweets."""
    features = {}

    if not tweets:
        return {
            'avg_tweet_len': 0.0,
            'tweet_len_std': 0.0,
            'excl_rate': 0.0,
            'question_rate': 0.0,
            'hashtag_rate': 0.0,
            'mid_hashtag_rate': 0.0,
            'mention_rate': 0.0,
            'url_rate': 0.0,
            'emoji_rate': 0.0,
            'ttr': 0.0,
            'self_similarity': 0.0,
            'self_similarity_std': 0.0,
            'self_similarity_max': 0.0,
        }

    texts = [t['text'] for t in tweets]
    n = len(texts)

    # tweet length stats
    lengths = [len(t) for t in texts]
    features['avg_tweet_len'] = np.mean(lengths)
    features['tweet_len_std'] = np.std(lengths) if n > 1 else 0.0

    # exclamation rate: fraction of tweets containing '!'
    features['excl_rate'] = sum(1 for t in texts if '!' in t) / n

    # question rate: fraction of tweets containing '?'
    features['question_rate'] = sum(1 for t in texts if '?' in t) / n

    # hashtag rate: average hashtags per tweet
    hashtag_counts = [len(re.findall(r'#\w+', t)) for t in texts]
    features['hashtag_rate'] = np.mean(hashtag_counts)

    # mid-hashtag rate: hashtags not at start/end of tweet
    mid_hashtag_count = 0
    for t in texts:
        matches = list(re.finditer(r'#\w+', t))
        for m in matches:
            before = t[:m.start()].strip()
            after = t[m.end():].strip()
            if before and after and not after.startswith('#'):
                mid_hashtag_count += 1
    features['mid_hashtag_rate'] = mid_hashtag_count / n

    # mention rate
    mention_counts = [len(re.findall(r'@\w+', t)) for t in texts]
    features['mention_rate'] = np.mean(mention_counts)

    # url rate: fraction of tweets with urls
    features['url_rate'] = sum(1 for t in texts if 'http' in t or 't.co' in t) / n

    # emoji rate: emoji characters per tweet
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f900-\U0001f9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002600-\U000026FF"
        "]+", flags=re.UNICODE
    )
    emoji_counts = [len(emoji_pattern.findall(t)) for t in texts]
    features['emoji_rate'] = np.mean(emoji_counts)

    # type-token ratio (vocabulary diversity)
    all_words = []
    for t in texts:
        clean = re.sub(r'https?://\S+|t\.co/\S+|@\w+', '', t)
        words = clean.lower().split()
        all_words.extend(words)
    if all_words:
        features['ttr'] = len(set(all_words)) / len(all_words)
    else:
        features['ttr'] = 0.0

    # self-similarity statistics (mean, std, max for template detection)
    sim_stats = _compute_self_similarity(texts)
    features.update(sim_stats)

    return features


def _compute_self_similarity(texts: list) -> dict:
    """compute self-similarity statistics using term frequency vectors. returns dict with mean, std, and max similarity for template detection."""
    if len(texts) < 2:
        return {
            'self_similarity': 0.0,
            'self_similarity_std': 0.0,
            'self_similarity_max': 0.0,
        }

    vocab = {}
    doc_vectors = []
    for t in texts:
        clean = re.sub(r'https?://\S+|t\.co/\S+|@\w+|#\w+', '', t).lower()
        words = clean.split()
        word_counts = Counter(words)
        for w in word_counts:
            if w not in vocab:
                vocab[w] = len(vocab)
        doc_vectors.append(word_counts)

    if not vocab:
        return {
            'self_similarity': 0.0,
            'self_similarity_std': 0.0,
            'self_similarity_max': 0.0,
        }

    n_docs = len(texts)
    vecs = np.zeros((n_docs, len(vocab)))
    for i, wc in enumerate(doc_vectors):
        for w, c in wc.items():
            vecs[i, vocab[w]] = c

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vecs = vecs / norms

    # sample pairwise similarities (cap at 50 pairs for speed)
    if n_docs <= 10:
        sims = []
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                sims.append(float(np.dot(vecs[i], vecs[j])))
    else:
        rng = np.random.RandomState(42)
        sims = []
        seen = set()
        n_pairs = min(50, n_docs * (n_docs - 1) // 2)
        while len(sims) < n_pairs:
            i, j = rng.randint(0, n_docs, 2)
            if i != j and (i, j) not in seen:
                seen.add((i, j))
                sims.append(float(np.dot(vecs[i], vecs[j])))

    if sims:
        return {
            'self_similarity': float(np.mean(sims)),
            'self_similarity_std': float(np.std(sims)),
            'self_similarity_max': float(np.max(sims)),
        }

    return {
        'self_similarity': 0.0,
        'self_similarity_std': 0.0,
        'self_similarity_max': 0.0,
    }


def extract_account_features(user: dict) -> dict:
    """extract account/profile features from user metadata."""
    features = {}

    features['tweet_count'] = user.get('tweet_count', 0)
    features['z_score'] = user.get('z_score', 0.0)

    username = user.get('username', '')
    features['username_len'] = len(username)
    features['username_digit_ratio'] = sum(c.isdigit() for c in username) / max(len(username), 1)

    if username:
        char_counts = Counter(username.lower())
        features['username_entropy'] = shannon_entropy(list(char_counts.values()))
    else:
        features['username_entropy'] = 0.0

    description = user.get('description', '') or ''
    features['has_description'] = 1 if description.strip() else 0
    features['description_length'] = len(description)

    return features


def extract_topic_features(tweets: list, topics: list) -> dict:
    """extract topic engagement features."""
    features = {}

    if not tweets or not topics:
        features['topic_diversity'] = 0.0
        return features

    topic_hits = set()
    for tweet in tweets:
        text_lower = tweet['text'].lower()
        for topic_info in topics:
            topic_name = topic_info.get('topic', '')
            keywords = topic_info.get('keywords', [])
            for kw in keywords:
                if kw.lower() in text_lower:
                    topic_hits.add(topic_name)
                    break

    features['topic_diversity'] = len(topic_hits)

    return features


def extract_language_features(tweets: list, lang: str) -> dict:
    """extract language-specific features."""
    features = {}

    if not tweets:
        return {'accent_rate': 0.0, 'bullet_rate': 0.0}

    texts = [t['text'] for t in tweets]
    n = len(texts)

    if lang == 'fr':
        accent_pattern = re.compile(r'[àâäéèêëïîôùûüÿçœæÀÂÄÉÈÊËÏÎÔÙÛÜŸÇŒÆ]')
        accent_counts = [len(accent_pattern.findall(t)) for t in texts]
        features['accent_rate'] = np.mean(accent_counts)

        bullet_count = sum(1 for t in texts if re.match(r'^\s*[-•●]\s', t) or re.match(r'^\s*\d+[\.\)]\s', t))
        features['bullet_rate'] = bullet_count / n
    else:
        features['accent_rate'] = 0.0
        features['bullet_rate'] = 0.0

    return features


FEATURE_CACHE_VERSION = "v3"


def extract_features(dataset_path: str, use_cache: bool = True) -> pd.DataFrame:
    """main entry point: extract all features for all users in a dataset. returns a dataframe with user_id as index and features as columns."""
    # check cache
    dataset_id = os.path.splitext(os.path.basename(dataset_path))[0]
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.tmp')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'features_{dataset_id}_{FEATURE_CACHE_VERSION}.json')

    if use_cache and os.path.exists(cache_path):
        df = pd.read_json(cache_path, orient='table')
        return df

    # load data
    data = load_dataset(dataset_path)
    posts_by_author = group_posts_by_author(data['posts'])
    lang = data.get('lang', 'en')
    metadata = data['metadata']
    topics = metadata.get('topics', [])

    rows = []
    for user in data['users']:
        uid = user['id']
        tweets = posts_by_author.get(uid, [])

        features = {'user_id': uid}
        features.update(extract_temporal_features(tweets, metadata))
        features.update(extract_timestamp_collision_features(tweets))
        features.update(extract_stylometric_features(tweets))
        features.update(extract_compression_features(tweets))
        features.update(extract_uniformity_features(tweets))
        features.update(extract_template_features(tweets))
        features.update(extract_text_features(tweets))
        features.update(extract_account_features(user))
        features.update(extract_topic_features(tweets, topics))
        features.update(extract_language_features(tweets, lang))
        features.update(extract_dataset_normalization_features(user, tweets, metadata))

        # derived features
        features['num_tweets'] = len(tweets)

        rows.append(features)

    df = pd.DataFrame(rows).set_index('user_id')

    # cache as json
    df.to_json(cache_path, orient='table')

    return df


def get_feature_columns(lang: str = 'en') -> list:
    """return the list of feature columns used for ml model training."""
    base_features = [
        # tier 1: temporal (auc > 0.78)
        'hour_entropy',
        'active_hours',
        'peak_hour_frac',
        'cv_intervals',
        'burstiness',
        # tier 1: text
        'excl_rate',
        'hashtag_rate',
        # tier 2: strong signal
        'topic_diversity',
        'tweet_count',
        'z_score',
        'avg_tweet_len',
        'tweet_len_std',
        'emoji_rate',
        # tier 3: supplementary
        'out_of_range',
        'ttr',
        'mid_hashtag_rate',
        'mean_interval',
        'std_interval',
        'night_ratio',
        'url_rate',
        'mention_rate',
        'self_similarity',
        'username_len',
        'username_entropy',
        'username_digit_ratio',
        'has_description',
        'description_length',
        'question_rate',
        'num_tweets',
    ]

    if lang == 'fr':
        base_features.extend(['accent_rate', 'bullet_rate'])

    return base_features


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('usage: python features.py <dataset_path>')
        sys.exit(1)

    path = sys.argv[1]
    print(f'extracting features from {path}...')
    df = extract_features(path, use_cache=False)
    print(f'extracted {len(df)} users x {len(df.columns)} features')
    print(f'\nfeature columns: {list(df.columns)}')
    print('\nsample (first 5 users):')
    print(df.head())
