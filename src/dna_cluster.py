"""digital dna clustering for bot detection. builds behavior sequences per user and clusters similar sequences."""

import re
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.cluster import DBSCAN


def _parse_timestamp(ts: str) -> datetime:
    ts = ts.replace('Z', '+00:00')
    return datetime.fromisoformat(ts)


def _tweet_token(text: str, created_at: str) -> str:
    """encode a tweet into a single character token (night = lowercase)."""
    base = 'O'
    if 'http' in text or 't.co' in text:
        base = 'U'
    elif re.search(r'#\w+', text):
        base = 'H'
    elif re.search(r'@\w+', text):
        base = 'M'
    elif '?' in text:
        base = 'Q'
    elif '!' in text:
        base = 'E'

    try:
        hour = _parse_timestamp(created_at).hour
    except Exception:
        hour = None

    if hour is not None and 2 <= hour <= 5:
        return base.lower()
    return base


def build_dna_sequences(posts_by_author: dict) -> dict:
    """return {user_id: sequence_string} for each author."""
    sequences = {}
    for uid, posts in posts_by_author.items():
        # preserve time order
        posts_sorted = sorted(posts, key=lambda p: p['created_at'])
        tokens = [_tweet_token(p['text'], p['created_at']) for p in posts_sorted]
        sequences[uid] = ''.join(tokens)
    return sequences


def _kgrams(seq: str, k: int) -> set:
    if len(seq) < k:
        return {seq} if seq else set()
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def jaccard_distance_matrix(sequences: dict, k: int = 3) -> tuple:
    """compute jaccard distance matrix for dna sequences."""
    users = list(sequences.keys())
    grams = {u: _kgrams(sequences[u], k) for u in users}

    n = len(users)
    dist = np.zeros((n, n))
    for i in range(n):
        gi = grams[users[i]]
        for j in range(i + 1, n):
            gj = grams[users[j]]
            if not gi and not gj:
                d = 1.0
            else:
                inter = len(gi & gj)
                union = len(gi | gj)
                d = 1.0 - (inter / union if union > 0 else 0.0)
            dist[i, j] = d
            dist[j, i] = d
    return users, dist


def cluster_sequences(sequences: dict, eps: float = 0.3, min_samples: int = 3, k: int = 3) -> tuple:
    """cluster users using dbscan on precomputed jaccard distances."""
    users, dist = jaccard_distance_matrix(sequences, k=k)
    if len(users) == 0:
        return users, np.array([])

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = db.fit_predict(dist)
    return users, labels


def expand_from_clusters(users: list, labels: np.ndarray, seed_mask: dict,
                         min_seed: int = 2, min_seed_ratio: float = 0.6,
                         max_cluster_size: int = 15) -> set:
    """return expanded set of user ids based on seeded clusters."""
    clusters = defaultdict(list)
    for u, lbl in zip(users, labels):
        if lbl == -1:
            continue
        clusters[lbl].append(u)

    expanded = set()
    for _, members in clusters.items():
        if len(members) > max_cluster_size:
            continue
        seeds = [u for u in members if seed_mask.get(u, False)]
        if len(seeds) < min_seed:
            continue
        if len(seeds) / max(len(members), 1) < min_seed_ratio:
            continue
        expanded.update(members)

    return expanded
