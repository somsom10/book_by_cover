"""
Unsupervised clustering of book summaries.

Pipeline: TF-IDF → LSA (TruncatedSVD, 100 dims) → L2-normalize → KMeans.
Analysis: do clusters correspond to genres?
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


def cluster_summaries(
    texts: pd.Series,
    n_clusters: int = 12,
    n_components: int = 100,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, Pipeline, KMeans]:
    """Return (labels, X_lsa, lsa_pipe, km)."""
    lsa_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30_000, sublinear_tf=True, min_df=2)),
        ("svd",   TruncatedSVD(n_components=n_components, random_state=random_state)),
        ("norm",  Normalizer(copy=False)),
    ])
    X_lsa = lsa_pipe.fit_transform(texts)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300)
    labels = km.fit_predict(X_lsa)
    return labels, X_lsa, lsa_pipe, km


def reduce_to_2d(X_lsa: np.ndarray, random_state: int = 42) -> np.ndarray:
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(X_lsa)


def analyze_cluster_genres(
    labels: np.ndarray,
    genre_lists: pd.Series,
    top_genres: list[str],
) -> dict:
    """For each cluster: dominant genre, purity, size, genre distribution."""
    n_clusters = int(labels.max()) + 1
    results = {}
    for c in range(n_clusters):
        mask = labels == c
        genres_in_cluster = [
            g
            for gl in genre_lists[mask]
            for g in (gl or [])
            if g in top_genres
        ]
        counter = Counter(genres_in_cluster)
        total = len(genres_in_cluster)
        if total == 0:
            results[c] = {"dominant": "Unknown", "purity": 0.0, "size": int(mask.sum()), "distribution": {}}
        else:
            dominant, dom_count = counter.most_common(1)[0]
            results[c] = {
                "dominant":     dominant,
                "purity":       dom_count / total,
                "size":         int(mask.sum()),
                "distribution": {g: cnt / total for g, cnt in counter.most_common(6)},
            }
    return results


def get_cluster_top_terms(lsa_pipe: Pipeline, km: KMeans, n_terms: int = 12) -> dict:
    """Map cluster centroids back to vocabulary; return top terms per cluster."""
    tfidf = lsa_pipe.named_steps["tfidf"]
    svd   = lsa_pipe.named_steps["svd"]
    terms = tfidf.get_feature_names_out()
    centroids_feat = km.cluster_centers_ @ svd.components_  # (n_clusters, vocab)
    return {
        i: list(terms[np.argsort(row)[-n_terms:][::-1]])
        for i, row in enumerate(centroids_feat)
    }


def overall_purity(
    labels: np.ndarray,
    genre_lists: pd.Series,
    top_genres: list[str],
) -> float:
    """Weighted-average cluster purity across all genre-labeled books."""
    n_clusters = int(labels.max()) + 1
    total_genre_items, total_correct = 0, 0
    for c in range(n_clusters):
        mask = labels == c
        genres_in_cluster = [
            g for gl in genre_lists[mask] for g in (gl or []) if g in top_genres
        ]
        if genres_in_cluster:
            total_genre_items += len(genres_in_cluster)
            total_correct += Counter(genres_in_cluster).most_common(1)[0][1]
    return total_correct / max(total_genre_items, 1)
