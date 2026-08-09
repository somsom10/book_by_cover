"""
Correlation analysis: linguistic features vs. genre and year.

Covers:
 - Vocabulary richness over decades (lexical diversity, avg sentence length, etc.)
 - Pearson/Spearman correlation matrix of text stats vs. year
 - Per-genre vocabulary profiles (TF-IDF discrimination)
 - Keyword temporal trends
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer


def correlate_text_stats_with_year(
    stats_df: pd.DataFrame, years: pd.Series
) -> pd.DataFrame:
    """Pearson and Spearman correlations of each text statistic with year."""
    rows = []
    for col in stats_df.columns:
        valid = stats_df[col].notna() & years.notna()
        x, y = stats_df.loc[valid, col], years[valid]
        if len(x) < 10:
            continue
        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)
        rows.append(
            {
                "feature": col,
                "pearson_r": r_p,
                "pearson_p": p_p,
                "spearman_r": r_s,
                "spearman_p": p_s,
            }
        )
    return pd.DataFrame(rows).sort_values("pearson_r", key=abs, ascending=False)


def decade_text_stats(stats_df: pd.DataFrame, decades: pd.Series) -> pd.DataFrame:
    """Mean text statistics per decade."""
    combined = stats_df.copy()
    combined["decade"] = decades.values
    return combined.groupby("decade").mean().sort_index()


def genre_discriminating_words(
    texts: pd.Series,
    genre_labels: pd.Series,
    n: int = 20,
) -> dict[str, list[str]]:
    """For each genre, find words with highest mean TF-IDF in that genre vs. rest."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30_000,
        sublinear_tf=True,
        min_df=5,
    )
    X = vectorizer.fit_transform(texts)
    vocab = np.array(vectorizer.get_feature_names_out())
    X_arr = X.toarray()

    result = {}
    for genre in genre_labels.unique():
        mask = (genre_labels == genre).values
        if mask.sum() < 20:
            continue
        mean_in = X_arr[mask].mean(axis=0)
        mean_out = X_arr[~mask].mean(axis=0)
        diff = mean_in - mean_out
        top_idx = np.argsort(diff)[-n:][::-1]
        result[genre] = list(vocab[top_idx])

    return result


def keyword_over_time(
    texts: pd.Series,
    decades: pd.Series,
    keywords: list[str],
) -> pd.DataFrame:
    """Relative frequency of each keyword per decade."""
    rows = []
    for decade in sorted(decades.unique()):
        mask = decades == decade
        decade_texts = " ".join(texts[mask].fillna("")).lower()
        total_words = len(decade_texts.split()) or 1
        row = {"decade": decade}
        for kw in keywords:
            count = decade_texts.count(f" {kw} ")
            row[kw] = count / total_words * 10_000  # per 10k words
        rows.append(row)
    return pd.DataFrame(rows).set_index("decade")


def top_tfidf_words_per_decade(
    texts: pd.Series,
    decades: pd.Series,
    n: int = 15,
) -> dict[int, list[str]]:
    """Most distinctive TF-IDF words for each decade (decade as 'document')."""
    decade_docs = {}
    for d in sorted(decades.unique()):
        decade_docs[d] = " ".join(texts[decades == d].fillna(""))

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1), max_features=20_000, sublinear_tf=True, min_df=2
    )
    labels = sorted(decade_docs.keys())
    corpus = [decade_docs[d] for d in labels]
    X = vectorizer.fit_transform(corpus)
    vocab = np.array(vectorizer.get_feature_names_out())

    result = {}
    for i, decade in enumerate(labels):
        top_idx = np.argsort(X[i].toarray().ravel())[-n:][::-1]
        result[decade] = list(vocab[top_idx])

    return result
