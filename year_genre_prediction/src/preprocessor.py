"""
Text preprocessing for book summaries.
Cleans text, removes stopwords, optionally lemmatizes.
"""

import re
import string

import nltk
import numpy as np

# Download on first use
for _pkg in ("stopwords", "wordnet", "punkt"):
    try:
        nltk.data.find(f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()
_PUNCT_RE = re.compile(r"[%s]" % re.escape(string.punctuation))
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str, lemmatize: bool = False, remove_stops: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    tokens = text.split()
    if remove_stops:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    if lemmatize:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def preprocess_series(series, lemmatize: bool = False, remove_stops: bool = True):
    return series.apply(
        lambda t: clean_text(t, lemmatize=lemmatize, remove_stops=remove_stops)
    )


def text_statistics(series) -> "pd.DataFrame":
    import pandas as pd

    raw = series.fillna("")
    cleaned = preprocess_series(raw, remove_stops=False)

    df = pd.DataFrame()
    df["char_count"] = raw.str.len()
    df["word_count"] = raw.str.split().str.len()
    df["unique_words"] = cleaned.str.split().apply(
        lambda t: len(set(t)) if isinstance(t, list) else 0
    )
    df["lexical_diversity"] = df["unique_words"] / df["word_count"].replace(0, np.nan)
    df["avg_word_len"] = raw.apply(
        lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 0
    )
    df["sentence_count"] = raw.str.count(r"[.!?]+")
    df["avg_sent_len"] = df["word_count"] / df["sentence_count"].replace(0, np.nan)
    return df
