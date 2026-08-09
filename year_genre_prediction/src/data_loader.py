"""
Loads and parses the CMU Book Summary Dataset.

Expected file: data/booksummaries.txt  (TSV, no header)
Columns: wiki_id, freebase_id, title, author, pub_date, genres, summary
"""

import ast
import json
import re

import numpy as np
import pandas as pd

CMU_COLUMNS = [
    "wiki_id",
    "freebase_id",
    "title",
    "author",
    "pub_date",
    "genres",
    "summary",
]

YEAR_MIN = 1950
YEAR_MAX = 2008
TOP_N_GENRES = 15
MIN_GENRE_BOOKS = 100


def load_cmu(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=CMU_COLUMNS,
        encoding="utf-8",
        on_bad_lines="skip",
    )
    df = _parse_genres(df)
    df = _parse_year(df)
    df = _clean_basics(df)
    return df


def _parse_genres(df: pd.DataFrame) -> pd.DataFrame:
    def extract(raw):
        if not isinstance(raw, str) or raw.strip() == "":
            return []
        try:
            d = json.loads(raw)
            return list(d.values())
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            d = ast.literal_eval(raw)
            return list(d.values())
        except Exception:
            return []

    df["genre_list"] = df["genres"].apply(extract)
    return df


def _parse_year(df: pd.DataFrame) -> pd.DataFrame:
    def extract_year(raw):
        if not isinstance(raw, str):
            return np.nan
        m = re.search(r"\b(1[5-9]\d{2}|20[01]\d)\b", raw)
        return int(m.group()) if m else np.nan

    df["year"] = df["pub_date"].apply(extract_year)
    return df


def _clean_basics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["summary"])
    df = df[df["summary"].str.strip().str.len() > 50]
    return df.reset_index(drop=True)


def filter_year_range(
    df: pd.DataFrame, lo: int = YEAR_MIN, hi: int = YEAR_MAX
) -> pd.DataFrame:
    mask = df["year"].notna() & df["year"].between(lo, hi)
    return df[mask].copy().reset_index(drop=True)


def assign_decade(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["decade"] = (df["year"] // 10 * 10).astype(int)
    return df


def get_top_genres(
    df: pd.DataFrame, n: int = TOP_N_GENRES, min_books: int = MIN_GENRE_BOOKS
):
    from collections import Counter

    counts = Counter(g for genres in df["genre_list"] for g in genres)
    top = [g for g, c in counts.most_common() if c >= min_books][:n]
    return top


def encode_genres(df: pd.DataFrame, top_genres: list[str]) -> pd.DataFrame:
    for g in top_genres:
        col = _genre_col(g)
        df[col] = df["genre_list"].apply(lambda lst: int(g in lst))
    return df


def _genre_col(name: str) -> str:
    return "genre_" + re.sub(r"[^a-z0-9]", "_", name.lower()).strip("_")


def genre_label_matrix(df: pd.DataFrame, top_genres: list[str]) -> np.ndarray:
    cols = [_genre_col(g) for g in top_genres]
    return df[cols].values


def books_with_genre(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["genre_list"].map(len) > 0].copy()
