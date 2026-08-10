import os
import gzip
import json
import math
import re
import random
from collections import Counter, defaultdict

import pandas as pd
import networkx as nx
import spacy
from itertools import combinations

# the English language model
nlp = spacy.load("en_core_web_sm")

# --- configuration ---
#
# The three raw files are downloaded into data/ by download_data.py. Scripts
# run from work/, so data/ is ../data - but several locations are tried, so
# running from another directory does not fail on a hardcoded path
_DATA_DIRS = ["../data", "data", ".",
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")]


def _find_data(name):
    for d in _DATA_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(_DATA_DIRS[0], name)


# CMU plot summaries (booksummaries.txt, from booksummaries.tar.gz)
CMU_PATH = _find_data("booksummaries.txt")
# the general Goodreads dump (all genres)
GOODREADS_PATH = _find_data("goodreads_books.json.gz")
# the Goodreads works file, which holds the original publication year
WORKS_PATH = _find_data("goodreads_book_works.json.gz")
# cache for the Goodreads sample, to avoid rescanning 2.36M lines every run
GOODREADS_CACHE = "goodreads_sample.pkl"

# Books sampled per decade. Sampling by decade rather than once globally stops
# the last two decades from making up almost the whole sample
PER_DECADE_SAMPLE = 250
# a decade with fewer books is flagged as unreliable but still shown
MIN_BOOKS_PER_DECADE = 30
# a word must appear in several blurbs per decade, so one book cannot make a keyword
MIN_DOC_FREQ = 3
# the dump was collected in 2017, so a later year is a data error
MAX_YEAR = 2017
# width of the word column in the output, so the notes beside it line up
_COL_WIDTH = 56

# Parts of speech kept: nouns, proper nouns, adjectives and verbs. Verbs are
# kept because they carry topical meaning (kill, escape, inherit); light verbs
# (find, take, tell) are handled by the distinctiveness score further down
_ALLOWED_POS = frozenset({"NOUN", "PROPN", "ADJ", "VERB"})


def preprocess_texts(texts):
    """Tokenise, drop stop words and lemmatise, in one nlp.pipe batch."""
    results = []
    # parser and ner are disabled for speed. The tagger stays on, so token.pos_
    # is available at no extra cost
    for doc in nlp.pipe(texts, disable=["parser", "ner"], batch_size=64):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and token.is_alpha and token.pos_ in _ALLOWED_POS
        ]
        results.append(tokens)
    return results


def _pagerank_scores(tokens, window_size=4):
    """Co-occurrence graph plus PageRank. Empty if the text is too short."""
    graph = nx.Graph()

    # edges from co-occurrence inside a sliding window
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        for w1, w2 in combinations(window, 2):
            if w1 == w2:
                continue
            if graph.has_edge(w1, w2):
                graph[w1][w2]['weight'] += 1
            else:
                graph.add_edge(w1, w2, weight=1)

    # edge case: text too short
    if len(graph.nodes) == 0:
        return {}

    # run PageRank over the graph
    return nx.pagerank(graph, weight='weight')


def extract_top_keywords(tokens, window_size=4, top_n=5):
    scores = _pagerank_scores(tokens, window_size=window_size)
    ranked_words = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [word for word, score in ranked_words[:top_n]]


def _extract_year(value):
    match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", str(value))
    if not match:
        return None
    year = int(match.group(1))
    # future years are data errors in a dump collected in 2017
    return year if year <= MAX_YEAR else None


def _stratify_by_decade(df, per_decade=PER_DECADE_SAMPLE, random_state=42):
    """Sample up to per_decade books per decade, so decades stay comparable."""
    parts = []
    for _, group in df.groupby(df["Year"] // 10 * 10):
        if len(group) > per_decade:
            group = group.sample(n=per_decade, random_state=random_state)
        parts.append(group)
    return pd.concat(parts).reset_index(drop=True)


def load_cmu(path=CMU_PATH):
    """Load the CMU summaries (headerless TSV): column 4 date, column 6 summary."""
    cols = ["wiki_id", "freebase_id", "title", "author", "pub_date", "genres", "summary"]
    df = pd.read_csv(path, sep="\t", header=None, names=cols, quoting=3)

    df["Year"] = df["pub_date"].apply(_extract_year)
    df["Summary"] = df["summary"].astype(str)
    df = df.dropna(subset=["Year"])
    df = df[df["Summary"].str.strip().str.len() > 0]
    df["Year"] = df["Year"].astype(int)
    return df[["Year", "Summary"]].reset_index(drop=True)


def _is_english_code(lang_code):
    """
    True for English codes and for an empty one: 45% of rows carry no
    language_code and most are still English, so empty means unknown and falls
    through to _is_english_text.
    """
    lang = str(lang_code).strip().lower()
    return lang == "" or lang in ("eng", "en") or lang.startswith("en-")


# Characters from non-Latin scripts: Greek, Cyrillic, Hebrew, Arabic,
# Devanagari, Thai, Chinese/Japanese and Korean.
# Accented Latin letters and typographic punctuation are deliberately excluded,
# so ordinary English blurbs with curly quotes or an em dash are not rejected.
_FOREIGN_SCRIPT_RE = re.compile(
    "["
    "Ͱ-Ͽ"  # Greek
    "Ѐ-ӿ"  # Cyrillic
    "֐-׿"  # Hebrew
    "؀-ۿ"  # Arabic
    "ऀ-ॿ"  # Devanagari
    "฀-๿"  # Thai
    "぀-ヿ"  # Japanese kana
    "㐀-鿿"  # Chinese ideographs
    "가-힯"  # Korean hangul
    "]"
)

# A short list of very common English function words. Real English text is full
# of them, while foreign text - including the Latin transliteration of
# Arabic/Hebrew that this dump stores as ASCII, e.g. "yHwl mw'lf lktb ldktwr" -
# barely contains them at all.
_ENGLISH_STOP_WORDS = frozenset(
    "the a an and or but of to in is was were be been are for with on at by from "
    "that this it as not his her he she they we you i had have has will would "
    "there their which who what when".split()
)
_WORD_RE = re.compile(r"[a-z']+")

# minimum share of function words for a blurb to count as English
_MIN_ENGLISH_RATIO = 0.10
# blurbs shorter than this are too short for the ratio to mean anything
_MIN_WORDS = 8


def _is_english_text(text):
    """
    True if the blurb itself looks English. language_code is missing or wrong
    across much of the dump, so this is the filter that actually removes foreign
    text - including foreign text written in Latin letters.
    """
    if _FOREIGN_SCRIPT_RE.search(text):
        return False
    words = _WORD_RE.findall(text.lower())
    if len(words) < _MIN_WORDS:
        return False
    stop_hits = sum(word in _ENGLISH_STOP_WORDS for word in words)
    return stop_hits / len(words) >= _MIN_ENGLISH_RATIO


def load_work_years(path=WORKS_PATH):
    """
    work_id -> the work's original publication year.

    Every row in the books file is an edition, so its publication_year is that
    printing; this file is what allows dating by first publication.
    """
    work_years = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # skip non-book items (an empty media_type counts as a book)
            media_type = str(obj.get("media_type", "")).strip()
            if media_type and media_type != "book":
                continue
            year = _extract_year(obj.get("original_publication_year", ""))
            if year is None:
                continue
            try:
                work_years[int(obj["work_id"])] = year
            except (KeyError, TypeError, ValueError):
                continue
    return work_years


def load_goodreads(path=GOODREADS_PATH, works_path=WORKS_PATH,
                   per_decade=PER_DECADE_SAMPLE, random_state=42):
    """
    The Goodreads dump as Year and Summary.

    Dated by the work's original publication year, deduplicated so a classic with
    60 editions is not counted 60 times, and reservoir-sampled per decade so recent
    decades cannot swamp the rest.
    """
    work_years = load_work_years(works_path)
    rng = random.Random(random_state)
    reservoirs = defaultdict(list)   # decade -> the books sampled
    seen_per_decade = Counter()      # decade -> eligible books seen so far
    seen_works = set()

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            try:
                work_id = int(obj.get("work_id"))
            except (TypeError, ValueError):
                continue
            # deduplicate: a work already taken is not counted again
            if work_id in seen_works:
                continue
            year = work_years.get(work_id)
            if year is None:
                continue
            # filters: English only, and a non-empty blurb
            if not _is_english_code(obj.get("language_code", "")):
                continue
            summary = str(obj.get("description", "")).strip()
            if not summary:
                continue
            # drop foreign text that slipped past the language_code filter
            if not _is_english_text(summary):
                continue

            seen_works.add(work_id)
            decade = year // 10 * 10
            seen_per_decade[decade] += 1
            bucket = reservoirs[decade]
            row = {"Year": year, "Summary": summary}
            # reservoir sampling within the decade: every eligible book has an
            # equal chance of inclusion
            if len(bucket) < per_decade:
                bucket.append(row)
            else:
                j = rng.randint(0, seen_per_decade[decade] - 1)
                if j < per_decade:
                    bucket[j] = row

    rows = [row for decade in sorted(reservoirs) for row in reservoirs[decade]]
    return pd.DataFrame(rows)


def load_goodreads_cached(cache_path=GOODREADS_CACHE, force_reload=False, **kwargs):
    """load_goodreads with an on-disk cache; the scan takes minutes."""
    if not force_reload and os.path.exists(cache_path):
        print(f"Loading cached Goodreads sample from {cache_path}")
        return pd.read_pickle(cache_path)
    df = load_goodreads(**kwargs)
    df.to_pickle(cache_path)
    print(f"Saved Goodreads sample to {cache_path}")
    return df


def analyze_dataframe(df, dataset_name, window_size=4, top_n=5):
    """
    Per decade, the central words by TextRank and the same words weighted by how
    specific they are to that decade (pagerank * log(1 + lift)).
    """
    print("=" * 78)
    print(f"Dataset: {dataset_name}  ({len(df)} books after sampling)")
    print("=" * 78)

    # bucket into decades
    df = df.copy()
    df["Decade"] = (df["Year"] // 10) * 10

    # preprocess every blurb in one batch
    df["Tokens"] = preprocess_texts(df["Summary"].tolist())

    # a second pass for corpus-wide frequencies, needed for lift
    corpus_counts = Counter()
    for tokens in df["Tokens"]:
        corpus_counts.update(tokens)
    corpus_total = sum(corpus_counts.values())
    if corpus_total == 0:
        print("No tokens after preprocessing.\n")
        return

    print("Top keywords per decade:")
    print("-" * 78)

    # walk the decades in order
    for decade in sorted(df["Decade"].unique()):
        decade_rows = df[df["Decade"] == decade]
        # pool every word from every blurb in the decade
        combined_tokens = [tok for tokens in decade_rows["Tokens"] for tok in tokens]

        scores = _pagerank_scores(combined_tokens, window_size=window_size)
        top_raw = [w for w, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]

        # frequency within the decade, and how many distinct blurbs hold each word
        decade_counts = Counter(combined_tokens)
        decade_total = len(combined_tokens)
        doc_freq = Counter()
        for tokens in decade_rows["Tokens"]:
            doc_freq.update(set(tokens))

        distinctive = []
        for word, pagerank in scores.items():
            # a word in few blurbs may come from a single book
            if doc_freq[word] < MIN_DOC_FREQ:
                continue
            p_decade = decade_counts[word] / decade_total
            p_corpus = corpus_counts[word] / corpus_total
            distinctive.append((word, pagerank * math.log(1 + p_decade / p_corpus)))
        top_distinctive = [w for w, _ in sorted(distinctive, key=lambda kv: kv[1], reverse=True)[:top_n]]

        # a thin decade is flagged rather than hidden: hiding it would hide how
        # thin the record is
        flag = "  [low-confidence]" if len(decade_rows) < MIN_BOOKS_PER_DECADE else ""
        print(f"Decade: {decade}s | Books: {len(decade_rows):4d}{flag}")
        # each column answers a different question, so both are printed
        print(f"    Top TextRank : {str(top_raw):<{_COL_WIDTH}} <- what vocabulary dominates")
        print(f"    Distinctive  : {str(top_distinctive):<{_COL_WIDTH}} <- what is unusual vs other decades")
    print()


def main():
    # process each corpus separately, one at a time
    print("Loading CMU book summaries...")
    cmu = _stratify_by_decade(load_cmu())
    analyze_dataframe(cmu, "CMU Book Summaries")

    print("Loading Goodreads (this streams 2.36M lines on the first run)...")
    goodreads = load_goodreads_cached()
    analyze_dataframe(goodreads, "Goodreads (all genres, English, dated by first publication)")


if __name__ == "__main__":
    main()
