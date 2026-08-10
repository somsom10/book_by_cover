"""
Theme detection over decades: NMF on a TF-IDF matrix of book descriptions.

One topic set is fixed for the whole corpus and each topic's weight is measured
per decade, so a perennial theme like love gets a curve over time instead of
disappearing for not being distinctive to any one decade.

TextRank was tried first and discarded: its PageRank score tracked node degree
almost exactly (CV 0.173), making it a weighted word count that returned the
same generic words in every decade.
"""

import hashlib
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

import text as T

# --- configuration ---
# The canonical corpus is the bounded one, so it is the default. The file is
# built by build_bounded_corpus.py; build_corpus alone does NOT bound, so it
# must never be allowed to produce a file under this name - see the check in main
CACHE_PATH = "themes_corpus_bounded.pkl"

# Output directory. The CMU control run writes to its own folder, or it would
# overwrite the Goodreads results - and comparing the two is its whole point
OUT_DIR = "."


def _out(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)
# contamination per decade, written during cleaning since only then is it known
CLEAN_STATS_PATH = "artifact_clean_stats.csv"

# Number of topics. A K sweep (12/15/18/22/25) showed topics widening and
# merging below 18, and splitting into remainders at the top of the range
N_TOPICS = 25

# Books per decade; smaller decades are taken whole. 10,000 rather than 5,000
# because seven of twelve decades were capped by the old quota - more books were
# available and simply were not sampled
PER_DECADE_CAP = 10000

# Separate cap for TRAINING, so the topic space is not defined by the 2010s
# alone. 4,000 rather than 1,500 was measured, not guessed: science fiction did
# not form at all at 1,500, was unstable at 2,500 (0.587), and reaches 0.990 at
# 4,000. Shares themselves are computed over the whole corpus
FIT_PER_DECADE = 4000

# minimum blurb length after cleaning
MIN_SUMMARY_CHARS = 200

# Earliest decade analysed. Raised from 1790: pre-1900 decades hold 66-984 books
# against 1,576-4,808 after, with 16% bibliographic noise against 6%, and are a
# canon of survivors rather than a sample of what was published. Applied before
# training, since those decades also skewed the topic space. The cache stays
# complete, so setting this back to 1790 needs no rebuild
MIN_DECADE = 1900

# --- the cleaning step ---

# Printer and scanner boilerplate found in the description field in place of a
# real blurb. These are not book descriptions at all, so they are deleted even
# when they appear only once. Measured: 7.7% of rows, 11% before 1900, up to 30%
# in individual decades.
_BOILERPLATE_RE = re.compile(
    "|".join([
        r"converted from its physical edition",
        r"pre-19\d\d historical reproduction",
        r"occasional imperfections",
        r"republishing these classic works",
        r"reproduction of a book published",
        r"digitization process",
        r"scanning process",
        r"kindle edition includes wireless",
        r"culturally important",
        r"print on demand",
        r"facsimile",
        r"we have elected to bring",
        r"original artifact",
        r"quality assurance was conducted",
        r"this book may have occasional",
        r"scanned (?:copy|image|version)",
        r"optical character recognition",
    ]),
    re.I,
)

# prefix length used to detect near-duplicates (see _dedupe)
_PREFIX_LEN = 120


def _normalise(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


# A sentence repeated across different books is not a description of a book but
# publisher text. The threshold was measured on the corpus: at 10 and above, all
# 55 sentence types found are marketing or series advertising, with no case of
# real content, and no document is emptied.
_SENTENCE_MIN_DOC_FREQ = 10
# shorter sentences are never counted and never removed ("full-color illustrations.")
_MIN_SENTENCE_CHARS = 25
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _sentence_parts(text):
    """
    Split into sentences as (original, normalised key). Matching uses the key,
    deletion uses the original, so capitalisation survives for spaCy's tagger.
    """
    parts = []
    for piece in _SENT_SPLIT_RE.split(str(text).strip()):
        piece = piece.strip()
        if piece:
            parts.append((piece, _normalise(piece)))
    return parts


def strip_repeated_sentences(df, min_df=_SENTENCE_MIN_DOC_FREQ, verbose=True):
    """
    Delete sentences that recur across different books; they are publisher text,
    not description. Removes paragraphs rather than whole rows, so a real blurb
    with advertising on the end survives. Adds StrippedChars per row.
    """
    split = df["Summary"].map(_sentence_parts)

    counts = Counter()
    for parts in split:
        # once per document: a book repeating its own sentence cannot push it
        # over the threshold
        counts.update({k for _, k in parts if len(k) >= _MIN_SENTENCE_CHARS})
    repeated = {k for k, n in counts.items() if n >= min_df}

    kept_text, stripped_chars = [], []
    for parts in split:
        keep, removed = [], 0
        for original, key in parts:
            if key in repeated:
                removed += len(original)
            else:
                keep.append(original)
        kept_text.append(" ".join(keep))
        stripped_chars.append(removed)

    out = df.copy()
    out["Summary"] = kept_text
    out["StrippedChars"] = stripped_chars

    if verbose:
        touched = int((out["StrippedChars"] > 0).sum())
        total = int(out["Summary"].str.len().sum() + out["StrippedChars"].sum())
        pct = out["StrippedChars"].sum() / total if total else 0.0
        print(f"    -{len(repeated):6d} repeated sentence types "
              f"(>={min_df} docs) removed from {touched} documents, {pct:.2%} of text")
    return out, repeated


def _decade(df):
    return df["Year"] // 10 * 10


def clean_corpus(df, verbose=True, stats_path=CLEAN_STATS_PATH, bound_fn=None):
    """
    Remove text that does not describe the book, in five ordered steps: printer
    boilerplate, exact duplicates, near duplicates, repeated sentences, then the
    length filter.

    Order is load-bearing. The length filter must come last, since the earlier
    steps and bound_fn shorten documents and push some below MIN_SUMMARY_CHARS.

    bound_fn takes the whole series, not one document, because bounding needs the
    repeated-sentence set derived from the corpus at this point.
    """
    n0 = len(df)
    books0 = _decade(df).value_counts()

    df = df[~df["Summary"].str.contains(_BOILERPLATE_RE, na=False)]
    n1 = len(df)

    norm = df["Summary"].map(_normalise)
    df = df[~norm.map(lambda s: hashlib.md5(s.encode()).hexdigest()).duplicated()]
    n2 = len(df)

    norm = df["Summary"].map(_normalise)
    df = df[~norm.str[:_PREFIX_LEN].duplicated()]
    n3 = len(df)

    if verbose:
        print(f"  cleaning: {n0} rows")
        print(f"    -{n0-n1:6d} printer/scanner boilerplate")
        print(f"    -{n1-n2:6d} exact duplicate text")
        print(f"    -{n2-n3:6d} near-duplicate text (shared opening)")

    df, removed_sentences = strip_repeated_sentences(df, verbose=verbose)

    # Step 4.5: bounding. It MUST run here, between repeated-sentence removal
    # and the length filter, not after clean_corpus - bounding shortens
    # documents and some drop below MIN_SUMMARY_CHARS only as a result. Running
    # it in another order produces a different corpus for 4% of documents, and
    # from there an entirely different model. bound_fn is passed in from outside
    # so that themes.py does not depend on bounding.py.
    # bound_fn takes the whole series rather than one document, because bounding
    # needs the repeated-sentence set - and that is derived from the corpus AT
    # THIS POINT, after cleaning. Computing it on the raw text returns too large
    # a set and deletes 35 documents too many
    if bound_fn is not None:
        df = df.copy()
        df["Summary"] = list(bound_fn(df["Summary"]))
        if verbose:
            print(f"    bounded {len(df)} summaries")

    # removed characters are counted before the length filter, or documents
    # deleted BECAUSE of the removal would not be counted in the contamination
    # that deleted them
    dec_strip = _decade(df)
    stripped = df["StrippedChars"].groupby(dec_strip).sum()
    kept_chars = df["Summary"].str.len().groupby(dec_strip).sum()

    df = df[df["Summary"].str.len() >= MIN_SUMMARY_CHARS]
    n4 = len(df)

    if verbose:
        print(f"    -{n3-n4:6d} shorter than {MIN_SUMMARY_CHARS} chars")
        print(f"  => {n4} rows kept ({n4/n0:.1%})")

    df = df.reset_index(drop=True)

    stats = pd.DataFrame({
        "n_books_before": books0,
        "n_books": _decade(df).value_counts(),
        "chars_stripped": stripped,
        "chars_kept": kept_chars,
    }).fillna(0).astype("int64")
    stats.index.name = "decade"
    stats = stats.sort_index()
    total_chars = (stats["chars_stripped"] + stats["chars_kept"]).replace(0, np.nan)
    stats["pct_chars_stripped"] = stats["chars_stripped"] / total_chars * 100
    stats["pct_docs_dropped"] = (
        (stats["n_books_before"] - stats["n_books"])
        / stats["n_books_before"].replace(0, np.nan) * 100
    )
    stats.to_csv(_out(stats_path))
    if verbose:
        print(f"  wrote per-decade cleaning stats to {stats_path} "
              f"({len(removed_sentences)} sentence types removed)")
    return df


# --- multi-word terms ---
#
# "science fiction" is a genre name, not two words. Left split, the model sends
# science to the philosophy topic (philosophy of science) and fiction to the
# bibliographic one - leaving science fiction with only generic worldbuilding
# vocabulary
# (world, earth, human, power) shared with fantasy and adventure. That is why
# the topic only scored 0.779 for reproducibility: it has no unique anchor.
#
# The same treatment "New York Times" gets in bounding.py, for exactly the same
# reason. Phrases are glued into one token AFTER lemmatisation, so spaCy does
# not have to be re-run. The numbers in brackets are how many documents contain
# each phrase.
MULTIWORD_TERMS = [
    ("science fiction", "science_fiction"),   # 625
    ("new york", "new_york"),                 # 1441
    ("world war", "world_war"),               # 1322
    ("short story", "short_story"),           # 1199
    ("civil war", "civil_war"),               # 630
    ("fairy tale", "fairy_tale"),             # 290
    ("picture book", "picture_book"),         # 169
    ("young adult", "young_adult"),           # 108
    ("graphic novel", "graphic_novel"),       # 92
    ("private eye", "private_eye"),           # 68
    ("wild west", "wild_west"),               # 38
    ("coming age", "coming_of_age"),          # 36
    ("true crime", "true_crime"),             # 22
]


def join_multiword(lemmas):
    for src, dst in MULTIWORD_TERMS:
        lemmas = lemmas.str.replace(src, dst, regex=False)
    return lemmas


def load_goodreads_full(path=T.GOODREADS_PATH, works_path=T.WORKS_PATH,
                        per_decade_cap=PER_DECADE_CAP, random_state=42):
    """Goodreads dated by original publication year and deduplicated by edition."""
    import gzip, json, random
    from collections import defaultdict, Counter

    work_years = T.load_work_years(works_path)
    rng = random.Random(random_state)
    buckets = defaultdict(list)
    seen_per_decade = Counter()
    seen_works = set()

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            try:
                work_id = int(obj.get("work_id"))
            except (TypeError, ValueError):
                continue
            if work_id in seen_works:
                continue
            year = work_years.get(work_id)
            if year is None:
                continue
            if not T._is_english_code(obj.get("language_code", "")):
                continue
            summary = str(obj.get("description", "")).strip()
            if not summary or not T._is_english_text(summary):
                continue

            seen_works.add(work_id)
            decade = year // 10 * 10
            seen_per_decade[decade] += 1
            bucket = buckets[decade]
            row = {"Year": year, "Summary": summary}
            if per_decade_cap is None or len(bucket) < per_decade_cap:
                bucket.append(row)
            else:
                j = rng.randint(0, seen_per_decade[decade] - 1)
                if j < per_decade_cap:
                    bucket[j] = row

    rows = [r for d in sorted(buckets) for r in buckets[d]]
    return pd.DataFrame(rows)


def _apply_multiword(df):
    if "Lemmas" in df.columns:
        df = df.copy()
        df["Lemmas"] = join_multiword(df["Lemmas"])
    return df


def _apply_min_decade(df, min_decade=MIN_DECADE):
    """
    Drop decades too thin to read. Applied at load, so the cache stays complete
    and changing MIN_DECADE needs no rebuild.
    """
    if min_decade is None:
        return df
    n0 = len(df)
    out = df[df["Decade"] >= min_decade].reset_index(drop=True)
    if len(out) < n0:
        print(f"  dropped {n0 - len(out)} documents before {min_decade} "
              f"(too few books per decade to estimate a share)")
    return out


def build_corpus(cache_path=CACHE_PATH, force_reload=False, min_decade=MIN_DECADE):
    """Load, clean and lemmatise. Cached: lemmatising is the slow part."""
    if not force_reload and os.path.exists(cache_path):
        print(f"Loading cached corpus from {cache_path}")
        return _apply_min_decade(_apply_multiword(pd.read_pickle(cache_path)), min_decade)

    print("Streaming Goodreads (2.36M lines)...")
    df = load_goodreads_full()
    print(f"  {len(df)} works after de-duplicating editions")
    df = clean_corpus(df)

    print("Lemmatising...")
    df["Lemmas"] = [" ".join(toks) for toks in T.preprocess_texts(df["Summary"].tolist())]
    df["Decade"] = df["Year"] // 10 * 10
    df = df[df["Lemmas"].str.len() > 0].reset_index(drop=True)
    df.to_pickle(cache_path)
    print(f"Saved corpus to {cache_path}")
    return _apply_min_decade(df, min_decade)


CMU_CACHE_PATH = "cmu_corpus.pkl"


def build_cmu_corpus(cache_path=CMU_CACHE_PATH, force_reload=False,
                     min_decade=MIN_DECADE):
    """
    The same pipeline over CMU summaries, as a control.

    CMU summaries are reader-written plot descriptions with no marketing register,
    so this answers one question: does the pipeline manufacture metadata topics, or
    find them because they exist in Goodreads? No parameter changes - only the
    source of the text.
    """
    if not force_reload and os.path.exists(cache_path):
        print(f"Loading cached CMU corpus from {cache_path}")
        return _apply_min_decade(_apply_multiword(pd.read_pickle(cache_path)), min_decade)

    print("Loading CMU book summaries...")
    df = T.load_cmu()
    print(f"  {len(df)} summaries with a usable year")
    df = clean_corpus(df)

    print("Lemmatising...")
    df["Lemmas"] = [" ".join(toks) for toks in T.preprocess_texts(df["Summary"].tolist())]
    df["Decade"] = df["Year"] // 10 * 10
    df = df[df["Lemmas"].str.len() > 0].reset_index(drop=True)
    df.to_pickle(cache_path)
    print(f"Saved CMU corpus to {cache_path}")
    return _apply_min_decade(df, min_decade)


# --- the blocked word list ---
# Words that describe the book as a product rather than its content: publisher
# marketing and edition statements. They are removed from the vocabulary before
# TF-IDF, so they cannot form a topic. Deliberately a static list - simple to
# read and to edit by hand.

# Group A: measured, see keyness.py. At least 1.5x more frequent on the
# publisher side in ~85% of decades, and predicting nothing about the book
# underneath (phi < 0.05). Words that DID predict the plot - war, adventure,
# mystery, world, america, century, history - were left out despite also being
# significant
_KEYNESS_REGISTER = frozenset("""
available beloved classic compelling delight depiction edition endure
exciting extraordinary fan feature genre illuminate insight inspire
masterpiece original print profound range reader reading realism remarkable
series style text unforgettable unique vivid weave
""".split())

# Group B: not measured, but cannot be content - the book as a physical object
# and the publishing apparatus. Several scored a near-zero phi in passing (isbn
# -0.003, paperback 0.010, bestseller 0.013), confirming the reading.
# award, prize and winner were added later: they formed a whole topic -
# "author, win, award, biography, experience, bestselle, prize, account" -
# whose share is flat across all twelve decades (3.6% -> 2.4%), i.e. noise
# eating 3% of every decade rather than a trend. "win" was kept - winning a
# battle or a race is content
_BIBLIOGRAPHIC_STOPWORDS = frozenset("""
isbn ebook paperback hardcover hardback audiobook facsimile reprint reissue
printing imprint typo typography typeset pagination page pages
foreword afterword preface appendix bibliography glossary footnote endnote
errata imperfection scanned scan ocr blur blurred
bestseller bestselling bestselle award awards prize winner fascinating
""".split())

# Group C: book-apparatus vocabulary, added by hand although phi put it in the
# grey zone (0.05-0.11). The measurement misses precisely this because CMU
# summaries carry bibliographic framing of their own ("in the final chapter"),
# so when the reference corpus is contaminated in the same direction a publisher
# word's phi is pulled up
_APPARATUS_STOPWORDS = frozenset("""
publish publication volume illustration translation novella prose
literature introduction
""".split())

# Group D: contentless vocabulary - neither metadata nor register, simply words
# that do not distinguish one topic from another. Found by measurement: in the
# previous run four of the 25 topics were built entirely from them -
#   thing, good, go, come, day, want, get, time
#   find, way, help, discover, search, place, turn, home
#   know, want, secret, people, feel, answer, need, fact
#   world, people, great, ii, live, create, change, human
# i.e. 16% of model capacity went on empty words, leaving science fiction no
# topic of its own. max_df=0.5 filtered nothing at all, since no word appears in
# more than half the documents, so this has to be explicit.
# Kept deliberately: world, people, human, place, home, secret, life, age,
# struggle, hope, country, death - topical even though common
_EMPTY_VOCABULARY = frozenset("""
go come get take give find know want feel need tell begin bring turn help
make look follow continue draw choose remain lead move call act reveal fill
serve deal prove appear hold grow form bear capture encounter set discover
search seem become use provide mark note cause result point order thing way
sense group name interest good great different large strong special free
kind late second
""".split()) | {"new"}

# "new" is removed for its own reason: in 9,223 documents (22%) it held together
# one incoherent topic (new, york, city, testament, orleans) by bridging New
# York, New Orleans, New Testament, New England and the generic "a new X". A
# homonym, not a topic

# what actually goes into TfidfVectorizer
STOPWORDS = frozenset(
    _KEYNESS_REGISTER | _BIBLIOGRAPHIC_STOPWORDS | _APPARATUS_STOPWORDS
    | _EMPTY_VOCABULARY)


def fit_topics(df, n_topics=N_TOPICS, fit_per_decade=FIT_PER_DECADE, random_state=42):
    """
    TF-IDF, then NMF into W (document x topic) and H (topic x word).

    Fitted on a decade-balanced sample so the topic space is not defined by the
    largest decades, then all documents are projected onto it.
    """
    fit_idx = []
    for _, group in df.groupby("Decade"):
        fit_idx.extend(group.sample(min(len(group), fit_per_decade),
                                    random_state=random_state).index)
    fit_rows = df.loc[fit_idx]
    print(f"Fitting on a decade-balanced sample of {len(fit_rows)} documents "
          f"(<= {fit_per_decade} per decade)")

    vectorizer = TfidfVectorizer(min_df=5, max_df=0.5, sublinear_tf=True,
                                 stop_words=list(STOPWORDS))
    X_fit = vectorizer.fit_transform(fit_rows["Lemmas"])
    print(f"  TF-IDF matrix: {X_fit.shape[0]} docs x {X_fit.shape[1]} words "
          f"({len(STOPWORDS)} words blacklisted)")

    # no regularisation: sklearn multiplies alpha_H by the number of samples,
    # so a seemingly small value becomes a large L1 penalty and zeroes H entirely
    nmf = NMF(n_components=n_topics, init="nndsvda", random_state=random_state,
              max_iter=800, tol=1e-5)
    nmf.fit(X_fit)

    X_all = vectorizer.transform(df["Lemmas"])
    W = nmf.transform(X_all)
    # row-normalise so each document distributes as percentages across topics
    row_sums = W.sum(axis=1, keepdims=True)
    W = np.divide(W, row_sums, out=np.zeros_like(W), where=row_sums > 0)
    return vectorizer, nmf, W


def topic_labels(vectorizer, nmf, top_n=12):
    words = np.array(vectorizer.get_feature_names_out())
    return [", ".join(words[np.argsort(-row)[:top_n]]) for row in nmf.components_]


def decade_profiles(df, W):
    """
    Mean share per topic per decade, with a 95% interval. The interval states how
    thin a decade is rather than hiding it behind uniform sampling.
    """
    means, cis, counts = {}, {}, {}
    for decade, idx in df.groupby("Decade").indices.items():
        block = W[idx]
        n = len(idx)
        means[decade] = block.mean(axis=0)
        cis[decade] = 1.96 * block.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(W.shape[1], np.nan)
        counts[decade] = n
    return means, cis, counts


# A document left with less mass than this after the artifact topics are
# removed cannot be renormalised: dividing by a tiny number inflates numerical
# noise into a full unit vector. It is dropped, and the count is reported per
# decade as another measure of contamination
_MIN_CONTENT_MASS = 0.01


def exclude_artifacts(df, W, flagged, min_mass=_MIN_CONTENT_MASS):
    """
    Drop the artifact columns from W and renormalise each row to 1.

    Without this, shares would sum to under 1 by an amount that varies across
    decades, distorting each end of the timeline differently. Returns
    (df, Wc, keep, alive); keep maps new column -> original topic index.
    """
    keep = [j for j in range(W.shape[1]) if j not in flagged]
    Wc = W[:, keep]
    row_sums = Wc.sum(axis=1, keepdims=True)
    alive = row_sums[:, 0] > min_mass
    Wc = Wc[alive] / row_sums[alive]
    out = df[alive].reset_index(drop=True)
    assert len(out) == Wc.shape[0], "df and W are out of sync"
    assert np.allclose(Wc.sum(axis=1), 1.0), "rows do not sum to 1"
    return out, Wc, keep, alive


def artifact_report(df, labels, W, flagged, alive,
                    clean_stats_path=CLEAN_STATS_PATH,
                    out_path="artifact_share_by_decade.csv"):
    """
    How much of each decade is not book content, measured two independent ways:
    at the text level (what cleaning deleted) and at the topic level (artifact
    share before renormalisation).
    """
    print("\n" + "=" * 78)
    print("Artifact topics — excluded from every number that follows")
    print("=" * 78)

    means, _, counts = decade_profiles(df, W)
    overall = W.mean(axis=0)
    for i in sorted(flagged):
        reason, hits = flagged[i]
        print(f"  T{i:02d}  {labels[i]}")
        print(f"        {reason:<14} {overall[i]:6.2%} of all text   "
              f"matched: {', '.join(hits)}")
    if not flagged:
        print("  none flagged")
    dropped = int((~alive).sum())
    print(f"\n  {dropped} documents dropped: under {_MIN_CONTENT_MASS:.0%} content "
          f"mass once artifact topics were removed")

    # Broken down by kind, not only summed: bibliographic noise falls over time
    # (16.1% -> 5.9%) while the generic-verb topic rises (2.6% -> 14.4%), and
    # the two nearly cancel in the total
    by_reason = {}
    for reason in sorted({r for r, _ in flagged.values()}):
        cols = [i for i in flagged if flagged[i][0] == reason]
        key = "pct_" + reason.replace(" ", "_")
        by_reason[key] = pd.Series({d: float(sum(means[d][i] for i in cols)) * 100
                                    for d in means})
    art = {d: float(sum(means[d][i] for i in flagged)) * 100 for d in means}
    table = pd.DataFrame({
        "n_books": pd.Series(counts),
        **by_reason,
        "pct_artifact_topic_share": pd.Series(art),
    })
    table.index.name = "decade"
    reason_cols = list(by_reason)

    if os.path.exists(_out(clean_stats_path)):
        clean = pd.read_csv(_out(clean_stats_path), index_col="decade")
        table = table.join(clean[["pct_chars_stripped", "pct_docs_dropped"]])
    else:
        print(f"  ({clean_stats_path} missing — text-level columns unavailable; "
              f"rebuild the corpus to regenerate it)")
        table["pct_chars_stripped"] = np.nan
        table["pct_docs_dropped"] = np.nan

    table = table[["n_books", "pct_chars_stripped", "pct_docs_dropped"]
                  + reason_cols + ["pct_artifact_topic_share"]].sort_index()
    table.to_csv(_out(out_path))

    shown = [d for d in table.index
             if d >= TREND_FROM_DECADE and table.loc[d, "n_books"] >= TREND_MIN_BOOKS]
    head = (f"  {'decade':>7} {'books':>7} {'stripped':>9} {'dropped':>8}"
            + "".join(f"{c.replace('pct_', '')[:13]:>14}" for c in reason_cols)
            + f"{'all artifact':>14}")
    print("\n" + head)
    for d in shown:
        r = table.loc[d]
        print(f"  {str(d) + 's':>7} {int(r['n_books']):>7} "
              f"{r['pct_chars_stripped']:>8.2f}% {r['pct_docs_dropped']:>7.1f}%"
              + "".join(f"{r[c]:>13.1f}%" for c in reason_cols)
              + f"{r['pct_artifact_topic_share']:>13.1f}%")
    print(f"\n  Wrote {out_path}")


# First decade shown in the trend table. Aligned with MIN_DECADE: no point
# filtering decades out of training and then displaying them
TREND_FROM_DECADE = MIN_DECADE
# a decade is only shown if it holds at least this many books
TREND_MIN_BOOKS = 50
# sparkline characters, to show the shape of a trend
_SPARK = " ▁▂▃▄▅▆▇█"


def _sparkline(values):
    lo, hi = min(values), max(values)
    if hi <= lo:
        return _SPARK[1] * len(values)
    span = len(_SPARK) - 1
    return "".join(_SPARK[max(1, round((v - lo) / (hi - lo) * span))] for v in values)


# --- further measures over the same matrix ---
# Everything from here is computed from the existing W and changes no number
# already reported. The key distinction: "share" is measured at the WORD level,
# "prevalence" at the BOOK level.

# The decade from which the dataset resembles a census rather than a sample of
# survivors. Earlier rows are flagged rather than dropped, so they are not read
# as representative. With MIN_DECADE=1900 none are displayed and the flag never
# fires; setting MIN_DECADE back to 1790 brings it straight back
CANON_DECADE = 1900

# Thresholds for "this book is about this topic". Three are reported rather than
# one chosen silently: the trend is the same under all of them (rank correlation
# 0.81-0.95) while the level varies eightfold. A dial, not a finding
PREVALENCE_THRESHOLDS = (0.05, 0.10, 0.20)
PREVALENCE_HEADLINE = 0.10

# bootstrap resamples for the lift interval, and a fixed seed for reproducibility
LIFT_BOOTSTRAP = 300
LIFT_SEED = 0


def decade_lift(means, decades):
    """
    How unusual a topic is in a decade relative to itself: its share there over
    its mean share across displayed decades. 1.0 is typical.

    The baseline is an unweighted mean across decades, not across documents, or it
    would be dominated by the modern decades and every early decade would look
    unusual against a baseline that is really the present.
    """
    matrix = np.array([means[d] for d in decades])
    base = matrix.mean(axis=0)
    return matrix / np.where(base == 0, 1.0, base)


def decade_lift_ci(df, W, decades, n_boot=LIFT_BOOTSTRAP, seed=LIFT_SEED):
    """
    Bootstrap interval for lift, resampling books within each decade.

    Needed because the thin pre-1900 decades are three times wider and several are
    a genuine tie: a set is reported rather than a ranking, and a topic whose
    interval crosses 1.0 is not called distinctive. Returns (lo, hi).
    """
    rng = np.random.default_rng(seed)
    dec = df["Decade"].values
    blocks = [W[dec == d] for d in decades]
    boot = np.empty((n_boot, len(decades), W.shape[1]))
    for b in range(n_boot):
        m = np.array([blk[rng.integers(0, len(blk), len(blk))].mean(axis=0)
                      for blk in blocks])
        base = m.mean(axis=0)
        boot[b] = m / np.where(base == 0, 1.0, base)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return lo, hi


def decade_prevalence(df, W, decades, thresholds=PREVALENCE_THRESHOLDS):
    """
    How many of a decade's BOOKS are about a topic, as opposed to how many of its
    WORDS belong to it.

    dominant is threshold-free (each book's largest topic) and sums to 100% per
    decade; prevalent depends on the threshold, so all three are returned.
    """
    dec = df["Decade"].values
    n_topics = W.shape[1]
    dominant = np.empty((len(decades), n_topics))
    prevalent = {t: np.empty((len(decades), n_topics)) for t in thresholds}
    for i, d in enumerate(decades):
        block = W[dec == d]
        top = block.argmax(axis=1)
        dominant[i] = np.bincount(top, minlength=n_topics) / len(block)
        for t in thresholds:
            prevalent[t][i] = (block >= t).mean(axis=0)
    return dominant, prevalent


def decade_movers(means, decades):
    """
    Largest rise and fall in pp against the previous DISPLAYED decade. The gap in
    years is returned too, so non-consecutive decades are not read as adjacent.
    """
    out = {}
    for i in range(1, len(decades)):
        d, prev = decades[i], decades[i - 1]
        delta = (np.asarray(means[d]) - np.asarray(means[prev])) * 100
        up, down = int(delta.argmax()), int(delta.argmin())
        out[d] = (up, float(delta[up]), down, float(delta[down]), int(d - prev))
    return out


# How many distinctive topics are shown per decade. Half the (decade x topic)
# cells depart significantly from 1.0, so a full list would be a wall of text.
# The remainder are counted and reported
_DIGEST_TOP = 4


def _digest_section(df, labels, W, keep, decades, counts, means, name):
    lift = decade_lift(means, decades)
    lo, hi = decade_lift_ci(df, W, decades)
    movers = decade_movers(means, decades)
    dominant, prevalent = decade_prevalence(df, W, decades)
    n_cols = len(keep)
    idx = {d: i for i, d in enumerate(decades)}
    # one word per topic: these rows are dense, and three words make them unreadable
    word = {j: labels[keep[j]].split(",")[0].strip() for j in range(n_cols)}

    print("\n" + "=" * 78)
    print("DECADE DIGEST — what each decade was unusual for, and what changed")
    print("=" * 78)
    print("'distinctive' = the topic's share of this decade divided by that topic's own")
    print("average across the decades shown; 1.0x is a typical decade. Only topics whose")
    print("95% bootstrap CI excludes 1.0 appear. Ordered by lift but NOT ranked — where the")
    print("CIs overlap, which one is 'first' is a coin flip.")
    # these two lines only apply when decades before CANON_DECADE are shown.
    # With MIN_DECADE=1900 there are none, and printing them would warn about
    # something not in the table
    if any(d < CANON_DECADE for d in decades):
        print(f"'~' marks the canon sample before {CANON_DECADE}, where CIs are ~3x wider.")
    print()

    rows = []
    for d in decades:
        i = idx[d]
        sig = [j for j in range(n_cols) if lo[i, j] > 1.0]
        sig.sort(key=lambda j: -lift[i, j])
        mark = "~" if d < CANON_DECADE else " "
        shown = sig[:_DIGEST_TOP]
        body = " | ".join(f"{word[j]} {lift[i, j]:.2f}x[{lo[i, j]:.2f}-{hi[i, j]:.2f}]"
                          for j in shown) or "(nothing distinguishable from a typical decade)"
        extra = f"  +{len(sig) - len(shown)} more" if len(sig) > len(shown) else ""
        print(f"{d}s{mark} n={counts[d]:<5d} {body}{extra}")
        for rank, j in enumerate(shown, 1):
            rows.append(dict(decade=d, n=counts[d], canon=d < CANON_DECADE,
                             kind="distinctive", rank=rank, topic=f"T{keep[j]:02d}",
                             label=labels[keep[j]], lift=lift[i, j],
                             lift_lo=lo[i, j], lift_hi=hi[i, j], delta_pp=""))
        if d in movers:
            up, up_pp, dn, dn_pp, span = movers[d]
            span_note = "" if span == 10 else f" (over {span} years — gap in the series)"
            print(f"{'':>15}changed: {up_pp:+.1f}pp {word[up]}   "
                  f"{dn_pp:+.1f}pp {word[dn]}{span_note}")
            for kind, j, pp in (("mover_up", up, up_pp), ("mover_down", dn, dn_pp)):
                rows.append(dict(decade=d, n=counts[d], canon=d < CANON_DECADE,
                                 kind=kind, rank=1, topic=f"T{keep[j]:02d}",
                                 label=labels[keep[j]], lift=lift[i, j],
                                 lift_lo=lo[i, j], lift_hi=hi[i, j], delta_pp=pp))
    pd.DataFrame(rows).to_csv(_out("decade_digest.csv"), index=False)

    # --- book-level prevalence ---
    hl = PREVALENCE_HEADLINE
    others = [t for t in PREVALENCE_THRESHOLDS if t != hl]
    print("\n" + "=" * 78)
    print("TOPIC PREVALENCE — % of a decade's BOOKS about a topic, at its peak decade")
    print("=" * 78)
    print("The tables above measure WORDS: what fraction of a decade's text is this topic.")
    print("This one measures BOOKS. W has one row per book summing to 1 (a 1943 novel might")
    print("read family .28, war .19, adventure .14); a book counts as 'about' a topic when")
    print(f"its cell clears a bar. The bar is a knob, not a finding — at {hl:.0%} war is 12% of")
    print("1940s books, at 5% it is 20%, at 20% it is 4%. Every threshold peaks in the same")
    print("decade, so this changes the LEVEL and never the TREND.\n")
    head = (f"{'topic':<26}{'peak':>7}{'% words':>9}{'% books':>9}"
            + "".join(f"{'@' + format(t, '.0%'):>9}" for t in others)
            + f"{'dominant':>10}")
    print(head)
    print("-" * len(head))
    order = sorted(range(n_cols), key=lambda j: -prevalent[hl][:, j].max())
    for j in order:
        i = int(prevalent[hl][:, j].argmax())
        print(f"{name[j][:25]:<26}{str(decades[i]) + 's':>7}"
              f"{means[decades[i]][j] * 100:>8.1f}%{prevalent[hl][i, j] * 100:>8.1f}%"
              + "".join(f"{prevalent[t][i, j] * 100:>8.1f}%" for t in others)
              + f"{dominant[i, j] * 100:>9.1f}%")

    prow = []
    for i, d in enumerate(decades):
        for j in range(n_cols):
            r = dict(decade=d, n=counts[d], canon=d < CANON_DECADE,
                     topic=f"T{keep[j]:02d}", label=labels[keep[j]],
                     mean_share=means[d][j], lift=lift[i, j],
                     lift_lo=lo[i, j], lift_hi=hi[i, j], dominant=dominant[i, j])
            for t in PREVALENCE_THRESHOLDS:
                r[f"p{int(t * 100):02d}"] = prevalent[t][i, j]
            prow.append(r)
    pd.DataFrame(prow).to_csv(_out("topic_prevalence_by_decade.csv"), index=False)

    lift_tab = pd.DataFrame(lift, index=decades,
                            columns=[f"T{keep[j]:02d}" for j in range(n_cols)])
    lift_tab.index.name = "Decade"
    for suffix, arr in (("_lo", lo), ("_hi", hi)):
        for j in range(n_cols):
            lift_tab[f"T{keep[j]:02d}{suffix}"] = arr[:, j]
    lift_tab.to_csv(_out("topic_lift_by_decade.csv"))

    print("\nWrote decade_digest.csv, topic_prevalence_by_decade.csv "
          "and topic_lift_by_decade.csv")

    # internal consistency checks: these must hold by definition, and a failure
    # means the calculation itself is wrong rather than the data being surprising
    assert np.allclose(lift.mean(axis=0), 1.0), "lift does not average to 1 per topic"
    assert np.allclose(dominant.sum(axis=1), 1.0), "dominant does not sum to 100% per decade"
    return lift, lo, hi, movers, dominant, prevalent


def report(df, labels, W, keep, top_k=3):
    """
    Content topics only, after renormalisation. keep maps a W column to the
    original topic index so names (T05 etc.) match the full topic list.
    """
    means, cis, counts = decade_profiles(df, W)
    n_cols = len(keep)

    print("\n" + "=" * 78)
    print(f"{n_cols} content topics over {len(df)} book summaries")
    print("=" * 78)
    for j, i in enumerate(keep):
        print(f"  T{i:02d}  {labels[i]}")

    # decades represented well enough to talk about a trend
    decades = [d for d in sorted(means)
               if d >= TREND_FROM_DECADE and counts[d] >= TREND_MIN_BOOKS]
    if not decades:
        print("\nNo decade has enough books for a trend table.")
        return

    series = {j: [means[d][j] * 100 for d in decades] for j in range(n_cols)}
    # sorted by how much they moved: the biggest movers come first
    order = sorted(range(n_cols), key=lambda j: max(series[j]) - min(series[j]), reverse=True)

    short = {j: ", ".join(labels[i].split(", ")[:3]) for j, i in enumerate(keep)}
    name = {j: f"T{keep[j]:02d} {short[j]}" for j in range(n_cols)}
    width = max(len(x) for x in name.values()) + 2

    canon = [d for d in decades if d < CANON_DECADE]
    if canon:
        print("\n" + "-" * 78)
        print(f"NOTE — the {len(canon)} decades before {CANON_DECADE} are a CANON sample, "
              "not a census.")
        print("Goodreads catalogues books that reached a modern edition, so its early decades")
        print("hold what was reprinted or digitised, not what was published. They are kept —")
        print("they carry the strongest historical validation — but marked '~' throughout.")
        print("  " + "  ".join(f"{d}s n={counts[d]}" for d in canon))
        print("-" * 78)

    print("\n" + "=" * 78)
    print(f"Topic trends, {decades[0]}s-{decades[-1]}s — biggest movers first")
    print("RELATIVE shares: each decade's topics sum to 100%, so a topic can fall\n"
          "purely because others rose. Read as composition, not volume.")
    print("=" * 78)
    print(f"{'topic':<{width}} {'shape':<{len(decades)}}   peak      range")
    for j in order:
        vals = series[j]
        peak = decades[vals.index(max(vals))]
        print(f"{name[j]:<{width}} {_sparkline(vals)}   {max(vals):4.1f}% {peak}s  "
              f"{min(vals):.1f}-{max(vals):.1f}")

    print("\n" + "=" * 78)
    print("Share of each decade's content text (%) — columns sum to 100 within a decade")
    print("=" * 78)
    print(" " * width + "".join(f"{str(d)[2:]:>5}" for d in decades))
    for j in order:
        print(f"{name[j]:<{width}}" + "".join(f"{v:5.1f}" for v in series[j]))
    print("\n" + " " * width + "".join(f"{counts[d]:>5}" for d in decades) + "   <- books per decade")
    print(" " * width + "(column headings are decade last-two-digits: 00 = 1800s ... 10 = 2010s)")

    print("\n" + "=" * 78)
    print("Top topics per decade, as a share of that decade "
          f"(95% CI; ~ = canon sample pre-{CANON_DECADE}; [thin] = under 100 books)")
    print("=" * 78)
    for decade in sorted(means):
        share, ci, n = means[decade], cis[decade], counts[decade]
        top = np.argsort(-share)[:top_k]
        parts = " | ".join(
            f"T{keep[j]:02d} {share[j]:.1%}+-{ci[j]:.1%}" if not np.isnan(ci[j])
            else f"T{keep[j]:02d} {share[j]:.1%}"
            for j in top
        )
        flag = ("~" if decade < CANON_DECADE else " ") + (" [thin]" if n < 100 else "")
        print(f"{decade}s n={n:<6d}{flag:<9} {parts}")
        print(f"{'':>21} -> {short[top[0]]}")

    _digest_section(df, labels, W, keep, decades, counts, means, name)

    cols = [f"T{i:02d}" for i in keep]
    table = pd.DataFrame({d: means[d] for d in sorted(means)}, index=cols).T
    table.index.name = "Decade"
    table.to_csv(_out("topic_shares_by_decade.csv"))
    pd.Series(labels, index=[f"T{i:02d}" for i in range(len(labels))]).to_csv(_out("topic_labels.csv"))
    print("\nWrote topic_shares_by_decade.csv (content topics, renormalised) "
          "and topic_labels.csv (all topics)")



# --- PDF export ---

PDF_PATH = "topic_trends.pdf"

# Words marking a topic that is about the book rather than its content.
# Rewritten after STOPWORDS was added: 13 of the original 16 markers were
# deleted from the vocabulary itself, leaving the detector blind, so these are
# markers that survived the blacklist. Several could be content on their own
# (write, author), hence the three-hit requirement below
_BIBLIOGRAPHIC_MARKERS = frozenset(
    "book books write read author writer novelist title review revise update "
    "copy publisher press chapter".split()
)
# generic words that form a leftover topic of contentless verbs
_GENERIC_MARKERS = frozenset("know want thing good come go day get take look".split())
# how many of a topic's top words must be flagged to disqualify it
_ARTIFACT_MIN_HITS = 3


def artifact_topics(labels, top_n=8):
    """
    Identify artifact topics by their top words, never by index - NMF numbering
    is not stable between runs. Returns {index: (reason, words that fired)} so the
    decision can be audited.
    """
    flagged = {}
    for i, label in enumerate(labels):
        words = [w.strip() for w in label.split(",")[:top_n]]
        bib = [w for w in words if w in _BIBLIOGRAPHIC_MARKERS]
        gen = [w for w in words if w in _GENERIC_MARKERS]
        if len(bib) >= _ARTIFACT_MIN_HITS:
            flagged[i] = ("bibliographic", bib)
        elif len(gen) >= _ARTIFACT_MIN_HITS:
            flagged[i] = ("generic verbs", gen)
    return flagged


def export_pdf(df, labels, W, keep, flagged, df_raw, W_raw, path=PDF_PATH):
    """
    Illustrated report over the content topics. Artifact topics appear only on a
    separate summary page: their size is a finding, so it is shown not hidden.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    means, cis, counts = decade_profiles(df, W)
    n_cols = len(keep)

    decades = [d for d in sorted(means)
               if d >= TREND_FROM_DECADE and counts[d] >= TREND_MIN_BOOKS]
    series = {j: np.array([means[d][j] * 100 for d in decades]) for j in range(n_cols)}
    errors = {j: np.array([cis[d][j] * 100 for d in decades]) for j in range(n_cols)}
    order = sorted(range(n_cols), key=lambda j: series[j].max() - series[j].min(), reverse=True)
    short = {j: ", ".join(labels[keep[j]].split(", ")[:3]) for j in range(n_cols)}

    with PdfPages(_out(path)) as pdf:
        # page 1: title and the list of content topics
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.06, .94, "Book themes by decade", size=22, weight="bold")
        fig.text(.06, .90,
                 f"{len(df):,} English book summaries, dated by first publication. "
                 f"{n_cols} content topics via NMF on TF-IDF.", size=10, color="#444")
        fig.text(.06, .875,
                 f"{len(flagged)} further topics were publisher/edition copy or generic-verb "
                 "residue; they are excluded here and shown on the last page.",
                 size=9, color="#a33")
        y = .82
        for j in order:
            fig.text(.06, y, f"T{keep[j]:02d}  {labels[keep[j]]}", size=8.5,
                     family="DejaVu Sans")
            y -= .0285
        pdf.savefig(fig); plt.close(fig)

        # page 2: one small plot per topic, including the artifact topics
        # marked with an asterisk. Artifact topics are drawn from the raw W -
        # they do not exist in the renormalised matrix - so the two scales
        # differ, which is stated explicitly at the foot of the page.
        raw_means, raw_cis, raw_counts = decade_profiles(df_raw, W_raw)
        panels = [("content", j, keep[j], series[j], errors[j]) for j in order]
        for i in sorted(flagged):
            vals = np.array([raw_means[d][i] * 100 for d in decades])
            errs = np.array([raw_cis[d][i] * 100 for d in decades])
            panels.append(("artifact", None, i, vals, errs))

        side = int(np.ceil(np.sqrt(len(panels))))
        fig, axes = plt.subplots(side, side, figsize=(11.7, 8.3), sharex=True,
                                 squeeze=False)
        fig.suptitle("Share of each decade's text (%) — biggest movers first",
                     size=13, weight="bold")
        fig.text(.5, .952,
                 "RELATIVE, not absolute: within a decade all topics sum to 100%, so a "
                 "line can fall only because other topics rose. Not a count of books.",
                 size=8.5, ha="center", color="#444")
        for ax, (kind, _, tid, vals, errs) in zip(axes.ravel(), panels):
            colour = "#c0392b" if kind == "artifact" else "#2c6fbb"
            # shade the decades that are a canon sample rather than a census
            if decades[0] < CANON_DECADE:
                ax.axvspan(decades[0], CANON_DECADE, color="#000000", alpha=.055, lw=0)
            ax.plot(decades, vals, lw=1.6, color=colour)
            ax.fill_between(decades, vals - errs, vals + errs, alpha=.2, color=colour, lw=0)
            ax.axvline(decades[int(vals.argmax())], color="#999", lw=.6, ls=":")
            star = " *" if kind == "artifact" else ""
            ax.set_title(f"T{tid:02d}{star} " + ", ".join(labels[tid].split(", ")[:3]),
                         size=6.5, color=colour)
            ax.tick_params(labelsize=6)
            ax.margins(x=.02)
        for ax in axes.ravel()[::side]:
            ax.set_ylabel("% of decade", size=6)
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        fig.text(.5, .022,
                 "* excluded artifact topic — publisher/edition copy or generic-verb residue, "
                 "not a theme. Plotted as a share of ALL text (pre-exclusion); "
                 "unmarked topics are shares of content text only, so the two scales differ.",
                 size=7.5, ha="center", color="#a33")
        if decades[0] < CANON_DECADE:
            fig.text(.5, .006,
                     f"Shaded region (before {CANON_DECADE}) is a CANON sample, not a census: "
                     "Goodreads holds what reached a modern edition, so those decades are "
                     "66-984 books each and 16% publisher copy against 6% in the 2010s.",
                     size=7.5, ha="center", color="#444")
        fig.tight_layout(rect=[0, .035, 1, .95])
        pdf.savefig(fig); plt.close(fig)

        # page 3: heatmap of the whole matrix
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        matrix = np.array([series[j] for j in order])
        # normalise per row so small topics do not vanish beside large ones.
        # a perfectly flat topic would divide by zero and leave an empty row
        spread = np.ptp(matrix, axis=1, keepdims=True)
        norm = (matrix - matrix.min(axis=1, keepdims=True)) / np.where(spread == 0, 1, spread)
        im = ax.imshow(norm, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(decades)))
        ax.set_xticklabels([f"{d}s" for d in decades], rotation=90, size=7)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"T{keep[j]:02d} {short[j]}" for j in order], size=7)
        ax.set_title("When each topic peaks — each row rescaled to its own min/max.\n"
                     "Colour compares a topic to ITSELF over time, never to other topics.",
                     size=11, weight="bold")
        fig.colorbar(im, ax=ax, shrink=.6, label="low → high (within topic)")
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # page 4: the numeric table
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.5, .965, "Share of each decade's content text (%)", size=13,
                 weight="bold", ha="center")
        fig.text(.5, .938, "Each COLUMN sums to 100 — these are proportions within a "
                 "decade, not numbers of books.", size=8.5, ha="center", color="#444")
        head = "topic".ljust(30) + "".join(f"{str(d)[2:]:>5}" for d in decades)
        lines = [head, "-" * len(head)]
        for j in order:
            name = f"T{keep[j]:02d} " + ", ".join(labels[keep[j]].split(", ")[:2])
            lines.append(name[:29].ljust(30) + "".join(f"{v:5.1f}" for v in series[j]))
        lines += ["-" * len(head),
                  "books".ljust(30) + "".join(f"{counts[d]:5d}" for d in decades)]
        fig.text(.03, .90, "\n".join(lines), family="monospace", size=6.2, va="top")
        pdf.savefig(fig); plt.close(fig)

        # page 5: the digest - what was distinctive about each decade and what
        # changed in it. This is the page answering "what does this decade tell
        # me", as opposed to "when was a topic common"
        lift = decade_lift(means, decades)
        lo, hi = decade_lift_ci(df, W, decades)
        movers = decade_movers(means, decades)
        word = {j: labels[keep[j]].split(",")[0].strip() for j in range(n_cols)}
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.5, .968, "What each decade was unusual for", size=14,
                 weight="bold", ha="center")
        fig.text(.5, .943,
                 "Each topic's share of the decade divided by that topic's own average "
                 "across decades. 1.0x = a typical decade.",
                 size=8.5, ha="center", color="#444")
        fig.text(.5, .924,
                 "Only topics whose 95% bootstrap CI excludes 1.0 are listed. Ordered by "
                 "lift but NOT ranked — where CIs overlap the order is a coin flip.",
                 size=8.5, ha="center", color="#444")
        lines = []
        for i, d in enumerate(decades):
            sig = sorted((j for j in range(n_cols) if lo[i, j] > 1.0),
                         key=lambda j: -lift[i, j])[:_DIGEST_TOP]
            mark = "~" if d < CANON_DECADE else " "
            body = "  ".join(f"{word[j]} {lift[i, j]:.2f}x" for j in sig) or "(none)"
            lines.append(f"{d}s{mark} n={counts[d]:<5d} {body}")
            if d in movers:
                up, up_pp, dn, dn_pp, _ = movers[d]
                lines.append(f"{'':>14}changed: {up_pp:+.1f}pp {word[up]}    "
                             f"{dn_pp:+.1f}pp {word[dn]}")
        fig.text(.06, .898, "\n".join(lines), family="monospace", size=8.4, va="top")
        if any(d < CANON_DECADE for d in decades):
            fig.text(.5, .02,
                     f"~ = canon sample before {CANON_DECADE}: fewer books per decade and 3x "
                     "wider confidence intervals, so the SET is reportable but not the winner.",
                     size=7.5, ha="center", color="#444")
        pdf.savefig(fig); plt.close(fig)

        # page 6: the noise that was removed, before renormalisation
        if flagged:
            raw_means, _, raw_counts = decade_profiles(df_raw, W_raw)
            rdec = [d for d in sorted(raw_means)
                    if d >= TREND_FROM_DECADE and raw_counts[d] >= TREND_MIN_BOOKS]
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            for i in sorted(flagged):
                ax.plot(rdec, [raw_means[d][i] * 100 for d in rdec], lw=1.6,
                        label=f"T{i:02d} [{flagged[i][0]}] " +
                              ", ".join(labels[i].split(", ")[:4]))
            ax.plot(rdec, [sum(raw_means[d][i] for i in flagged) * 100 for d in rdec],
                    lw=2.4, color="black", ls="--", label="all artifact topics")
            ax.set_title("Excluded artifact topics — share of each decade's text "
                         "BEFORE renormalisation", size=12, weight="bold")
            ax.set_ylabel("% of decade's text")
            ax.legend(fontsize=7, loc="upper center")
            ax.grid(alpha=.25)
            fig.text(.5, .02,
                     "These are not themes: they are publisher and edition copy, and "
                     "generic-verb residue. Their size is itself a finding about the corpus — "
                     "the pre-1900 decades are largely marketing text, not summaries.",
                     size=8, ha="center", color="#a33")
            fig.tight_layout(rect=[0, .05, 1, 1])
            pdf.savefig(fig); plt.close(fig)

        pdf.infodict()["Title"] = "Book themes by decade"

    print(f"Wrote {_out(path)} ({n_cols} content topics; {len(flagged)} excluded as artifacts: "
          f"{', '.join('T%02d' % i for i in sorted(flagged))})")


def main(source="goodreads"):
    """source='goodreads' (default), or 'cmu' for the control run."""
    global OUT_DIR
    if source == "cmu":
        OUT_DIR = "cmu_control"
        df = build_cmu_corpus()
    else:
        # without this check, a run on a clean machine would build a corpus
        # WITHOUT bounding here and save it under the bounded corpus's name,
        # completely silently
        if not os.path.exists(CACHE_PATH):
            raise SystemExit(
                f"{CACHE_PATH} is missing. Build it first:\n"
                f"    python build_bounded_corpus.py\n"
                f"(build_corpus alone does not apply bounding, so letting it "
                f"create this file would silently produce a different corpus.)")
        df = build_corpus()
    print(f"\n[{source}] Corpus: {len(df)} documents, {df['Decade'].nunique()} decades, "
          f"{df['Year'].min()}-{df['Year'].max()}")
    vectorizer, nmf, W = fit_topics(df)
    labels = topic_labels(vectorizer, nmf)
    flagged = artifact_topics(labels)

    # order matters: the noise is measured on the raw W, before renormalisation
    dfc, Wc, keep, alive = exclude_artifacts(df, W, flagged)
    artifact_report(df, labels, W, flagged, alive)
    report(dfc, labels, Wc, keep)
    export_pdf(dfc, labels, Wc, keep, flagged, df, W)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "goodreads")
