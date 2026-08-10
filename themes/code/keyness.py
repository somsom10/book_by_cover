"""
Keyness: which words are consistently common in Goodreads blurbs but not in
CMU plot summaries, FOR THE SAME BOOKS.

A blurb is publisher marketing, a CMU summary is a reader's plot description.
Comparing the two descriptions of one book, any vocabulary difference cannot
come from the plot - only from register. Those words are what has to be
neutralised before topics are extracted.

Books are paired on a normalised title, dropping ambiguous titles on both sides
and checking the year where known.
"""

import gzip
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import text as T

MATCH_CACHE = "keyness_matched.pkl"
# Largest year gap allowed when both sides carry a year. CMU sometimes gives
# the edition year rather than the year of composition, so zero is too strict
YEAR_TOLERANCE = 2
# very short titles ("Home", "1984") match too many different books
MIN_TITLE_CHARS = 6

_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")


def norm_title(value):
    s = unicodedata.normalize("NFKD", str(value))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def cmu_title_index(path=T.CMU_PATH):
    """
    CMU records with a normalised title, dropping non-unique titles: there is no
    way to tell which book a Goodreads record refers to.
    """
    cols = ["wiki_id", "freebase_id", "title", "author", "pub_date", "genres", "summary"]
    df = pd.read_csv(path, sep="\t", header=None, names=cols, quoting=3)
    df["Summary"] = df["summary"].astype(str)
    df["Year"] = df["pub_date"].apply(T._extract_year)
    df["TitleKey"] = df["title"].apply(norm_title)
    df = df[df["Summary"].str.strip().str.len() > 0]
    df = df[df["TitleKey"].str.len() >= MIN_TITLE_CHARS]

    counts = df["TitleKey"].value_counts()
    dupes = set(counts[counts > 1].index)
    n0 = len(df)
    df = df[~df["TitleKey"].isin(dupes)]
    print(f"CMU: {n0} rows with summaries -> {len(df)} unambiguous titles "
          f"({len(dupes)} titles dropped as duplicated)")
    return df[["TitleKey", "title", "author", "Year", "Summary"]].reset_index(drop=True)


def stream_goodreads_by_title(title_keys, path=T.GOODREADS_PATH, works_path=T.WORKS_PATH):
    """
    One pass collecting every work whose title is in title_keys. No sampling -
    this is about register, not trends. Titles keep all their matches so ambiguous
    ones can be dropped later.
    """
    work_years = T.load_work_years(works_path)
    print(f"  loaded {len(work_years)} work years")
    seen_works = set()
    hits = defaultdict(list)
    n_lines = 0

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            obj = json.loads(line)
            key = norm_title(obj.get("title_without_series", "") or obj.get("title", ""))
            if key not in title_keys:
                continue
            try:
                work_id = int(obj.get("work_id"))
            except (TypeError, ValueError):
                continue
            if work_id in seen_works:
                continue
            if not T._is_english_code(obj.get("language_code", "")):
                continue
            summary = str(obj.get("description", "")).strip()
            if not summary or not T._is_english_text(summary):
                continue
            seen_works.add(work_id)
            hits[key].append({
                "TitleKey": key,
                "work_id": work_id,
                "gr_title": obj.get("title_without_series", ""),
                "Year": work_years.get(work_id),
                "Summary": summary,
            })
    print(f"  scanned {n_lines} editions, matched {len(hits)} titles "
          f"({sum(len(v) for v in hits.values())} distinct works)")
    return hits


def build_matched_pairs(cache_path=MATCH_CACHE, force_reload=False):
    """
    One Goodreads blurb and one CMU summary per title.

    Dropped: titles ambiguous in CMU, titles matching several Goodreads works
    unless the year settles it, and pairs whose known years differ by more than
    YEAR_TOLERANCE.
    """
    if not force_reload and os.path.exists(cache_path):
        print(f"Loading cached matched pairs from {cache_path}")
        return pd.read_pickle(cache_path)

    cmu = cmu_title_index()
    cmu_by_key = {r.TitleKey: r for r in cmu.itertuples()}
    print("Streaming Goodreads for title matches (2.36M lines)...")
    hits = stream_goodreads_by_title(set(cmu_by_key))

    rows, dropped_multi, dropped_year = [], 0, 0
    for key, cands in hits.items():
        ref = cmu_by_key[key]
        if len(cands) > 1:
            if ref.Year is None:
                dropped_multi += 1
                continue
            near = [c for c in cands
                    if c["Year"] is not None and abs(c["Year"] - ref.Year) <= YEAR_TOLERANCE]
            if len(near) != 1:
                dropped_multi += 1
                continue
            cands = near
        gr = cands[0]
        if ref.Year is not None and gr["Year"] is not None \
                and abs(gr["Year"] - ref.Year) > YEAR_TOLERANCE:
            dropped_year += 1
            continue
        # prefer the Goodreads year (the work's original publication year);
        # CMU is the fallback when it is missing
        year = gr["Year"] if gr["Year"] is not None else ref.Year
        rows.append({
            "TitleKey": key,
            "Title": ref.title,
            "Author": ref.author,
            "Year": year,
            "GR_Summary": gr["Summary"],
            "CMU_Summary": ref.Summary,
        })

    df = pd.DataFrame(rows).dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)
    df["Decade"] = df["Year"] // 10 * 10
    df = df.sort_values("Year").reset_index(drop=True)
    print(f"  dropped {dropped_multi} titles matching several Goodreads works, "
          f"{dropped_year} on year disagreement")
    print(f"  {len(df)} matched pairs, {df['Decade'].min()}s-{df['Decade'].max()}s")
    df.to_pickle(cache_path)
    print(f"Saved matched pairs to {cache_path}")
    return df




LEMMA_CACHE = "keyness_lemmas.pkl"
KEYNESS_CSV = "keyness_goodreads_vs_cmu.csv"

# Thresholds for the suggested removal list.
# G2 is a log-likelihood ratio test with one degree of freedom; 15.13 is p<0.0001
G2_MIN = 15.13
# A log ratio of 0.585 means "at least 1.5x as frequent in Goodreads as in CMU,
# after length normalisation". It is deliberately loose: the filter that protects
# content words is phi below, not keyness. A stricter keyness cut protected no
# additional content word - it only lost genuine marketing words (unforgettable,
# compelling)
LOG_RATIO_MIN = 0.585
# a word in under 0.5% of blurbs (about 43 books) is a property of those books
# rather than of the register
MIN_DOC_SHARE = 0.005
# a decade with fewer pairs than this does not count toward the consistency test
CONSISTENCY_MIN_PAIRS = 30
# "consistently" means more frequent in Goodreads in at least 85% of counted
# decades. Requiring 100% rejects words over a single decade holding 30 pairs
CONSISTENCY_MIN = 0.85


def matched_lemmas(cache_path=LEMMA_CACHE, force_reload=False):
    """Lemmatise both sides through the pipeline used for the model."""
    if not force_reload and os.path.exists(cache_path):
        print(f"Loading cached lemmas from {cache_path}")
        return pd.read_pickle(cache_path)
    df = build_matched_pairs()
    print(f"Lemmatising {len(df)} pairs on both sides...")
    df["GR_Lemmas"] = [" ".join(t) for t in T.preprocess_texts(df["GR_Summary"].tolist())]
    df["CMU_Lemmas"] = [" ".join(t) for t in T.preprocess_texts(df["CMU_Summary"].tolist())]
    df.to_pickle(cache_path)
    print(f"Saved lemmas to {cache_path}")
    return df


def _counts(series):
    tokens, docs = Counter(), Counter()
    for text in series:
        toks = text.split()
        tokens.update(toks)
        docs.update(set(toks))
    return tokens, docs


def keyness_table(df):
    """
    Keyness of Goodreads against CMU over the same books.

    G2 is computed on token frequency, not document frequency: a CMU summary is
    2.6x longer at the median, so a document-level measure would credit CMU for
    length alone. log_ratio is the log2 ratio of relative frequencies, smoothed by
    0.5 on whichever side is absent.
    """
    gr_tok, gr_doc = _counts(df["GR_Lemmas"])
    cmu_tok, cmu_doc = _counts(df["CMU_Lemmas"])
    n_gr, n_cmu = sum(gr_tok.values()), sum(cmu_tok.values())
    n_docs = len(df)
    print(f"  {n_gr} Goodreads tokens vs {n_cmu} CMU tokens "
          f"({n_cmu / n_gr:.2f}x longer on the CMU side)")

    # consistency: per decade with enough pairs, is the word commoner in Goodreads
    decades = [d for d, n in df["Decade"].value_counts().items() if n >= CONSISTENCY_MIN_PAIRS]
    decades.sort()
    per_decade = {}
    for d in decades:
        block = df[df["Decade"] == d]
        g, _ = _counts(block["GR_Lemmas"])
        c, _ = _counts(block["CMU_Lemmas"])
        per_decade[d] = (g, sum(g.values()), c, sum(c.values()))
    print(f"  consistency measured over {len(decades)} decades "
          f"({decades[0]}s-{decades[-1]}s, >= {CONSISTENCY_MIN_PAIRS} pairs each)")

    rows = []
    for word in set(gr_tok) | set(cmu_tok):
        a, b = gr_tok[word], cmu_tok[word]
        if a + b < 20:          # too rare to estimate a ratio
            continue
        e1 = n_gr * (a + b) / (n_gr + n_cmu)
        e2 = n_cmu * (a + b) / (n_gr + n_cmu)
        g2 = 2 * ((a * np.log(a / e1) if a else 0) + (b * np.log(b / e2) if b else 0))
        # smoothing is needed only when one side is zero
        ra = (a if a else 0.5) / n_gr
        rb = (b if b else 0.5) / n_cmu
        higher_in_gr = 0
        for d in decades:
            g, gt, c, ct = per_decade[d]
            if (g[word] / gt if gt else 0) > (c[word] / ct if ct else 0):
                higher_in_gr += 1
        rows.append({
            "word": word,
            "g2": g2,
            "log_ratio": np.log2(ra / rb),
            "gr_per_10k": ra * 1e4,
            "cmu_per_10k": rb * 1e4,
            "gr_tokens": a,
            "cmu_tokens": b,
            "gr_doc_share": gr_doc[word] / n_docs,
            "cmu_doc_share": cmu_doc[word] / n_docs,
            "decades_higher_in_gr": higher_in_gr / len(decades),
        })

    out = pd.DataFrame(rows)
    # direction is carried by the sign of log_ratio; g2 itself is undirected
    out["direction"] = np.where(out["log_ratio"] > 0, "goodreads", "cmu")
    return out.sort_values("g2", ascending=False).reset_index(drop=True)


def suggested_removals(table):
    """
    Words consistently common in Goodreads and not in CMU. Every condition must
    hold at once: significance, effect size, spread across books, consistency
    across decades.
    """
    keep = (
        (table["direction"] == "goodreads")
        & (table["g2"] >= G2_MIN)
        & (table["log_ratio"] >= LOG_RATIO_MIN)
        & (table["gr_doc_share"] >= MIN_DOC_SHARE)
        & (table["decades_higher_in_gr"] >= CONSISTENCY_MIN)
    )
    return table[keep].sort_values("log_ratio", ascending=False).reset_index(drop=True)


def _fmt(row):
    return (f"    {row.word:<16} {row.log_ratio:5.2f}x2  "
            f"gr {row.gr_per_10k:7.1f} vs cmu {row.cmu_per_10k:6.1f} /10k  "
            f"in {row.gr_doc_share * 100:4.1f}% of blurbs  G2 {row.g2:8.0f}")


def report(table, removals, top_n=200):
    print()
    print("=" * 78)
    print("KEYNESS: Goodreads blurb vs CMU plot summary, same books")
    print("=" * 78)
    print("Both texts describe the same book, so any vocabulary difference is")
    print("register, not plot. log_ratio is log2 of the length-normalised")
    print("frequency ratio: 1.0 = twice as frequent in Goodreads.")
    print()
    print(f"--- SUGGESTED REMOVALS ({len(removals)} words) ---")
    print(f"    filters: G2>={G2_MIN}, log_ratio>={LOG_RATIO_MIN}, "
          f">={MIN_DOC_SHARE * 100:.1f}% of blurbs, higher in Goodreads in "
          f"{CONSISTENCY_MIN * 100:.0f}% of decades")
    tiers = (
        ("register", f"REGISTER - remove (phi < {PHI_REGISTER_MAX})",
         "boilerplate: the word in the blurb tells you nothing about which\n"
         "    book is underneath it, so no theme is lost by dropping it."),
        ("borderline", f"BORDERLINE ({PHI_REGISTER_MAX} <= phi < {PHI_CONTENT_MIN})",
         "above the boilerplate cluster but below every calibration content\n"
         "    word - a judgement call, not a measurement."),
        ("content-bearing", f"CONTENT-BEARING - keep (phi >= {PHI_CONTENT_MIN})",
         "keyness flags these, but the word in the blurb predicts the same\n"
         "    word in that book's plot summary, so it carries theme."),
    )
    for tier, label, why in tiers:
        block = removals[removals["tier"] == tier]
        print()
        print(f"  [{label}]  ({len(block)} words)")
        print(f"    {why}")
        for row in block.itertuples():
            # phi is unstable when a word appears on both sides in few books.
            # "penguin" for instance scores phi 0.18 on only 5 overlaps out of
            # 8,680. The flag matters only when a word was SAVED from removal by
            # a high phi: a low phi on few overlaps is exactly what a register
            # word looks like, and is not suspicious
            weak = ("  (!) phi on few co-occurrences"
                    if tier != "register" and row.both < MIN_BOTH_FOR_PHI else "")
            print(_fmt(row) + f"  phi {row.phi:5.2f}{weak}")

    print()
    print("--- CONTROL: strongest words on the CMU side ---")
    print("    these should be plot vocabulary. If they are, the method is")
    print("    separating register and not simply separating the two files.")
    cmu_side = table[table["direction"] == "cmu"].sort_values("g2", ascending=False)
    for row in cmu_side.head(20).itertuples():
        print(_fmt(row))

    reg = sorted(removals.loc[removals["tier"] == "register", "word"])
    print()
    print("--- copy-paste: the register tier ---")
    print("KEYNESS_STOPWORDS = {" + ", ".join(f'"{w}"' for w in reg) + "}")
    print()


def calibrate_phi(df):
    """
    phi over the reference set. Without this calibration any threshold would be a
    guess; with it the split is visibly bimodal.
    """
    print("--- phi calibration on reference words ---")
    for kind, words in _PHI_CALIBRATION.items():
        a = paired_association(df, words).sort_values("phi", ascending=False)
        pairs = ", ".join(f"{r.word} {r.phi:.3f}" for r in a.itertuples())
        print(f"  {kind:<9}: {pairs}")
    print()


def main():
    force = "--force" in os.sys.argv
    df = matched_lemmas(force_reload=force)
    table = keyness_table(df)
    removals = suggested_removals(table)
    assoc = paired_association(df, removals["word"].tolist())
    removals = removals.merge(assoc, on="word")
    # ranked by phi: safest removals first, contested ones after
    removals = removals.sort_values("phi").reset_index(drop=True)
    removals["tier"] = np.select(
        [removals["phi"] < PHI_REGISTER_MAX, removals["phi"] < PHI_CONTENT_MIN],
        ["register", "borderline"], default="content-bearing")
    table = table.merge(assoc, on="word", how="left")
    table.to_csv(KEYNESS_CSV, index=False)
    removals.to_csv("keyness_suggested_removals.csv", index=False)
    print(f"  wrote {KEYNESS_CSV} ({len(table)} words) and "
          f"keyness_suggested_removals.csv ({len(removals)} words)")
    calibrate_phi(df)
    report(table, removals)



# Both thresholds were calibrated on a reference set, not chosen by feel. See
# calibrate_phi(): metadata words cluster at phi 0.00-0.05 (isbn -0.003,
# paperback 0.010, introduction 0.050) while content words start at 0.12 and
# rise (story 0.14, love 0.23, war 0.37, vampire 0.70). The 0.05-0.12 range is
# nearly empty, so it is a reported grey zone rather than an arbitrary cut
PHI_REGISTER_MAX = 0.05
PHI_CONTENT_MIN = 0.12
# below this many books where the word appears in both descriptions, phi rests
# on too small a sample to trust, and the word is flagged in the output
MIN_BOTH_FOR_PHI = 20

# calibration reference set: unambiguous content words against bibliographic metadata
_PHI_CALIBRATION = {
    "content": ["vampire", "dragon", "island", "king", "war", "murder", "school",
                "space", "detective", "ship", "family", "horse", "love",
                "soldier", "doctor", "prison", "spy", "marriage", "life",
                "story", "money"],
    "metadata": ["isbn", "reprint", "anthology", "paperback", "bestseller",
                 "edition", "introduction"],
}


def paired_association(df, words):
    """
    Does a word in the blurb PREDICT the plot?

    A 2x2 table per word over the book pairs, scored by phi. A real content word
    (war, murder) appears in both descriptions of the same book; a register word
    (classic, edition) appears only on the publisher's side. Low phi is the
    evidence that a word can be removed without losing a topic.
    """
    gr_sets = [set(t.split()) for t in df["GR_Lemmas"]]
    cmu_sets = [set(t.split()) for t in df["CMU_Lemmas"]]
    n = len(df)
    rows = []
    for w in words:
        a = sum((w in g) and (w in c) for g, c in zip(gr_sets, cmu_sets))
        b = sum((w in g) and (w not in c) for g, c in zip(gr_sets, cmu_sets))
        c_ = sum((w not in g) and (w in c) for g, c in zip(gr_sets, cmu_sets))
        d = n - a - b - c_
        denom = np.sqrt(float(a + b) * (c_ + d) * (a + c_) * (b + d))
        rows.append({
            "word": w,
            "phi": (a * d - b * c_) / denom if denom else 0.0,
            "both": a, "gr_only": b, "cmu_only": c_,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()


WORD_WEIGHTS_CSV = "keyness_word_weights.csv"


def _phi_all(df):
    """
    phi for the whole vocabulary as matrix algebra; the loop in
    paired_association is impractical for 28k words.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    n = len(df)
    cv = CountVectorizer(binary=True, min_df=3, token_pattern=r"\S+")
    cv.fit(pd.concat([df.GR_Lemmas, df.CMU_Lemmas]))
    A = cv.transform(df.GR_Lemmas).astype(np.float64)
    B = cv.transform(df.CMU_Lemmas).astype(np.float64)
    n11 = np.asarray(A.multiply(B).sum(axis=0)).ravel()
    na = np.asarray(A.sum(axis=0)).ravel()
    nb = np.asarray(B.sum(axis=0)).ravel()
    n10, n01 = na - n11, nb - n11
    n00 = n - n11 - n10 - n01
    den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    phi = np.divide(n11 * n00 - n10 * n01, den, out=np.zeros_like(den), where=den > 0)
    return dict(zip(cv.get_feature_names_out(), phi))


def export_word_weights(path=WORD_WEIGHTS_CSV):
    """
    A register weight per word: positive is publisher vocabulary, negative is
    plot. log_ratio is damped as phi rises, so war and adventure stay near zero.
    """
    df = matched_lemmas()
    phi = _phi_all(df)
    table = keyness_table(df)
    rows = []
    for r in table.itertuples():
        if r.g2 < G2_MIN:
            continue
        p = max(phi.get(r.word, 0.0), 0.0)
        w = (r.log_ratio * max(0.0, 1 - p / PHI_CONTENT_MIN)
             if r.log_ratio > 0 else r.log_ratio)
        if w != 0:
            rows.append({"word": r.word, "weight": w, "phi": phi.get(r.word, 0.0),
                         "log_ratio": r.log_ratio, "g2": r.g2})
    # mergesort is stable: quicksort orders exactly-tied weights differently on
    # each run, so the file differs byte for byte although its content is identical
    out = pd.DataFrame(rows).sort_values("weight", ascending=False,
                                         kind="mergesort")
    out.to_csv(path, index=False)
    print(f"wrote {path}: {len(out)} words "
          f"({(out.weight > 0).sum()} publisher-side, {(out.weight < 0).sum()} plot-side)")
    return out
