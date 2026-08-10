"""
Builds themes_corpus_bounded.pkl, the corpus the rest of the pipeline reads.

The one step not already in themes.build_corpus is bounding (see bounding.py).
Order decides the result: load -> clean_corpus -> bounding -> lemmatise.
Lemmatising must be last, or Lemmas describe the text from before the trim.

Runtime: tens of minutes, spaCy over ~73k documents.
"""
import sys

import pandas as pd

import bounding as B
import evaluate_bounding as E
import text as T
import themes as TH

OUT = "themes_corpus_bounded.pkl"
REPEAT_MIN_DOCS = 10


def main(out=OUT, limit=None):
    print("Streaming Goodreads (2.36M lines)...")
    df = TH.load_goodreads_full()
    if limit:
        # smoke mode: a small sample, only to prove the pipeline runs end to end
        df = df.sample(min(limit, len(df)), random_state=42).reset_index(drop=True)
        print(f"  smoke test on {len(df)} documents")
    print(f"  {len(df)} works after de-duplicating editions")

    # Trim the edges, then delete newspaper and retailer names as phrases.
    # Both steps together produce the canonical corpus; the first alone does not
    def bound(summaries):
        # The repeated-sentence set comes from the cleaned text; this is the
        # rule that finds publishers nobody listed. 10 rather than
        # evaluate_bounding's default of 5 was measured against the original
        # corpus: 5 drops 15 documents too many, 12 leaves 10 with advertising,
        # and 10 reproduces it exactly at 73,411 documents
        rep = E.repeated_sentences(summaries, min_docs=REPEAT_MIN_DOCS)
        return [B.strip_publication_names(
                    B.bound_summary(t, repeated=rep).text)[0]
                for t in summaries]

    # in smoke mode the stats go to a different filename: otherwise a 300
    # document test run overwrites the real run's per-decade cleaning stats
    print("Cleaning and bounding...")
    df = TH.clean_corpus(
        df, bound_fn=bound,
        stats_path="artifact_clean_stats_smoke.csv" if limit
        else TH.CLEAN_STATS_PATH)

    print("Lemmatising...")
    df["Lemmas"] = [" ".join(t) for t in T.preprocess_texts(df["Summary"].tolist())]
    df["Decade"] = df["Year"] // 10 * 10
    df = df[df["Lemmas"].str.len() > 0].reset_index(drop=True)

    df.to_pickle(out)
    print(f"Saved {len(df)} documents to {out}")
    return df


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=n)
