"""
What exactly is unstable about a topic, and what is not.

The project's reproducibility metric compares WORD LISTS, but the writeup
claims a curve shape. The two can come apart: if the boundary between space and
adventure/fantasy moves, the word list changes while the trend survives. Both
are measured here over the same refits, and both mean and worst case reported.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

import themes as TH

CORPUS = "themes_corpus_bounded.pkl"
SEEDS = [42, 7, 13, 99]
OUT = "sf_stability.csv"

# every topic the writeup names in Results, not only the convenient ones
REPORTED = {
    "T04": "detective", "T06": "war", "T13": "guides", "T21": "science fiction",
    "T08": "poetry", "T15": "hardboiled", "T20": "translation",
    "T24": "short-story collections", "T01": "social theory", "T03": "adventure",
}


def fit(df, seed):
    """Varying training sample, fixed NMF seed - the stability.py method."""
    idx = []
    for _, g in df.groupby("Decade"):
        idx.extend(g.sample(min(len(g), TH.FIT_PER_DECADE), random_state=seed).index)
    vec = TfidfVectorizer(min_df=5, max_df=0.5, sublinear_tf=True,
                          stop_words=list(TH.STOPWORDS))
    X = vec.fit_transform(df.loc[idx, "Lemmas"])
    nmf = NMF(n_components=TH.N_TOPICS, init="nndsvda", random_state=42,
              max_iter=800, tol=1e-5).fit(X)
    W = nmf.transform(vec.transform(df["Lemmas"]))
    rs = W.sum(1, keepdims=True)
    W = np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)
    return np.array(vec.get_feature_names_out()), nmf.components_, W


def aligned(H, names, vocab_pos, n_vocab):
    """H normalised onto a shared vocabulary, so cosine is defined."""
    M = np.zeros((H.shape[0], n_vocab))
    for c, w in enumerate(names):
        j = vocab_pos.get(w)
        if j is not None:
            M[:, j] = H[:, c]
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def main():
    df = TH.build_corpus(cache_path=CORPUS)
    decades = [d for d in sorted(df.Decade.unique()) if d < 2010]
    dec_mask = {d: (df.Decade == d).values for d in decades}
    ref_sh = pd.read_csv("final_refit/topic_shares_by_decade.csv", index_col=0) * 100
    ref_lab = pd.read_csv("final_refit/topic_labels.csv", index_col=0)["0"]

    runs = [fit(df, s) for s in SEEDS]
    print(f"{len(runs)} refits, K={TH.N_TOPICS}, {len(df):,} documents\n")

    # vocabulary shared by every run, including the anchor run
    vocab = sorted(set.intersection(*[set(names) for names, _, _ in runs]))
    pos = {w: i for i, w in enumerate(vocab)}
    Hs = [aligned(H, names, pos, len(vocab)) for names, H, _ in runs]

    # the anchor is the first refit; the reported topic is located in it by
    # word overlap
    anchor_names, anchor_H, anchor_W = runs[0]
    anchor = {}
    for tid in REPORTED:
        target = set(str(ref_lab[tid]).split(", ")[:12])
        best = max(range(TH.N_TOPICS),
                   key=lambda t: len(set(anchor_names[np.argsort(-anchor_H[t])[:12]]) & target))
        anchor[tid] = best

    rows = []
    for tid, name in REPORTED.items():
        a = anchor[tid]
        ref_curve = ref_sh.loc[decades, tid].values
        cos_list, cor_list = [], []
        for k in range(1, len(runs)):
            sim = Hs[0][a] @ Hs[k].T          # cosine against every topic in run k
            m = int(sim.argmax())
            cos_list.append(float(sim[m]))
            W = runs[k][2]
            curve = np.array([W[dec_mask[d], m].mean() * 100 for d in decades])
            cor_list.append(float(np.corrcoef(ref_curve, curve)[0, 1]))
        rows.append(dict(topic=tid, name=name,
                         cosine_mean=np.mean(cos_list), cosine_worst=np.min(cos_list),
                         curve_r_mean=np.mean(cor_list), curve_r_worst=np.min(cor_list)))
        print(f"  {name:24} cosine {np.mean(cos_list):.3f} (worst {np.min(cos_list):.3f})   "
              f"curve r {np.mean(cor_list):+.3f} (worst {np.min(cor_list):+.3f})")

    out = pd.DataFrame(rows).sort_values("curve_r_worst")
    out.to_csv(OUT, index=False)
    print("\n" + out.to_string(index=False, float_format="%.3f"))
    print(f"\ncurve r worst case across all reported themes: {out.curve_r_worst.min():.3f}")
    print(f"cosine worst case across all reported themes:  {out.cosine_worst.min():.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
