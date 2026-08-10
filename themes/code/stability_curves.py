"""
Reproducibility per reported topic, for both the word list and the curve.

  cosine  - does the topic's WORD LIST re-form on another training sample?
  curve r - does the DECADE CURVE the writeup quotes re-form? This is the
            actual claim, and can survive when two neighbouring topics swap
            their boundary.

Refits are matched by cosine of H rows over a shared vocabulary. Matching on
word overlap would flatter the result. Reported columns are the worst of the
three comparisons, not the mean.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

import themes as TH

CORPUS = "themes_corpus_bounded.pkl"
SRC = "final_refit"
SEEDS = [42, 7, 13, 99]
OUT = "all_topic_stability.csv"
PARTIAL = 2010          # 2010-2017, a partial decade - excluded everywhere


def fit(df, seed):
    """Varying training sample, fixed NMF seed, so only the sample changes."""
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


def aligned(H, names, pos, n_vocab):
    """H normalised onto a shared vocabulary, or the cosine is undefined."""
    M = np.zeros((H.shape[0], n_vocab))
    for c, w in enumerate(names):
        j = pos.get(w)
        if j is not None:
            M[:, j] = H[:, c]
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def main():
    df = TH.build_corpus(cache_path=CORPUS)
    sh = pd.read_csv(f"{SRC}/topic_shares_by_decade.csv", index_col=0) * 100
    lab = pd.read_csv(f"{SRC}/topic_labels.csv", index_col=0)["0"]
    dec = [d for d in sh.index if d < PARTIAL]
    dec_mask = {d: (df.Decade == d).values for d in dec}

    runs = [fit(df, s) for s in SEEDS]
    print(f"{len(runs)} refits, K={TH.N_TOPICS}, {len(df):,} documents\n")

    vocab = sorted(set.intersection(*[set(names) for names, _, _ in runs]))
    pos = {w: i for i, w in enumerate(vocab)}
    Hs = [aligned(H, names, pos, len(vocab)) for names, H, _ in runs]

    anchor_names, anchor_H, _ = runs[0]
    rows = []
    for tid in sh.columns:
        target = set(str(lab[tid]).split(", ")[:12])
        # locate the reported topic inside the anchor run. This is
        # identification, not measurement: agreement itself is the cosine
        # against the other runs
        a = max(range(TH.N_TOPICS),
                key=lambda t: len(set(anchor_names[np.argsort(-anchor_H[t])[:12]]) & target))
        ref = sh.loc[dec, tid].values
        cos_list, cor_list = [], []
        for k in range(1, len(runs)):
            sim = Hs[0][a] @ Hs[k].T
            m = int(sim.argmax())
            cos_list.append(float(sim[m]))
            W = runs[k][2]
            curve = np.array([W[dec_mask[d], m].mean() * 100 for d in dec])
            cor_list.append(float(np.corrcoef(ref, curve)[0, 1]))
        rows.append(dict(topic=tid,
                         label=", ".join(str(lab[tid]).split(", ")[:4]),
                         peak=dec[int(ref.argmax())],
                         peak_pct=ref.max(),
                         span_pp=ref.max() - ref.min(),
                         cos_worst=min(cos_list),
                         curve_r_worst=min(cor_list)))
        print(f"  {tid} {rows[-1]['label'][:34]:34} cos {min(cos_list):.3f}  "
              f"curve r {min(cor_list):+.3f}")

    out = pd.DataFrame(rows).sort_values("curve_r_worst", ascending=False)
    out.to_csv(OUT, index=False)
    ok = out[out.curve_r_worst >= 0.75]
    print(f"\nwrote {OUT}")
    print(f"curve r range over topics with r >= 0.75: "
          f"{ok.curve_r_worst.min():.3f} - {ok.curve_r_worst.max():.3f}")
    print(f"topics below 0.75: {list(out[out.curve_r_worst < 0.75].topic)}")


if __name__ == "__main__":
    main()
