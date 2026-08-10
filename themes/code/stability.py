"""
Which specific topics fail to reproduce across refits.

Run from work/ with code/ on the import path; all paths here are relative.
"""
import sys, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import themes as TH

df = TH.build_corpus(cache_path="themes_corpus_bounded.pkl")
K, SEEDS = TH.N_TOPICS, [42, 7, 13, 99]
def fit(seed):
    idx = []
    for _, g in df.groupby("Decade"):
        idx.extend(g.sample(min(len(g), TH.FIT_PER_DECADE), random_state=seed).index)
    vec = TfidfVectorizer(min_df=5, max_df=0.5, sublinear_tf=True,
                          stop_words=list(TH.STOPWORDS))
    X = vec.fit_transform(df.loc[idx, "Lemmas"])
    nmf = NMF(n_components=K, init="nndsvda", random_state=42, max_iter=800, tol=1e-5).fit(X)
    return vec.get_feature_names_out(), nmf.components_

runs = [fit(s) for s in SEEDS]
labels = [", ".join(runs[0][0][i] for i in np.argsort(-runs[0][1][t])[:8]) for t in range(K)]
vocab = sorted(set.intersection(*[set(n) for n, _ in runs]))
pos = {w: i for i, w in enumerate(vocab)}
Hs = []
for names, H in runs:
    M = np.zeros((K, len(vocab)))
    for c, w in enumerate(names):
        if w in pos: M[:, pos[w]] = H[:, c]
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    Hs.append(M)
rows = []
for t in range(K):
    sims = [(Hs[0][t] @ Hs[r].T).max() for r in range(1, 4)]
    rows.append((min(sims), labels[t]))
rows.sort()
out = pd.DataFrame(rows, columns=["reproducibility", "topic"])
# Thresholds were measured, not chosen. A seed sensitivity test put the
# metric's own noise at about 0.05 (K=30 gave 0.848/0.828/0.755 across three
# seed sets), so a 0.95 cut would be sharper than the measurement and would
# separate 0.94 from 0.97 for no reason. These cuts are wider than the noise.
out["verdict"] = np.where(out.reproducibility < 0.75, "unstable",
                 np.where(out.reproducibility < 0.90, "moderate", "solid"))
out.to_csv("topic_stability.csv", index=False)
print(f"per-topic reproducibility across {len(SEEDS)} sampling draws (K={K}):\n")
for r in out.itertuples():
    print(f"  {r.verdict:9} {r.reproducibility:.3f}  {r.topic[:62]}")
print("\nwrote topic_stability.csv")
print("\nTiers, not a ranking: seed-to-seed noise on this metric is about +/-0.05,")
print("so >=0.90 vs 0.88 is not a real difference, but >=0.90 vs <0.75 is.")
print("  solid    (>=0.90)  the word list re-forms under a different training sample")
print("  moderate (0.75-0.90) partially re-forms; report the trend with the caveat")
print("  unstable (<0.75)   real structure in THIS fit only; do not claim a trend")
