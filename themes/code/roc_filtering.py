"""
עקומות ROC ועקומת התמורה של הסינון.

שתי שאלות שונות, ולכן שתי עקומות:
1. האם ציון הרגיסטר (מבוסס מילים, רציף) מסכים עם הכללים (מבוססי תבנית,
   בינאריים)? שתי שיטות בלתי תלויות לאותה מטרה; הסכמה ביניהן היא ראיה.
2. כמה עלילה נהרסת ככל שמסננים יותר? זו העקומה שקובעת היכן לעצור.
"""
import re, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

import bounding as B, evaluate_bounding as E, text as T

OUT_PDF = "filtering_roc.pdf"


def sentence_scores(sents):
    return np.array([B.register_score(s)[0] for s in sents]), \
           np.array([B.register_score(s)[1] for s in sents])


def main():
    m = pd.read_pickle("keyness_matched.pkl")
    rep = E.repeated_sentences(m["GR_Summary"])

    gr, flagged = [], []
    for txt in m["GR_Summary"].head(4000):
        for _, _, s in B.split_sentences(str(txt)):
            if len(s) > 20:
                gr.append(s); flagged.append(bool(B.flag_sentence(s, rep)[0]))
    cmu = [s for txt in m["CMU_Summary"].head(700)
           for _, _, s in B.split_sentences(str(txt)) if len(s) > 20][:8000]
    flagged = np.array(flagged)
    gs, gn = sentence_scores(gr)
    cs, cn = sentence_scores(cmu)
    ok_g, ok_c = gn >= 3, cn >= 3
    print(f"{len(gr)} Goodreads sentences ({flagged.mean()*100:.1f}% rule-flagged), "
          f"{len(cmu)} CMU plot sentences")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))

    # --- עקומה 1: ROC ---
    curves = []
    y1 = np.r_[np.ones(int((flagged & ok_g).sum())), np.zeros(int(ok_c.sum()))]
    x1 = np.r_[gs[flagged & ok_g], cs[ok_c]]
    curves.append(("rule-flagged Goodreads vs CMU plot", y1, x1))
    y2 = np.r_[np.ones(int((flagged & ok_g).sum())), np.zeros(int((~flagged & ok_g).sum()))]
    x2 = np.r_[gs[flagged & ok_g], gs[~flagged & ok_g]]
    curves.append(("rule-flagged vs unflagged Goodreads", y2, x2))
    for name, y, x in curves:
        fpr, tpr, _ = roc_curve(y, x)
        auc = roc_auc_score(y, x)
        ax[0].plot(fpr, tpr, lw=2, label=f"{name}\nAUC = {auc:.3f}")
        print(f"AUC  {name:42} {auc:.3f}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
    ax[0].set_title("Does the word-based register score\nagree with the rule-based filter?")
    ax[0].legend(fontsize=7, loc="lower right"); ax[0].grid(alpha=0.3)

    # --- עקומה 2: כמה עלילה נהרסת ככל שמסננים יותר ---
    L = pd.read_csv("bounding_removed_log.csv")
    L = L[L.content_words >= 3].sort_values("register_weight", ascending=False)
    tot = L.chars.sum()
    frac = np.cumsum(L.chars.values) / tot
    # שיעור התווים שהוסרו והיו למעשה עלילה, מצטבר
    plotty = np.cumsum((L.overlap_with_plot.values >= 0.5) * L.chars.values) \
             / np.maximum(np.cumsum(L.chars.values), 1)
    ax[1].plot(frac * 100, plotty * 100, lw=2, color="crimson")
    ax[1].set_xlabel("% of flagged text removed (most register-like first)")
    ax[1].set_ylabel("% of removed text that was actually plot")
    ax[1].set_title("Cost of filtering harder")
    ax[1].grid(alpha=0.3)
    for q in (25, 50, 75, 100):
        i = min(int(len(frac) * q / 100) - 1, len(frac) - 1)
        ax[1].annotate(f"{plotty[i]*100:.1f}%", (frac[i]*100, plotty[i]*100),
                       fontsize=7, xytext=(3, 4), textcoords="offset points")
        print(f"  at {q:3d}% of removals: {plotty[i]*100:.2f}% of removed text was plot")

    fig.tight_layout(); fig.savefig(OUT_PDF)
    print(f"\nwrote {OUT_PDF}")


if __name__ == "__main__":
    main()
