"""
מדווח כל מגמה בשתי הגרסאות: לפני הנרמול-מחדש ואחריו.

הנרמול נכון עניינית - "מתוך מה שהתיאור באמת אומר על הספר" - אבל אינו
נייטרלי: שיעור הרעש יורד לאורך הזמן (12.9% ב-1900 מול 6.3% ב-2010), ולכן
המכפיל נסחף מ-1.148 ל-1.067 והעשורים המוקדמים מנופחים יותר. נושא שטוח
לחלוטין ייראה כיורד בכ-7%.

**הפער בין שתי העמודות הוא ההטיה עצמה, גלויה לעין.** אותו סימן וסדר גודל
דומה - הטענה בטוחה. סימן הפוך - אין טענה. שיאים אינם מושפעים, משום
שהמכפיל משתנה בהדרגה ואינו יכול להזיז מקסימום מקומי.
"""
import numpy as np
import pandas as pd
import themes as TH

OUT = "trends_raw_vs_renorm.csv"
CORPUS = "themes_corpus_bounded.pkl"


def main():
    df = TH.build_corpus(cache_path=CORPUS)
    vec, nmf, W = TH.fit_topics(df)
    labels = TH.topic_labels(vec, nmf, top_n=8)
    flagged = TH.artifact_topics(TH.topic_labels(vec, nmf))
    keep = [j for j in range(W.shape[1]) if j not in flagged]

    dec = df["Decade"].values
    decades = sorted(set(dec))
    Wk = W[:, keep]
    rs = Wk.sum(axis=1, keepdims=True)
    Wn = np.divide(Wk, rs, out=np.zeros_like(Wk), where=rs > 0)

    raw = np.array([Wk[dec == d].mean(axis=0) for d in decades]) * 100
    ren = np.array([Wn[dec == d].mean(axis=0) for d in decades]) * 100
    # ההשוואה מסתיימת ב-2000: 2010 הוא עשור חלקי (2010-2017, מוטה ל-2013)
    idx = [i for i, d in enumerate(decades) if d <= 2000]
    x = np.array([decades[i] for i in idx], dtype=float)

    rows = []
    for j, k in enumerate(keep):
        sr = np.polyfit(x, raw[idx, j], 1)[0] * 10
        sn = np.polyfit(x, ren[idx, j], 1)[0] * 10
        safe = (np.sign(sr) == np.sign(sn)) and abs(sn - sr) < 0.5 * max(abs(sn), 1e-9)
        rows.append({
            "topic": labels[k],
            "slope_raw_pp_per_decade": round(sr, 3),
            "slope_renorm_pp_per_decade": round(sn, 3),
            "bias_pp_per_decade": round(sn - sr, 3),
            "peak_decade": int(decades[int(np.argmax(ren[:, j]))]),
            "claimable_slope": bool(safe and abs(sn) >= 0.10),
        })
    out = pd.DataFrame(rows).sort_values("slope_renorm_pp_per_decade",
                                         key=abs, ascending=False)
    out.to_csv(OUT, index=False)

    print("slopes 1900s-2000s, both ways (2010s excluded: partial decade)\n")
    print(f"  {'topic':<40}{'raw':>8}{'renorm':>9}{'bias':>8}{'peak':>7}  claim?")
    for r in out.itertuples():
        print(f"  {r.topic[:38]:<40}{r.slope_raw_pp_per_decade:+8.2f}"
              f"{r.slope_renorm_pp_per_decade:+9.2f}{r.bias_pp_per_decade:+8.2f}"
              f"{str(r.peak_decade)+'s':>7}"
              + ("   YES" if r.claimable_slope else "    no"))
    n = int(out.claimable_slope.sum())
    print(f"\n{n}/{len(out)} slopes are claimable: same sign both ways, bias under "
          f"half the effect, and at least 0.10pp per decade.")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
