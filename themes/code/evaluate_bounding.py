"""
הערכת התיחום, ובעיקר: יומן של מה הוסר ולמה, וניתוח כמה מזה היה תוכן אמיתי.

השאלה "האם הסרנו בטעות עלילה?" ניתנת למדידה ולא רק לעיון, בזכות
keyness_matched.pkl: 8,680 ספרים שלכל אחד מהם יש גם תקציר מו"ל מגודריידס
וגם תיאור עלילה מוויקיפדיה. אם קטע שהוסר מהתקציר חולק הרבה מילים עם תיאור
העלילה של *אותו ספר*, הקטע תיאר את העלילה וההסרה הייתה שגיאה.

זה מה שהופך את היומן לכלי ולא לרשימה: לכל שורה בו יש ציון אובייקטיבי של
"כמה תוכן היה כאן".
"""

import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import bounding as B

MATCHED = "keyness_matched.pkl"
WEIGHTS = "keyness_word_weights.csv"
LOG_PATH = "bounding_removed_log.csv"
RULE_STATS = "bounding_rule_stats.csv"

# משפט שחוזר מילה במילה במספר הזה של ספרים שונים אינו מתאר אף אחד מהם
REPEAT_MIN_DOCS = 5
# קטע שהוסר וחולק לפחות שיעור זה ממילות התוכן שלו עם תיאור העלילה של אותו
# ספר נחשב חשוד: כנראה הסרנו עלילה
CONTENT_OVERLAP_ALERT = 0.50

_WORD_RE = re.compile(r"[a-z][a-z']+")
_STOP = frozenset("""
the a an and or but of to in is was were be been are for with on at by from that
this it as not his her he she they we you i had have has will would there their
which who what when who whom while into out up down over under again then than
so no nor own too very can just do does did doing about after before between
""".split())


def content_words(text):
    return {w for w in _WORD_RE.findall(str(text).lower())
            if w not in _STOP and len(w) > 2}


def load_weights():
    """טוען את משקלי הרגיסטר אם קיימים; אחרת מחזיר מילון ריק."""
    if not os.path.exists(WEIGHTS):
        print(f"  ({WEIGHTS} not found - register-weight columns will be 0)")
        return {}
    w = pd.read_csv(WEIGHTS)
    return dict(zip(w["word"], w["weight"]))


def repeated_sentences(texts, min_docs=REPEAT_MIN_DOCS):
    """
    שיטה 2ב: הקבוצה נגזרת מהקורפוס ולא נכתבת ביד. זהו החלק היחיד מבין
    שלוש השיטות שמוצא מו"לים שאיש לא רשם מראש.
    """
    counts = Counter()
    for t in texts:
        seen = {re.sub(r"\s+", " ", s.strip()).lower()
                for _, _, s in B.split_sentences(str(t))}
        counts.update(s for s in seen if len(s) >= 25)
    rep = {s for s, n in counts.items() if n >= min_docs}
    print(f"  {len(rep)} sentence types repeat in >={min_docs} books "
          f"(method 2b, corpus-derived)")
    return rep


def run(strict=False, limit=None):
    df = pd.read_pickle(MATCHED)
    if limit:
        df = df.head(limit)
    print(f"Evaluating bounding on {len(df)} matched books "
          f"(strict={strict})")
    weights = load_weights()
    rep = repeated_sentences(df["GR_Summary"])

    rows, log = [], []
    for r in df.itertuples():
        res = B.bound_summary(r.GR_Summary, repeated=rep, strict=strict)
        cmu_words = content_words(r.CMU_Summary)
        before = content_words(r.GR_Summary)
        after = content_words(res.text)

        def jac(a, b):
            return len(a & b) / len(a | b) if (a | b) else 0.0

        def reg(words):
            v = [weights[w] for w in words if w in weights]
            return float(np.mean(v)) if v else 0.0

        rows.append({
            "Decade": r.Decade,
            "chars_before": len(r.GR_Summary), "chars_after": len(res.text),
            "fallback": res.fallback, "n_removed": len(res.removed),
            "jac_before": jac(before, cmu_words), "jac_after": jac(after, cmu_words),
            "reg_before": reg(before), "reg_after": reg(after),
        })

        # היומן: כל קטע שהוסר, עם ציון אובייקטיבי לכמה תוכן היה בו
        for rm in res.removed:
            span = content_words(rm["text"])
            overlap = len(span & cmu_words) / len(span) if span else 0.0
            log.append({
                "Title": r.Title, "Year": r.Year, "side": rm["side"],
                "rules": rm["rules"], "applied": not res.fallback,
                "chars": len(rm["text"]),
                "content_words": len(span),
                "overlap_with_plot": overlap,
                "register_weight": reg(span),
                "text": rm["text"][:300],
            })

    stats = pd.DataFrame(rows)
    L = pd.DataFrame(log)
    L.to_csv(LOG_PATH, index=False)
    print(f"\nwrote {LOG_PATH} ({len(L)} removed spans)")
    return stats, L


def report(stats, L):
    kept = stats[~stats.fallback]
    print("\n" + "=" * 78)
    print("BOUNDING EVALUATION")
    print("=" * 78)

    print(f"\n1. Does bounding move the blurb toward the plot summary?")
    print(f"   token overlap with the CMU plot summary of the same book")
    print(f"     before {kept.jac_before.mean():.4f}   "
          f"after {kept.jac_after.mean():.4f}   "
          f"{'UP' if kept.jac_after.mean() > kept.jac_before.mean() else 'DOWN'}")

    print(f"\n2. Register gap (negative = plot-like; CMU baseline is about -0.64)")
    print(f"     before {kept.reg_before.mean():+.4f}   after {kept.reg_after.mean():+.4f}")

    print(f"\n3. Trim rate")
    trim = 1 - kept.chars_after.sum() / kept.chars_before.sum()
    print(f"     {trim*100:.1f}% of characters removed, "
          f"{(stats.n_removed > 0).mean()*100:.1f}% of books touched")

    print(f"\n4. Trim rate by decade (the confound: must not rise with time)")
    by = stats[stats.Decade >= 1900].groupby("Decade").apply(
        lambda g: 100 * (1 - g.chars_after.sum() / g.chars_before.sum()))
    for d, v in by.items():
        print(f"     {int(d)}s  {v:5.1f}%")
    if len(by) > 2:
        r = np.corrcoef(by.index.values.astype(float), by.values)[0, 1]
        print(f"     correlation with decade: r = {r:+.2f}   "
              f"({'SAFE' if r <= 0 else 'WARNING - cuts modern text harder'})")

    print(f"\n5. Fallback rate (safety rail fired): "
          f"{stats.fallback.mean()*100:.1f}%")

    print(f"\n6. Rule firing counts and how much plot each rule ate")
    per = []
    for rule in B.ALL_RULE_NAMES:
        hit = L[L.rules.str.contains(rf"\b{re.escape(rule)}\b", regex=True, na=False)]
        if len(hit) == 0:
            per.append((rule, 0, np.nan, np.nan, 0))
            continue
        per.append((rule, len(hit), hit.overlap_with_plot.mean(),
                    hit.register_weight.mean(),
                    int((hit.overlap_with_plot >= CONTENT_OVERLAP_ALERT).sum())))
    P = pd.DataFrame(per, columns=["rule", "n_fired", "mean_plot_overlap",
                                   "mean_register_weight", "n_suspicious"])
    P = P.sort_values("n_fired", ascending=False)
    P.to_csv(RULE_STATS, index=False)
    print(P.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print(f"\n   wrote {RULE_STATS}")
    print("   mean_plot_overlap = share of the removed span's content words that")
    print("   also appear in that book's Wikipedia plot summary. High = the rule")
    print("   is eating plot. n_suspicious counts spans above "
          f"{CONTENT_OVERLAP_ALERT:.0%}.")

    dead = P[P.n_fired == 0].rule.tolist()
    if dead:
        print(f"\n   rules that never fired: {', '.join(dead)}")


def audit(L, top=25):
    """הביקורת שהתבקשה: אילו הסרות היו למעשה תוכן, ומאיזה כלל הן באו."""
    print("\n" + "=" * 78)
    print("AUDIT - removed spans that look like real content")
    print("=" * 78)
    bad = L[(L.overlap_with_plot >= CONTENT_OVERLAP_ALERT) & (L.content_words >= 4)]
    print(f"{len(bad)} of {len(L)} removed spans ({len(bad)/max(len(L),1)*100:.1f}%) "
          f"share >={CONTENT_OVERLAP_ALERT:.0%} of their content words with the plot")
    print(f"they account for {bad.chars.sum()/max(L.chars.sum(),1)*100:.1f}% "
          f"of all removed characters\n")
    print("worst offenders by rule:")
    print(bad.rules.value_counts().head(12).to_string())
    print("\nexamples:")
    for r in bad.nlargest(top, "chars").itertuples():
        print(f"  [{r.rules}] overlap {r.overlap_with_plot:.2f}  {r.Title[:40]}")
        print(f"      {r.text[:200]}")


if __name__ == "__main__":
    strict = "--strict" in sys.argv
    limit = None
    for a in sys.argv[1:]:
        if a.startswith("--limit="):
            limit = int(a.split("=")[1])
    stats, L = run(strict=strict, limit=limit)
    report(stats, L)
    audit(L)
