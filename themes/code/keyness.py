"""
Keyness: אילו מילים שכיחות בעקביות בתקצירי Goodreads ולא בתקצירי CMU,
**באותם ספרים עצמם**.

הרעיון: תקציר Goodreads הוא טקסט שיווקי שכתב מו"ל, ותקציר CMU הוא תיאור
עלילה שכתב קורא בוויקיפדיה. אם משווים את שני התיאורים של *אותו* ספר, כל
הבדל בין אוצר המילים אינו יכול לנבוע מהעלילה - העלילה זהה - אלא רק
מהרגיסטר. המילים שיוצאות מכך הן אוצר המילים של המו"ל, והן אלו שיש
לנטרל לפני חילוץ הנושאים.

ההצמדה נעשית לפי כותרת מנורמלת, עם פסילת כותרות דו-משמעיות בשני הצדדים
ואימות שנה כשהיא ידועה. אין כאן שום שימוש במגמות של CMU לאורך זמן -
הביקורת היחידה שנדרשת היא "אותו ספר, שני מתארים".
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
# פער השנים המרבי המותר בין שתי הרשומות כשהשנה ידועה בשני הצדדים.
# CMU נותן לעיתים את שנת המהדורה ולא את שנת החיבור, ולכן אפס יהיה נוקשה מדי
YEAR_TOLERANCE = 2
# כותרות קצרות מדי ("Home", "1984") מתאימות ליותר מדי ספרים שונים
MIN_TITLE_CHARS = 6

_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")


def norm_title(value):
    """מנרמל כותרת להשוואה: ללא ניקוד, ללא פיסוק, רווחים מכווצים, אותיות קטנות."""
    s = unicodedata.normalize("NFKD", str(value))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def cmu_title_index(path=T.CMU_PATH):
    """
    מחזיר DataFrame של CMU עם כותרת מנורמלת, לאחר פסילת כותרות שאינן חד-משמעיות.
    כותרת שמופיעה ביותר משורה אחת ב-CMU נפסלת: אין דרך לדעת לאיזה ספר
    הרשומה ב-Goodreads מתאימה.
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
    מעבר יחיד על goodreads_books.json.gz שאוסף **כל** יצירה שכותרתה נמצאת
    ב-title_keys. אין כאן מכסה לעשור ואין דגימה: המחקר הזה עוסק ברגיסטר
    ולא במגמות, ולכן כל התאמה שנמצאת שווה משהו.

    כותרת נשמרת עם כל היצירות שנמצאו לה, כדי שאפשר יהיה לפסול בהמשך
    כותרת שהתאימה ליותר מיצירה אחת ב-Goodreads.
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
    בונה את קבוצת הזוגות: לכל כותרת, תקציר Goodreads אחד ותקציר CMU אחד.

    פסילות (לפי הסדר):
    1. כותרת דו-משמעית ב-CMU  - נפסלה כבר ב-cmu_title_index.
    2. כותרת שהתאימה ליותר מיצירת Goodreads אחת - אם השנה מכריעה בין
       המועמדים, נבחר המועמד היחיד שבטווח YEAR_TOLERANCE; אחרת נפסלת.
    3. שתי השנים ידועות ורחוקות זו מזו יותר מ-YEAR_TOLERANCE - כנראה
       שני ספרים שונים בעלי אותה כותרת.
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
        # השנה המועדפת היא זו של Goodreads (שנת הפרסום המקורית של היצירה);
        # CMU משמש כגיבוי כשהיא חסרה
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

# ספי הסינון לרשימת ההסרה המוצעת.
# G2 הוא מבחן יחס-נראות עם דרגת חופש אחת; 15.13 הוא p<0.0001
G2_MIN = 15.13
# יחס-לוג 0.585 פירושו "שכיח פי 1.5 לפחות בגודריידס מאשר ב-CMU, לאחר נרמול
# אורך". הסף מכוון בכוונה נמוך: המסננת שמגינה על מילות תוכן היא phi
# שלהלן, ולא ה-keyness. סף keyness נוקשה יותר לא הגן על אף מילת תוכן
# נוספת - הוא רק העלים מילות שיווק אמיתיות (unforgettable, compelling)
LOG_RATIO_MIN = 0.585
# מילה שמופיעה בפחות מ-0.5% מהתקצירים (כ-43 ספרים) היא תכונה של אותם
# ספרים ולא של הרגיסטר
MIN_DOC_SHARE = 0.005
# עשור עם פחות זוגות מכך אינו נספר בבדיקת העקביות
CONSISTENCY_MIN_PAIRS = 30
# "בעקביות" - שכיח יותר בגודריידס ב-85% מהעשורים הנספרים לפחות.
# דרישה של 100% פוסלת מילים בגלל עשור בודד בעל 30 זוגות
CONSISTENCY_MIN = 0.85


def matched_lemmas(cache_path=LEMMA_CACHE, force_reload=False):
    """מלמטז את שני הצדדים באותו צינור בדיוק ששימש למודל הנושאים."""
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
    """מחזיר (ספירת מופעים, ספירת מסמכים) עבור עמודת מחרוזות של למות."""
    tokens, docs = Counter(), Counter()
    for text in series:
        toks = text.split()
        tokens.update(toks)
        docs.update(set(toks))
    return tokens, docs


def keyness_table(df):
    """
    מחשב keyness של Goodreads מול CMU על אותם ספרים.

    G2 (log-likelihood) נמדד על שכיחות המופעים ולא על שכיחות המסמכים, משום
    שתקציר CMU ארוך פי 2.6 בחציון מתקציר Goodreads: מדד ברמת המסמך היה
    מזכה את CMU רק בשל האורך, ואילו שכיחות יחסית מנוטרלת אורך מעצם הגדרתה.
    ספירת המסמכים עדיין מדווחת, כדי לחשוף מילה שכל מופעיה באים מספר בודד.

    log_ratio הוא log2 של יחס השכיחויות היחסיות, עם החלקת 0.5 לצד שבו
    המילה נעדרת לגמרי.
    """
    gr_tok, gr_doc = _counts(df["GR_Lemmas"])
    cmu_tok, cmu_doc = _counts(df["CMU_Lemmas"])
    n_gr, n_cmu = sum(gr_tok.values()), sum(cmu_tok.values())
    n_docs = len(df)
    print(f"  {n_gr} Goodreads tokens vs {n_cmu} CMU tokens "
          f"({n_cmu / n_gr:.2f}x longer on the CMU side)")

    # עקביות: לכל עשור בעל די זוגות, האם המילה שכיחה יותר בגודריידס
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
        if a + b < 20:          # מילים נדירות מדי מכדי להעריך יחס
            continue
        e1 = n_gr * (a + b) / (n_gr + n_cmu)
        e2 = n_cmu * (a + b) / (n_gr + n_cmu)
        g2 = 2 * ((a * np.log(a / e1) if a else 0) + (b * np.log(b / e2) if b else 0))
        # החלקה נדרשת רק כשאחד הצדדים אפס; אחרת היחס מחושב כפי שהוא
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
    # הכיוון נשמר בסימן של log_ratio; g2 עצמו חסר-כיוון
    out["direction"] = np.where(out["log_ratio"] > 0, "goodreads", "cmu")
    return out.sort_values("g2", ascending=False).reset_index(drop=True)


def suggested_removals(table):
    """
    רשימת ההסרה המוצעת: מילים שכיחות בעקביות בגודריידס ולא ב-CMU.
    כל התנאים חייבים להתקיים יחד - מובהקות, גודל אפקט, פיזור על פני ספרים
    ועקביות על פני העשורים.
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
            # phi אינו יציב כשמספר הספרים שבהם המילה מופיעה בשני הצדדים קטן.
            # "penguin" למשל מקבל phi 0.18 על סמך 5 חפיפות בלבד מתוך 8,680
            # הסימון רלוונטי רק כשמילה *ניצלה* מהסרה בזכות phi גבוה: phi נמוך
            # על מעט חפיפות הוא בדיוק מה שמצופה ממילת רגיסטר, ואינו חשוד
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
    מדפיס את phi על קבוצת הייחוס. זהו הבסיס לשני הספים: ללא הכיול הזה
    כל סף היה ניחוש, ועם הכיול רואים שהחלוקה עצמה בימודלית.
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
    # דירוג לפי phi: קודם מה שאפשר להסיר בבטחה, אחר כך מה ששנוי במחלוקת
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



# שני הספים כוילו על קבוצת ייחוס, ולא נבחרו לפי תחושה. ראו calibrate_phi():
# מילות מטא ביבליוגרפיות מובהקות מקבצות ב-phi שבין 0.00 ל-0.05
# (isbn -0.003, reprint -0.003, anthology -0.001, paperback 0.010,
#  bestseller 0.013, edition 0.021, introduction 0.050), ואילו מילות תוכן
# מובהקות מתחילות ב-0.12 ועולות (money 0.12, story 0.14, life 0.16,
# marriage 0.19, love 0.23, war 0.37, vampire 0.70). בין 0.05 ל-0.12 יש
# פער ריק כמעט, ולכן הוא מדווח כתחום ביניים במקום להיחתך בנקודה שרירותית
PHI_REGISTER_MAX = 0.05
PHI_CONTENT_MIN = 0.12
# מתחת למספר הזה של ספרים שבהם המילה מופיעה בשני התיאורים, phi מבוסס על
# מדגם קטן מדי מכדי לסמוך עליו, והמילה מסומנת בפלט
MIN_BOTH_FOR_PHI = 20

# קבוצת הייחוס לכיול: מילות תוכן חד-משמעיות מול מטא-דאטה ביבליוגרפי
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
    המבחן שההצמדה לפי ספר נועדה לאפשר: האם המילה בתקציר המו"ל **מנבאת**
    את העלילה?

    לכל מילה נבנית טבלת 2x2 על פני זוגות הספרים - האם הופיעה בצד גודריידס,
    האם הופיעה בצד CMU - ומחושב מקדם phi. ההיגיון: מילת תוכן אמיתית
    ("war", "murder") תופיע בשני התיאורים של אותו ספר, משום ששניהם מתארים
    את אותה עלילה; מילת רגיסטר ("classic", "edition") תופיע רק בצד המו"ל,
    ללא כל קשר לספר שמתחתיה. phi נמוך הוא הראיה שהמילה אינה נושאת תוכן,
    ולכן אפשר להסירה בלי לאבד נושא.
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
    phi לכל אוצר המילים, בחישוב מטריצי. הלולאה שב-paired_association היא
    O(מילים x מסמכים) ואינה מעשית ל-28 אלף מילים; כאן n11 מתקבל ממכפלה
    איבר-איבר של שתי מטריצות בינאריות
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
    משקל הרגיסטר של כל מילה, לשימוש מסננים אחרים.
    חיובי = אוצר מילים של מו"ל, שלילי = אוצר מילים של עלילה, אפס = נייטרלי.
    ה-log_ratio מדוכא ככל ש-phi גבוה יותר, כלומר ככל שהמילה כן מנבאת את
    הספר שמתחתיה, כדי ש-war ו-adventure יישארו קרובים לאפס.
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
    # mergesort יציב: quicksort מחזיר סדר אחר בכל ריצה עבור מילים בעלות
    # אותו משקל בדיוק, והקובץ יוצא שונה בבתים אף שתוכנו זהה
    out = pd.DataFrame(rows).sort_values("weight", ascending=False,
                                         kind="mergesort")
    out.to_csv(path, index=False)
    print(f"wrote {path}: {len(out)} words "
          f"({(out.weight > 0).sum()} publisher-side, {(out.weight < 0).sum()} plot-side)")
    return out
