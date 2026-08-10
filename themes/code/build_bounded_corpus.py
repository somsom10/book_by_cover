"""
בניית מטמון הקורפוס שכל שאר הצינור קורא ממנו: themes_corpus_bounded.pkl.

השלב היחיד שאינו נמצא ב-build_corpus של themes.py הוא bounding: חיתוך
הפסקאות השיווקיות מקצוות התקציר (ראו bounding.py). הוא נשמר בסקריפט נפרד
ולא נתפר לתוך clean_corpus כדי ש-themes.py יישאר בעל התנהגות אחת בלבד -
המטמון שנבנה כאן הוא מה שהוגש, ומטמון שנבנה בלעדיו הוא ריצת ביקורת.

סדר הפעולות קובע ולכן הוא מפורש כאן:
  טעינה מהמאגר הגולמי  ->  clean_corpus  ->  bounding  ->  למטיזציה
הלמטיזציה אחרונה, אחרת ה-Lemmas היו של הטקסט שלפני החיתוך והחיתוך לא היה
מגיע למודל כלל.

זמן ריצה: עשרות דקות. הקובץ הגולמי הוא 2GB ו-spaCy רץ על ~73 אלף מסמכים.
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
        # מצב עשן: מריצים על מדגם קטן רק כדי לוודא שהצינור עובר מקצה לקצה
        df = df.sample(min(limit, len(df)), random_state=42).reset_index(drop=True)
        print(f"  smoke test on {len(df)} documents")
    print(f"  {len(df)} works after de-duplicating editions")

    # התיחום עצמו: חיתוך הקצוות, ואז מחיקת שמות עיתונים וקמעונאים כצירוף
    # ("New York Times", "Barnes & Noble") וכיווץ רווחים. שתי הפעולות יחד
    # הן מה שמייצר את הקורפוס הקנוני; הראשונה לבדה מייצרת קורפוס אחר
    def bound(summaries):
        # קבוצת המשפטים החוזרים נגזרת מהטקסט הנקי, וזהו הכלל שמוצא מו"לים
        # שאיש לא רשם ("An NYRB Classics Original", "From the Paperback edition")
        # 10 ולא 5, שהוא ברירת המחדל של evaluate_bounding: הסף נמדד מול
        # הקורפוס המקורי ולא נבחר. 5 מוחק 15 מסמכים עודפים, 12 משאיר 10
        # מסמכים עם פרסומת, ו-10 משחזר את הקורפוס המקורי בדיוק - 73,411
        # מסמכים, אפס הבדלים. זהו גם הסף של strip_repeated_sentences
        rep = E.repeated_sentences(summaries, min_docs=REPEAT_MIN_DOCS)
        return [B.strip_publication_names(
                    B.bound_summary(t, repeated=rep).text)[0]
                for t in summaries]

    # במצב עשן הסטטיסטיקה נכתבת לשם אחר: אחרת ריצת בדיקה על 300 מסמכים
    # דורסת את קובץ הזיהום-לפי-עשור של הריצה האמיתית
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
