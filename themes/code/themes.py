"""
זיהוי נושאים (themes) והמגמות שלהם לאורך העשורים, באמצעות NMF על מטריצת TF-IDF.

למה לא TextRank: נמדד שציון ה-PageRank על גרף ההופעה המשותפת פרופורציוני
כמעט לחלוטין לדרגת הצומת (מקדם שונות 0.173), ודרגת הצומת בתורה עוקבת אחרי
תדירות המילה. כלומר TextRank כאן שקול בקירוב לספירת מילים משוקללת, ולכן החזיר
את אותן מילים כלליות (book, story, life) בכל 52 העשורים.

הרעיון כאן הפוך: במקום לדרג מילים *שונות* בכל עשור, קובעים אוסף נושאים אחד
לכל המאגר, ומודדים את *המשקל* של כל נושא בכל עשור. כך נושא נצחי כמו אהבה נשאר
בניתוח תמיד ומקבל עקומה לאורך הזמן, במקום להיעלם מפני שהוא אינו "ייחודי"
לאף עשור. פירוט מלא ב-METHODS.md.
"""

import hashlib
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

import text as T

# --- הגדרות ---
# הקורפוס הקנוני הוא זה שעבר bounding, ולכן הוא ברירת המחדל. הקובץ נבנה
# על ידי build_bounded_corpus.py; build_corpus לבדו *אינו* מבצע bounding,
# ולכן אסור לתת לו לייצר קובץ בשם הזה - ראו הבדיקה ב-main
CACHE_PATH = "themes_corpus_bounded.pkl"

# תיקיית הפלט. ריצת הביקורת על CMU כותבת לתיקייה נפרדת, אחרת היא הייתה
# דורסת את התוצאות של גודריידס - וההשוואה בין השתיים היא כל מטרתה
OUT_DIR = "."


def _out(name):
    """נתיב פלט בתוך OUT_DIR, שנוצרת אם אינה קיימת."""
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)
# היקף הזיהום לפי עשור. נכתב בזמן הניקוי, כי רק אז ידוע מה נמחק
CLEAN_STATS_PATH = "artifact_clean_stats.csv"

# מספר הנושאים. 20-30 הוא טווח סביר לקורפוס בסדר גודל כזה; יש לקרוא את רשימות
# המילים שמתקבלות ולהתאים אם הנושאים אינם מובנים
# 22 ולא 25. סריקת K על הקורפוס הנקי (12/15/18/22/25) הראתה ש-25 מפצל
# נושאים אמיתיים לשאריות: מדע בדיוני נשבר לשניים (earth/planet מול
# time/travel/space), נושא הנשים משוכפל ל-"young, boy, lady, handsome",
# ונוצרים שני נושאים ריקים. ב-22 כל נושא ניתן לשיום, ומדע בדיוני, אמנות,
# שפה ואגדות מופיעים כל אחד בנפרד ובבירור. מתחת ל-18 קורה ההפך - הנושאים
# מתרחבים ומתמזגים (משפחה, ילדים ונערות הופכים לנושא אחד)
N_TOPICS = 25

# תקרת ספרים לעשור. אין כאן ויתור על דיוק לטובת מהירות: בגודל מדגם 5000
# רווח הסמך של נתח נושא הוא כבר +-0.75 נקודות אחוז, ותוספת ספרים מעבר לכך
# אינה משנה את התוצאה בפועל. עשורים קטנים יותר נלקחים במלואם.
# 10,000 ולא 5,000. שבעה מתוך שנים-עשר העשורים נחסמו על ידי המכסה הקודמת,
# כלומר היו זמינים עוד ספרים והם פשוט לא נדגמו. המדידה הראתה שהגדלת מדגם
# האימון מ-1,500 ל-4,000 לעשור העלתה את שחזור נושא המדע הבדיוני מ"לא נוצר
# כלל" ל-0.990, ולכן יש טעם ממשי בעוד נתונים. העשורים המוקדמים אינם
# מושפעים - הם ממילא מכילים את כל מה שקיים
PER_DECADE_CAP = 10000

# תקרה נפרדת לשלב *אימון* המודל. הנושאים נלמדים ממדגם מאוזן בין העשורים,
# אחרת 390 אלף ספרי שנות ה-2010 היו קובעים לבדם את מרחב הנושאים, ולתוכן של
# המאה ה-19 לא היה נושא להישען עליו. הנתחים עצמם מחושבים על כל הקורפוס.
# 4,000 ולא 1,500. הסף הקודם נבחר כ"המספר הגדול ביותר ששומר על איזון"
# (העשור הדל ביותר מחזיק 1,474), אך מדידה הראתה שהוא היה יקר: נושא המדע
# הבדיוני כלל לא נוצר ב-1,500, נוצר ולא היה יציב ב-2,500 (0.587), וב-4,000
# הוא מגיע ל-0.990 ומזהה את עצמו - earth, human, planet, space, alien,
# science_fiction, universe. הבעיה לא הייתה פרופורציה אלא רעש דגימה.
# האיזון נשמר בקירוב: העשורים המוקדמים תורמים את כל מה שיש להם (1,474-1,574)
# והמאוחרים 4,000, כלומר יחס 2.7:1 במקום 300:1 ללא הגבלה
FIT_PER_DECADE = 4000

# אורך מזערי של תקציר אחרי הניקוי
MIN_SUMMARY_CHARS = 200

# העשור המוקדם ביותר שנכנס לניתוח.
#
# הסף היה 1790 ועודכן ל-1900 (2026-08-07) בהחלטה מפורשת: העשורים שלפני 1900
# דקים מדי. 66 עד 984 ספרים לעשור מול 1,576 עד 4,808 מ-1900 ואילך, רווח סמך
# של 4.4 נקודות אחוז על נתח של 11.5% בשנות ה-1800, ורעש ביבליוגרפי של 16%
# מול 6% בשנות ה-2010. מעבר לדקות יש הטיית שרידות: גודריידס מקטלג ספרים
# שהגיעו למהדורה מודרנית, ולכן עשור מוקדם הוא הקאנון ומה שהודפס מחדש,
# ולא מדגם של מה שראה אור.
#
# מה שנאבד בכך: פילוסופיה 1790, שירה רומנטית 1810, הרומן הוויקטוריאני,
# וזינוק המדע הבדיוני של ורן ב-1860. אלה היו האימות ההיסטורי החזק ביותר של
# השיטה, והם אינם ניתנים למדידה עוד. הבדיקות שנשארו בטווח מפורטות ב-
# METADATA_VOCABULARY.md.
#
# הסינון נעשה לפני האימון ולא רק בדיווח, כי העשורים האלה גם השתתפו באימון
# והטו את מרחב הנושאים. המטמון נשאר מלא, כדי שאפשר יהיה לשנות את הסף בלי
# לבנות אותו מחדש - שינוי הערך הזה מספיק כדי לחזור ל-1790.
MIN_DECADE = 1900

# --- שלב הניקוי ---

# טקסט של בתי דפוס וסריקה שנמצא בשדה description במקום תקציר אמיתי.
# אלה אינם תיאורי ספרים כלל, ולכן נמחקים גם כשהם מופיעים פעם אחת בלבד.
# נמדד: 7.7% מהשורות, 11% לפני 1900, עד 30% בעשורים בודדים.
_BOILERPLATE_RE = re.compile(
    "|".join([
        r"converted from its physical edition",
        r"pre-19\d\d historical reproduction",
        r"occasional imperfections",
        r"republishing these classic works",
        r"reproduction of a book published",
        r"digitization process",
        r"scanning process",
        r"kindle edition includes wireless",
        r"culturally important",
        r"print on demand",
        r"facsimile",
        r"we have elected to bring",
        r"original artifact",
        r"quality assurance was conducted",
        r"this book may have occasional",
        r"scanned (?:copy|image|version)",
        r"optical character recognition",
    ]),
    re.I,
)

# אורך הפתיח המשמש לזיהוי כפילות חלקית (ראו _dedupe)
_PREFIX_LEN = 120


def _normalise(text):
    """צמצום רווחים והורדה לאותיות קטנות, לצורך השוואת טקסטים."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


# משפט שחוזר על עצמו בין ספרים שונים אינו תיאור של ספר אלא טקסט של המוציא לאור.
# הסף נמדד על הקורפוס: מ-10 ומעלה כל 55 סוגי המשפטים שנמצאו הם טקסט שיווקי או
# פרסומת לסדרה, בלי אף מקרה של תוכן אמיתי, ואף מסמך אינו מתרוקן.
_SENTENCE_MIN_DOC_FREQ = 10
# משפטים קצרים מכך אינם נספרים ואינם נמחקים לעולם ("full-color illustrations.")
_MIN_SENTENCE_CHARS = 25
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _sentence_parts(text):
    """
    מפצל טקסט למשפטים ומחזיר לכל אחד (המקור, מפתח מנורמל).
    המפתח משמש להשוואה בלבד; המחיקה נעשית מהטקסט המקורי, כדי לשמר אותיות
    גדולות - בלעדיהן מנתח חלקי הדיבר של spaCy אינו מזהה שמות פרטיים (PROPN).
    """
    parts = []
    for piece in _SENT_SPLIT_RE.split(str(text).strip()):
        piece = piece.strip()
        if piece:
            parts.append((piece, _normalise(piece)))
    return parts


def strip_repeated_sentences(df, min_df=_SENTENCE_MIN_DOC_FREQ, verbose=True):
    """
    מסיר משפטים החוזרים על עצמם בין ספרים שונים, ומשאיר את שאר המסמך במקום
    למחוק אותו כולו - כך תקציר אמיתי שהודבקה בסופו פסקת פרסומת נשמר.

    זה מחליף רשימת ביטויים קבועה, שאינה יכולה לנצח: מדידה על הקורפוס הראתה
    לפחות שמונה הוצאות לאור עם אותה תבנית (Forgotten Books, Kessinger,
    Endeavour, Oxford World's Classics, Penguin Classics, HarperPerennial,
    Pushkin, General Books). ספירת תדירות מוצאת גם הוצאות שטרם נראו.

    מחזיר (DataFrame מעודכן, קבוצת המשפטים שהוסרו). לכל שורה נוסף
    StrippedChars, מספר התווים שהוסרו ממנה, לצורך מדידת היקף הזיהום.
    """
    split = df["Summary"].map(_sentence_parts)

    counts = Counter()
    for parts in split:
        # פעם אחת למסמך: ספר שחוזר על משפט משלו לא ידחוף אותו מעל הסף
        counts.update({k for _, k in parts if len(k) >= _MIN_SENTENCE_CHARS})
    repeated = {k for k, n in counts.items() if n >= min_df}

    kept_text, stripped_chars = [], []
    for parts in split:
        keep, removed = [], 0
        for original, key in parts:
            if key in repeated:
                removed += len(original)
            else:
                keep.append(original)
        kept_text.append(" ".join(keep))
        stripped_chars.append(removed)

    out = df.copy()
    out["Summary"] = kept_text
    out["StrippedChars"] = stripped_chars

    if verbose:
        touched = int((out["StrippedChars"] > 0).sum())
        total = int(out["Summary"].str.len().sum() + out["StrippedChars"].sum())
        pct = out["StrippedChars"].sum() / total if total else 0.0
        print(f"    -{len(repeated):6d} repeated sentence types "
              f"(>={min_df} docs) removed from {touched} documents, {pct:.2%} of text")
    return out, repeated


def _decade(df):
    """עשור לכל שורה, בלי לשנות את ה-DataFrame עצמו."""
    return df["Year"] // 10 * 10


def clean_corpus(df, verbose=True, stats_path=CLEAN_STATS_PATH, bound_fn=None):
    """
    מסיר טקסט שאינו תיאור של הספר. חמישה שלבים, בסדר הזה:

    1. סינון טקסט של בתי דפוס - תוכן שידוע שהוא פסול, נמחק גם בעותק יחיד.
       נשאר בתוקף גם אחרי שלב 4, משום שהוא תופס רשומות שכולן הודעת סריקה
       ושאף משפט בהן אינו חוזר מספיק פעמים כדי לעבור את הסף.
    2. ניכוי כפילויות מדויקות - אותו טקסט בדיוק על ספרים שונים. תופס למשל
       פסקת ביוגרפיה של מחבר שהודבקה על כל ספריו (פסקה אחת על בלזק הופיעה
       7 פעמים בין 1831 ל-1843), שאינה מכילה אף ביטוי מהרשימה שלמעלה.
    3. ניכוי כפילויות חלקיות - אותה תבנית עם שינוי קל, למשל שם הוצאה אחר
       ("hesperides press are republishing" מול "we are republishing").
       ההשוואה נעשית לפי פתיח זהה, שהוא O(n) ותופס את התבנית שנמדדה.
    4. הסרת משפטים חוזרים - ראו strip_repeated_sentences. שלב זה מוחק פסקאות
       ולא שורות שלמות, ולכן תקציר אמיתי עם פרסומת בסופו שורד.
    5. סינון לפי אורך - *חייב* לבוא אחרי שלב 4, שכן ההסרה מקצרת מסמכים
       וחלקם יורדים מתחת לסף רק בעקבותיה.

    בדרך נאספת חלוקה לפי עשור של היקף הזיהום, ונכתבת ל-stats_path. זהו ממצא
    על המאגר ולא רק אבחון: הוא מכמת כמה מ"התקציר" בעשורים המוקדמים אינו
    תקציר כלל.
    """
    n0 = len(df)
    books0 = _decade(df).value_counts()

    df = df[~df["Summary"].str.contains(_BOILERPLATE_RE, na=False)]
    n1 = len(df)

    norm = df["Summary"].map(_normalise)
    df = df[~norm.map(lambda s: hashlib.md5(s.encode()).hexdigest()).duplicated()]
    n2 = len(df)

    norm = df["Summary"].map(_normalise)
    df = df[~norm.str[:_PREFIX_LEN].duplicated()]
    n3 = len(df)

    if verbose:
        print(f"  cleaning: {n0} rows")
        print(f"    -{n0-n1:6d} printer/scanner boilerplate")
        print(f"    -{n1-n2:6d} exact duplicate text")
        print(f"    -{n2-n3:6d} near-duplicate text (shared opening)")

    df, removed_sentences = strip_repeated_sentences(df, verbose=verbose)

    # שלב 4.5: תיחום. הוא *חייב* לרוץ כאן, בין הסרת המשפטים החוזרים לבין
    # סינון האורך, ולא אחרי clean_corpus - התיחום מקצר מסמכים, וחלקם יורדים
    # מתחת ל-MIN_SUMMARY_CHARS רק בעקבותיו. הרצה בסדר אחר מייצרת קורפוס
    # אחר ב-4% מהמסמכים, ומכאן מודל אחר לגמרי. bound_fn מועבר מבחוץ כדי
    # ש-themes.py לא יהיה תלוי ב-bounding.py
    # bound_fn מקבל את כל הסדרה ולא מסמך בודד, משום שהתיחום זקוק לקבוצת
    # המשפטים החוזרים - והיא נגזרת מהקורפוס *בשלב הזה*, אחרי הניקוי. חישוב
    # שלה על הטקסט הגולמי מחזיר קבוצה גדולה מדי ומוחק 35 מסמכים עודפים
    if bound_fn is not None:
        df = df.copy()
        df["Summary"] = list(bound_fn(df["Summary"]))
        if verbose:
            print(f"    bounded {len(df)} summaries")

    # התווים שהוסרו נספרים לפני סינון האורך, אחרת מסמכים שנמחקו *בגלל* ההסרה
    # לא ייספרו בזיהום שגרם למחיקתם
    dec_strip = _decade(df)
    stripped = df["StrippedChars"].groupby(dec_strip).sum()
    kept_chars = df["Summary"].str.len().groupby(dec_strip).sum()

    df = df[df["Summary"].str.len() >= MIN_SUMMARY_CHARS]
    n4 = len(df)

    if verbose:
        print(f"    -{n3-n4:6d} shorter than {MIN_SUMMARY_CHARS} chars")
        print(f"  => {n4} rows kept ({n4/n0:.1%})")

    df = df.reset_index(drop=True)

    stats = pd.DataFrame({
        "n_books_before": books0,
        "n_books": _decade(df).value_counts(),
        "chars_stripped": stripped,
        "chars_kept": kept_chars,
    }).fillna(0).astype("int64")
    stats.index.name = "decade"
    stats = stats.sort_index()
    total_chars = (stats["chars_stripped"] + stats["chars_kept"]).replace(0, np.nan)
    stats["pct_chars_stripped"] = stats["chars_stripped"] / total_chars * 100
    stats["pct_docs_dropped"] = (
        (stats["n_books_before"] - stats["n_books"])
        / stats["n_books_before"].replace(0, np.nan) * 100
    )
    stats.to_csv(_out(stats_path))
    if verbose:
        print(f"  wrote per-decade cleaning stats to {stats_path} "
              f"({len(removed_sentences)} sentence types removed)")
    return df


# --- ביטויים רב-מיליים ---
#
# "science fiction" הוא שם ז'אנר, לא שתי מילים. כשהוא נשאר מפוצל, המודל
# שולח את science לנושא הפילוסופיה (פילוסופיה של המדע) ואת fiction לנושא
# הביבליוגרפי - ולנושא המדע הבדיוני נשארות רק מילים גנריות של בניית עולם
# (world, earth, human, power), שמשותפות לפנטזיה ולהרפתקאות. זו הסיבה
# שהנושא קיבל ציון שחזור של 0.779 בלבד: אין לו עוגן ייחודי.
#
# אותו טיפול שניתן ל"New York Times" ב-bounding.py, מאותה סיבה בדיוק.
# הצירופים נדבקים לטוקן אחד *אחרי* הלמטיזציה, ולכן אין צורך להריץ את
# spaCy מחדש. המספרים בסוגריים הם מספר המסמכים שבהם הצירוף מופיע.
MULTIWORD_TERMS = [
    ("science fiction", "science_fiction"),   # 625
    ("new york", "new_york"),                 # 1441
    ("world war", "world_war"),               # 1322
    ("short story", "short_story"),           # 1199
    ("civil war", "civil_war"),               # 630
    ("fairy tale", "fairy_tale"),             # 290
    ("picture book", "picture_book"),         # 169
    ("young adult", "young_adult"),           # 108
    ("graphic novel", "graphic_novel"),       # 92
    ("private eye", "private_eye"),           # 68
    ("wild west", "wild_west"),               # 38
    ("coming age", "coming_of_age"),          # 36
    ("true crime", "true_crime"),             # 22
]


def join_multiword(lemmas):
    """מדביק צירופים קבועים לטוקן אחד בעמודת הלמות."""
    for src, dst in MULTIWORD_TERMS:
        lemmas = lemmas.str.replace(src, dst, regex=False)
    return lemmas


def load_goodreads_full(path=T.GOODREADS_PATH, works_path=T.WORKS_PATH,
                        per_decade_cap=PER_DECADE_CAP, random_state=42):
    """
    טוען את מאגר Goodreads כשהוא מתוארך לפי שנת הפרסום המקורית ומנוכה כפילויות
    מהדורה, ללא הדגימה האגרסיבית שבצינור ה-TextRank. הניקוי מתבצע אחרי הטעינה,
    כדי שספירות הניקוי יהיו על כל מה שנאסף.
    """
    import gzip, json, random
    from collections import defaultdict, Counter

    work_years = T.load_work_years(works_path)
    rng = random.Random(random_state)
    buckets = defaultdict(list)
    seen_per_decade = Counter()
    seen_works = set()

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            try:
                work_id = int(obj.get("work_id"))
            except (TypeError, ValueError):
                continue
            if work_id in seen_works:
                continue
            year = work_years.get(work_id)
            if year is None:
                continue
            if not T._is_english_code(obj.get("language_code", "")):
                continue
            summary = str(obj.get("description", "")).strip()
            if not summary or not T._is_english_text(summary):
                continue

            seen_works.add(work_id)
            decade = year // 10 * 10
            seen_per_decade[decade] += 1
            bucket = buckets[decade]
            row = {"Year": year, "Summary": summary}
            if per_decade_cap is None or len(bucket) < per_decade_cap:
                bucket.append(row)
            else:
                j = rng.randint(0, seen_per_decade[decade] - 1)
                if j < per_decade_cap:
                    bucket[j] = row

    rows = [r for d in sorted(buckets) for r in buckets[d]]
    return pd.DataFrame(rows)


def _apply_multiword(df):
    """מוחל בטעינה ולא בבנייה, כדי שהמטמון יישאר תקף בלי בנייה מחדש."""
    if "Lemmas" in df.columns:
        df = df.copy()
        df["Lemmas"] = join_multiword(df["Lemmas"])
    return df


def _apply_min_decade(df, min_decade=MIN_DECADE):
    """
    מסנן עשורים דלים מדי מכדי להיות ניתנים לקריאה. הסינון נעשה בטעינה ולא
    בכתיבה למטמון, כדי שהמטמון יישאר מלא ושינוי הסף לא יחייב בנייה מחדש.
    """
    if min_decade is None:
        return df
    n0 = len(df)
    out = df[df["Decade"] >= min_decade].reset_index(drop=True)
    if len(out) < n0:
        print(f"  dropped {n0 - len(out)} documents before {min_decade} "
              f"(too few books per decade to estimate a share)")
    return out


def build_corpus(cache_path=CACHE_PATH, force_reload=False, min_decade=MIN_DECADE):
    """טעינה, ניקוי ולמטיזציה. נשמר במטמון משום שהלמטיזציה היא החלק האיטי."""
    if not force_reload and os.path.exists(cache_path):
        print(f"Loading cached corpus from {cache_path}")
        return _apply_min_decade(_apply_multiword(pd.read_pickle(cache_path)), min_decade)

    print("Streaming Goodreads (2.36M lines)...")
    df = load_goodreads_full()
    print(f"  {len(df)} works after de-duplicating editions")
    df = clean_corpus(df)

    print("Lemmatising...")
    df["Lemmas"] = [" ".join(toks) for toks in T.preprocess_texts(df["Summary"].tolist())]
    df["Decade"] = df["Year"] // 10 * 10
    df = df[df["Lemmas"].str.len() > 0].reset_index(drop=True)
    df.to_pickle(cache_path)
    print(f"Saved corpus to {cache_path}")
    return _apply_min_decade(df, min_decade)


CMU_CACHE_PATH = "cmu_corpus.pkl"


def build_cmu_corpus(cache_path=CMU_CACHE_PATH, force_reload=False,
                     min_decade=MIN_DECADE):
    """
    אותו מסלול בדיוק על תקצירי CMU, כקבוצת ביקורת.

    תקצירי CMU הם תיאורי עלילה מוויקיפדיה, שנכתבו על ידי קוראים ולא על ידי
    מוציאים לאור, ולכן אין בהם רגיסטר של טקסט שיווקי. השאלה שהריצה הזו
    עונה עליה צרה ומוגדרת: **האם הצינור מייצר נושאי מטא-דאטה בעצמו, או
    שהוא מוצא אותם משום שהם קיימים בגודריידס?** אם גם כאן ייווצרו נושאים
    של מהדורות והוצאות לאור, הכשל הוא בשיטה. אם לא ייווצרו, הזיהום שנמדד
    בגודריידס הוא תכונה אמיתית של המאגר.

    כדי שההשוואה תהיה תקפה *שום* פרמטר אינו משתנה: אותו ניקוי, אותה
    למטיזציה, אותו MIN_DECADE, אותו N_TOPICS, אותו זרע ואותו גלאי רעש.
    ההבדל היחיד הוא מקור הטקסט.
    """
    if not force_reload and os.path.exists(cache_path):
        print(f"Loading cached CMU corpus from {cache_path}")
        return _apply_min_decade(_apply_multiword(pd.read_pickle(cache_path)), min_decade)

    print("Loading CMU book summaries...")
    df = T.load_cmu()
    print(f"  {len(df)} summaries with a usable year")
    df = clean_corpus(df)

    print("Lemmatising...")
    df["Lemmas"] = [" ".join(toks) for toks in T.preprocess_texts(df["Summary"].tolist())]
    df["Decade"] = df["Year"] // 10 * 10
    df = df[df["Lemmas"].str.len() > 0].reset_index(drop=True)
    df.to_pickle(cache_path)
    print(f"Saved CMU corpus to {cache_path}")
    return _apply_min_decade(df, min_decade)


# --- רשימת המילים החסומות ---
# מילים שאינן מתארות את תוכן הספר אלא את הספר כמוצר: טקסט שיווקי של המו"ל
# ותיאורי מהדורה. הן מוסרות מאוצר המילים לפני ה-TF-IDF, ולכן אינן יכולות
# להרכיב נושא. זו רשימה סטטית בכוונה - פשוטה לקריאה ולעריכה ביד.

# קבוצה א: נמדדה. ראו keyness.py ו-METADATA_VOCABULARY.md sec. 2.2b.
# 8,680 ספרים הוצמדו בין Goodreads ל-CMU לפי כותרת, כך שלכל ספר יש שני
# תיאורים של אותה עלילה - אחד של המו"ל ואחד של קורא בוויקיפדיה. המילים
# שלהלן שכיחות פי 1.5 ומעלה בצד המו"ל בכ-85% מהעשורים לפחות, ובמקביל
# הופעתן בתקציר אינה מנבאת דבר על הספר שמתחתיה (phi < 0.05, כמו isbn).
# מילים שכן ניבאו את העלילה - war, adventure, mystery, world, america,
# century, history - נשארו בחוץ במתכוון, אף שגם הן יצאו מובהקות
_KEYNESS_REGISTER = frozenset("""
available beloved classic compelling delight depiction edition endure
exciting extraordinary fan feature genre illuminate insight inspire
masterpiece original print profound range reader reading realism remarkable
series style text unforgettable unique vivid weave
""".split())

# קבוצה ב: לא נמדדה, אך אינה יכולה להיות תוכן. אוצר המילים של הספר כחפץ
# פיזי ושל מנגנון ההוצאה לאור. שלוש מהן נמדדו אגב כך ב-keyness וקיבלו
# phi אפסי (isbn -0.003, reprint -0.003, anthology -0.001, paperback 0.010,
# bestseller 0.013), מה שמאשר את הקריאה.
# award, prize ו-winner נוספו בסבב האחרון: הן יצרו נושא שלם -
# "author, win, award, biography, experience, bestselle, prize, account" -
# שחלקו שטוח לאורך כל שנים-עשר העשורים (3.6% -> 2.4%), כלומר אינו מגמה
# אלא רעש שאוכל 3% מכל עשור. ה-keyness מאשר: award ‎+4.16 (10.0 מול 0.6
# ל-10k ב-CMU), winner ‎+3.39, prize ‎+2.52. "win" נשארה - ניצחון בקרב
# או במרוץ הוא תוכן
_BIBLIOGRAPHIC_STOPWORDS = frozenset("""
isbn ebook paperback hardcover hardback audiobook facsimile reprint reissue
printing imprint typo typography typeset pagination page pages
foreword afterword preface appendix bibliography glossary footnote endnote
errata imperfection scanned scan ocr blur blurred
bestseller bestselling bestselle award awards prize winner fascinating
""".split())

# קבוצה ג: אוצר המילים של מנגנון הספר, שנוסף ביד למרות ש-phi הציב אותו
# בתחום הביניים (0.05-0.11). הסיבה לכך שהמדידה מחמיצה דווקא אותו: תקצירי
# CMU אינם עלילה טהורה, ויש בהם מסגור ביבליוגרפי משלהם - "the novel was
# published in 1961", "in the final chapter". ואכן chapter הוא מהמילים
# החזקות בצד ה-CMU (9.1 מול 2.3 ל-10k), ו-novel עומד שם על 26.4 ל-10k.
# כשגם קורפוס הייחוס מזוהם באותו כיוון, phi של מילת מו"ל נמשך למעלה,
# ולכן כאן ההכרעה ידנית. page כבר נמצא בקבוצה ב
_APPARATUS_STOPWORDS = frozenset("""
publish publication volume illustration translation novella prose
literature introduction
""".split())

# קבוצה ד: אוצר מילים חסר תוכן. לא מטא-דאטה ולא רגיסטר של מו"ל - פשוט
# מילים שאינן מבחינות בין נושא לנושא.
#
# הן אותרו במדידה ולא בניחוש: בהרצה הקודמת ארבעה מתוך 25 הנושאים היו
# בנויים כולם מהן -
#   thing, good, go, come, day, want, get, time
#   find, way, help, discover, search, place, turn, home
#   know, want, secret, people, feel, answer, need, fact
#   world, people, great, ii, live, create, change, human
# כלומר 16% מקיבולת המודל הלכה על מילים ריקות, ולכן למדע בדיוני לא נשאר
# נושא משלו. max_df=0.5 לא סינן אף מילה (אפס!), משום שאף מילה אינה מופיעה
# ביותר מחצי מהמסמכים, ולכן הסינון חייב להיות מפורש.
#
# מה שנשאר בכוונה: world, people, human, place, home, secret, life, age,
# struggle, hope, country, death - אלה נושאיים גם אם הם נפוצים
_EMPTY_VOCABULARY = frozenset("""
go come get take give find know want feel need tell begin bring turn help
make look follow continue draw choose remain lead move call act reveal fill
serve deal prove appear hold grow form bear capture encounter set discover
search seem become use provide mark note cause result point order thing way
sense group name interest good great different large strong special free
kind late second
""".split()) | {"new"}

# "new" מוסרת בנפרד ומסיבה משלה. היא הופיעה ב-9,223 מסמכים (22%) והחזיקה
# יחד נושא אחד שאין בו היגיון: new, york, city, update, generation,
# testament, orleans. מדידה על הקורפוס מסבירה למה - המילה מגשרת בין
# ארבעה דברים שאינם קשורים: New York (1,335), New Orleans (120),
# New Testament (143), New England (159), New World (237), ו-"a new X"
# הסתמי (2,589). זהו הומונים, ולא נושא. בלעדיה york ו-orleans אמורים
# ליפול לנושא מקומות, ו-testament לנושא הדת

# מה שנכנס בפועל ל-TfidfVectorizer
STOPWORDS = frozenset(
    _KEYNESS_REGISTER | _BIBLIOGRAPHIC_STOPWORDS | _APPARATUS_STOPWORDS
    | _EMPTY_VOCABULARY)


def fit_topics(df, n_topics=N_TOPICS, fit_per_decade=FIT_PER_DECADE, random_state=42):
    """
    בונה מטריצת TF-IDF ומפרק אותה ל-W (מסמך x נושא) ו-H (נושא x מילה).

    ה-NMF מאומן על מדגם מאוזן בין העשורים כדי שמרחב הנושאים לא ייקבע על ידי
    העשורים הגדולים, ואז כל המסמכים מוטלים עליו (transform) כדי שהנתחים לכל
    עשור יחושבו על כל הנתונים הזמינים.
    """
    fit_idx = []
    for _, group in df.groupby("Decade"):
        fit_idx.extend(group.sample(min(len(group), fit_per_decade),
                                    random_state=random_state).index)
    fit_rows = df.loc[fit_idx]
    print(f"Fitting on a decade-balanced sample of {len(fit_rows)} documents "
          f"(<= {fit_per_decade} per decade)")

    vectorizer = TfidfVectorizer(min_df=5, max_df=0.5, sublinear_tf=True,
                                 stop_words=list(STOPWORDS))
    X_fit = vectorizer.fit_transform(fit_rows["Lemmas"])
    print(f"  TF-IDF matrix: {X_fit.shape[0]} docs x {X_fit.shape[1]} words "
          f"({len(STOPWORDS)} words blacklisted)")

    # ללא רגולריזציה: sklearn מכפיל את alpha_H במספר הדגימות, כך שערך קטן
    # לכאורה הופך לקנס L1 גדול ומאפס את H לחלוטין
    nmf = NMF(n_components=n_topics, init="nndsvda", random_state=random_state,
              max_iter=800, tol=1e-5)
    nmf.fit(X_fit)

    X_all = vectorizer.transform(df["Lemmas"])
    W = nmf.transform(X_all)
    # נרמול לשורה כדי שכל מסמך יתפלג כאחוזים בין הנושאים
    row_sums = W.sum(axis=1, keepdims=True)
    W = np.divide(W, row_sums, out=np.zeros_like(W), where=row_sums > 0)
    return vectorizer, nmf, W


def topic_labels(vectorizer, nmf, top_n=12):
    """שם לכל נושא = המילים בעלות המשקל הגבוה ביותר בשורה שלו ב-H."""
    words = np.array(vectorizer.get_feature_names_out())
    return [", ".join(words[np.argsort(-row)[:top_n]]) for row in nmf.components_]


def decade_profiles(df, W):
    """
    נתח ממוצע של כל נושא בכל עשור, יחד עם רווח סמך של 95%.
    הנתח הוא ממוצע של פרופורציות לכל מסמך, ולכן שגיאת התקן היא std/sqrt(n)
    ורווח הסמך מצהיר בעצמו כמה דל העשור - במקום להסתיר זאת בדגימה אחידה.
    """
    means, cis, counts = {}, {}, {}
    for decade, idx in df.groupby("Decade").indices.items():
        block = W[idx]
        n = len(idx)
        means[decade] = block.mean(axis=0)
        cis[decade] = 1.96 * block.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(W.shape[1], np.nan)
        counts[decade] = n
    return means, cis, counts


# מסמך שנשארה בו פחות ממסה כזו של תוכן אחרי הוצאת נושאי הרעש אינו ניתן
# לנרמול: החלוקה במספר זעיר מנפחת רעש נומרי לווקטור יחידה שלם. הוא מוסר,
# וכמותם מדווחת לפי עשור כמדד נוסף לזיהום
_MIN_CONTENT_MASS = 0.01


def exclude_artifacts(df, W, flagged, min_mass=_MIN_CONTENT_MASS):
    """
    מוציא את עמודות נושאי-הרעש מ-W ומנרמל כל שורה מחדש לסכום 1.

    בלי הנרמול הנתחים היו מסתכמים בפחות מ-1 בשיעור *משתנה* בין העשורים
    (רעש ביבליוגרפי מרוכז לפני 1900, נושא הפעלים הגנריים גדל לקראת ההווה),
    וכל קו מגמה היה מעוות אחרת בכל קצה של ציר הזמן. אחרי הנרמול המשמעות היא
    "מתוך הטקסט שהוא באמת תיאור של ספר, כך וכך אחוז הוא בלשי" - וזה בר-השוואה.

    מחזיר (df מסונן, Wc, keep, alive). keep ממפה עמודה חדשה -> אינדקס הנושא
    המקורי, כדי שמספרי הנושאים בפלט יישארו יציבים.
    """
    keep = [j for j in range(W.shape[1]) if j not in flagged]
    Wc = W[:, keep]
    row_sums = Wc.sum(axis=1, keepdims=True)
    alive = row_sums[:, 0] > min_mass
    Wc = Wc[alive] / row_sums[alive]
    out = df[alive].reset_index(drop=True)
    assert len(out) == Wc.shape[0], "df ו-W יצאו מסנכרון"
    assert np.allclose(Wc.sum(axis=1), 1.0), "שורות שאינן מסתכמות ב-1"
    return out, Wc, keep, alive


def artifact_report(df, labels, W, flagged, alive,
                    clean_stats_path=CLEAN_STATS_PATH,
                    out_path="artifact_share_by_decade.csv"):
    """
    מכמת כמה מכל עשור אינו תוכן ספר. זהו ממצא על המאגר, ולא אבחון פנימי:
    זו המידה הישירה ביותר לכך שה"תקצירים" של המאה ה-19 הם ברובם טקסט שיווקי,
    והסתייגות שצריכה להתלוות לכל שורה מוקדמת בטבלאות המגמה.

    שתי מדידות בלתי תלויות: ברמת הטקסט (מה נמחק בניקוי) וברמת הנושאים
    (הנתח המצטבר של נושאי הרעש, *לפני* הנרמול מחדש).
    """
    print("\n" + "=" * 78)
    print("Artifact topics — excluded from every number that follows")
    print("=" * 78)

    means, _, counts = decade_profiles(df, W)
    overall = W.mean(axis=0)
    for i in sorted(flagged):
        reason, hits = flagged[i]
        print(f"  T{i:02d}  {labels[i]}")
        print(f"        {reason:<14} {overall[i]:6.2%} of all text   "
              f"matched: {', '.join(hits)}")
    if not flagged:
        print("  none flagged")
    dropped = int((~alive).sum())
    print(f"\n  {dropped} documents dropped: under {_MIN_CONTENT_MASS:.0%} content "
          f"mass once artifact topics were removed")

    # פירוק לפי סוג, ולא רק סכום: המדידה הראתה שהסך הכל מטעה. הרעש
    # הביבליוגרפי יורד לאורך הזמן (16.1% בשנות ה-1790 -> 11.4% ב-1900 ->
    # 5.9% ב-2010) בעוד נושא הפעלים הגנריים עולה (2.6% -> 5.2% -> 14.4%),
    # והשניים כמעט מבטלים זה את זה בסכום. זו בדיוק הסיבה שההוצאה נחוצה:
    # כל קצה של ציר הזמן מעוות אחרת.
    by_reason = {}
    for reason in sorted({r for r, _ in flagged.values()}):
        cols = [i for i in flagged if flagged[i][0] == reason]
        key = "pct_" + reason.replace(" ", "_")
        by_reason[key] = pd.Series({d: float(sum(means[d][i] for i in cols)) * 100
                                    for d in means})
    art = {d: float(sum(means[d][i] for i in flagged)) * 100 for d in means}
    table = pd.DataFrame({
        "n_books": pd.Series(counts),
        **by_reason,
        "pct_artifact_topic_share": pd.Series(art),
    })
    table.index.name = "decade"
    reason_cols = list(by_reason)

    if os.path.exists(_out(clean_stats_path)):
        clean = pd.read_csv(_out(clean_stats_path), index_col="decade")
        table = table.join(clean[["pct_chars_stripped", "pct_docs_dropped"]])
    else:
        print(f"  ({clean_stats_path} missing — text-level columns unavailable; "
              f"rebuild the corpus to regenerate it)")
        table["pct_chars_stripped"] = np.nan
        table["pct_docs_dropped"] = np.nan

    table = table[["n_books", "pct_chars_stripped", "pct_docs_dropped"]
                  + reason_cols + ["pct_artifact_topic_share"]].sort_index()
    table.to_csv(_out(out_path))

    shown = [d for d in table.index
             if d >= TREND_FROM_DECADE and table.loc[d, "n_books"] >= TREND_MIN_BOOKS]
    head = (f"  {'decade':>7} {'books':>7} {'stripped':>9} {'dropped':>8}"
            + "".join(f"{c.replace('pct_', '')[:13]:>14}" for c in reason_cols)
            + f"{'all artifact':>14}")
    print("\n" + head)
    for d in shown:
        r = table.loc[d]
        print(f"  {str(d) + 's':>7} {int(r['n_books']):>7} "
              f"{r['pct_chars_stripped']:>8.2f}% {r['pct_docs_dropped']:>7.1f}%"
              + "".join(f"{r[c]:>13.1f}%" for c in reason_cols)
              + f"{r['pct_artifact_topic_share']:>13.1f}%")
    print(f"\n  Wrote {out_path}")


# העשור הראשון שמוצג בטבלת המגמות. מיושר עם MIN_DECADE: אין טעם לסנן
# עשורים מהאימון ואז להציג אותם בטבלה
TREND_FROM_DECADE = MIN_DECADE
# עשור מוצג בטבלה רק אם יש בו לפחות כך ספרים
TREND_MIN_BOOKS = 50
# תווי גרף זעיר להמחשת צורת המגמה
_SPARK = " ▁▂▃▄▅▆▇█"


def _sparkline(values):
    """גרף זעיר בשורה אחת. כל נושא מנורמל לטווח שלו עצמו, כדי שתיראה הצורה."""
    lo, hi = min(values), max(values)
    if hi <= lo:
        return _SPARK[1] * len(values)
    span = len(_SPARK) - 1
    return "".join(_SPARK[max(1, round((v - lo) / (hi - lo) * span))] for v in values)


# --- מדדים נוספים על אותה מטריצה ---
# כל מה שמכאן ואילך מחושב מ-W הקיים ואינו משנה אף מספר שכבר מדווח.
# ההבחנה המרכזית: "נתח" נמדד ברמת ה*מילה*, ו"שכיחות" ברמת ה*ספר*.
# פירוט מלא של ההגדרות בסעיף "What each view measures" ב-notes_for_next_session.md

# העשור שממנו ואילך המאגר מתקרב למפקד ולא למדגם שרידים. לפני 1900 יש
# 66-984 ספרים לעשור מול 1,576-4,808 מ-1900 ואילך, והרעש הביבליוגרפי עומד
# על 16% מול 6% בשנות ה-2010. השורות המוקדמות *נשארות* - הן נושאות את
# האימות ההיסטורי החזק ביותר (פילוסופיה 1790, שירה רומנטית 1810, ורן 1860)
# - אבל הן מסומנות, כדי שלא ייקראו כמדגם מייצג של מה שראה אור אז
# כרגע MIN_DECADE=1900 ולכן אין עשורי קאנון מוצגים והסימון אינו מופיע.
# הקבוע והלוגיקה נשארים: החזרת MIN_DECADE ל-1790 מחזירה אותם מיד
CANON_DECADE = 1900

# ספי ה"ספר עוסק בנושא". אין ערך נכון יחיד, ולכן שלושה מדווחים ואחד אינו
# נבחר בשקט: נמדד שהמגמה זהה בכולם (המתאם הדרגתי מול הנתח 0.81-0.95)
# בעוד הרמה משתנה פי שמונה. הסף הוא ידית, לא ממצא, ולכן הוא גלוי
PREVALENCE_THRESHOLDS = (0.05, 0.10, 0.20)
PREVALENCE_HEADLINE = 0.10

# מספר דגימות ה-bootstrap לרווח הסמך של ה-lift, וזרע קבוע לשחזוריות הדוח
LIFT_BOOTSTRAP = 300
LIFT_SEED = 0


def decade_lift(means, decades):
    """
    עד כמה כל נושא חריג בעשור נתון, ביחס לעצמו: הנתח בעשור חלקי הנתח
    הממוצע של אותו נושא על פני העשורים המוצגים. 1.0 = עשור טיפוסי.

    למה זה נחוץ: הנושאים הגדולים גדולים בכל עשור (משפחה, אהבה, חיים
    מופיעים כמעט בכל תקציר), ולכן "שלושת הנושאים הגדולים של העשור" כמעט
    זהה בשנות ה-30 ובשנות ה-70. החלוקה בבסיס מבטלת את הקבועים האלה
    ומשאירה את מה שמייחד את העשור עצמו.

    הבסיס הוא ממוצע *לא משוקלל* על פני העשורים, ולא ממוצע על פני המסמכים:
    ממוצע לפי מסמכים היה נשלט על ידי העשורים המודרניים (4,808 ספרים בשנות
    ה-2010 מול 124 בשנות ה-1790), וכל עשור מוקדם היה יוצא "חריג" מול בסיס
    שהוא למעשה ההווה.
    """
    matrix = np.array([means[d] for d in decades])
    base = matrix.mean(axis=0)
    return matrix / np.where(base == 0, 1.0, base)


def decade_lift_ci(df, W, decades, n_boot=LIFT_BOOTSTRAP, seed=LIFT_SEED):
    """
    רווח סמך ל-lift ב-bootstrap: דגימה חוזרת של ספרים *בתוך* כל עשור.

    זו אינה זהירות סתמית. ניסיון קודם חישב lift על *מילים* בודדות ב-100
    ספרים לעשור והחזיר dorothy, tarzan, templar, valerian - שמות דמויות
    מספרים בודדים, כי למילה נדירה יש מכנה זעיר. ברמת הנושא אין את הכשל
    הזה (נושא מצרף אלפי מילים על פני אלפי ספרים), אבל יש כשל אחר: בעשורים
    הדקים שלפני 1900 רוחב רווח הסמך גדול פי שלושה, וכמה עשורים הם תיקו
    אמיתי בין שלושה נושאים. לכן מדווחת *קבוצה* ולא דירוג, ונושא שרווח
    הסמך שלו חוצה את 1.0 אינו מוצג כמייחד.

    מחזיר (lo, hi) בצורת (עשורים x נושאים).
    """
    rng = np.random.default_rng(seed)
    dec = df["Decade"].values
    blocks = [W[dec == d] for d in decades]
    boot = np.empty((n_boot, len(decades), W.shape[1]))
    for b in range(n_boot):
        m = np.array([blk[rng.integers(0, len(blk), len(blk))].mean(axis=0)
                      for blk in blocks])
        base = m.mean(axis=0)
        boot[b] = m / np.where(base == 0, 1.0, base)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return lo, hi


def decade_prevalence(df, W, decades, thresholds=PREVALENCE_THRESHOLDS):
    """
    כמה מ*ספרי* העשור עוסקים בנושא - להבדיל מכמה מ*מילות* העשור שייכות לו.

    W הוא שורה לספר ועמודה לנושא, וכל שורה מסתכמת ב-1: רומן משנת 1943
    עשוי להיראות כך - משפחה 0.28, מלחמה 0.19, הרפתקה 0.14. הנתח מְמַצֵּעַ
    את *עמודת* המלחמה; השכיחות *סופרת ספרים* שהתא שלהם חוצה סף.

    dominant נטול-סף (הנושא הגדול ביותר של הספר) ומסתכם ב-100% בכל עשור.
    prevalent תלוי בסף, ולכן מוחזר לכל הספים ולא רק לאחד.

    מחזיר (dominant, {סף: prevalent}), שניהם בצורת (עשורים x נושאים).
    """
    dec = df["Decade"].values
    n_topics = W.shape[1]
    dominant = np.empty((len(decades), n_topics))
    prevalent = {t: np.empty((len(decades), n_topics)) for t in thresholds}
    for i, d in enumerate(decades):
        block = W[dec == d]
        top = block.argmax(axis=1)
        dominant[i] = np.bincount(top, minlength=n_topics) / len(block)
        for t in thresholds:
            prevalent[t][i] = (block >= t).mean(axis=0)
    return dominant, prevalent


def decade_movers(means, decades):
    """
    מה השתנה: העלייה והירידה הגדולות ביותר בנקודות אחוז מול העשור *המוצג*
    הקודם. אם שני עשורים מוצגים אינם רצופים, המרווח מוחזר ונדפס במפורש
    כדי שלא תיווצר אשליה של השוואה בין עשורים סמוכים.

    מחזיר {עשור: (j_עלייה, pp, j_ירידה, pp, מרווח_בשנים)}.
    """
    out = {}
    for i in range(1, len(decades)):
        d, prev = decades[i], decades[i - 1]
        delta = (np.asarray(means[d]) - np.asarray(means[prev])) * 100
        up, down = int(delta.argmax()), int(delta.argmin())
        out[d] = (up, float(delta[up]), down, float(delta[down]), int(d - prev))
    return out


# כמה נושאים מייחדים מוצגים לכל עשור. חצי מהתאים (עשור x נושא) חורגים
# מובהקית מ-1.0, ולכן רשימה מלאה הייתה קיר של טקסט. השאר נספרים ומדווחים
_DIGEST_TOP = 4


def _digest_section(df, labels, W, keep, decades, counts, means, name):
    """
    שני הסעיפים החדשים, שניהם מחושבים מ-W הקיים ואינם משנים אף מספר קיים:
    "במה העשור חריג" (lift עם רווח סמך) ו"כמה מספרי העשור עוסקים בנושא".
    """
    lift = decade_lift(means, decades)
    lo, hi = decade_lift_ci(df, W, decades)
    movers = decade_movers(means, decades)
    dominant, prevalent = decade_prevalence(df, W, decades)
    n_cols = len(keep)
    idx = {d: i for i, d in enumerate(decades)}
    # מילה אחת לנושא: השורות כאן צפופות, ושלוש מילים לנושא הופכות אותן ללא קריאות
    word = {j: labels[keep[j]].split(",")[0].strip() for j in range(n_cols)}

    print("\n" + "=" * 78)
    print("DECADE DIGEST — what each decade was unusual for, and what changed")
    print("=" * 78)
    print("'distinctive' = the topic's share of this decade divided by that topic's own")
    print("average across the decades shown; 1.0x is a typical decade. Only topics whose")
    print("95% bootstrap CI excludes 1.0 appear. Ordered by lift but NOT ranked — where the")
    print("CIs overlap, which one is 'first' is a coin flip.")
    # שתי השורות האלה נכונות רק כשמוצגים עשורים שלפני CANON_DECADE. עם
    # MIN_DECADE=1900 אין כאלה, והדפסתן הייתה מייצרת אזהרה על מה שאינו בטבלה
    if any(d < CANON_DECADE for d in decades):
        print(f"'~' marks the canon sample before {CANON_DECADE}, where CIs are ~3x wider.")
    print()

    rows = []
    for d in decades:
        i = idx[d]
        sig = [j for j in range(n_cols) if lo[i, j] > 1.0]
        sig.sort(key=lambda j: -lift[i, j])
        mark = "~" if d < CANON_DECADE else " "
        shown = sig[:_DIGEST_TOP]
        body = " | ".join(f"{word[j]} {lift[i, j]:.2f}x[{lo[i, j]:.2f}-{hi[i, j]:.2f}]"
                          for j in shown) or "(nothing distinguishable from a typical decade)"
        extra = f"  +{len(sig) - len(shown)} more" if len(sig) > len(shown) else ""
        print(f"{d}s{mark} n={counts[d]:<5d} {body}{extra}")
        for rank, j in enumerate(shown, 1):
            rows.append(dict(decade=d, n=counts[d], canon=d < CANON_DECADE,
                             kind="distinctive", rank=rank, topic=f"T{keep[j]:02d}",
                             label=labels[keep[j]], lift=lift[i, j],
                             lift_lo=lo[i, j], lift_hi=hi[i, j], delta_pp=""))
        if d in movers:
            up, up_pp, dn, dn_pp, span = movers[d]
            span_note = "" if span == 10 else f" (over {span} years — gap in the series)"
            print(f"{'':>15}changed: {up_pp:+.1f}pp {word[up]}   "
                  f"{dn_pp:+.1f}pp {word[dn]}{span_note}")
            for kind, j, pp in (("mover_up", up, up_pp), ("mover_down", dn, dn_pp)):
                rows.append(dict(decade=d, n=counts[d], canon=d < CANON_DECADE,
                                 kind=kind, rank=1, topic=f"T{keep[j]:02d}",
                                 label=labels[keep[j]], lift=lift[i, j],
                                 lift_lo=lo[i, j], lift_hi=hi[i, j], delta_pp=pp))
    pd.DataFrame(rows).to_csv(_out("decade_digest.csv"), index=False)

    # --- שכיחות ברמת הספר ---
    hl = PREVALENCE_HEADLINE
    others = [t for t in PREVALENCE_THRESHOLDS if t != hl]
    print("\n" + "=" * 78)
    print("TOPIC PREVALENCE — % of a decade's BOOKS about a topic, at its peak decade")
    print("=" * 78)
    print("The tables above measure WORDS: what fraction of a decade's text is this topic.")
    print("This one measures BOOKS. W has one row per book summing to 1 (a 1943 novel might")
    print("read family .28, war .19, adventure .14); a book counts as 'about' a topic when")
    print(f"its cell clears a bar. The bar is a knob, not a finding — at {hl:.0%} war is 12% of")
    print("1940s books, at 5% it is 20%, at 20% it is 4%. Every threshold peaks in the same")
    print("decade, so this changes the LEVEL and never the TREND.\n")
    head = (f"{'topic':<26}{'peak':>7}{'% words':>9}{'% books':>9}"
            + "".join(f"{'@' + format(t, '.0%'):>9}" for t in others)
            + f"{'dominant':>10}")
    print(head)
    print("-" * len(head))
    order = sorted(range(n_cols), key=lambda j: -prevalent[hl][:, j].max())
    for j in order:
        i = int(prevalent[hl][:, j].argmax())
        print(f"{name[j][:25]:<26}{str(decades[i]) + 's':>7}"
              f"{means[decades[i]][j] * 100:>8.1f}%{prevalent[hl][i, j] * 100:>8.1f}%"
              + "".join(f"{prevalent[t][i, j] * 100:>8.1f}%" for t in others)
              + f"{dominant[i, j] * 100:>9.1f}%")

    prow = []
    for i, d in enumerate(decades):
        for j in range(n_cols):
            r = dict(decade=d, n=counts[d], canon=d < CANON_DECADE,
                     topic=f"T{keep[j]:02d}", label=labels[keep[j]],
                     mean_share=means[d][j], lift=lift[i, j],
                     lift_lo=lo[i, j], lift_hi=hi[i, j], dominant=dominant[i, j])
            for t in PREVALENCE_THRESHOLDS:
                r[f"p{int(t * 100):02d}"] = prevalent[t][i, j]
            prow.append(r)
    pd.DataFrame(prow).to_csv(_out("topic_prevalence_by_decade.csv"), index=False)

    lift_tab = pd.DataFrame(lift, index=decades,
                            columns=[f"T{keep[j]:02d}" for j in range(n_cols)])
    lift_tab.index.name = "Decade"
    for suffix, arr in (("_lo", lo), ("_hi", hi)):
        for j in range(n_cols):
            lift_tab[f"T{keep[j]:02d}{suffix}"] = arr[:, j]
    lift_tab.to_csv(_out("topic_lift_by_decade.csv"))

    print("\nWrote decade_digest.csv, topic_prevalence_by_decade.csv "
          "and topic_lift_by_decade.csv")

    # בדיקות שקילות פנימיות: אלה תנאים שחייבים להתקיים בהגדרה, ונפילה
    # שלהם פירושה שהחישוב עצמו שגוי ולא שהנתונים מפתיעים
    assert np.allclose(lift.mean(axis=0), 1.0), "lift אינו ממוצע ל-1 לכל נושא"
    assert np.allclose(dominant.sum(axis=1), 1.0), "dominant אינו מסתכם ב-100% לעשור"
    return lift, lo, hi, movers, dominant, prevalent


def report(df, labels, W, keep, top_k=3):
    """
    כל המספרים כאן הם על נושאי תוכן בלבד, אחרי נרמול מחדש. keep ממפה עמודה
    ב-W לאינדקס הנושא המקורי, כדי שהשמות בפלט (T05 וכו') יישארו זהים לרשימת
    הנושאים המלאה ולא ימוספרו מחדש אחרי ההוצאה.
    """
    means, cis, counts = decade_profiles(df, W)
    n_cols = len(keep)

    print("\n" + "=" * 78)
    print(f"{n_cols} content topics over {len(df)} book summaries")
    print("=" * 78)
    for j, i in enumerate(keep):
        print(f"  T{i:02d}  {labels[i]}")

    # העשורים שמספיק מיוצגים כדי שאפשר יהיה לדבר על מגמה
    decades = [d for d in sorted(means)
               if d >= TREND_FROM_DECADE and counts[d] >= TREND_MIN_BOOKS]
    if not decades:
        print("\nNo decade has enough books for a trend table.")
        return

    series = {j: [means[d][j] * 100 for d in decades] for j in range(n_cols)}
    # מיון לפי גודל התנועה: הנושאים שזזו הכי הרבה מופיעים ראשונים
    order = sorted(range(n_cols), key=lambda j: max(series[j]) - min(series[j]), reverse=True)

    short = {j: ", ".join(labels[i].split(", ")[:3]) for j, i in enumerate(keep)}
    name = {j: f"T{keep[j]:02d} {short[j]}" for j in range(n_cols)}
    width = max(len(x) for x in name.values()) + 2

    canon = [d for d in decades if d < CANON_DECADE]
    if canon:
        print("\n" + "-" * 78)
        print(f"NOTE — the {len(canon)} decades before {CANON_DECADE} are a CANON sample, "
              "not a census.")
        print("Goodreads catalogues books that reached a modern edition, so its early decades")
        print("hold what was reprinted or digitised, not what was published. They are kept —")
        print("they carry the strongest historical validation — but marked '~' throughout.")
        print("  " + "  ".join(f"{d}s n={counts[d]}" for d in canon))
        print("-" * 78)

    print("\n" + "=" * 78)
    print(f"Topic trends, {decades[0]}s-{decades[-1]}s — biggest movers first")
    print("RELATIVE shares: each decade's topics sum to 100%, so a topic can fall\n"
          "purely because others rose. Read as composition, not volume.")
    print("=" * 78)
    print(f"{'topic':<{width}} {'shape':<{len(decades)}}   peak      range")
    for j in order:
        vals = series[j]
        peak = decades[vals.index(max(vals))]
        print(f"{name[j]:<{width}} {_sparkline(vals)}   {max(vals):4.1f}% {peak}s  "
              f"{min(vals):.1f}-{max(vals):.1f}")

    print("\n" + "=" * 78)
    print("Share of each decade's content text (%) — columns sum to 100 within a decade")
    print("=" * 78)
    print(" " * width + "".join(f"{str(d)[2:]:>5}" for d in decades))
    for j in order:
        print(f"{name[j]:<{width}}" + "".join(f"{v:5.1f}" for v in series[j]))
    print("\n" + " " * width + "".join(f"{counts[d]:>5}" for d in decades) + "   <- books per decade")
    print(" " * width + "(column headings are decade last-two-digits: 00 = 1800s ... 10 = 2010s)")

    print("\n" + "=" * 78)
    print("Top topics per decade, as a share of that decade "
          f"(95% CI; ~ = canon sample pre-{CANON_DECADE}; [thin] = under 100 books)")
    print("=" * 78)
    for decade in sorted(means):
        share, ci, n = means[decade], cis[decade], counts[decade]
        top = np.argsort(-share)[:top_k]
        parts = " | ".join(
            f"T{keep[j]:02d} {share[j]:.1%}+-{ci[j]:.1%}" if not np.isnan(ci[j])
            else f"T{keep[j]:02d} {share[j]:.1%}"
            for j in top
        )
        flag = ("~" if decade < CANON_DECADE else " ") + (" [thin]" if n < 100 else "")
        print(f"{decade}s n={n:<6d}{flag:<9} {parts}")
        print(f"{'':>21} -> {short[top[0]]}")

    _digest_section(df, labels, W, keep, decades, counts, means, name)

    cols = [f"T{i:02d}" for i in keep]
    table = pd.DataFrame({d: means[d] for d in sorted(means)}, index=cols).T
    table.index.name = "Decade"
    table.to_csv(_out("topic_shares_by_decade.csv"))
    pd.Series(labels, index=[f"T{i:02d}" for i in range(len(labels))]).to_csv(_out("topic_labels.csv"))
    print("\nWrote topic_shares_by_decade.csv (content topics, renormalised) "
          "and topic_labels.csv (all topics)")



# --- ייצוא ל-PDF ---

PDF_PATH = "topic_trends.pdf"

# מילים המסמנות נושא ש"עוסק בספר" ולא בתוכנו: טקסט של הוצאות לאור ותיאורי
# מהדורה, שהם שריד לכך ששדה description הוא חומר שיווקי. ראו notes_for_next_session.md
# הרשימה נכתבה מחדש אחרי הוספת STOPWORDS: 13 מתוך 16 הסמנים המקוריים
# (edition, page, reader, text, print, classic, original...) נמחקו מאוצר
# המילים עצמו, ולכן הם אינם יכולים להופיע באף נושא והגלאי נותר עיוור.
# הסמנים כאן הם מילים ששרדו את הרשימה השחורה. שימו לב שכמה מהן יכולות
# להיות תוכן בפני עצמן (write, author) - הגלאי דורש שלוש פגיעות, וההנחה
# היא שנושא שרוב מילותיו המובילות הן אלה עוסק בספרים ולא בנושא כלשהו
_BIBLIOGRAPHIC_MARKERS = frozenset(
    "book books write read author writer novelist title review revise update "
    "copy publisher press chapter".split()
)
# מילים גנריות שיוצרות "נושא שארית" של פעלים חסרי תוכן
_GENERIC_MARKERS = frozenset("know want thing good come go day get take look".split())
# כמה מילים מהמובילות בנושא צריכות להיות מסומנות כדי לפסול אותו
_ARTIFACT_MIN_HITS = 3


def artifact_topics(labels, top_n=8):
    """
    מזהה נושאי-רעש לפי המילים המובילות שלהם ולא לפי מספרם.
    מספור הנושאים ב-NMF אינו יציב בין ריצות, ולכן אסור לקבע אינדקסים.

    מחזיר {אינדקס: (סיבה, המילים שהופעלו)}. המילים המפעילות מוחזרות ונדפסות
    כדי שההחלטה תהיה ניתנת לביקורת - הרשימות למעלה נכתבו ביד, והן החוליה
    החלשה בשלב הזה.
    """
    flagged = {}
    for i, label in enumerate(labels):
        words = [w.strip() for w in label.split(",")[:top_n]]
        bib = [w for w in words if w in _BIBLIOGRAPHIC_MARKERS]
        gen = [w for w in words if w in _GENERIC_MARKERS]
        if len(bib) >= _ARTIFACT_MIN_HITS:
            flagged[i] = ("bibliographic", bib)
        elif len(gen) >= _ARTIFACT_MIN_HITS:
            flagged[i] = ("generic verbs", gen)
    return flagged


def export_pdf(df, labels, W, keep, flagged, df_raw, W_raw, path=PDF_PATH):
    """
    כותב דוח מאויר על נושאי התוכן בלבד: גרפי מגמה, מפת חום וטבלת מספרים.
    נושאי הרעש אינם מופיעים בעמודי המגמה כלל, אלא בעמוד סיכום נפרד - הגודל
    שלהם הוא ממצא על המאגר, ולכן הוא מוצג ולא מוסתר.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    means, cis, counts = decade_profiles(df, W)
    n_cols = len(keep)

    decades = [d for d in sorted(means)
               if d >= TREND_FROM_DECADE and counts[d] >= TREND_MIN_BOOKS]
    series = {j: np.array([means[d][j] * 100 for d in decades]) for j in range(n_cols)}
    errors = {j: np.array([cis[d][j] * 100 for d in decades]) for j in range(n_cols)}
    order = sorted(range(n_cols), key=lambda j: series[j].max() - series[j].min(), reverse=True)
    short = {j: ", ".join(labels[keep[j]].split(", ")[:3]) for j in range(n_cols)}

    with PdfPages(_out(path)) as pdf:
        # עמוד 1: כותרת ורשימת נושאי התוכן
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.06, .94, "Book themes by decade", size=22, weight="bold")
        fig.text(.06, .90,
                 f"{len(df):,} English book summaries, dated by first publication. "
                 f"{n_cols} content topics via NMF on TF-IDF.", size=10, color="#444")
        fig.text(.06, .875,
                 f"{len(flagged)} further topics were publisher/edition copy or generic-verb "
                 "residue; they are excluded here and shown on the last page.",
                 size=9, color="#a33")
        y = .82
        for j in order:
            fig.text(.06, y, f"T{keep[j]:02d}  {labels[keep[j]]}", size=8.5,
                     family="DejaVu Sans")
            y -= .0285
        pdf.savefig(fig); plt.close(fig)

        # עמוד 2: גרף קטן לכל נושא, כולל נושאי הרעש המסומנים בכוכבית.
        # נושאי הרעש מוצגים מתוך W הגולמי - הם אינם קיימים במטריצה המנורמלת -
        # ולכן שתי הסקאלות שונות, וזה נאמר במפורש בתחתית העמוד.
        raw_means, raw_cis, raw_counts = decade_profiles(df_raw, W_raw)
        panels = [("content", j, keep[j], series[j], errors[j]) for j in order]
        for i in sorted(flagged):
            vals = np.array([raw_means[d][i] * 100 for d in decades])
            errs = np.array([raw_cis[d][i] * 100 for d in decades])
            panels.append(("artifact", None, i, vals, errs))

        side = int(np.ceil(np.sqrt(len(panels))))
        fig, axes = plt.subplots(side, side, figsize=(11.7, 8.3), sharex=True,
                                 squeeze=False)
        fig.suptitle("Share of each decade's text (%) — biggest movers first",
                     size=13, weight="bold")
        fig.text(.5, .952,
                 "RELATIVE, not absolute: within a decade all topics sum to 100%, so a "
                 "line can fall only because other topics rose. Not a count of books.",
                 size=8.5, ha="center", color="#444")
        for ax, (kind, _, tid, vals, errs) in zip(axes.ravel(), panels):
            colour = "#c0392b" if kind == "artifact" else "#2c6fbb"
            # הצללה על העשורים שהם מדגם קאנון ולא מפקד
            if decades[0] < CANON_DECADE:
                ax.axvspan(decades[0], CANON_DECADE, color="#000000", alpha=.055, lw=0)
            ax.plot(decades, vals, lw=1.6, color=colour)
            ax.fill_between(decades, vals - errs, vals + errs, alpha=.2, color=colour, lw=0)
            ax.axvline(decades[int(vals.argmax())], color="#999", lw=.6, ls=":")
            star = " *" if kind == "artifact" else ""
            ax.set_title(f"T{tid:02d}{star} " + ", ".join(labels[tid].split(", ")[:3]),
                         size=6.5, color=colour)
            ax.tick_params(labelsize=6)
            ax.margins(x=.02)
        for ax in axes.ravel()[::side]:
            ax.set_ylabel("% of decade", size=6)
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        fig.text(.5, .022,
                 "* excluded artifact topic — publisher/edition copy or generic-verb residue, "
                 "not a theme. Plotted as a share of ALL text (pre-exclusion); "
                 "unmarked topics are shares of content text only, so the two scales differ.",
                 size=7.5, ha="center", color="#a33")
        if decades[0] < CANON_DECADE:
            fig.text(.5, .006,
                     f"Shaded region (before {CANON_DECADE}) is a CANON sample, not a census: "
                     "Goodreads holds what reached a modern edition, so those decades are "
                     "66-984 books each and 16% publisher copy against 6% in the 2010s.",
                     size=7.5, ha="center", color="#444")
        fig.tight_layout(rect=[0, .035, 1, .95])
        pdf.savefig(fig); plt.close(fig)

        # עמוד 3: מפת חום של כל המטריצה
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        matrix = np.array([series[j] for j in order])
        # נרמול לכל שורה, כדי שנושאים קטנים לא ייעלמו מול הגדולים.
        # נושא שטוח לחלוטין היה מייצר חלוקה באפס ושורה ריקה במפה
        spread = np.ptp(matrix, axis=1, keepdims=True)
        norm = (matrix - matrix.min(axis=1, keepdims=True)) / np.where(spread == 0, 1, spread)
        im = ax.imshow(norm, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(decades)))
        ax.set_xticklabels([f"{d}s" for d in decades], rotation=90, size=7)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"T{keep[j]:02d} {short[j]}" for j in order], size=7)
        ax.set_title("When each topic peaks — each row rescaled to its own min/max.\n"
                     "Colour compares a topic to ITSELF over time, never to other topics.",
                     size=11, weight="bold")
        fig.colorbar(im, ax=ax, shrink=.6, label="low → high (within topic)")
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # עמוד 4: הטבלה המספרית
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.5, .965, "Share of each decade's content text (%)", size=13,
                 weight="bold", ha="center")
        fig.text(.5, .938, "Each COLUMN sums to 100 — these are proportions within a "
                 "decade, not numbers of books.", size=8.5, ha="center", color="#444")
        head = "topic".ljust(30) + "".join(f"{str(d)[2:]:>5}" for d in decades)
        lines = [head, "-" * len(head)]
        for j in order:
            name = f"T{keep[j]:02d} " + ", ".join(labels[keep[j]].split(", ")[:2])
            lines.append(name[:29].ljust(30) + "".join(f"{v:5.1f}" for v in series[j]))
        lines += ["-" * len(head),
                  "books".ljust(30) + "".join(f"{counts[d]:5d}" for d in decades)]
        fig.text(.03, .90, "\n".join(lines), family="monospace", size=6.2, va="top")
        pdf.savefig(fig); plt.close(fig)

        # עמוד 5: הדיג'סט - מה ייחד כל עשור ומה השתנה בו.
        # זהו העמוד שעונה על "מה העשור הזה אומר לי", להבדיל מ"מתי נושא היה נפוץ"
        lift = decade_lift(means, decades)
        lo, hi = decade_lift_ci(df, W, decades)
        movers = decade_movers(means, decades)
        word = {j: labels[keep[j]].split(",")[0].strip() for j in range(n_cols)}
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.5, .968, "What each decade was unusual for", size=14,
                 weight="bold", ha="center")
        fig.text(.5, .943,
                 "Each topic's share of the decade divided by that topic's own average "
                 "across decades. 1.0x = a typical decade.",
                 size=8.5, ha="center", color="#444")
        fig.text(.5, .924,
                 "Only topics whose 95% bootstrap CI excludes 1.0 are listed. Ordered by "
                 "lift but NOT ranked — where CIs overlap the order is a coin flip.",
                 size=8.5, ha="center", color="#444")
        lines = []
        for i, d in enumerate(decades):
            sig = sorted((j for j in range(n_cols) if lo[i, j] > 1.0),
                         key=lambda j: -lift[i, j])[:_DIGEST_TOP]
            mark = "~" if d < CANON_DECADE else " "
            body = "  ".join(f"{word[j]} {lift[i, j]:.2f}x" for j in sig) or "(none)"
            lines.append(f"{d}s{mark} n={counts[d]:<5d} {body}")
            if d in movers:
                up, up_pp, dn, dn_pp, _ = movers[d]
                lines.append(f"{'':>14}changed: {up_pp:+.1f}pp {word[up]}    "
                             f"{dn_pp:+.1f}pp {word[dn]}")
        fig.text(.06, .898, "\n".join(lines), family="monospace", size=8.4, va="top")
        if any(d < CANON_DECADE for d in decades):
            fig.text(.5, .02,
                     f"~ = canon sample before {CANON_DECADE}: fewer books per decade and 3x "
                     "wider confidence intervals, so the SET is reportable but not the winner.",
                     size=7.5, ha="center", color="#444")
        pdf.savefig(fig); plt.close(fig)

        # עמוד 6: הרעש שהוצא, לפני הנרמול מחדש
        if flagged:
            raw_means, _, raw_counts = decade_profiles(df_raw, W_raw)
            rdec = [d for d in sorted(raw_means)
                    if d >= TREND_FROM_DECADE and raw_counts[d] >= TREND_MIN_BOOKS]
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            for i in sorted(flagged):
                ax.plot(rdec, [raw_means[d][i] * 100 for d in rdec], lw=1.6,
                        label=f"T{i:02d} [{flagged[i][0]}] " +
                              ", ".join(labels[i].split(", ")[:4]))
            ax.plot(rdec, [sum(raw_means[d][i] for i in flagged) * 100 for d in rdec],
                    lw=2.4, color="black", ls="--", label="all artifact topics")
            ax.set_title("Excluded artifact topics — share of each decade's text "
                         "BEFORE renormalisation", size=12, weight="bold")
            ax.set_ylabel("% of decade's text")
            ax.legend(fontsize=7, loc="upper center")
            ax.grid(alpha=.25)
            fig.text(.5, .02,
                     "These are not themes: they are publisher and edition copy, and "
                     "generic-verb residue. Their size is itself a finding about the corpus — "
                     "the pre-1900 decades are largely marketing text, not summaries.",
                     size=8, ha="center", color="#a33")
            fig.tight_layout(rect=[0, .05, 1, 1])
            pdf.savefig(fig); plt.close(fig)

        pdf.infodict()["Title"] = "Book themes by decade"

    print(f"Wrote {_out(path)} ({n_cols} content topics; {len(flagged)} excluded as artifacts: "
          f"{', '.join('T%02d' % i for i in sorted(flagged))})")


def main(source="goodreads"):
    """source='goodreads' (ברירת המחדל) או 'cmu' לריצת הביקורת."""
    global OUT_DIR
    if source == "cmu":
        OUT_DIR = "cmu_control"
        df = build_cmu_corpus()
    else:
        # בלי הבדיקה הזו ריצה על מחשב נקי הייתה בונה כאן קורפוס *ללא*
        # bounding ושומרת אותו תחת השם של הקורפוס החתוך, בשקט מוחלט
        if not os.path.exists(CACHE_PATH):
            raise SystemExit(
                f"{CACHE_PATH} is missing. Build it first:\n"
                f"    python build_bounded_corpus.py\n"
                f"(build_corpus alone does not apply bounding, so letting it "
                f"create this file would silently produce a different corpus.)")
        df = build_corpus()
    print(f"\n[{source}] Corpus: {len(df)} documents, {df['Decade'].nunique()} decades, "
          f"{df['Year'].min()}-{df['Year'].max()}")
    vectorizer, nmf, W = fit_topics(df)
    labels = topic_labels(vectorizer, nmf)
    flagged = artifact_topics(labels)

    # סדר קריטי: המדידה של הרעש נעשית על W הגולמי, לפני הנרמול מחדש
    dfc, Wc, keep, alive = exclude_artifacts(df, W, flagged)
    artifact_report(df, labels, W, flagged, alive)
    report(dfc, labels, Wc, keep)
    export_pdf(dfc, labels, Wc, keep, flagged, df, W)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "goodreads")
