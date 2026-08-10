import os
import gzip
import json
import math
import re
import random
from collections import Counter, defaultdict

import pandas as pd
import networkx as nx
import spacy
from itertools import combinations

# טעינת מודל השפה האנגלי (English language model)
nlp = spacy.load("en_core_web_sm")

# --- הגדרות (Configuration) ---
#
# שלושת הקבצים הגולמיים יורדים מהרשת אל data/ על ידי download_data.py.
# הסקריפטים רצים מתוך work/, ולכן data/ הוא ../data - אבל מחפשים בכמה
# מקומות, כדי שהרצה מתיקייה אחרת לא תיפול על נתיב קשיח. זה בדיוק מה
# שקרה קודם: הנתיב הצביע על מטמון של kagglehub שקיים במחשב אחד בלבד
_DATA_DIRS = ["../data", "data", ".",
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")]


def _find_data(name):
    """הנתיב הראשון שקיים; אם אין - מוחזר הנתיב המועדף, כדי שהשגיאה תהיה קריאה."""
    for d in _DATA_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(_DATA_DIRS[0], name)


# מאגר תקצירי העלילה של CMU (booksummaries.txt מתוך booksummaries.tar.gz)
CMU_PATH = _find_data("booksummaries.txt")
# מאגר ה-Goodreads הכללי (כל הז'אנרים)
GOODREADS_PATH = _find_data("goodreads_books.json.gz")
# קובץ ה"יצירות" (works) של Goodreads, המחזיק את שנת הפרסום המקורית
WORKS_PATH = _find_data("goodreads_book_works.json.gz")
# מטמון לדגימת ה-Goodreads, כדי לא לסרוק 2.36M שורות בכל ריצה
GOODREADS_CACHE = "goodreads_sample.pkl"

# מספר הספרים המרבי שנדגום מכל עשור. דגימה לפי עשור (ולא דגימה אחת גלובלית)
# מונעת מצב שבו שני העשורים האחרונים מהווים כמעט את כל המדגם
PER_DECADE_SAMPLE = 250
# עשור עם פחות ספרים מכך מסומן כלא-אמין בפלט, אך עדיין מוצג
MIN_BOOKS_PER_DECADE = 30
# מילה חייבת להופיע במספר תקצירים שונים בעשור, כדי שספר בודד לא ייצור מילת מפתח
MIN_DOC_FREQ = 3
# המאגר נאסף ב-2017, ולכן שנה מאוחרת יותר היא שגיאת נתונים
MAX_YEAR = 2017
# רוחב עמודת המילים בפלט, כדי שההסברים שלצידה יתיישרו
_COL_WIDTH = 56

# חלקי הדיבר שנשמרים בניתוח: שמות עצם, שמות פרטיים, שמות תואר ופעלים.
# פעלים נשמרים משום שהם נושאים משמעות נושאית (kill, escape, inherit);
# פעלים "קלים" (find, take, tell) מטופלים בציון הייחודיות שבהמשך
_ALLOWED_POS = frozenset({"NOUN", "PROPN", "ADJ", "VERB"})


def preprocess_texts(texts):
    """
    שלב 1: ניקוי הטקסט והכנתו (Tokenization, Stop Words, Lemmatization)
    מעבד רשימת טקסטים באצווה (batch) בעזרת nlp.pipe לשם יעילות,
    ומחזיר עבור כל טקסט רשימת מילים בצורת הבסיס שלהן.
    """
    results = []
    # מנטרלים רכיבים שאינם נחוצים (parser, ner) כדי להאיץ את העיבוד.
    # מתייג חלקי הדיבר (tagger) נשאר פעיל, ולכן token.pos_ זמין ללא עלות נוספת
    for doc in nlp.pipe(texts, disable=["parser", "ner"], batch_size=64):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and token.is_alpha and token.pos_ in _ALLOWED_POS
        ]
        results.append(tokens)
    return results


def _pagerank_scores(tokens, window_size=4):
    """
    שלבים 2 ו-3: בניית רשת המילים והרצת TextRank (PageRank).
    מחזיר מילון של מילה -> ציון חשיבות (מילון ריק אם הטקסט קצר מדי).
    """
    graph = nx.Graph()

    # בניית הקשתות (Edges) בגרף על בסיס הופעה משותפת בחלון קריאה
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        for w1, w2 in combinations(window, 2):
            if w1 == w2:
                continue
            if graph.has_edge(w1, w2):
                graph[w1][w2]['weight'] += 1
            else:
                graph.add_edge(w1, w2, weight=1)

    # טיפול במקרה קצה של טקסט קצר מדי
    if len(graph.nodes) == 0:
        return {}

    # הרצת PageRank על הגרף
    return nx.pagerank(graph, weight='weight')


def extract_top_keywords(tokens, window_size=4, top_n=5):
    """
    מחזיר את המילים המובילות לפי TextRank בלבד (ללא שקלול ייחודיות).
    """
    scores = _pagerank_scores(tokens, window_size=window_size)
    ranked_words = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [word for word, score in ranked_words[:top_n]]


def _extract_year(value):
    """שולף שנה בת 4 ספרות מתוך ערך תאריך (מחזיר None אם לא נמצא)."""
    match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", str(value))
    if not match:
        return None
    year = int(match.group(1))
    # שנים עתידיות הן שגיאת נתונים במאגר שנאסף ב-2017
    return year if year <= MAX_YEAR else None


def _stratify_by_decade(df, per_decade=PER_DECADE_SAMPLE, random_state=42):
    """
    דוגם עד per_decade ספרים מכל עשור, כדי שכל העשורים יהיו בני-השוואה.
    משמש למאגר CMU; ב-Goodreads הדגימה נעשית כבר בזמן הזרימה מהקובץ.
    """
    parts = []
    for _, group in df.groupby(df["Year"] // 10 * 10):
        if len(group) > per_decade:
            group = group.sample(n=per_decade, random_state=random_state)
        parts.append(group)
    return pd.concat(parts).reset_index(drop=True)


def load_cmu(path=CMU_PATH):
    """
    טעינת מאגר תקצירי הספרים של CMU (קובץ מופרד-טאבים ללא כותרות).
    עמודה 4 = תאריך פרסום, עמודה 6 = תקציר העלילה.
    מחזיר DataFrame עם העמודות Year ו-Summary.
    """
    cols = ["wiki_id", "freebase_id", "title", "author", "pub_date", "genres", "summary"]
    df = pd.read_csv(path, sep="\t", header=None, names=cols, quoting=3)

    df["Year"] = df["pub_date"].apply(_extract_year)
    df["Summary"] = df["summary"].astype(str)
    df = df.dropna(subset=["Year"])
    df = df[df["Summary"].str.strip().str.len() > 0]
    df["Year"] = df["Year"].astype(int)
    return df[["Year", "Summary"]].reset_index(drop=True)


def _is_english_code(lang_code):
    """
    מחזיר True עבור קודי שפה אנגליים, וגם עבור קוד ריק.
    בכ-45% מהשורות אין language_code כלל, ורובן בכל זאת באנגלית,
    ולכן ערך ריק נחשב "לא ידוע" ומועבר לבדיקת הטקסט שב-_is_english_text.
    """
    lang = str(lang_code).strip().lower()
    return lang == "" or lang in ("eng", "en") or lang.startswith("en-")


# תווים ממערכות כתב שאינן לטיניות: יוונית, קירילית, עברית, ערבית,
# דוונגרית, תאית, סינית/יפנית וקוריאנית.
# אותיות לטיניות מנוקדות וסימני פיסוק טיפוגרפיים לא נכללים בכוונה,
# כדי לא לפסול תקצירים אנגליים רגילים שמכילים גרשיים מסולסלים או מקף ארוך.
_FOREIGN_SCRIPT_RE = re.compile(
    "["
    "Ͱ-Ͽ"  # יוונית
    "Ѐ-ӿ"  # קירילית
    "֐-׿"  # עברית
    "؀-ۿ"  # ערבית
    "ऀ-ॿ"  # דוונגרית
    "฀-๿"  # תאית
    "぀-ヿ"  # קאנה יפנית
    "㐀-鿿"  # סימניות סיניות
    "가-힯"  # האנגול הקוריאני
    "]"
)

# רשימה קצרה של מילות תפקוד אנגליות נפוצות מאוד. טקסט אנגלי אמיתי מלא בהן,
# ואילו טקסט זר – כולל התעתיק הלטיני של ערבית/עברית שהמאגר הזה שומר ב-ASCII
# (למשל "yHwl mw'lf lktb ldktwr") – כמעט ואינו מכיל אותן.
_ENGLISH_STOP_WORDS = frozenset(
    "the a an and or but of to in is was were be been are for with on at by from "
    "that this it as not his her he she they we you i had have has will would "
    "there their which who what when".split()
)
_WORD_RE = re.compile(r"[a-z']+")

# השיעור המזערי של מילות תפקוד הנדרש כדי שתקציר ייחשב אנגלי
_MIN_ENGLISH_RATIO = 0.10
# תקצירים קצרים מכך אינם ארוכים מספיק כדי שהיחס יהיה משמעותי
_MIN_WORDS = 8


def _is_english_text(text):
    """
    מחזיר True אם התקציר עצמו נראה אנגלי.
    השדה language_code חסר או שגוי בחלק ניכר מהמאגר, ולכן זהו הסינון שבפועל
    מסיר את הטקסט הזר. הוא פוסל כתבים שאינם לטיניים, ובנוסף דורש שיעור סביר
    של מילות תפקוד אנגליות – מה שתופס גם טקסט זר הכתוב באותיות לטיניות
    (ספרדית, צרפתית, רומנית, פולנית, ערבית בתעתיק וכו').
    """
    if _FOREIGN_SCRIPT_RE.search(text):
        return False
    words = _WORD_RE.findall(text.lower())
    if len(words) < _MIN_WORDS:
        return False
    stop_hits = sum(word in _ENGLISH_STOP_WORDS for word in words)
    return stop_hits / len(words) >= _MIN_ENGLISH_RATIO


def load_work_years(path=WORKS_PATH):
    """
    מחזיר מיפוי work_id -> שנת הפרסום המקורית של היצירה.

    כל שורה ב-goodreads_books.json.gz היא מהדורה (edition) ולא יצירה (work),
    ולכן publication_year שם הוא שנת ההדפסה הזו – "גאווה ודעה קדומה" מופיעה
    כ-2006 או 2012. הקובץ הזה מחזיק את original_publication_year לכל יצירה,
    וזה מה שמאפשר לתארך ספר לפי פרסומו הראשון.

    כ-1.27M ערכים, בערך 150-250MB בזיכרון.
    """
    work_years = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # מדלגים על פריטים שאינם ספרים (media_type ריק נחשב ספר)
            media_type = str(obj.get("media_type", "")).strip()
            if media_type and media_type != "book":
                continue
            year = _extract_year(obj.get("original_publication_year", ""))
            if year is None:
                continue
            try:
                work_years[int(obj["work_id"])] = year
            except (KeyError, TypeError, ValueError):
                continue
    return work_years


def load_goodreads(path=GOODREADS_PATH, works_path=WORKS_PATH,
                   per_decade=PER_DECADE_SAMPLE, random_state=42):
    """
    טעינת מאגר ה-Goodreads הכללי (JSON דחוס, אובייקט אחד בכל שורה, ~2.36M מהדורות).

    שלושה תיקונים מרכזיים ביחס לגרסה הקודמת:
    1. תיארוך לפי שנת הפרסום המקורית של היצירה (דרך work_id), ולא לפי המהדורה.
    2. ניכוי כפילויות: המהדורה הכשירה הראשונה של כל יצירה זוכה, כדי שקלאסיקה
       עם 60 מהדורות לא תיספר 60 פעמים.
    3. דגימת מאגר (reservoir sampling) נפרדת לכל עשור, כך שכל עשור תורם
       לכל היותר per_decade ספרים ואינו נבלע על ידי העשורים האחרונים.

    מחזיר DataFrame עם העמודות Year ו-Summary.
    """
    work_years = load_work_years(works_path)
    rng = random.Random(random_state)
    reservoirs = defaultdict(list)   # עשור -> רשימת הספרים שנדגמו
    seen_per_decade = Counter()      # עשור -> כמה ספרים כשירים נראו עד כה
    seen_works = set()

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            try:
                work_id = int(obj.get("work_id"))
            except (TypeError, ValueError):
                continue
            # ניכוי כפילויות: יצירה שכבר נבחרה לא נספרת שוב
            if work_id in seen_works:
                continue
            year = work_years.get(work_id)
            if year is None:
                continue
            # מסננים: שפה אנגלית בלבד ותקציר לא ריק
            if not _is_english_code(obj.get("language_code", "")):
                continue
            summary = str(obj.get("description", "")).strip()
            if not summary:
                continue
            # סילוק טקסט זר שחמק מהסינון לפי language_code
            if not _is_english_text(summary):
                continue

            seen_works.add(work_id)
            decade = year // 10 * 10
            seen_per_decade[decade] += 1
            bucket = reservoirs[decade]
            row = {"Year": year, "Summary": summary}
            # Reservoir sampling בתוך העשור: לכל ספר כשיר סיכוי שווה להיכלל
            if len(bucket) < per_decade:
                bucket.append(row)
            else:
                j = rng.randint(0, seen_per_decade[decade] - 1)
                if j < per_decade:
                    bucket[j] = row

    rows = [row for decade in sorted(reservoirs) for row in reservoirs[decade]]
    return pd.DataFrame(rows)


def load_goodreads_cached(cache_path=GOODREADS_CACHE, force_reload=False, **kwargs):
    """
    עוטף את load_goodreads במטמון על הדיסק. הסריקה עוברת על 2.36M שורות
    ונמשכת מספר דקות, ולכן שומרים את המדגם ומשתמשים בו שוב.
    force_reload=True מכריח בנייה מחדש.
    """
    if not force_reload and os.path.exists(cache_path):
        print(f"Loading cached Goodreads sample from {cache_path}")
        return pd.read_pickle(cache_path)
    df = load_goodreads(**kwargs)
    df.to_pickle(cache_path)
    print(f"Saved Goodreads sample to {cache_path}")
    return df


def analyze_dataframe(df, dataset_name, window_size=4, top_n=5):
    """
    שלב 4: ניתוח ציר הזמן (Temporal Analysis) עבור מאגר נתונים בודד.

    לכל עשור מודפסות שתי רשימות:
    - Top TextRank: המילים המרכזיות בגרף ההופעה המשותפת של אותו עשור.
    - Distinctive: אותן מילים משוקללות במידת הייחודיות שלהן לעשור, לפי
      score = pagerank * log(1 + lift), כאשר lift הוא שכיחות המילה בעשור
      חלקי שכיחותה בכל המאגר. מילה שמופיעה בקצב הממוצע מקבלת log(2) בלבד,
      ולכן מילים נפוצות-בכל-עשור (book, story, life) אמורות לרדת.
    """
    print("=" * 78)
    print(f"Dataset: {dataset_name}  ({len(df)} books after sampling)")
    print("=" * 78)

    # חלוקה לעשורים
    df = df.copy()
    df["Decade"] = (df["Year"] // 10) * 10

    # שלב 1: עיבוד מקדים של כל התקצירים באצווה אחת (יעיל יותר)
    df["Tokens"] = preprocess_texts(df["Summary"].tolist())

    # מעבר נוסף לבניית שכיחות המילים בכל המאגר, לצורך חישוב ה-lift
    corpus_counts = Counter()
    for tokens in df["Tokens"]:
        corpus_counts.update(tokens)
    corpus_total = sum(corpus_counts.values())
    if corpus_total == 0:
        print("No tokens after preprocessing.\n")
        return

    print("Top keywords per decade:")
    print("-" * 78)

    # מעבר על העשורים לפי סדר כרונולוגי
    for decade in sorted(df["Decade"].unique()):
        decade_rows = df[df["Decade"] == decade]
        # קיבוץ כל המילים של כל תקצירי העשור לרשימה אחת
        combined_tokens = [tok for tokens in decade_rows["Tokens"] for tok in tokens]

        scores = _pagerank_scores(combined_tokens, window_size=window_size)
        top_raw = [w for w, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]

        # שכיחות בתוך העשור, ומספר התקצירים השונים שבהם הופיעה כל מילה
        decade_counts = Counter(combined_tokens)
        decade_total = len(combined_tokens)
        doc_freq = Counter()
        for tokens in decade_rows["Tokens"]:
            doc_freq.update(set(tokens))

        distinctive = []
        for word, pagerank in scores.items():
            # מילה שמופיעה במעט תקצירים עלולה לנבוע מספר בודד
            if doc_freq[word] < MIN_DOC_FREQ:
                continue
            p_decade = decade_counts[word] / decade_total
            p_corpus = corpus_counts[word] / corpus_total
            distinctive.append((word, pagerank * math.log(1 + p_decade / p_corpus)))
        top_distinctive = [w for w, _ in sorted(distinctive, key=lambda kv: kv[1], reverse=True)[:top_n]]

        # עשור דל בספרים מסומן, אך אינו מוסתר – ההסתרה תסתיר עד כמה הרשומה דלילה
        flag = "  [low-confidence]" if len(decade_rows) < MIN_BOOKS_PER_DECADE else ""
        print(f"Decade: {decade}s | Books: {len(decade_rows):4d}{flag}")
        # כל עמודה עונה על שאלה אחרת, ולכן שתיהן מודפסות עם ההסבר לצידן
        print(f"    Top TextRank : {str(top_raw):<{_COL_WIDTH}} <- what vocabulary dominates")
        print(f"    Distinctive  : {str(top_distinctive):<{_COL_WIDTH}} <- what is unusual vs other decades")
    print()


def main():
    # מעבדים כל מאגר בנפרד, אחד אחרי השני (one at a time)
    print("Loading CMU book summaries...")
    cmu = _stratify_by_decade(load_cmu())
    analyze_dataframe(cmu, "CMU Book Summaries")

    print("Loading Goodreads (this streams 2.36M lines on the first run)...")
    goodreads = load_goodreads_cached()
    analyze_dataframe(goodreads, "Goodreads (all genres, English, dated by first publication)")


if __name__ == "__main__":
    main()
