"""
תיחום הטקסט: איתור ההתחלה והסוף של תיאור העלילה בתוך תקציר Goodreads.

ההנחה המבנית שעליה הכל נשען: הזבל מצטבר בקצוות. תקציר מזוהם טיפוסי בנוי
[פתיח פרסי/רבי-מכר] [העלילה] [על המחבר / שבחים / מדריך לקבוצת קריאה],
ולכן מציאת גבול התחלה וגבול סוף מחזירה את רוב העלילה בלי לסווג כל משפט.

שלוש משפחות של כללים, כל אחת עם שם לכל כלל, כדי שכל הסרה תהיה ניתנת
לביקורת בדיעבד. זו הדרישה המרכזית: לא מספיק שהמסנן יעבוד, צריך יומן שאפשר
לעבור עליו ולראות מה נמחק ולמה.

מגבלה ידועה מראש: תיחום אינו יכול להגיע לפרסומת שיושבת *בתוך* משפט עלילה.
"In this unforgettable novel, a soldier returns from Vietnam" הוא משפט אחד,
רובו עלילה, והוא שורד בשלמותו.
"""

import re
from collections import namedtuple

# --- פרמטרים ---

# מתחת לכמות הזו של תווים אחרי התיחום, מוותרים ומחזירים את המקור.
# מסמך ריק גרוע יותר ממסמך עם רעש
MIN_KEPT_CHARS = 200
# אותו רעיון כשיעור: תיחום שמוחק יותר מ-60% מהתקציר כנראה טעה
MIN_KEPT_RATIO = 0.40
# פסקה שרוב אותיותיה גדולות היא כותרת או שורת צעקה, ולא עלילה
_CAPS_RATIO = 0.80
_CAPS_MIN_CHARS = 20
# כותרת קצרה המסתיימת בנקודתיים ("About the Author:")
_HEADING_MAX_CHARS = 60


# --- שיטה 1: ביטויים רגולריים לפתיחים ולסיומות פרסומיות ---
#
# הכיוון חשוב ולכן הרשימות נפרדות: כלל "פתיח" אומר "חתוך את הכל עד כאן",
# וכלל "סיומת" אומר "חתוך את הכל מכאן והלאה". ביטוי שמופיע רק כשהוא בקצה
# מעוגן ל-^ - "Praise for" באמצע משפט יכול להיות תוכן אמיתי.

_PREFACE_RULES = [
    # "From the #1 New York Times bestselling author of..." - הצגת המחבר,
    # שקודמת לעלילה ואינה חלק ממנה
    ("bestselling_author",
     r"\b(#\s*1\s+)?(new york times|usa today|sunday times|wall street journal|"
     r"internationally|nationally|globally)?\s*best[\s-]?selling author\b"),
    # פרסים מוכרזים לפני התקציר, כמעט תמיד כמשפט נפרד
    ("award_winner",
     r"^\s*\W*(winner of|shortlisted for|longlisted for|finalist for|"
     r"nominated for|winner:)\b"),
    # "From the author of Gone Girl" - הפניה לספר אחר
    ("from_the_author_of",
     r"^\s*\W*from the (#\s*1\s+)?(new york times\s+)?(best[\s-]?selling\s+)?"
     r"(author|creator|writer) of\b"),
    # עיבוד קולנועי הוא עובדה על הספר, לא על העלילה
    ("now_a_film",
     r"\bnow a (major )?(motion picture|film|netflix (series|original|film)|"
     r"hbo (series|film)|major (television|tv) (series|event))\b"
     r"|\bsoon to be a (major )?(motion picture|film|series)\b"),
    # תגי "ספר השנה" למיניהם
    ("notable_book",
     r"^\s*\W*(an? )?(new york times )?(notable book|book of the year|"
     r"best book of|editors'? choice|oprah'?s book club)\b"),
    # "The instant #1 New York Times bestseller"
    ("instant_bestseller",
     r"^\s*\W*(the )?(instant )?(#\s*1\s+)?(new york times |usa today )?"
     r"best[\s-]?seller\b"),
]

# סיומות מסוג *כותרת*: הן אינן משפט זבל בודד אלא נקודת חיתוך. כל מה שאחרי
# "About the Author" הוא ביוגרפיה, וכל מה שאחרי "Praise for" הוא ציטוטי
# ביקורת - גם אם אף אחד מהמשפטים האלה אינו מפעיל בעצמו שום כלל. זה היה
# הכשל המרכזי בגרסה הראשונה: הסריקה לאחור נעצרה מייד, משום שהמשפט האחרון
# בביוגרפיה נראה תמים, והכותרת נשארה באמצע הטקסט בלי שהוסרה
_HARD_POSTSCRIPT_RULES = [
    # הביוגרפיה של המחבר - הסיומת הנפוצה ביותר
    ("about_the_author",
     r"^\s*\W*about the (author|translator|illustrator|editor|book)\b"),
    # תוספות מסחריות בסוף הכרך
    ("includes_extra",
     r"\bincludes? (a |an )?(excerpt|preview|bonus|sneak peek|"
     r"reading group guide|discussion guide|readers guide)\b"),
    ("reading_guide",
     r"\b(reading (group|club) guide|discussion questions|"
     r"questions for discussion|a readers guide)\b"),
    # פנייה ישירה לקורא לקנות עוד - לעולם לא עלילה
    ("dont_miss",
     r"^\s*\W*(don'?t miss|look for|also by|also available|perfect for fans of|"
     r"if you (loved|liked|enjoyed))\b"),
    ("praise_block", r"^\s*\W*praise for\b"),
]

_POSTSCRIPT_RULES = [
    # מנגנון המהדורה
    ("with_introduction",
     r"\bwith an? (new |brand[\s-]new )?(introduction|foreword|afterword|"
     r"preface|essay) by\b"),
    ("translated_by",
     r"\btranslated (from the \w+ )?by\b|^\s*\W*edited by\b"),
    ("edition_stmt",
     r"^\s*\W*(this|the) (revised |deluxe |anniversary |special |new |"
     r"expanded |updated )*(edition|printing|reissue|impression)\b"),
    # רק כמשפט קצר ועצמאי. "First published in 1952, this classic recounts
    # the story of Basil, a young silversmith..." הוא משפט עלילה שנפתח
    # בהערה ביבליוגרפית, והסרתו מוחקת את העלילה איתה
    ("first_published", r"^.{0,110}\b(first|originally) published (in|by)\b.{0,60}$"),
    ("cover_art",
     r"\bcover (art|design|illustration|photograph) by\b|\bjacket (design|art) by\b"),
    ("page_count", r"^\s*\W*\d{2,4} pages\b|\b\d{2,4} pages\.\s*$"),
]


# --- שיטה 2א: תבניות חוזרות קבועות - כותרות תחתונות של מו"לים ---
#
# טקסט משפטי ויצירת קשר. אלו יורים בכל מיקום, לא רק בקצה, משום שהם
# לעולם אינם עלילה - גם אם נתקעו באמצע.

# נמצאו בסריקת המשפטים בעלי ציון הרגיסטר הגבוה ביותר שאף כלל לא תפס.
# כל אחת מהן היא משפחה שחוזרת בקורפוס, לא מקרה בודד.
_JUNK_RULES = [
    # א. תגי פרסים ועיטורים, לרוב כמשפט ללא פועל כלל:
    # "An ALA Best Book for Young Adults", "New York Times and Publishers
    # Weekly Bestseller"
    ("award_badge",
     r"\b(ala|hugo|nebula|pulitzer|booker|newbery|caldecott|edgar|whitbread|"
     r"costa|orange|nobel|national book|man booker|pen/faulkner)\b"
     r"[^.!?]{0,40}\b(award|prize|winner|honou?r|best books?|notable)\b"
     r"|^\W*an? [A-Z][^.!?]{0,50}\bbest books? for young adults\b"
     r"|^[^.!?]{0,60}\bbest[\s-]?seller\b\s*[.!]?\s*$"),
    # ב. הצהרות מהדורה והדפסה מחדש - עשירות הרבה יותר ממה שהיה ברשימה:
    # "Unabridged republication of the classic 1931 edition.",
    # "Unabridged, slightly corrected reprint of the 2nd, 1957 edition."
    ("reprint_stmt",
     r"\b(unabridged|abridged|facsimile)\b[^.!?]{0,80}"
     r"\b(republication|reprint|reproduction|edition|republished)\b"
     r"|\b(republication|reprint) of the\b"
     r"|\b(reissued|republished|reprinted) (by|in|as)\b"
     r"|\bnow available in (paperback|hardcover|hardback|ebook|print)\b"
     r"|\bavailable in (paperback|hardcover|ebook)[^.!?]{0,30}for the first time\b"
     r"|\bmass market edition\b|\bdigitally (enlarged|reproduced|remastered)\b"
     r"|\bupdated (typeface|layout)\b"),
    # ג. ציטוט ביקורת שמיוחס בסוגריים או בשם עיתון בלבד, בלי מקף:
    # "'superb!' (Australian SF News)", "'Entirely original' Spectator"
    ("quote_attribution",
     r"[\"'“”‘’][^\"'“”‘’]{3,}[\"'“”‘’]\s*\([^)]{3,40}\)\s*$"
     r"|[\"'“”‘’][^\"'“”‘’]{6,}[\"'“”‘’]\s+[A-Z][\w' ]{2,30}\s*$"
     r"|^\W*from \d+ stars? reviews?\b"),
    # ד. מסחר טהור
    ("commerce",
     r"\bfree to download\b|\bplus excerpts? from\b|\bbonus (material|content)\b"
     r"|\border (your copy|now)\b|\bon sale now\b|\bbuy (it |the )?now\b"),
    # ו. תג פרס כשבר משפט, בלי פועל ובלי שם פרס מוכר:
    # "Winner, Best Books 2010", "Fehrenbach Award, Best Ethnic, Minority,
    # And Women's History Publication, 1987"
    ("award_fragment",
     r"^\W*winner\s*[,:]\s*\w"
     r"|^[^.!?]{0,60}\baward\b\s*[,:][^.!?]{0,80}\d{4}\s*[.!]?\s*$"
     r"|^\W*(one of |an? )?[A-Z][\w' ]{2,30}('s)?\s*best\b[^.!?]{0,50}"
     r"\b(books?|novels?|of \d{4})\b"),
    # ז. חזרה לדפוס. ההבדל מ-reprint_stmt: כאן המו"ל הוא הנושא הדקדוקי,
    # "The OSU Press is proud to reissue this...", "is back in print with
    # a new afterword", "This Swallow Press reissue of Ladders to Fire"
    ("back_in_print",
     r"\bback in print\b"
     r"|\bis (proud|pleased|delighted) to (reissue|present|publish|offer)\b"
     r"|\b[A-Z][\w]* (Press|Books|Publishing|House) reissue of\b"
     r"|\bnow in (paperback|hardcover|hardback)\b"
     r"|\bhas been reissued\b"),
    # ח. טקסט של המרה לספר אלקטרוני
    ("ebook_format",
     r"\bcarefully crafted ebook\b|\bformatted for your (ereader|kindle|nook)\b"
     r"|\bfunctional table of contents\b|\bactive table of contents\b"
     r"|\bconverted from its physical edition\b"),
    # ט. מיצוב הקריירה של המחבר. שני תנאים בביטוי עצמו - ניסוח של מעמד
    # *וגם* הקשר של כתיבה - כדי ש-"Anna is known for her temper" לא ייפול
    ("author_positioning",
     r"\bis (widely |generally |universally )?"
     r"(known|regarded|recognized|considered|celebrated|acclaimed|hailed)"
     r"\s+(for|as)\b[^.!?]{0,80}"
     r"\b(writer|author|novelist|poet|work|writing|fiction|literature|"
     r"novels?|books?|stories|storytelling|prose)\b"
     r"|\bhas established (himself|herself|themselves) as\b"
     r"|\bone of the (foremost|greatest|finest|leading|most \w+) "
     r"(writers?|authors?|novelists?|poets?|storytellers?|authorities)\b"),
    # י. היסטוריית הפרסום של הכרך
    ("publication_history",
     r"\bbecame an? (immediate |instant |international )?best[\s-]?seller\b"
     r"|\bwas adapted into (a )?(film|movie|television|tv)\b"
     r"|\bwas selected by\b[^.!?]{0,60}\bfor\b[^.!?]{0,40}\b(annual|year'?s best)\b"
     r"|\bwent on to (sell|become)\b"),
    # ה. רשימת חומרי הפתיחה של הכרך, שהיא תוכן העניינים ולא תיאור:
    # "Foreword to the Basic Books Paperback Edition, 1974 (Gardner);
    #  Preface (Carnap); Foreword to the Dover Edition (Gardner)."
    ("front_matter_list",
     r"^\W*(foreword|preface|introduction|contents|translator'?s note)\b"
     r"[^.!?]{0,80};"),
]


_FOOTER_RULES = [
    ("copyright", r"copyright\s*(©|\(c\))|©\s*\d{4}|\ball rights reserved\b"),
    ("printed_in", r"\bprinted (and bound )?in (the )?[A-Z]"),
    ("isbn", r"\bisbn\b"),
    ("library_of_congress", r"\blibrary of congress\b"),
    # כתב הוויתור המשפטי שמופיע בעמוד זכויות היוצרים
    ("fiction_disclaimer",
     r"\bis a work of fiction\b|\bany resemblance to (actual|real) (persons|events)\b"),
    ("url", r"www\.\S+|https?://\S+"),
    ("social", r"\b(visit|follow|find) (the author|us|him|her|them)?\s*"
               r".{0,40}(at |on )(twitter|instagram|facebook|www\.|http)"),
]


# --- שיטה 3: היסקים מבניים ולשוניים ---
#
# מילות קישור, פיסוק ואותיות גדולות שמסמנים מעבר מהעלילה למטא-דאטה.

# פעלים בקול של מוצר: המו"ל מתאר את החפץ ולא את הסיפור.
# מסוכן יותר משאר הכללים - "Includes twelve stories about grief" הוא תוכן -
# ולכן משקלו נמוך והוא אינו יורה לבדו במצב הרגיל
_PRODUCT_VOICE_RE = re.compile(
    r"^\s*\W*(includes?|featuring|features?|contains?|presents?|offers?|"
    r"provides?|collects?|compiles?|gathers?)\b", re.I)

# ציטוט ואחריו מקף וייחוס - חתימת ציטוט הביקורת.
# הגרסה הראשונה הייתה רחבה מדי: הדפוס "מרכאות ... מקף ... אות גדולה" תפס
# ציטוטים מתוך *הספר עצמו* שהודבקו כתקציר, משום שמקף באמצע משפט נחשב
# ייחוס. כאן נדרש שהייחוס יהיה קצר ויסיים את המשפט, כפי שייחוס אמיתי נראה
# שני ביטויים ולא אחד, משום ש-re.I על ביטוי משותף מבטל את הדרישה לאות
# גדולה: כך '"Genesis" -- before it is too late' נחשב ייחוס של מבקר.
# צורת הייחוס תלוית-רישיות ולכן היא נבדקת בלי re.I
_REVIEW_ATTR_CASED_RE = re.compile(
    r"[\"“”]\s*[-–—]{1,2}\s*[A-Z][\w.'’&\- ]{2,40}\s*$")
_REVIEW_SOURCE_RE = re.compile(
    r"[-–—]{1,2}\s*(the )?(kirkus|publishers weekly|booklist|"
    r"library journal|new york times|washington post|guardian|observer|locus|"
    r"times literary supplement|entertainment weekly|people magazine|usa today|"
    r"boston globe|san francisco chronicle|npr|the atlantic|"
    r"\w+ review of books)\b", re.I)

# פנייה בגוף שני או ציווי. "Meet Anna" נשאר בחוץ בכוונה - זו פתיחה
# לגיטימית של עלילה
_READER_ADDRESS_RE = re.compile(
    r"^\s*\W*(don'?t miss|get ready (for|to)|if you (loved|liked|enjoyed)|"
    r"perfect for (fans|readers|anyone)|a must[\s-]read for|"
    r"you'?ll (love|never)|prepare to be)\b", re.I)

# הצהרת מהדורה. שימו לב למה שאינו כאן: book, novel, story, collection.
# הגרסה המתבקשת של הכלל - "סמן כל משפט שנושאו הוא הספר" - הורסת תוכן,
# משום ש-"This novel explores the life of a Nigerian immigrant in 1970s
# London" הוא משפט העלילה המרכזי בחלק גדול מהתקצירים. הנושא אינו האות;
# המושא הוא. ולכן נדרשים שלושה תנאים יחד ולא אחד מהם
_EDITION_SUBJECT_RE = re.compile(
    r"\bthis (\w+\s+)?(edition|volume|printing|reissue|impression|reprint)\b", re.I)
_EDITION_VERB_RE = re.compile(
    r"\b(includes?|contains?|features?|reprints?|restores?|corrects?|adds?|"
    r"presents?|reproduces?|incorporates?)\b", re.I)
_EDITION_OBJECT_RE = re.compile(
    r"\b(introduction|notes?|index|appendix|appendices|illustrations?|"
    r"translation|foreword|afterword|errata|revisions?|bibliography|"
    r"glossary|preface|typography|annotations?)\b", re.I)

_HEADING_RE = re.compile(r"^\s*[A-Z][^.!?]{0,%d}:\s*$" % _HEADING_MAX_CHARS)


# --- שיטה 3ב: צפיפות רגיסטר, לכללים שאי אפשר לנסח כביטוי רגולרי ---
#
# המשפחה הגדולה ביותר שנמצאה בקריאת הטקסט אינה בעלת תבנית מילולית קבועה:
# "It's an audacious, at times hilarious story that is ultimately
# heartbreaking and unforgettable." או "Here is a rich and moving story, a
# superbly readable one, a remarkable evocation of the native South."
# אין כאן שום ביטוי שאפשר לתפוס - יש שם עצם של ספר, ערימת תארי שבח, ואפס
# עלילה. לכן הכלל הזה מסתמך על משקלי הרגיסטר מ-keyness_word_weights.csv.

WEIGHTS_PATH = "keyness_word_weights.csv"
# הסף כויל על 60 המשפטים בעלי הציון הגבוה שאף כלל לא תפס
REGISTER_DENSE_MIN = 2.2
REGISTER_DENSE_MIN_WORDS = 4
# ציון גבוה במיוחד, שמאפשר להסתפק בשלוש מילים מדורגות
REGISTER_DENSE_STRONG = 3.0
# מילה בעלת משקל שלילי חזק היא ראיה לעלילה, ומבטלת את הכלל
PLOT_EVIDENCE_MAX = -0.40

# הדבר שמשבחים. בלי שם עצם של ספר, משפט עתיר תארים הוא כנראה תיאור של
# העלילה עצמה - "Their monumental achievement ... led to each being awarded
# the Distinguished Flying Cross" מדבר על אנשים, לא על הכרך
_BOOK_NOUN_RE = re.compile(
    r"\b(story|stories|novel|book|work|masterpiece|masterwork|portrait|"
    r"account|saga|study|collection|memoir|biography|read|volume|epic|"
    r"debut|prose|narrative|edition|anthology|classic)\b", re.I)

# שמות ז'אנר מגנים על המשפט גם כשציון הרגיסטר גבוה. "A tense and
# nerve-shattering classic from the highly acclaimed master of action and
# suspense" הוא אכן טקסט עטיפה, אך הז'אנר עצמו הוא מידע נושאי: "מתח עולה
# בשנות התשעים" הוא ממצא. הערה: ה-keyness חולק - suspense מקבל שם 3.96 -
# וההכרעה כאן היא של החוקרת ולא של המדידה
_GENRE_NOUN_RE = re.compile(
    r"\b(action|suspense|adventure|mystery|romance|horror|fantasy|thriller|"
    r"western|crime|noir|gothic|satire|comedy|tragedy|science fiction|"
    r"historical|detective|espionage|dystopian)\b", re.I)

# פותח פסוקית סיפורית: "the story of *how* Wol and Weeps turn the town
# upside down" הוא עלילה שנארזה במשפט משבח, ואסור להסירו
_NARRATIVE_CLAUSE_RE = re.compile(
    r"\b(how|when|after|before|while|who|whose|where)\b\s+\w+", re.I)

_TOKEN_RE = re.compile(r"[a-z][a-z']+")
_WEIGHTS = None


def _resolve_weights(path):
    """
    הטבלה נטענה קודם לפי תיקיית העבודה בלבד, ובהיעדרה הכלל register_dense
    פשוט לא ירה - בלי הודעה, ועם קורפוס שונה בסופו של דבר. לכן מחפשים גם
    ליד הקוד עצמו, וכשלא נמצא כלום מזהירים במקום לשתוק.
    """
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (path, os.path.join(here, "..", "work", path),
                 os.path.join(here, path)):
        if os.path.exists(cand):
            return cand
    return None


def load_weights(path=WEIGHTS_PATH):
    """טוען את טבלת המשקלים פעם אחת. בהיעדרה הכלל אינו פועל, וזה נאמר בקול."""
    global _WEIGHTS
    if _WEIGHTS is None:
        found = _resolve_weights(path)
        if found is None:
            import warnings
            warnings.warn(
                f"{path} not found - the register_dense rule will not fire and "
                f"the corpus will differ from the published one. Run keyness.py "
                f"first (run_all.py does this in the right order).", stacklevel=2)
            _WEIGHTS = {}
        else:
            import csv
            with open(found, newline="", encoding="utf-8") as f:
                _WEIGHTS = {r["word"]: float(r["weight"]) for r in csv.DictReader(f)}
    return _WEIGHTS


def _lookup(word, W):
    """התאמה גסה ללמה: הטבלה בנויה על למות, והטקסט כאן אינו מלומטז."""
    if word in W:
        return W[word]
    for suf, rep in (("ies", "y"), ("es", ""), ("s", ""), ("ing", ""),
                     ("ed", ""), ("ly", "")):
        if word.endswith(suf):
            cand = word[:-len(suf)] + rep
            if cand in W:
                return W[cand]
    return None


def register_score(sentence):
    """(ציון ממוצע, מספר מילים מדורגות, המשקל הנמוך ביותר שנמצא)."""
    W = load_weights()
    if not W:
        return 0.0, 0, 0.0
    vals = [v for v in (_lookup(w, W) for w in _TOKEN_RE.findall(sentence.lower()))
            if v is not None]
    if not vals:
        return 0.0, 0, 0.0
    return sum(vals) / len(vals), len(vals), min(vals)


def is_register_dense(sentence):
    """
    ארבעה תנאים יחד. כל אחד מהם לבדו מוחק תוכן:
    ציון גבוה, די מילים כדי שהממוצע יהיה יציב, שם עצם של ספר כמושא השבח,
    ואין ראיה לעלילה - לא מילת עלילה חזקה ולא פסוקית סיפורית.
    """
    score, n, lo = register_score(sentence)
    if score < REGISTER_DENSE_MIN:
        return False
    # שלוש מילים מדורגות מספיקות רק כשהציון גבוה מאוד, אחרת נדרשות ארבע
    if n < 3 or (n == 3 and score < REGISTER_DENSE_STRONG):
        return False
    if lo <= PLOT_EVIDENCE_MAX:
        return False
    if not _BOOK_NOUN_RE.search(sentence):
        return False
    if _NARRATIVE_CLAUSE_RE.search(sentence) or _GENRE_NOUN_RE.search(sentence):
        return False
    return True


# --- שמות עיתונים וכתבי עת, כביטויים ולא כמילים ---
#
# "New York Times" הוא שם עיתון, אבל york ו-times כמילים בודדות הן תוכן
# לגיטימי: "new york" ללא "times" מופיע ב-1,384 מסמכים כזירת התרחשות.
# מדידה על הקורפוס: 36% מכל האזכורים של york ו-34% מאלה של times יושבים
# בתוך "New York Times". לכן ההסרה היא של הצירוף בלבד, ולא של המילים -
# הרשימה השחורה ברמת המילה הייתה מוחקת את ניו יורק עצמה.
_PUBLICATION_PHRASES = [
    r"#?\s*1?\s*new york times", r"n\.?y\.?\s+times", r"los angeles times",
    r"sunday times", r"times literary supplement", r"wall street journal",
    r"washington post", r"publishers?'? weekly", r"kirkus reviews?",
    r"(school )?library journal", r"\bbooklist\b", r"usa today",
    r"entertainment weekly", r"\bbook review\b", r"oprah'?s book club",
    r"\bgoodreads\b", r"amazon\.com", r"barnes ?& ?noble",
    # ניסוחי מהדורה שהחזיקו את אותו נושא: revised / updated.
    # דפוס אחד ולא ארבעה, אחרת "newly revised and updated" נחתך לחצי
    r"(newly |fully |completely |thoroughly )?"
    r"(revis\w+|updated)( and (revis\w+|updated))?(?=\s+(edition|version|and))",
]
_PUBLICATION_RE = re.compile("|".join(_PUBLICATION_PHRASES), re.I)


def strip_publication_names(text):
    """מוחק שמות עיתונים כצירוף. מחזיר (טקסט, כמה הוסרו)."""
    out, n = _PUBLICATION_RE.subn(" ", str(text))
    return re.sub(r"\s{2,}", " ", out).strip(), n


def _compile(rules):
    return [(name, re.compile(pat, re.I)) for name, pat in rules]


_PREFACE = _compile(_PREFACE_RULES)
_POSTSCRIPT = _compile(_POSTSCRIPT_RULES)
_HARD_POSTSCRIPT = _compile(_HARD_POSTSCRIPT_RULES)
_FOOTER = _compile(_FOOTER_RULES)
_JUNK = _compile(_JUNK_RULES)


# --- פיצול למשפטים עם שמירת היסטים ---

# מפצלים גם לפי שורה חדשה ולא רק לפי סימן סיום. הביקורת על הריצה השנייה
# הראתה שזה מקור הטעויות הנותר: "ISBN: 9780099602019\nTHE SECOND WAR OF THE
# RACES\nHorrified by the misuse of magic..." הוא משפט אחד בעיני מפצל
# שמחפש נקודה, ולכן הסרת ה-ISBN גררה איתה את פתיחת העלילה
_SENT_END_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text):
    """
    מפצל למשפטים ומחזיר [(start, end, sentence)] עם היסטים לתוך המחרוזת
    המקורית. ההיסטים נדרשים כדי שאפשר יהיה להחזיר גבולות ולתעד ביומן בדיוק
    איזה קטע הוסר.
    """
    out, pos = [], 0
    for piece in _SENT_END_RE.split(text):
        idx = text.find(piece, pos)
        if idx < 0:
            idx = pos
        if piece.strip():
            out.append((idx, idx + len(piece), piece))
        pos = idx + len(piece)
    return out


def _is_all_caps(s):
    letters = [c for c in s if c.isalpha()]
    if len(letters) < _CAPS_MIN_CHARS:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= _CAPS_RATIO


def flag_sentence(sentence, repeated=frozenset(), strict=False):
    """
    מחזיר (רשימת שמות הכללים שירו, kind) עבור משפט אחד.

    kind הוא "preface", "postscript" או "any" - הוא קובע מאיזה כיוון מותר
    לחתוך. משפט שסומן על ידי כלל דו-כיווני ("any") נחשב זבל בכל מיקום.
    """
    hits, kinds = [], set()
    norm = re.sub(r"\s+", " ", sentence.strip())

    for name, rx in _PREFACE:
        if rx.search(sentence):
            hits.append(name); kinds.add("preface")
    for name, rx in _POSTSCRIPT + _HARD_POSTSCRIPT:
        if rx.search(sentence):
            hits.append(name); kinds.add("postscript")
    for name, rx in _FOOTER + _JUNK:
        if rx.search(sentence):
            hits.append(name); kinds.add("any")

    # שיטה 2ב: משפט שחוזר מילה במילה בין ספרים שונים אינו מתאר אף אחד מהם.
    # הקבוצה מגיעה מהקורפוס ולא מרשימה שנכתבה ביד, ולכן היא מוצאת גם
    # מו"לים שאיש לא רשם
    if norm.lower() in repeated:
        hits.append("repeated_across_books"); kinds.add("any")

    # all_caps אינו יורה לבדו. הביקורת על הריצה הראשונה הראתה שהוא אכל
    # עלילה: "WHO KNOWS WHAT EVIL LURKS IN THE HEARTS OF MEN?" ו-"A
    # LUFTWAFFE ACE WHO WOULDN'T DIE..." הן שורות פתיחה של מגזיני עיסה,
    # כלומר תוכן בצעקות, ורק "OVER TWO AND ONE-HALF MILLION COPIES IN
    # PRINT!" היה פרסומת. אותיות גדולות מסמנות רגיסטר, לא מטא-דאטה
    caps = _is_all_caps(norm)
    if _REVIEW_ATTR_CASED_RE.search(sentence) or _REVIEW_SOURCE_RE.search(sentence):
        hits.append("review_attribution"); kinds.add("any")
    if _READER_ADDRESS_RE.search(sentence):
        hits.append("reader_address"); kinds.add("any")
    if _HEADING_RE.match(norm):
        hits.append("heading_colon"); kinds.add("any")
    if (_EDITION_SUBJECT_RE.search(sentence)
            and _EDITION_VERB_RE.search(sentence)
            and _EDITION_OBJECT_RE.search(sentence)):
        hits.append("edition_statement"); kinds.add("any")

    # שני הכללים החלשים. הם מחזקים סימון קיים ואינם יוצרים אחד חדש, משום
    # ששניהם מסמנים רגיסטר ("קול של מוצר", "צעקה") ולא מטא-דאטה
    if is_register_dense(sentence):
        hits.append("register_dense"); kinds.add("any")

    weak = []
    if _PRODUCT_VOICE_RE.match(sentence):
        weak.append("product_voice")
    if caps:
        weak.append("all_caps")
    if weak and (hits or strict):
        hits.extend(weak); kinds.add("any")

    if not hits:
        return [], None
    if "any" in kinds:
        return hits, "any"
    if "preface" in kinds and "postscript" in kinds:
        return hits, "any"
    return hits, kinds.pop()


# כללים שמבוססים על ראיה חד-משמעית ולא על היסק לשוני
_HARD_EVIDENCE = frozenset({
    "repeated_across_books", "isbn", "copyright", "url", "library_of_congress",
    "printed_in", "fiction_disclaimer", "social", "about_the_author",
    "cover_art", "page_count",
})


BoundResult = namedtuple(
    "BoundResult", "start end text removed fallback n_sentences")


def bound_summary(text, repeated=frozenset(), strict=False):
    """
    מחזיר BoundResult עם גבולות התיאור, הטקסט המתוחם, ויומן ההסרות.

    removed הוא רשימה של dict-ים - קטע, מיקום, הכללים שירו וצד החיתוך.
    זהו התוצר החשוב: בלעדיו אי אפשר לבדוק בדיעבד מה נמחק בטעות.

    במצב הרגיל חותכים רק מהקצוות ולעולם לא מהאמצע, וזו בדיוק המשמעות של
    "תיחום". strict=True מסיר גם משפטים מסומנים מהאמצע, משום שציטוטי
    ביקורת אכן מופיעים באמצע התקציר.
    """
    text = str(text)
    sents = split_sentences(text)
    if not sents:
        return BoundResult(0, len(text), text, [], False, 0)

    flags = [flag_sentence(s, repeated, strict) for _, _, s in sents]
    n = len(sents)

    # גבול התחלה: מתקדמים כל עוד המשפט מסומן כפתיח או כזבל דו-כיווני
    lo = 0
    while lo < n and flags[lo][1] in ("preface", "any"):
        lo += 1
    # גבול סוף: נסוגים כל עוד המשפט מסומן כסיומת או כזבל דו-כיווני
    hi = n
    while hi > lo and flags[hi - 1][1] in ("postscript", "any"):
        hi -= 1

    # נקודת חיתוך קשה: הכותרת עצמה ומה שאחריה יורדים, גם אם המשפטים
    # שאחריה תמימים למראה. מחפשים את המוקדמת ביותר שאינה המשפט הראשון -
    # תקציר שכולו "About the Author" עדיף להשאיר לגלאי הבטיחות
    for i in range(1, hi):
        if any(rx.search(sents[i][2]) for _, rx in _HARD_POSTSCRIPT):
            hi = i
            break

    removed = []
    for i in range(n):
        inside = lo <= i < hi
        hits, kind = flags[i]
        if inside and not (strict and kind == "any" and hits):
            continue
        if not hits:
            continue
        side = "start" if i < lo else ("end" if i >= hi else "interior")
        removed.append({
            "sent_index": i, "side": side, "rules": ",".join(hits),
            "char_start": sents[i][0], "char_end": sents[i][1],
            "text": sents[i][2].strip(),
        })

    if strict:
        keep = [sents[i] for i in range(lo, hi)
                if not (flags[i][1] == "any" and flags[i][0])]
        bounded = " ".join(s for _, _, s in keep).strip()
        start = keep[0][0] if keep else sents[lo][0]
        end = keep[-1][1] if keep else sents[lo][0]
    else:
        if lo >= hi:
            bounded = ""
            start = end = sents[0][0]
        else:
            start, end = sents[lo][0], sents[hi - 1][1]
            bounded = text[start:end].strip()

    # מעקה בטיחות: תיחום שמוחק כמעט הכל כנראה טעה, ואז עדיף להחזיר את המקור.
    #
    # יוצא דופן אחד, שנלמד מהיומן: כשכל ההסרות באו מכללי ראיה קשה - משפט
    # שחוזר מילה במילה בין ספרים, ISBN, זכויות יוצרים - אין מה להציל.
    # 112 מסמכים נפלו למעקה, ורובם היו טקסט מו"ל במלואם ("This is a pre-1923
    # historical reproduction that was curated for quality..."). החזרתם
    # מכניסה את הזבל בחזרה; עדיף להחזיר ריק ולתת למסנן האורך למחוק אותם
    hard = all(set(r["rules"].split(",")) <= _HARD_EVIDENCE for r in removed)
    fallback = (len(bounded) < MIN_KEPT_CHARS
                or (len(text) and len(bounded) / len(text) < MIN_KEPT_RATIO))
    if fallback and removed and hard:
        return BoundResult(start if lo < hi else 0, end if lo < hi else 0,
                           bounded, removed, False, n)
    if fallback:
        return BoundResult(0, len(text), text, removed, True, n)
    return BoundResult(start, end, bounded, removed, False, n)


ALL_RULE_NAMES = (
    [n for n, _ in _PREFACE_RULES]
    + [n for n, _ in _POSTSCRIPT_RULES]
    + [n for n, _ in _HARD_POSTSCRIPT_RULES]
    + [n for n, _ in _FOOTER_RULES]
    + [n for n, _ in _JUNK_RULES]
    + ["register_dense"]
    + ["repeated_across_books", "all_caps", "review_attribution",
       "reader_address", "heading_colon", "edition_statement", "product_voice"]
)
