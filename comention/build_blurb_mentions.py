"""Extract author mentions from every blurb.

Pass 1 builds an author dictionary from goodreads/*.csv (which carry names);
pass 2 streams goodreads_books.json (which carries the blurbs), finds runs of
capitalised words and longest-matches them against the dictionary.

Self-mentions are dropped -- a blurb naming its own book's author says nothing
about who is named *together*, and wires every author to their own marketing
comparisons. The JSON identifies a book's own authors only by id, which is what
author_id_names.csv is for.

Output: blurb_mentions.csv"""
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from collections import defaultdict

from name_filters import exclude_author, clean_ws

ROOT = Path(__file__).parent
CSV_DIR = ROOT / "goodreads"
JSON_PATH = ROOT / "goodreads_books.json"
AID_PATH = ROOT / "author_id_names.csv"
OUT_PATH = ROOT / "blurb_mentions.csv"

POP_MIN = 100            # min RatingDistTotal for a CSV book to seed the dictionary
MIN_DESC_LEN = 80        # skip near-empty blurbs
MAX_NGRAM = 4            # author names are 2-4 tokens

csv.field_size_limit(sys.maxsize)

# A run of >=2 capitalized words, allowing interior lowercase joiner words so
# names like "Charles de Lint" or "Ursula K. Le Guin" stay whole.
_CAP = r"[A-Z][A-Za-z.'’-]*"
_JOIN = r"of|the|and|in|to|for|a|an|de|van|von|le|la|du|del|der|on|at|by|with"
CAP_RUN_RE = re.compile(rf"{_CAP}(?:\s+(?:{_JOIN}|{_CAP}))*\s+{_CAP}")
TOKEN_RE = re.compile(r"[a-z0-9]+")


def norm_tokens(s):
    """Lowercase, fold initials (J.K. -> jk), return content tokens."""
    return TOKEN_RE.findall(s.replace(".", "").lower())


def name_key(s):
    """Accent-folded token identity for self-mention comparison ("Carré" ==
    "Carre"); same rule comention.py uses to canonicalize downstream."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return tuple(norm_tokens(s))


# Initials handling mirrors comention.py's merge rule, but per-book: a
# spelled-out mention ("Jane Green") must match the own-author name exactly,
# while an initials-form mention ("J.K. Rowling" against own "Joanne K.
# Rowling") may match on the (surname, given-initials) signature.
VOWELS = set("aeiouy")


def given_initials(tokens):
    out = []
    for t in tokens[:-1]:
        if len(t) <= 4 and not set(t) & VOWELS:
            out.extend(t)          # run-together initials: 'jrr' -> j,r,r
        else:
            out.append(t[0])
    return "".join(out)


def is_spelled(tokens):
    return any(len(t) >= 3 and set(t) & VOWELS for t in tokens[:-1])


# ---------------------------------------------------------------------------
# Pass 1: build the author dictionary from the CSV book dump (has names).
# ---------------------------------------------------------------------------
print("Pass 1: building author dictionary from goodreads/*.csv ...")

author_disp = defaultdict(lambda: defaultdict(int))   # tuple -> {display: count}

csv_files = sorted(CSV_DIR.glob("book*.csv"))
for cf in csv_files:
    with open(cf, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rdt = row.get("RatingDistTotal") or ""
            m = re.search(r"\d+", rdt)
            pop = int(m.group()) if m else 0
            if pop < POP_MIN:
                continue
            authors = row.get("Authors") or ""
            for a in re.split(r"\s*[,/&]\s*| and ", authors):
                a = a.strip()
                if not a:
                    continue
                a_toks = norm_tokens(a)
                if 2 <= len(a_toks) <= MAX_NGRAM and not exclude_author(a):
                    author_disp[tuple(a_toks)][clean_ws(a)] += 1

# Resolve each key to its most common original spelling.
author_set = {k: max(v, key=v.get) for k, v in author_disp.items()}
del author_disp
print(f"  authors in dictionary: {len(author_set)}")


def find_mentions(desc):
    """Yield canonical author names mentioned in a blurb (longest match wins)."""
    out = []
    for run in CAP_RUN_RE.findall(desc):
        toks = norm_tokens(run)
        i = 0
        L = len(toks)
        while i < L:
            advanced = False
            for n in range(min(MAX_NGRAM, L - i), 1, -1):
                win = tuple(toks[i:i + n])
                if win in author_set:
                    out.append(author_set[win])
                    i += n
                    advanced = True
                    break
            if not advanced:
                i += 1
    return out


# ---------------------------------------------------------------------------
# Pass 2: stream the blurbs and emit the edge list.
# ---------------------------------------------------------------------------
aid_name = {}
if AID_PATH.exists():
    with open(AID_PATH, newline="") as f:
        r = csv.reader(f)
        next(r)
        aid_name = {row[0]: row[1] for row in r}
    print(f"Loaded {len(aid_name)} author_id -> name entries")
else:
    print(f"WARNING: {AID_PATH.name} missing (run build_author_id_map.py); "
          "self-mentions will NOT be dropped")

print("Pass 2: scanning blurbs in goodreads_books.json ...")

HTML_RE = re.compile(r"<[^>]+>")
seen_works = set()
n_total = n_books = n_edges = n_self = 0
n_books_owned = 0                    # books whose own authors we could name

with open(JSON_PATH, "r") as f, open(OUT_PATH, "w", newline="") as out:
    w = csv.writer(out)
    w.writerow(["src_book_id", "src_work_id", "src_title", "mention_type", "mention_name"])
    for line in f:
        n_total += 1
        if n_total % 200000 == 0:
            print(f"  scanned {n_total:>8d}  books {n_books:>7d}  edges {n_edges:>8d}")
        try:
            b = json.loads(line)
        except json.JSONDecodeError:
            continue
        desc = b.get("description") or ""
        if len(desc) < MIN_DESC_LEN:
            continue
        work_id = b.get("work_id") or b.get("book_id")
        if not work_id or work_id in seen_works:
            continue
        seen_works.add(work_id)
        n_books += 1
        desc = HTML_RE.sub(" ", desc)
        title = (b.get("title") or "").strip()
        book_id = b.get("book_id") or ""
        own_keys = {name_key(aid_name[a["author_id"]])
                    for a in (b.get("authors") or [])
                    if a.get("author_id") in aid_name}
        own_sigs = {(k[-1], given_initials(k)) for k in own_keys if len(k) >= 2}
        if own_keys:
            n_books_owned += 1
        seen = set()
        for name in find_mentions(desc):
            if name in seen:
                continue
            seen.add(name)
            k = name_key(name)
            if k in own_keys or (len(k) >= 2 and not is_spelled(k)
                                 and (k[-1], given_initials(k)) in own_sigs):
                n_self += 1
                continue
            w.writerow([book_id, work_id, title, "author", name])
            n_edges += 1

print(f"Total records:  {n_total}")
print(f"Books w/ blurb: {n_books}")
print(f"  own author known: {n_books_owned}")
print(f"Author mentions: {n_edges}")
print(f"Self-mentions dropped: {n_self}")
print(f"Saved: {OUT_PATH}")
