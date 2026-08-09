"""Give every author a genre profile from what readers shelve them as.

The yardstick for eval_genres.py: the shelves never enter the clustering, so
they are an independent test of it. Each book is a distribution over
genre_vocab's categories, normalised within the book so one bestseller cannot
drown out an author's list; each author is the mean over their books.

Output: author_genres.csv"""
import csv
import json
from collections import defaultdict
from pathlib import Path

from genre_vocab import GENRE_NAMES, SHELF_GENRE

ROOT = Path(__file__).parent
JSON_PATH = ROOT / "goodreads_books.json"
AID_PATH = ROOT / "author_id_names.csv"
OUT_PATH = ROOT / "author_genres.csv"

MIN_BOOK_VOTES = 5      # genre-shelf votes a book needs to count at all
MIN_BOOKS = 3           # such books an author needs to get a profile

NG = len(GENRE_NAMES)
COL = {g: i for i, g in enumerate(GENRE_NAMES)}
# shelf -> [(column, share)], share split evenly over a shelf's genres so a
# compound shelf ("paranormal-romance") is counted for both, not for a winner
SHELF_COLS = {s: [(COL[g], 1.0 / len(gs)) for g in gs]
              for s, gs in SHELF_GENRE.items()}

with open(AID_PATH, newline="") as f:
    r = csv.reader(f)
    next(r)
    aid_name = {row[0]: row[1] for row in r}
print(f"Loaded {len(aid_name)} author_id -> name entries")

print("Scanning popular_shelves in goodreads_books.json ...")
totals = defaultdict(lambda: [0.0] * NG)    # author_id -> genre mass
n_books = defaultdict(int)                  # author_id -> books counted
n_votes = defaultdict(int)                  # author_id -> shelf votes behind it
seen_works = set()
n_total = n_shelved = n_used = 0

with open(JSON_PATH) as f:
    for line in f:
        n_total += 1
        if n_total % 200000 == 0:
            print(f"  scanned {n_total:>8d}  books used {n_used:>7d}  "
                  f"authors {len(totals):>6d}")
        try:
            b = json.loads(line)
        except json.JSONDecodeError:
            continue
        shelves = b.get("popular_shelves") or []
        if not shelves:
            continue
        n_shelved += 1
        work_id = b.get("work_id") or b.get("book_id")
        if not work_id or work_id in seen_works:
            continue
        seen_works.add(work_id)
        # a book is evidence about the people who wrote it; role is "" for an
        # author and names the job ("Illustrator", "Translator") otherwise
        aids = [a["author_id"] for a in (b.get("authors") or [])
                if not a.get("role") and a.get("author_id") in aid_name]
        if not aids:
            continue

        vec = [0.0] * NG
        votes = 0
        for sh in shelves:
            cols = SHELF_COLS.get(sh["name"])
            if not cols:
                continue
            try:
                c = int(sh["count"])
            except (TypeError, ValueError):
                continue
            votes += c
            for i, share in cols:
                vec[i] += c * share
        if votes < MIN_BOOK_VOTES:
            continue
        n_used += 1
        # normalise within the book: what we take from it is its genre mix,
        # not how many readers it has
        scale = 1.0 / sum(vec)
        for aid in aids:
            t = totals[aid]
            for i, v in enumerate(vec):
                if v:
                    t[i] += v * scale
            n_books[aid] += 1
            n_votes[aid] += votes

with open(OUT_PATH, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["author_id", "name", "books", "votes"] + GENRE_NAMES)
    kept = 0
    for aid in sorted(totals, key=lambda a: (-n_books[a], a)):
        nb = n_books[aid]
        if nb < MIN_BOOKS:
            continue
        kept += 1
        w.writerow([aid, aid_name[aid], nb, n_votes[aid]]
                   + [f"{v / nb:.6f}" for v in totals[aid]])

print(f"Records:            {n_total}")
print(f"  with any shelf:   {n_shelved}")
print(f"  used (deduped by work, >= {MIN_BOOK_VOTES} genre votes): {n_used}")
print(f"Authors with any genre-shelved book: {len(totals)}")
print(f"  kept (>= {MIN_BOOKS} books): {kept}")
print(f"Saved: {OUT_PATH}")
