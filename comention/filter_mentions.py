"""blurb_mentions.csv -> blurb_mentions_clean.csv: authors only.

Drops organisations, publishers and non-literary public figures via
name_filters."""
import csv
from pathlib import Path

from name_filters import exclude_author, clean_ws

ROOT = Path(__file__).parent
IN_PATH = ROOT / "blurb_mentions.csv"
OUT_PATH = ROOT / "blurb_mentions_clean.csv"

kept = dropped_book = dropped_org = 0
with open(IN_PATH, newline="") as fin, open(OUT_PATH, "w", newline="") as fout:
    r = csv.reader(fin)
    w = csv.writer(fout)
    w.writerow(next(r))
    for row in r:
        if len(row) < 5:
            continue
        if row[3] != "author":
            dropped_book += 1
            continue
        if exclude_author(row[4]):
            dropped_org += 1
            continue
        row[2] = clean_ws(row[2])          # tidy mangled whitespace in names
        row[4] = clean_ws(row[4])
        w.writerow(row)
        kept += 1

print(f"Kept:    {kept} author mentions")
print(f"Dropped: {dropped_book} book-title rows (not used by this pipeline)")
print(f"Dropped: {dropped_org} organisation/non-person authors")
print(f"Saved: {OUT_PATH}")
