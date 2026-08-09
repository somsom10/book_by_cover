"""author_id -> display name, by joining the two dumps.

goodreads_books.json has only author_ids; goodreads/*.csv has names under the
same book id. For books in both, zip the two author lists positionally -- only
when the counts match, so the pairing is unambiguous -- and vote.

Output: author_id_names.csv"""
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from name_filters import clean_ws

ROOT = Path(__file__).parent
CSV_DIR = ROOT / "goodreads"
JSON_PATH = ROOT / "goodreads_books.json"
OUT_PATH = ROOT / "author_id_names.csv"

csv.field_size_limit(sys.maxsize)

# book_id appears exactly once per record; authors is a list of
# {"author_id": "...", "role": "..."} objects.
BID_RE = re.compile(r'"book_id": "(\d+)"')
AUTHORS_SEG_RE = re.compile(r'"authors": \[(.*?)\]')
AID_RE = re.compile(r'"author_id": "(\d+)"')

print("Reading author names from goodreads/*.csv ...")
csv_names = {}                       # book id -> [name, ...] in field order
for cf in sorted(CSV_DIR.glob("book*.csv")):
    with open(cf, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            names = [clean_ws(a) for a in
                     re.split(r"\s*[,/&]\s*| and ", row.get("Authors") or "")
                     if a.strip()]
            if names and (row.get("Id") or "").isdigit():
                csv_names[row["Id"]] = names
print(f"  books with author names: {len(csv_names)}")

print("Scanning goodreads_books.json for author_ids ...")
votes = defaultdict(Counter)         # author_id -> {name: votes}
n_seen = n_joined = 0
with open(JSON_PATH) as f:
    for line in f:
        n_seen += 1
        if n_seen % 200000 == 0:
            print(f"  scanned {n_seen:>8d}  joined {n_joined:>8d}")
        m = BID_RE.search(line)
        if not m or m.group(1) not in csv_names:
            continue
        seg = AUTHORS_SEG_RE.search(line)
        if not seg:
            continue
        aids = AID_RE.findall(seg.group(1))
        names = csv_names[m.group(1)]
        if len(aids) != len(names):
            continue                 # ambiguous pairing -> skip
        n_joined += 1
        for aid, name in zip(aids, names):
            votes[aid][name] += 1

with open(OUT_PATH, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["author_id", "name", "votes"])
    for aid in sorted(votes, key=int):
        name, n = votes[aid].most_common(1)[0]
        w.writerow([aid, name, n])
print(f"Books joined:  {n_joined}")
print(f"Author ids named: {len(votes)}")
print(f"Saved: {OUT_PATH}")
