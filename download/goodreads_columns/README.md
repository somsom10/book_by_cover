# goodreads_columns — the three columns the pipeline actually reads

These 23 files are a distilled copy of the Kaggle dump
[Goodreads Book Datasets With User Rating 2M](https://www.kaggle.com/datasets/bahramjannesarr/goodreads-book-datasets-10m)
by Bahram Jannesar, reduced to the only columns `comention/` ever touches:

    Id, Authors, RatingDistTotal

`build_author_id_map.py` reads `Id` and `Authors`; `build_blurb_mentions.py`
reads `Authors` and `RatingDistTotal` (the `>= 100 ratings` gate). Both use
`csv.DictReader` with named lookups, so the other 15 columns — `Name`,
`pagesNumber`, `Publisher`, `ISBN`, the per-star `RatingDist1..5` and the rest —
are never referenced. `year_genre_prediction/` does not read these files at all.

**Why they are committed.** The Kaggle original needs an account, which is the
one thing that stopped `main.py` being end-to-end. Three columns gzipped is
19.8 MB against 1,110.7 MB for the full dump — 56x smaller, small enough to
live in the repo, so a fresh clone runs without a login.

**Results are unchanged.** Both pipeline inputs were rebuilt from the raw CSVs
and from these files and compared as data structures, not counts:

    raw       : 1,850,115 id->names, 56,134 dictionary entries
    distilled : 1,850,115 id->names, 56,134 dictionary entries
    identical : True (both)

`fetch_data.py` unpacks these into `comention/goodreads/` as plain `.csv`, so
the pipeline code reads exactly what it always read. If you would rather use the
full original, put the real `book*.csv` in `comention/goodreads/` and
`fetch_data.py` will leave them alone.

Source data is scraped public Goodreads metadata; redistributed here as a
minimal derivative for coursework, with attribution above.
