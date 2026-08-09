# Who gets name-dropped together: author co-mention communities in Goodreads blurbs

Book blurbs constantly name-drop other authors — "for fans of X and Y", "in the
tradition of Z", award names, anthology contents. This pipeline extracts those
mentions from ~1.2M Goodreads blurbs, links two authors whenever the same blurb
names both, and clusters the resulting graph. The clusters land on **marketing
scenes, genres and eras** — who publishers compare you to, not literary
influence.

That claim is tested rather than asserted: authors are given genre profiles
from Goodreads reader shelves, which no part of the clustering ever sees, and
two authors in one community turn out to be far more alike than two authors in
a random group of the same size. Current numbers are in `genre_eval.txt`; they
are not repeated here, because prose goes stale and that file is regenerated.

## Data

```
python3 fetch_data.py            # download everything (~2 GB over the wire)
python3 fetch_data.py --check    # what's present, downloads nothing
```

Two dumps that complement each other, keyed by the same Goodreads book id —
which is what makes the author-id join possible.

| Source | Has | Lacks | Where from |
|---|---|---|---|
| `goodreads/book*.csv` (23 files, ~1.85M books) | author **names**, popularity (`RatingDistTotal`) | blurbs | [Kaggle](https://www.kaggle.com/datasets/bahramjannesarr/goodreads-book-datasets-10m) |
| `goodreads_books.json` (~2.36M records, 8.6 GB) | blurbs (`description`), `author_id`s | author names | [UCSD Book Graph](https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/) |

The JSON downloads directly (2.0 GB gzipped). **Kaggle needs an account**, so
`fetch_data.py` uses the `kaggle` CLI when it is configured and prints manual
instructions when it isn't. Downloads resume if interrupted and are checked by
size, so re-running after a failure is safe.

`fetch_data.py` also pulls the CMU Book Summary Dataset into `other/`. The
co-mention pipeline doesn't use it; it belongs to the plot-summary half of the
project, and is here so one script sets up everything.

## Pipeline

```
python3 main.py          # the whole thing, ~2 min once the data is on disk
python3 main.py --dry-run / --force / --only STAGE / --from STAGE
```

A stage runs when an output is missing, when an output is older than an input
or than the code that produced it, or when an earlier stage re-ran — so an
interrupted run resumes, and editing a plotting parameter redraws the figures
without re-streaming the 9 GB JSON. Staleness is by mtime, so a fresh clone may
re-run more than it needs.

The stages, also runnable on their own, in order:

```
build_author_id_map.py    ~15 s   -> author_id_names.csv
build_author_genres.py    ~40 s   -> author_genres.csv
build_blurb_mentions.py   ~55 s   -> blurb_mentions.csv
filter_mentions.py         ~1 s   -> blurb_mentions_clean.csv
comention.py               ~4 s   -> edges, groups, titles
viz_variants.py           ~11 s   -> the two network figures
eval_genres.py             ~2 s   -> the genre evaluation
```

Measured end to end on an NVMe laptop with the 8.6 GB JSON already in page
cache. The three heavy stages each stream that whole file, so on slower storage
they are bound by how fast it can be read, not by the work they do.

`build_author_genres.py` sits second only because `main.py` re-runs everything
downstream of a stage that re-ran; its output is not used until the last stage.

## The decisions that matter

- **Dictionary from names, ids only for self-mentions.** A blurb contains names
  and nothing else, so mentions are matched against a dictionary built from the
  CSVs' `Authors` column, gated at ≥100 ratings and 2–4 tokens. The id map
  exists for one reason: the JSON identifies a book's *own* authors only by id,
  and you need their names to drop self-mentions — 40% of raw mentions.
- **`MIN_EDGE_W = 2`.** A pair co-mentioned once is one over-stuffed blurb.
  Cuts ~85% of pairs, leaves 6,930 authors and 39,315 edges.
- **Leiden at γ = 8**, one flat pass, fixed seed. γ = 1 hits modularity's
  resolution limit and collapses everything literary into 800+-author blobs;
  γ = 8 is the smallest value keeping every community under ~180 while coverage
  stays at 94%. Louvain does not guarantee internally connected communities and
  here emitted a six-author "community" that was three unconnected pairs.
- **The evaluation's yardstick is a random group of the same size** — community
  labels permuted with sizes held fixed, giving both a pooled null and a
  size-matched null per community.

## Outputs

| File | What |
|---|---|
| `blurb_mentions.csv` / `_clean.csv` | one row per (blurb, mentioned author) |
| `comention_author_edges.csv` | weighted edge list, canonical names |
| `author_groups.csv` / `.txt` | author → community, with the community's title |
| `comention_groups_full.png` | the 7 biggest communities, all 1,003 members |
| `comention_all.png` | every author with a tie into a community, two-level layout |
| `author_genres.csv` | per author: books counted, genre distribution |
| `community_genres.csv` | per community: coverage, coherence, z, dominant genres |
| `genre_eval.txt` / `genre_eval.png` | the evaluation, as printed and as drawn |
| `author_group_summaries.md` | hand interpretation of every community — **read back in** as the figures' titles |

## Dependencies

`numpy`, `matplotlib`, `networkx`, `python-igraph`, `leidenalg`
(the last two: `pip install python-igraph leidenalg`).

## Known limitations

- **Award names are people.** "Bram Stoker", "Coretta Scott King" are real
  authors *and* award names; each hubs the community competing for its award.
  Unfixable without dropping real authors.
- **Name collisions.** "John Williams" (composer vs novelist), "David
  Copperfield" and "John Carter" (characters matched as people).
- **Only full names are matched.** The 2–4 token rule means a blurb saying
  "for fans of Chandler" contributes nothing; mononyms can never be matched.
- **The genre yardstick is itself marketing.** Reader shelves are shaped by how
  a book was sold — the same process that writes the blurbs. The evaluation
  shows two marketing signals agree, which is weaker than showing the
  communities are "really" genres. It remains a fair test of the *clustering*,
  since no shelf ever enters it.
- **Genre coverage is ~77%** of graph authors and not missing at random: it
  favours the recent and the popular. Communities are scored on covered members
  only; `community_genres.csv` reports the covered count per group.
- **The vocabulary includes nationality, era and language** alongside genre, so
  for communities organised by country of origin the measure partly confirms
  what binds them rather than testing it independently. Single-genre-dominant
  fields (romance, comics) also score above genuinely mixed ones (literary
  fiction), which is a property of the measure.
- **Reproducible is not stable.** Output is byte-identical for a fixed input,
  but collapsing one duplicate node can move boundary authors between adjacent
  communities. Don't over-read assignments at the borders.

Pre-2026-07 outputs (before the self-mention drop and the Leiden switch) are
archived in `other/pre_self_drop/`.
