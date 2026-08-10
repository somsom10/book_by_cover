# Themes by decade

Themes in a century of Goodreads book descriptions, found with NMF on TF-IDF and
tracked by decade.

The reported decade curves rest on refitting the model on four independent
training samples and requiring each curve to reproduce. Separately,
`roc_filtering.py` validates the text filter that feeds the model (AUC 0.986).

## Layout — code, inputs, and outputs are kept apart

```
run_all.py          the whole pipeline, one stage at a time
requirements.txt
code/               the code. 14 scripts and their own README, nothing generated
data/               the three raw datasets
work/               everything the code produces — caches, CSVs, figures, PDFs
```

Nothing in `code/` is generated and nothing in `work/` is written by hand. On a
clean checkout `data/` and `work/` both start empty.

## Running it

From the repository root, as one of the three pipelines:

```bash
python3 main.py --only themes     # fetch this pipeline's data, then run it
```

Or standalone, from this directory:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

Both routes end in the same place. The difference is only where the data comes
from: `run_all.py`'s own `download` stage fetches into `data/`, while
`main.py` fetches through `download/fetch_data.py` and then starts this
pipeline at `keyness`. Either way nothing is downloaded twice — both writers
check byte counts first and skip files already in `data/`.

Individual stages, for when you do not want to sit through the download:

```bash
python run_all.py --list           # the six stages and what each produces
python run_all.py --only figures   # just redraw, needs only work/final_refit/
python run_all.py --from model     # skip download and corpus building
```

| stage | produces | cost |
|---|---|---|
| `download` | `data/goodreads_books.json.gz`, `..._works.json.gz`, `booksummaries.txt` | 2.2 GB, resumable |
| `keyness` | `keyness_matched.pkl`, `keyness_word_weights.csv` | minutes |
| `corpus` | `themes_corpus_bounded.pkl` | ~19 min — spaCy over ~73k blurbs |
| `model` | `final_refit/*.csv` | ~25 s |
| `figures` | `wfig1`–`wfig3` (png + pdf), `topic_trends_v2.pdf` | seconds |
| `checks` | stability, bounding and renormalisation checks | ~6 min — four NMF refits, three times over |

**Stage order is not cosmetic.** `keyness` has to precede `corpus`, because
`bounding.py` reads `keyness_word_weights.csv`; without it one of the trimming
rules silently does not fire and the corpus comes out different, with no error.
`run_all.py` enforces the order, and `themes.py` refuses to run if the bounded
corpus is missing rather than quietly rebuilding an unbounded one under the same
name.

Inside `checks`, `stability_curves.py` runs first because the two scripts after
it read `all_topic_stability.csv`.

## Where the data comes from

| file | size | source |
|---|---|---|
| `goodreads_books.json.gz` | 2.1 GB | UCSD McAuley lab, `mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/` |
| `goodreads_book_works.json.gz` | 75 MB | same — the works file, which is what dates a book by *first* publication rather than by the reprint in hand |
| `booksummaries.txt` | 43 MB | CMU Book Summaries (Bamman), `cs.cmu.edu/~dbamman/data/booksummaries.tar.gz` |

Two of these are shared with the rest of the repository, and neither is fetched
twice. `comention/` reads the Goodreads archive expanded rather than compressed,
so the download lands here in `data/` and `comention/` expands its copy from it.
`booksummaries.txt` is downloaded once for `year_genre_prediction/` and
hard-linked here, so the two paths are one file on disk.

`python code/download_data.py --check` reports what is present without
downloading anything; `python3 download/fetch_data.py --check` from the
repository root does the same for all three pipelines at once.

## Which script produces which result

| result | script | output |
|---|---|---|
| 2.36M editions joined to the works file, dated by first publication | `themes.py` (`load_goodreads_full`) | — |
| one edition per work, English only, 10,000/decade cap, 68,961 descriptions | `build_bounded_corpus.py` | `themes_corpus_bounded.pkl` |
| jacket copy trimmed from the edges of each blurb | `bounding.py` | — |
| that filter validated against 8,680 matched books, **AUC 0.986** | `roc_filtering.py` | `filtering_roc.pdf` |
| 25 topics, shares and lift per decade, bootstrap intervals | `themes.py` | `final_refit/*.csv` |
| era map, four-panel figure, rise/fall | `writeup_figures.py` | `wfig1`–`wfig3` |
| the long-form report | `trends_report.py` | `topic_trends_v2.pdf` |
| refits reproduce every reported curve, **r = 0.85–0.999** | `stability_curves.py` | `all_topic_stability.csv` |
| word lists re-form: solid / moderate / unstable tiers | `stability.py` | `topic_stability.csv` |
| sci-fi and fantasy, **r = 0.49 mean, −0.39 worst** | `sf_stability.py` | `sf_stability.csv` |
| how much of each removed span was really plot | `evaluate_bounding.py` | `bounding_removed_log.csv`, `bounding_rule_stats.csv` |
| marketing 10.6% of 1900s text vs 5.9% of 2010s | `themes.py` | `final_refit/artifact_share_by_decade.csv` |
| renormalisation inflates early decades, so only 10 of 23 slopes are claimable while peaks survive | `raw_vs_renorm.py` | `trends_raw_vs_renorm.csv` |

## Notes

**Multi-word terms are joined at load time, not in the cache.** `world_war` and
`science_fiction` do not exist inside `themes_corpus_bounded.pkl`; `build_corpus`
creates them on the way out. Anything reading the pickle directly must call
`themes.join_multiword`, or those terms silently read 0.0 in every decade.

The exploratory work for hyperparameter calibration is not included.
