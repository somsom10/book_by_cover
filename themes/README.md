# Judging a decade by its covers — code for the writeup

Themes in a century of Goodreads book descriptions, found with NMF on TF-IDF and
tracked by decade. 

The reported decade curves rest on refitting the model on four independent 
training samples and requiring each curve to reproduce. Separately, 
`roc_filtering.py` validates the text filter that feeds
the model (AUC 0.986, also quoted in the writeup).

Source comments and docstrings are in Hebrew.

## Layout — code, inputs, and outputs are kept apart

```
run_all.py          the whole pipeline, one stage at a time
requirements.txt
code/               the code. 15 scripts and their own README, nothing generated
data/               the three raw datasets, downloaded by code/download_data.py
work/               everything the code produces — caches, CSVs, figures, PDFs
writeup/            the document, and the two figures it embeds
```

Nothing in `code/` is generated and nothing in `work/` is written by hand. On a
clean machine `data/` and `work/` both start empty.

## Running it on a machine that has never seen this project

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

`run_all.py` downloads the datasets, builds the corpus, fits the model, 
draws the figures and runs every validation check, writing all of
it into `work/`.

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
| `corpus` | `themes_corpus_bounded.pkl`, `cmu_corpus.pkl` | ~19 min — spaCy over ~73k blurbs |
| `model` | `final_refit/*.csv` | ~25 s |
| `figures` | `wfig1`–`wfig3` (png + pdf), `topic_trends_v2.pdf` | seconds |
| `checks` | the validation numbers quoted in the writeup | ~7 min — four NMF refits, three times over |

**Stage order is not cosmetic.** `keyness` has to precede `corpus`, because
`bounding.py` reads `keyness_word_weights.csv`; without it one of the trimming
rules silently does not fire and the corpus comes out different, with no error.
`run_all.py` enforces the order, and `themes.py` refuses to run if the bounded
corpus is missing rather than quietly rebuilding an unbounded one under the same
name.

One ordering inside a stage matters for the same reason: `verify_writeup.py`
runs last, after `stability_curves.py` has written `all_topic_stability.csv` —
run it earlier and it skips its four stability checks and reports 32/32 as
though nothing were missing.

## Where the data comes from

| file | size | source |
|---|---|---|
| `goodreads_books.json.gz` | 2.1 GB | UCSD McAuley lab, `mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/` |
| `goodreads_book_works.json.gz` | 75 MB | same — the works file, which is what dates a book by *first* publication rather than by the reprint in hand |
| `booksummaries.txt` | 43 MB | CMU Book Summaries (Bamman), `cs.cmu.edu/~dbamman/data/booksummaries.tar.gz` |

`download_data.py` checks the byte count of each file and resumes a partial
download instead of restarting it. `python code/download_data.py --check`
reports what is present without downloading anything.

## Which script backs which claim

| claim in the writeup | script | output |
|---|---|---|
| 2.36M editions joined to the works file, dated by first publication | `themes.py` (`load_goodreads_full`) | — |
| one edition per work, English only, 10,000/decade cap, 68,961 descriptions | `build_bounded_corpus.py` | `themes_corpus_bounded.pkl` |
| jacket copy trimmed from the edges of each blurb | `bounding.py` | — |
| that filter validated against 8,680 matched books, **AUC 0.986** | `roc_filtering.py` | `filtering_roc.pdf` |
| 25 topics, shares and lift per decade, bootstrap intervals | `themes.py` | `final_refit/*.csv` |
| **Figure 1** era map, **Figure 2** four panels, peak spans | `writeup_figures.py` | `wfig1`, `wfig2` |
| early intervals **2.3x** the width of late ones | `verify_writeup.py` | recomputed from `topic_lift_by_decade.csv` |
| adventure **-2.2pp**, translation -1.9, collections -1.3, poetry -0.7 | `verify_writeup.py` | mean(1900-1940) - mean(1960-2000), 1950 and 2010 excluded |
| every number in the writeup, re-derived and compared | `verify_writeup.py` | 36 checks, non-zero exit on drift |
| refits reproduce every reported curve, **r = 0.85–0.999** | `stability_curves.py` | `all_topic_stability.csv` |
| word lists re-form: solid / moderate / unstable tiers | `stability.py` | `topic_stability.csv` |
| sci-fi and fantasy, **r = 0.49 mean, −0.39 worst** | `sf_stability.py` | `sf_stability.csv` |
| marketing 10.6% of 1900s text vs 5.9% of 2010s | `themes.py` | `final_refit/artifact_share_by_decade.csv` |
| renormalisation inflates early decades, so only 10 of 23 slopes are claimable while peaks survive | `raw_vs_renorm.py` | `trends_raw_vs_renorm.csv` |

`verify_writeup.py` is the fastest way to see that the document and the data
still agree: it recomputes every share, lift, interval and count the writeup
quotes and fails loudly if one has drifted.

## Notes:

**Multi-word terms are joined at load time, not in the cache.** `world_war` and
`science_fiction` do not exist inside `themes_corpus_bounded.pkl`; `build_corpus`
creates them on the way out. Anything reading the pickle directly must call
`themes.join_multiword`, or those terms silently read 0.0 in every decade.
Additionally, the exploratory work for hyperparameter calibration is not included.