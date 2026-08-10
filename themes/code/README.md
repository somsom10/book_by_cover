# code/

Every script here is run from `work/` with `code/` on the import path — which is
exactly what `run_all.py` does, so the normal way to run any of this is

```bash
python run_all.py              # all six stages
python run_all.py --only figures
```

To run one script by hand:

```bash
cd work
PYTHONPATH=../code python ../code/writeup_figures.py
```

Raw data is read from `../data/`, everything written lands in `work/`.

## The pipeline, in the order it has to run

| # | script | reads | writes |
|---|---|---|---|
| 1 | `download_data.py` | the internet | `data/goodreads_books.json.gz`, `data/goodreads_book_works.json.gz`, `data/booksummaries.txt` |
| 2 | `keyness.py` | both raw corpora | `keyness_matched.pkl`, `keyness_word_weights.csv`, `keyness_goodreads_vs_cmu.csv` |
| 3 | `build_bounded_corpus.py` | raw Goodreads + the weights from 2 | `themes_corpus_bounded.pkl` |
| 4 | `themes.py` | the corpus from 3 | `topic_*.csv`, `decade_digest.csv`, `artifact_share_by_decade.csv`, `topic_trends.pdf` |
| 5 | `writeup_figures.py` | `final_refit/` | `wfig1_era_map`, `wfig2_headline`, `wfig3_rise_fall` |
| 6 | `trends_report.py` | `final_refit/` (+ `topic_stability.csv` if present) | `topic_trends_v2.pdf` |

Step 2 is the one place `run_all.py` calls a function rather than a script:
`keyness.export_word_weights()`, straight after `keyness.py`.

`run_all.py` copies step 4's CSVs into `work/final_refit/` before step 5.
Everything downstream reads `final_refit/`, never the loose CSVs, so a new fit
cannot half-overwrite a published one.

**Step 2 must precede step 3.** `bounding.py` reads `keyness_word_weights.csv`;
without it the `register_dense` rule never fires, the corpus comes out different,
and nothing raises an error.

Among the checks, `stability_curves.py` runs first: `stability.py` and
`sf_stability.py` both read `all_topic_stability.csv`.

## What each file is

**The pipeline**

- `download_data.py` — fetches the three raw datasets, verifies byte counts,
  resumes partial downloads. `--check` reports what is present.
- `text.py` — data loading and spaCy lemmatisation. Holds the paths to the raw
  files and the English/language filters.
- `bounding.py` — trims jacket copy from the edges of a blurb: bestseller
  prefaces, "About the Author", review quotes, edition statements. Standalone
  and importable; `bound_summary()` returns the text plus a trace of which rule
  fired where.
- `keyness.py` — matches Goodreads blurbs to CMU plot summaries of the same book
  and measures which words mark publisher register rather than plot. Produces
  the weights `bounding.py` uses.
- `build_bounded_corpus.py` — raw → clean → bound → lemmatise → the corpus
  pickle. Takes an optional document count for a smoke run.
- `themes.py` — the model. TF-IDF, NMF with K=25, artifact-topic exclusion,
  renormalisation, per-decade shares and lift with bootstrap intervals.

**Figures and the report**

- `writeup_figures.py` — the era map (`wfig1`), the four-panel figure (`wfig2`)
  and rise/fall (`wfig3`), as standalone png and pdf.
- `trends_report.py` — the 6-page long-form version, `topic_trends_v2.pdf`.
  (`topic_trends.pdf`, also 6 pages, is `themes.py`'s own diagnostic plot.)

**Checks**

- `stability_curves.py` — refits on four independent training samples and reports
  both word-list cosine and decade-curve correlation per topic.
- `stability.py` — the same idea reported as tiers (solid / moderate / unstable).
- `sf_stability.py` — the same four refits reported per named theme, giving both
  the mean and the worst case. Science fiction scores `curve_r_mean` 0.49 and
  −0.39 at worst, the clearest example of a topic that does not reproduce.
- `evaluate_bounding.py` — does bounding move a blurb toward the plot summary of
  the same book, and which rules earn their place.
- `roc_filtering.py` — the AUC 0.986 agreement between the rule-based filter and
  the word-based register score.
- `raw_vs_renorm.py` — every trend before and after renormalisation, so the bias
  is visible rather than argued about.

Comments and docstrings are in Hebrew.
