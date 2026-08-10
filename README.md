# Judging a Book by Its Cover

What a book's own words say about it. Three pipelines on two datasets:

```
year_genre_prediction/   CMU plot summaries -> genre and publication-year models
comention/               Goodreads blurbs   -> author co-mention communities
themes/                  Goodreads blurbs   -> themes by decade (NMF on TF-IDF)
download/                fetch_data.py -- pulls the data all three of them need
```

## Run it

```
python3 main.py                  # fetch the data, then run all three pipelines
python3 main.py --check          # what data is present; runs nothing
python3 main.py --only year_genre     # one pipeline, and only its data
python3 main.py --only comention
python3 main.py --only themes
python3 main.py --skip-download       # data already in place
```

`--only` fetches just what that pipeline needs, which matters: the CMU dataset
is 16 MB and the Goodreads dumps are ~10 GB.

Each pipeline can also be run on its own from inside its directory; see the
three READMEs linked at the bottom. `themes/` is the one that keeps its own
orchestrator, `themes/run_all.py`, and `main.py` enters it at the `keyness`
stage because the data is already in place by then.

## Data

| Dataset | For | Size | Source |
|---|---|---|---|
| `booksummaries.txt` | year_genre_prediction, themes | 41 MB | [CMU Book Summary Dataset](https://www.cs.cmu.edu/~dbamman/booksummaries.html) |
| `goodreads_books.json.gz` | themes, and comention expands from it | 2.0 GB | [UCSD Book Graph](https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/) |
| `goodreads_books.json` | comention | 8.6 GB | expanded from the archive above |
| `goodreads_book_works.json.gz` | themes | 75 MB | same UCSD source — first-publication years |
| `goodreads/book*.csv` | comention | 20 MB | in this repo, see below |

**Nothing is fetched twice.** Two pipelines want the same Goodreads dump in
different forms: `themes/` reads the compressed archive, `comention/` reads it
expanded. So the download lands in `themes/data/` and `comention/` expands its
copy from there. `booksummaries.txt` is downloaded once into
`year_genre_prediction/data/` and hard-linked into `themes/data/`, so the two
paths are one file on disk. Running all three costs one 2.0 GB transfer, not
two.

The downloads need no account. The CSVs are **already in the repo**:
`download/goodreads_columns/` holds the three columns the pipeline actually
reads, gzipped — 19.8 MB against 1.1 GB for the full Kaggle dump,
which needs a login. `fetch_data.py` unpacks them, so a fresh clone runs
end-to-end with no sign-up anywhere. Results are unchanged; the equivalence is
verified in that folder's README.

Downloads resume if interrupted and are verified by size, so re-running after a
failure is safe.

## Dependencies

```
pip install -r year_genre_prediction/requirements.txt   # pandas, scikit-learn, seaborn, ...
pip install python-igraph leidenalg                     # comention only
pip install -r themes/requirements.txt                  # themes only; adds spaCy
```

`themes/` needs the spaCy English model as well as the library; its
requirements file pins both, so the one `pip install` covers it.

On a distro with an externally-managed Python (PEP 668), use a virtualenv.

## What is not in this repo

The datasets (~11 GB), the trained models and figures under
`year_genre_prediction/outputs/`, three large regenerable intermediates in
`comention/`, and everything `themes/` generates under `themes/work/`.
`fetch_data.py` restores the data; the pipelines rebuild the rest.

Everything else is committed — including `comention/`'s edge list, communities,
evaluation and all three figures, so those results are readable without running
anything. See [comention/README.md](comention/README.md),
[year_genre_prediction/README.md](year_genre_prediction/README.md) and
[themes/README.md](themes/README.md).
