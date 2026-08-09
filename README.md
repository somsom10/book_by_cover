# Judging a Book by Its Cover

What a book's own words say about it. Two pipelines on two datasets:

```
year_genre_prediction/   CMU plot summaries -> genre and publication-year models
comention/               Goodreads blurbs   -> author co-mention communities
download/                fetch_data.py -- pulls the data both of them need
```

## Run it

```
python3 main.py                  # fetch the data, then run both pipelines
python3 main.py --check          # what data is present; runs nothing
python3 main.py --only year_genre     # one pipeline, and only its data
python3 main.py --only comention
python3 main.py --skip-download       # data already in place
```

`--only` fetches just what that pipeline needs, which matters: the CMU dataset
is 16 MB and the Goodreads dumps are ~10 GB.

## Data

| Dataset | For | Size | Source |
|---|---|---|---|
| `booksummaries.txt` | year_genre_prediction | 41 MB | [CMU Book Summary Dataset](https://www.cs.cmu.edu/~dbamman/booksummaries.html) |
| `goodreads_books.json` | comention | 8.6 GB | [UCSD Book Graph](https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/) |
| `goodreads/book*.csv` | comention | 1.2 GB | [Kaggle](https://www.kaggle.com/datasets/bahramjannesarr/goodreads-book-datasets-10m) |

The first two download directly. **The Kaggle dump needs an account** —
`fetch_data.py` uses the `kaggle` CLI when configured and prints setup
instructions when it isn't. Downloads resume if interrupted and are verified by
size, so re-running after a failure is safe.

## Dependencies

```
pip install -r year_genre_prediction/requirements.txt   # pandas, scikit-learn, seaborn, ...
pip install python-igraph leidenalg                     # comention only
```

On a distro with an externally-managed Python (PEP 668), use a virtualenv.

## What is not in this repo

All three datasets (~10 GB), the trained models and figures under
`year_genre_prediction/outputs/`, and three large regenerable intermediates in
`comention/`. `fetch_data.py` restores the data; the pipelines rebuild the rest.

Everything else is committed — including `comention/`'s edge list, communities,
evaluation and all three figures, so those results are readable without running
anything. See [comention/README.md](comention/README.md) and
[year_genre_prediction/README.md](year_genre_prediction/README.md).
