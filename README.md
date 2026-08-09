# Judging a Book by Its Cover — author co-mention communities

Book blurbs constantly name-drop other authors — "for fans of X and Y", "in the
tradition of Z", award names, anthology contents. This repo extracts those
mentions from ~1.2M Goodreads blurbs, links two authors whenever the same blurb
names both, clusters the resulting graph, and tests the communities against a
signal the clustering never sees.

```
download/    fetch_data.py — pulls the two Goodreads dumps (~2 GB over the wire)
comention/   the pipeline, its outputs and figures
```

## Run it

```
python3 download/fetch_data.py     # data (~10 GB on disk once unpacked)
python3 comention/main.py          # the whole pipeline, ~28 min from scratch
```

`fetch_data.py --check` reports what is present without downloading anything.
See [comention/README.md](comention/README.md) for the pipeline itself: the
stages, the decisions behind them, the outputs and the known limitations.

## What is not in this repo

The raw dumps (~10 GB) and three large regenerable intermediates
(`author_genres.csv`, `blurb_mentions.csv`, `blurb_mentions_clean.csv`) are
gitignored. `fetch_data.py` restores the data; `main.py` rebuilds the rest.
Everything else — the edge list, the communities, the evaluation and all three
figures — is committed, so the results are readable without running anything.

**The Goodreads CSV dump is on Kaggle and needs an account.** The JSON dump
downloads directly. `fetch_data.py` uses the `kaggle` CLI when configured and
prints setup instructions when it isn't.
