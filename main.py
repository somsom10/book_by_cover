"""Run the whole project: fetch the data, then every pipeline.

    python3 main.py                  # everything, in order
    python3 main.py --check          # what data is present; runs nothing
    python3 main.py --skip-download  # data is already in place
    python3 main.py --only year_genre|comention|themes

Three independent pipelines on the same two datasets:

    year_genre_prediction/   CMU plot summaries -> genre and publication-year
                             models. Needs booksummaries.txt (16 MB).
    comention/               Goodreads blurbs -> author co-mention communities.
                             Needs the two Goodreads dumps (~10 GB).
    themes/                  Goodreads blurbs -> themes by decade, NMF on
                             TF-IDF. Needs the Goodreads archive, the works
                             file and booksummaries.txt.

No pipeline reads another's outputs, so `--only` runs one and fetches just its
inputs. Where two pipelines want the same source file, download/fetch_data.py
gets it once - see its docstring.

themes/ is run through its own orchestrator, starting after its download stage
because the data is already in place by then. Each pipeline decides for itself
what is up to date: comention/main.py skips stages whose outputs are current,
and themes/run_all.py reuses its corpus cache.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FETCH = ROOT / "download" / "fetch_data.py"

PIPELINES = {
    # key: (directory, argv, what it needs fetched)
    "year_genre": ("year_genre_prediction",
                   ["main.py", "--data", "data/booksummaries.txt"], "cmu"),
    "comention": ("comention", ["main.py"], None),
    # --from keyness skips themes' own download stage: fetch_data.py has
    # already put the three files in themes/data/, and running the stage would
    # only re-verify them
    "themes": ("themes", ["run_all.py", "--from", "keyness"], "themes"),
}


def run(argv, cwd, what):
    """Run a step in its own directory; stop the whole thing if it fails."""
    print(f"\n{'=' * 72}\n{what}\n  $ {' '.join(argv)}   (in {cwd.name}/)\n"
          f"{'=' * 72}", flush=True)
    r = subprocess.run([sys.executable, *argv], cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"\n{what} failed (exit {r.returncode}); stopping.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="report what data is present, run nothing")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--only", choices=list(PIPELINES))
    a = ap.parse_args()

    if a.check:
        sys.exit(subprocess.run([sys.executable, str(FETCH), "--check"]).returncode)

    todo = [a.only] if a.only else list(PIPELINES)

    if not a.skip_download:
        # --only fetches just what that pipeline needs; the Goodreads dumps are
        # ~10 GB, so don't pull them to run the CMU half.
        needed = {PIPELINES[k][2] for k in todo}
        if a.only and needed != {None}:
            for what in sorted(x for x in needed if x):
                run([str(FETCH), "--only", what], ROOT, f"fetch data ({what})")
        else:
            run([str(FETCH)], ROOT, "fetch data")

    for key in todo:
        d, argv, _ = PIPELINES[key]
        run(argv, ROOT / d, key)

    print("\nDone. Figures and outputs are in year_genre_prediction/outputs/, "
          "comention/ and themes/work/.")


if __name__ == "__main__":
    main()
