"""
The whole pipeline, from downloading the data to the figures and the checks.

  python run_all.py                # everything, from scratch
  python run_all.py --list         # the stages and what each one produces
  python run_all.py --from model   # resume from a given stage
  python run_all.py --only figures

Code lives in code/, raw data in data/, everything generated in work/.

STAGE ORDER IS NOT ARBITRARY. keyness must run before the corpus is built,
because bounding.py reads keyness_word_weights.csv - without it one trimming
rule does not fire, with no error, and the corpus comes out different.
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE, DATA, WORK = (os.path.join(ROOT, d) for d in ("code", "data", "work"))

# Files themes.py writes to the current directory, which the figures and
# checks then read from final_refit/
REFIT_FILES = ["topic_shares_by_decade.csv", "topic_labels.csv",
               "topic_lift_by_decade.csv", "topic_prevalence_by_decade.csv",
               "decade_digest.csv", "artifact_share_by_decade.csv"]


def sh(*args, **kw):
    env = dict(os.environ, PYTHONPATH=CODE)
    r = subprocess.run(args, cwd=WORK, env=env, **kw)
    if r.returncode:
        raise SystemExit(f"stage failed: {' '.join(args)}")


def py(script, *args):
    sh(sys.executable, os.path.join(CODE, script), *args)


def snap_refit():
    """Copy the fit outputs into final_refit/."""
    dest = os.path.join(WORK, "final_refit")
    os.makedirs(dest, exist_ok=True)
    for f in REFIT_FILES:
        src = os.path.join(WORK, f)
        if not os.path.exists(src):
            raise SystemExit(f"themes.py did not produce {f}")
        shutil.copy2(src, os.path.join(dest, f))
    print(f"  copied {len(REFIT_FILES)} files into work/final_refit/")


def stage_download():
    py("download_data.py")


def stage_keyness():
    # match Goodreads to CMU, then the register weight table bounding needs
    py("keyness.py")
    sh(sys.executable, "-c",
       "import keyness; keyness.export_word_weights()")


def stage_corpus():
    py("build_bounded_corpus.py")


def stage_model():
    py("themes.py")
    snap_refit()


def stage_figures():
    py("writeup_figures.py")
    py("trends_report.py")


def stage_checks():
    # stability_curves first: the two after it read all_topic_stability.csv
    py("stability_curves.py")
    py("stability.py")
    py("sf_stability.py")
    py("evaluate_bounding.py")
    py("roc_filtering.py")
    py("raw_vs_renorm.py")


STAGES = [
    ("download", stage_download, "data/*.json.gz, data/booksummaries.txt"),
    ("keyness", stage_keyness, "keyness_matched.pkl, keyness_word_weights.csv"),
    ("corpus", stage_corpus, "themes_corpus_bounded.pkl"),
    ("model", stage_model, "final_refit/*.csv"),
    ("figures", stage_figures, "wfig1-wfig3, topic_trends_v2.pdf"),
    ("checks", stage_checks, "stability, bounding and renormalisation checks"),
]


def main():
    names = [n for n, _, _ in STAGES]
    if "--list" in sys.argv:
        for n, _, produces in STAGES:
            print(f"  {n:9} -> {produces}")
        return
    todo = names
    if "--from" in sys.argv:
        todo = names[names.index(sys.argv[sys.argv.index("--from") + 1]):]
    if "--only" in sys.argv:
        todo = [sys.argv[sys.argv.index("--only") + 1]]

    os.makedirs(WORK, exist_ok=True)
    for name, fn, _ in STAGES:
        if name not in todo:
            continue
        print(f"\n{'=' * 70}\n== {name}\n{'=' * 70}")
        t = time.time()
        fn()
        print(f"-- {name} finished in {time.time() - t:.0f}s")
    print("\nall requested stages completed")


if __name__ == "__main__":
    main()
