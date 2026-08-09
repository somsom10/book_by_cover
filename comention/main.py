"""Run the pipeline: raw Goodreads dumps -> groups, figures, evaluation.

    python3 main.py              # run whatever is missing or out of date
    python3 main.py --dry-run    # print the plan
    python3 main.py --force      # ignore staleness
    python3 main.py --only S / --from S

A stage runs when an output is missing, an output is older than an input or the
code that produced it, or an earlier stage re-ran. Staleness is by mtime, so a
fresh clone may re-run more than it needs."""
import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent


@dataclass
class Stage:
    key: str
    script: str
    blurb: str
    outputs: list           # written by the stage, relative to ROOT
    data: list = field(default_factory=list)    # raw inputs (may be absent)
    inputs: list = field(default_factory=list)  # outputs of earlier stages
    code: list = field(default_factory=list)    # modules it imports
    minutes: str = ""

    def paths(self, patterns):
        """Resolve names/globs against ROOT, keeping globs that match nothing
        visible as an empty result rather than a phantom path."""
        out = []
        for p in patterns:
            out.extend(sorted(ROOT.glob(p)) if "*" in p else [ROOT / p])
        return out


STAGES = [
    Stage("authors", "build_author_id_map.py",
          "name the author_ids (joins the CSV dump to the JSON dump)",
          outputs=["author_id_names.csv"],
          data=["goodreads/book*.csv", "goodreads_books.json"],
          code=["name_filters.py"], minutes="~5 min"),
    # Second, not last, although the evaluation it feeds is the last thing that
    # happens: a stage re-runs when anything before it re-ran, so parking this
    # one after the figures would put a 7-minute pass over the 9 GB JSON behind
    # every tweak to a plotting parameter. Its only input is the id map above.
    Stage("genres", "build_author_genres.py",
          "genre-profile every author from the Goodreads shelves",
          outputs=["author_genres.csv"],
          data=["goodreads_books.json"],
          inputs=["author_id_names.csv"],
          code=["genre_vocab.py"], minutes="~7 min"),
    Stage("mentions", "build_blurb_mentions.py",
          "extract author mentions from every blurb",
          outputs=["blurb_mentions.csv"],
          data=["goodreads/book*.csv", "goodreads_books.json"],
          inputs=["author_id_names.csv"],
          code=["name_filters.py"], minutes="~15 min"),
    Stage("filter", "filter_mentions.py",
          "drop organisations and non-literary figures",
          outputs=["blurb_mentions_clean.csv"],
          inputs=["blurb_mentions.csv"],
          code=["name_filters.py"], minutes="seconds"),
    Stage("cluster", "comention.py",
          "canonicalize, build the graph, cluster",
          outputs=["comention_author_edges.csv", "author_groups.csv",
                   "author_groups.txt"],
          inputs=["blurb_mentions_clean.csv"],
          # the summaries file supplies the group titles, so editing a title
          # is a reason to rewrite the group files
          code=["name_filters.py", "author_group_summaries.md"],
          minutes="~1 min"),
    Stage("variants", "viz_variants.py",
          "draw the two network figures (the 7 big communities, everything)",
          outputs=["comention_groups_full.png", "comention_all.png"],
          # reads the previous stage's CSVs, never the raw dumps
          inputs=["comention_author_edges.csv", "author_groups.csv"],
          code=["viz_common.py"],
          minutes="~1 min"),
    Stage("evaluate", "eval_genres.py",
          "test the communities against the genre profiles",
          outputs=["community_genres.csv", "genre_eval.txt", "genre_eval.png"],
          inputs=["comention_author_edges.csv", "author_groups.csv",
                  "author_genres.csv"],
          code=["author_ident.py", "genre_vocab.py", "viz_common.py"],
          minutes="seconds"),
]
BY_KEY = {s.key: s for s in STAGES}


def decide(stage, upstream_ran, force):
    """(run?, reason). Reason is shown in the plan, so keep it concrete."""
    if force:
        return True, "forced"
    outs = stage.paths(stage.outputs)
    gone = [p for p in outs if not p.exists()]
    if gone:
        return True, f"missing {gone[0].name}"
    if upstream_ran:
        return True, "an earlier stage re-ran"
    oldest = min(p.stat().st_mtime for p in outs)
    # the stage's own script counts as an input: editing it is the most common
    # reason to want the outputs rebuilt
    for src in stage.paths([stage.script] + stage.data + stage.inputs
                           + stage.code):
        if src.exists() and src.stat().st_mtime > oldest:
            return True, f"{src.name} is newer"
    return False, "up to date"


def missing_data(stage):
    """Raw inputs the repo does not ship, absent from disk."""
    return [p for p in stage.data
            if ("*" in p and not sorted(ROOT.glob(p)))
            or ("*" not in p and not (ROOT / p).exists())]


def run(stage):
    # flush: the child writes straight to the terminal, so unflushed
    # parent prints would land out of order in a piped log
    print(f"\n\033[1m=== {stage.key}: {stage.script}\033[0m  "
          f"({stage.blurb})", flush=True)
    t0 = time.time()
    code = subprocess.call([sys.executable, stage.script], cwd=ROOT)
    dt = time.time() - t0
    if code != 0:
        print(f"\n!! {stage.script} exited {code} after {dt:.0f}s -- stopping "
              f"here; later stages would read a half-written file.")
        sys.exit(code)
    print(f"--- {stage.key} done in {dt:.0f}s", flush=True)
    return dt


def main():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="stages: " + ", ".join(f"{s.key} ({s.minutes})"
                                      for s in STAGES))
    p.add_argument("--force", action="store_true",
                   help="re-run every selected stage, fresh or not")
    p.add_argument("--only", metavar="STAGE", choices=list(BY_KEY),
                   help="run just this stage")
    p.add_argument("--from", dest="start", metavar="STAGE",
                   choices=list(BY_KEY),
                   help="start at this stage instead of the first")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and exit")
    a = p.parse_args()

    if a.only:
        selected = [BY_KEY[a.only]]
        a.force = True          # asking for one stage by name means run it
    else:
        keys = [s.key for s in STAGES]
        selected = STAGES[keys.index(a.start) if a.start else 0:]

    plan, upstream_ran, coming = [], False, set()
    for s in selected:
        go, why = decide(s, upstream_ran, a.force)
        if go:
            # a stage can't run without its raw dumps, or without an input no
            # earlier stage in this plan is going to write
            absent = missing_data(s) or [
                i for i in s.inputs
                if not (ROOT / i).exists() and i not in coming]
            if absent:
                go, why = False, f"needs {absent[0]}, not on disk"
        plan.append((s, go, why))
        upstream_ran = upstream_ran or go
        if go:
            coming.update(s.outputs)

    print("Pipeline plan:")
    for s, go, why in plan:
        mark = "run " if go else "skip"
        print(f"  [{mark}] {s.key:<9} {s.script:<24} {why}"
              f"{'  (' + s.minutes + ')' if go else ''}")
    todo = [s for s, go, _ in plan if go]
    sys.stdout.flush()
    if a.dry_run or not todo:
        print("\nNothing to do." if not todo else "\nDry run, stopping.")
        return

    t0 = time.time()
    for s in todo:
        run(s)
    print(f"\nPipeline finished in {time.time() - t0:.0f}s. "
          f"Figures: comention_groups_full.png, comention_all.png, "
          f"genre_eval.png (see README)")


if __name__ == "__main__":
    main()
