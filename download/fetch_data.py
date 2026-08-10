"""Download the raw data the pipeline runs on.

    python3 fetch_data.py             # everything still missing
    python3 fetch_data.py --check     # report what is present, download nothing
    python3 fetch_data.py --only json|csv|cmu|themes

Four things the repo needs, from three sources:

  goodreads_books.json  UCSD Book Graph -- blurbs and author_ids, for
                        comention/. Direct download, 2.0 GB -> 8.6 GB on disk.
  goodreads/*.csv       author names and popularity, for comention/. Shipped
                        in download/goodreads_columns/ as the three columns the
                        pipeline reads (19.8 MB gzipped, vs 1.1 GB for the full
                        Kaggle dump, which needs an account) -- so this is
                        unpacked, not downloaded. See that folder's README.
  booksummaries.txt     CMU Book Summary Dataset -- plot summaries with genre
                        and publication date, for year_genre_prediction/.
                        Direct download, 16 MB.
  themes/data/          the same two sources again, in the forms themes/ reads:
                        the Goodreads archive uncompressed, the works file, and
                        booksummaries.txt.

Nothing is fetched twice. themes/ reads goodreads_books.json.gz as an archive
while comention/ reads it expanded, so the download lands in themes/data/ and
comention/ expands from there. booksummaries.txt is downloaded once for
year_genre_prediction/ and hard-linked into themes/data/, so the two paths are
one file on disk.

Downloads resume if interrupted and are verified by size before use, so
re-running after a failure is safe.
"""
import argparse
import gzip
import os
import shutil
import subprocess
import tarfile
import sys
import urllib.request
from pathlib import Path

# This script lives in download/; each dataset goes next to the code it feeds.
REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "comention"
YGP = REPO / "year_genre_prediction"
THEMES = REPO / "themes"
THEMES_DATA = THEMES / "data"

JSON_URL = ("https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"
            "goodreads_books.json.gz")
JSON_GZ_BYTES = 2_083_197_934          # verified against the server
JSON_BYTES = 9_202_235_168             # after gunzip

KAGGLE_SLUG = "bahramjannesarr/goodreads-book-datasets-10m"
BUNDLED = Path(__file__).resolve().parent / "goodreads_columns"
CSV_DIR = ROOT / "goodreads"
CSV_MIN_FILES = 23                     # the dump ships 23 book*.csv (plus 7
                                       # user_rating_*.csv the pipeline never reads)

CMU_URL = "https://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz"
CMU_DEST = YGP / "data" / "booksummaries.txt"
CMU_BYTES = 43_461_583                 # md5 f8a38037d88988596bdc097c1ad4c65d

# themes/ dates a book by the work's first publication rather than by the
# edition in hand, which is what this second file is for. comention/ and
# year_genre_prediction/ do not use it.
WORKS_URL = ("https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"
             "goodreads_book_works.json.gz")
WORKS_DEST = THEMES_DATA / "goodreads_book_works.json.gz"
WORKS_BYTES = 75_397_299

# The 2 GB archive. themes/ reads it as-is, so it lives in themes/data/ and
# comention/ expands its copy from there instead of fetching it again.
JSON_GZ = THEMES_DATA / "goodreads_books.json.gz"



def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1024


def download(url, dest, expect=None):
    """Resumable GET. Returns dest. Verifies size when `expect` is given."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if expect and have == expect:
        print(f"  {dest.name}: already complete")
        return dest

    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"  resuming at {human(have)}")
    with urllib.request.urlopen(req) as r:
        # A server that ignores Range replies 200 and restarts the body.
        mode = "ab" if (have and r.status == 206) else "wb"
        if mode == "wb":
            have = 0
        total = int(r.headers.get("Content-Length", 0)) + have
        done = have
        with open(dest, mode) as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {dest.name}: {human(done)} / {human(total)} "
                          f"({pct:.1f}%)", end="", flush=True)
        print()

    got = dest.stat().st_size
    if expect and got != expect:
        raise SystemExit(
            f"{dest.name}: expected {expect} bytes, got {got}. Delete it and "
            f"re-run to start over.")
    return dest


def get_json_gz():
    """The shared 2 GB archive, fetched once. Both pipelines start here."""
    if JSON_GZ.exists() and JSON_GZ.stat().st_size == JSON_GZ_BYTES:
        print(f"{JSON_GZ.name}: present")
        return JSON_GZ
    print(f"{JSON_GZ.name}  <- {JSON_URL}")
    return download(JSON_URL, JSON_GZ, JSON_GZ_BYTES)


def get_json():
    out = ROOT / "goodreads_books.json"
    if out.exists() and out.stat().st_size == JSON_BYTES:
        print("goodreads_books.json: present")
        return
    gz = get_json_gz()
    print("  decompressing (8.6 GB, a few minutes) ...")
    with gzip.open(gz, "rb") as fi, open(out, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 22)
    if out.stat().st_size != JSON_BYTES:
        raise SystemExit(f"decompressed to {out.stat().st_size} bytes, "
                         f"expected {JSON_BYTES}")
    # the archive is kept: themes/ reads it directly, and re-downloading 2 GB
    # to get it back would be the exact duplication this avoids
    print("  done")


MANUAL = f"""
  The CSV dump lives on Kaggle, which requires an account:

      https://www.kaggle.com/datasets/{KAGGLE_SLUG}

  Either set up the CLI once --

      pip install kaggle
      # Kaggle -> Settings -> API -> "Create New Token" -> kaggle.json
      mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
      chmod 600 ~/.kaggle/kaggle.json

  -- and re-run this script, or download the zip by hand and unpack the
  book*.csv files into {CSV_DIR}/
"""


def get_csvs():
    have = sorted(CSV_DIR.glob("book*.csv"))
    if len(have) >= CSV_MIN_FILES:
        print(f"goodreads/*.csv: present ({len(have)} files)")
        return
    # Prefer the columns committed to this repo: no Kaggle account needed, and
    # they are unpacked as plain .csv so the pipeline reads what it always read.
    packed = sorted(BUNDLED.glob("book*.csv.gz"))
    if len(packed) >= CSV_MIN_FILES:
        print(f"goodreads/*.csv  <- unpacking {len(packed)} files from "
              f"{BUNDLED.name}/")
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        for gz in packed:
            with gzip.open(gz, "rb") as fi, open(CSV_DIR / gz.name[:-3], "wb") as fo:
                shutil.copyfileobj(fi, fo, 1 << 20)
        print(f"  done, {len(list(CSV_DIR.glob('book*.csv')))} csv files")
        return

    if not shutil.which("kaggle"):
        print("goodreads/*.csv: MISSING, and the kaggle CLI is not installed.")
        print(MANUAL)
        return
    print(f"goodreads/*.csv  <- kaggle datasets download {KAGGLE_SLUG}")
    CSV_DIR.mkdir(exist_ok=True)
    r = subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_SLUG,
                        "--unzip", "-p", str(CSV_DIR)])
    if r.returncode != 0:
        print("\n  kaggle CLI failed (usually missing or unreadable "
              "~/.kaggle/kaggle.json).")
        print(MANUAL)
        return
    n = len(list(CSV_DIR.glob("book*.csv")))
    print(f"  done, {n} csv files")



def get_cmu():
    if CMU_DEST.exists() and CMU_DEST.stat().st_size == CMU_BYTES:
        print(f"{CMU_DEST.name}: present")
        return
    tgz = REPO / "booksummaries.tar.gz"
    print(f"{CMU_DEST.name}  <- {CMU_URL}")
    download(CMU_URL, tgz)
    with tarfile.open(tgz) as t:
        member = next(m for m in t.getmembers()
                      if m.name.endswith("booksummaries.txt"))
        member.name = CMU_DEST.name
        CMU_DEST.parent.mkdir(parents=True, exist_ok=True)
        t.extract(member, CMU_DEST.parent)
    tgz.unlink()
    got = CMU_DEST.stat().st_size
    if got != CMU_BYTES:
        raise SystemExit(f"{CMU_DEST.name}: expected {CMU_BYTES} bytes, got {got}")
    print("  done")


def link(src, dest):
    """Hard-link src to dest so one file serves two paths; copy if that fails."""
    if dest.exists() and dest.stat().st_size == src.stat().st_size:
        print(f"  {dest.name}: present")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
        print(f"  {dest.name}: linked to {src.relative_to(REPO)}")
    except OSError:
        shutil.copy2(src, dest)
        print(f"  {dest.name}: copied from {src.relative_to(REPO)}")


def get_themes():
    THEMES_DATA.mkdir(parents=True, exist_ok=True)
    get_json_gz()
    if WORKS_DEST.exists() and WORKS_DEST.stat().st_size == WORKS_BYTES:
        print(f"{WORKS_DEST.name}: present")
    else:
        print(f"{WORKS_DEST.name}  <- {WORKS_URL}")
        download(WORKS_URL, WORKS_DEST, WORKS_BYTES)
    get_cmu()
    link(CMU_DEST, THEMES_DATA / CMU_DEST.name)


def check():
    j = ROOT / "goodreads_books.json"
    tcmu = THEMES_DATA / CMU_DEST.name
    n = len(list(CSV_DIR.glob("book*.csv")))
    rows = [
        ("goodreads_books.json", j.exists() and j.stat().st_size == JSON_BYTES,
         human(j.stat().st_size) if j.exists() else "missing"),
        ("goodreads/*.csv", n >= CSV_MIN_FILES, f"{n} files"),
        ("year_genre_prediction/data/booksummaries.txt",
         CMU_DEST.exists() and CMU_DEST.stat().st_size == CMU_BYTES,
         human(CMU_DEST.stat().st_size) if CMU_DEST.exists() else "missing"),
        ("themes/data/goodreads_books.json.gz",
         JSON_GZ.exists() and JSON_GZ.stat().st_size == JSON_GZ_BYTES,
         human(JSON_GZ.stat().st_size) if JSON_GZ.exists() else "missing"),
        ("themes/data/goodreads_book_works.json.gz",
         WORKS_DEST.exists() and WORKS_DEST.stat().st_size == WORKS_BYTES,
         human(WORKS_DEST.stat().st_size) if WORKS_DEST.exists() else "missing"),
        ("themes/data/booksummaries.txt",
         tcmu.exists() and tcmu.stat().st_size == CMU_BYTES,
         human(tcmu.stat().st_size) if tcmu.exists() else "missing"),
    ]
    for name, ok, detail in rows:
        print(f"  [{'ok' if ok else '  '}] {name:<40} {detail}")
    return all(ok for _, ok, _ in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="report what is present, download nothing")
    ap.add_argument("--only", choices=["json", "csv", "cmu", "themes"])
    a = ap.parse_args()

    if a.check:
        sys.exit(0 if check() else 1)

    if a.only in (None, "json"):
        get_json()
    if a.only in (None, "csv"):
        get_csvs()
    if a.only in (None, "cmu"):
        get_cmu()
    if a.only in (None, "themes"):
        get_themes()

    print("\nState:")
    ready = check()
    print("\nReady for `python3 main.py`." if ready else
          "\nSomething is still missing; see above.")


if __name__ == "__main__":
    main()
