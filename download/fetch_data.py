"""Download the raw data the pipeline runs on.

    python3 fetch_data.py             # everything still missing
    python3 fetch_data.py --check     # report what is present, download nothing
    python3 fetch_data.py --only json|csv|cmu

Three sources, one per thing the repo needs:

  goodreads_books.json  UCSD Book Graph -- blurbs and author_ids, for
                        comention/. Direct download, 2.0 GB -> 8.6 GB on disk.
  goodreads/*.csv       Kaggle -- author names and popularity, for comention/.
                        Kaggle needs an account, so this one cannot be fully
                        automated; the script uses the `kaggle` CLI if it is
                        set up and prints manual instructions if not.
  booksummaries.txt     CMU Book Summary Dataset -- plot summaries with genre
                        and publication date, for year_genre_prediction/.
                        Direct download, 16 MB.

Downloads resume if interrupted and are verified by size before use, so
re-running after a failure is safe.
"""
import argparse
import gzip
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

JSON_URL = ("https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"
            "goodreads_books.json.gz")
JSON_GZ_BYTES = 2_083_197_934          # verified against the server
JSON_BYTES = 9_202_235_168             # after gunzip

KAGGLE_SLUG = "bahramjannesarr/goodreads-book-datasets-10m"
CSV_DIR = ROOT / "goodreads"
CSV_MIN_FILES = 23                     # the dump ships 23 book*.csv (plus 7
                                       # user_rating_*.csv the pipeline never reads)

CMU_URL = "https://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz"
CMU_DEST = YGP / "data" / "booksummaries.txt"
CMU_BYTES = 43_461_583                 # md5 f8a38037d88988596bdc097c1ad4c65d



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


def get_json():
    out = ROOT / "goodreads_books.json"
    if out.exists() and out.stat().st_size == JSON_BYTES:
        print("goodreads_books.json: present")
        return
    gz = ROOT / "goodreads_books.json.gz"
    print(f"goodreads_books.json  <- {JSON_URL}")
    download(JSON_URL, gz, JSON_GZ_BYTES)
    print("  decompressing (8.6 GB, a few minutes) ...")
    with gzip.open(gz, "rb") as fi, open(out, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 22)
    if out.stat().st_size != JSON_BYTES:
        raise SystemExit(f"decompressed to {out.stat().st_size} bytes, "
                         f"expected {JSON_BYTES}")
    gz.unlink()
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


def check():
    j = ROOT / "goodreads_books.json"
    n = len(list(CSV_DIR.glob("book*.csv")))
    rows = [
        ("goodreads_books.json", j.exists() and j.stat().st_size == JSON_BYTES,
         human(j.stat().st_size) if j.exists() else "missing"),
        ("goodreads/*.csv", n >= CSV_MIN_FILES, f"{n} files"),
        ("year_genre_prediction/data/booksummaries.txt",
         CMU_DEST.exists() and CMU_DEST.stat().st_size == CMU_BYTES,
         human(CMU_DEST.stat().st_size) if CMU_DEST.exists() else "missing"),
    ]
    for name, ok, detail in rows:
        print(f"  [{'ok' if ok else '  '}] {name:<26} {detail}")
    return all(ok for _, ok, _ in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="report what is present, download nothing")
    ap.add_argument("--only", choices=["json", "csv", "cmu"])
    a = ap.parse_args()

    if a.check:
        sys.exit(0 if check() else 1)

    if a.only in (None, "json"):
        get_json()
    if a.only in (None, "csv"):
        get_csvs()
    if a.only in (None, "cmu"):
        get_cmu()

    print("\nState:")
    ready = check()
    print("\nReady for `python3 main.py`." if ready else
          "\nSomething is still missing; see above.")


if __name__ == "__main__":
    main()
