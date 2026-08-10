"""
Fetches the three raw datasets into data/. The only step that needs a network.

A script rather than a README line because the source must be explicit, sizes
verified, and an interrupted transfer resumed - the largest file is 2GB.

  python download_data.py            # everything
  python download_data.py --check    # report what is present, download nothing
"""
import os
import sys
import tarfile
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# Sizes are what the server reports, and are checked after each download. A
# file of any other length is truncated - a failure that is easy to miss, since
# a truncated gzip opens, reads partially, and yields a smaller corpus silently
FILES = [
    ("goodreads_books.json.gz",
     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"
     "goodreads_books.json.gz", 2083197934),
    ("goodreads_book_works.json.gz",
     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"
     "goodreads_book_works.json.gz", 75397299),
]

CMU_URL = "http://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz"
CMU_MEMBER = "booksummaries/booksummaries.txt"
CMU_OUT = "booksummaries.txt"


def _human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


def fetch(url, dest, expect=None):
    """Download with resume: a partial file is continued via a Range request."""
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if expect and have == expect:
        print(f"  {os.path.basename(dest)}: already complete ({_human(have)})")
        return dest
    if have and expect and have > expect:
        print(f"  {os.path.basename(dest)}: larger than expected, re-downloading")
        os.remove(dest); have = 0

    req = urllib.request.Request(url)
    mode = "wb"
    if have:
        req.add_header("Range", f"bytes={have}-")
        mode = "ab"
        print(f"  resuming at {_human(have)}")

    with urllib.request.urlopen(req, timeout=60) as r:
        # a server that ignores Range answers 200 with the whole file: restart
        if have and r.status == 200:
            mode, have = "wb", 0
        total = int(r.headers.get("Content-Length", 0)) + have
        done = have
        with open(dest, mode) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    print(f"\r  {os.path.basename(dest)} {pct:5.1f}%  "
                          f"{_human(done)} / {_human(total)}", end="", flush=True)
    print()
    got = os.path.getsize(dest)
    if expect and got != expect:
        raise IOError(f"{dest}: got {got} bytes, expected {expect}. "
                      f"Run again to resume.")
    return dest


def fetch_cmu():
    out = os.path.join(DATA_DIR, CMU_OUT)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"  {CMU_OUT}: already present ({_human(os.path.getsize(out))})")
        return out
    tgz = os.path.join(DATA_DIR, "booksummaries.tar.gz")
    fetch(CMU_URL, tgz)
    with tarfile.open(tgz) as t:
        member = t.getmember(CMU_MEMBER)
        with t.extractfile(member) as src, open(out, "wb") as dst:
            dst.write(src.read())
    os.remove(tgz)
    print(f"  extracted {CMU_OUT} ({_human(os.path.getsize(out))})")
    return out


def status():
    print(f"data directory: {os.path.abspath(DATA_DIR)}")
    for name, _, expect in FILES:
        p = os.path.join(DATA_DIR, name)
        have = os.path.getsize(p) if os.path.exists(p) else 0
        state = "ok" if have == expect else (f"partial {_human(have)}" if have else "missing")
        print(f"  {name:32} {state}")
    p = os.path.join(DATA_DIR, CMU_OUT)
    print(f"  {CMU_OUT:32} "
          f"{'ok' if os.path.exists(p) else 'missing'}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if "--check" in sys.argv:
        status()
        return
    print("Goodreads (UCSD McAuley lab) - 2.1GB, this is the slow one")
    for name, url, expect in FILES:
        fetch(url, os.path.join(DATA_DIR, name), expect)
    print("CMU Book Summaries (Bamman, CMU)")
    fetch_cmu()
    print()
    status()


if __name__ == "__main__":
    main()
