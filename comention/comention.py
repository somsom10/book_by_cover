"""Build and cluster the co-mention graph.

Two authors are linked when the same blurb name-drops both; edge weight is the
number of distinct blurbs. Names are canonicalized first (pseudonyms, initials
vs. spelled-out) so one author is one node.

Clustering is Leiden (RBConfiguration = modularity with a resolution knob) in
one flat pass. Two choices worth knowing: gamma=8 because gamma=1 hits
modularity's resolution limit and collapses everything literary into
800+-author blobs; Leiden over Louvain because Louvain does not guarantee
internally connected communities and here emitted a six-author "community" that
was three unconnected pairs.

Reproducible for a fixed input (sorted relabel + seed) but not stable under
small input changes -- collapsing one duplicate node can move boundary authors
between adjacent communities. Don't over-read assignments at the borders.

Outputs: comention_author_edges.csv, author_groups.csv, author_groups.txt"""
import csv
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import igraph as ig
import leidenalg as la

from author_ident import CURATED_ALIASES, author_key, build_canonical_map
from name_filters import exclude_author, clean_ws

ROOT = Path(__file__).parent
MENTIONS = ROOT / "blurb_mentions_clean.csv"
SUMMARIES = ROOT / "author_group_summaries.md"
EDGES_OUT = ROOT / "comention_author_edges.csv"
GROUPS_CSV = ROOT / "author_groups.csv"
GROUPS_TXT = ROOT / "author_groups.txt"

MIN_EDGE_W = 2        # pair must be co-mentioned in >= this many blurbs
MIN_GROUP = 3         # smallest community worth reporting
RESOLUTION = 8        # Leiden gamma: largest community ~170, coverage ~94%
SEED = 42

csv.field_size_limit(sys.maxsize)

# ── 0. community titles, read back from the hand interpretation ─────────────────
# author_group_summaries.md is the only place the groups have names; figures
# that say "G12 · Conan Doyle, Christie …" make the reader redo the reading the
# doc already did. Three notations are used there, all keyed by group id:
#     - **G0 (177) Contemporary American poetry** — Billy Collins, ...
#     - **G31 (70)** Big-house US literary — Oates, ...
#     | G64 | 37 | Paris avant-garde salon (Stein, Duchamp) |
BULLET_RE = re.compile(r"^\s*-\s*\*\*G(\d+)\s*\((\d+)\)\s*(.*?)\*\*\s*(.*)$")
ROW_RE = re.compile(r"^\s*\|\s*G(\d+)\s*\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*$")
TRAIL_PAREN_RE = re.compile(r"\s*\([^()]*\)\s*$")


def load_titles(path=SUMMARIES):
    """{group id: (title, size as documented)} from the summaries markdown.

    The documented size is kept only to warn when a title has drifted onto a
    group this run numbered differently: ids are stable for a fixed input, not
    across inputs (see the stability note above), so a re-run on new data
    renumbers everything and the titles must be re-checked by hand."""
    if not path.exists():
        return {}
    titles = {}
    for line in path.read_text().splitlines():
        m = BULLET_RE.match(line)
        if m:
            gid, size, inside, after = m.groups()
            title = inside.strip() or after.lstrip("—– ").split("—")[0]
        else:
            m = ROW_RE.match(line)
            if not m:                              # prose, header, separator
                continue
            gid, size, title = m.groups()
            title = TRAIL_PAREN_RE.sub("", title)  # drop the "(Stein, …)" gloss
        title = title.strip().rstrip(".")
        if title:
            titles.setdefault(int(gid), (title, int(size)))
    return titles


GROUP_TITLES = load_titles()
print(f"Community titles from {SUMMARIES.name}: {len(GROUP_TITLES)}")


# ── 1. read mentions, canonicalize authors, collect per-blurb sets ──────────
print("Reading blurb mentions ...")
raw_rows = []                       # (blurb_key, author_key) after aliasing
counts = Counter()                  # author_key -> total mentions
disp = defaultdict(Counter)         # author_key -> {display: count}
with open(MENTIONS, newline="") as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        if len(row) < 5 or row[3] != "author" or exclude_author(row[4]):
            continue
        name_raw = clean_ws(row[4])
        alias = CURATED_ALIASES.get(name_raw.lower())
        if alias:
            name_raw = alias
        k = author_key(name_raw)
        if not k:
            continue
        raw_rows.append((row[1] or row[0], k))   # key blurbs by work id
        counts[k] += 1
        disp[k][name_raw] += 1

merge = build_canonical_map(counts)
print(f"  surface forms: {len(counts)}   merged by initials rule: {len(merge)}")

per_blurb = defaultdict(set)
canon_counts = Counter()
for blurb, k in raw_rows:
    k = merge.get(k, k)
    per_blurb[blurb].add(k)
    canon_counts[k] += 1

# display name: most common surface form across everything merged into the key
canon_disp = defaultdict(Counter)
for k, c in disp.items():
    canon_disp[merge.get(k, k)].update(c)
name = {k: c.most_common(1)[0][0] for k, c in canon_disp.items()}
print(f"  authors: {len(name)}   blurbs with >=1 author: {len(per_blurb)}")

# ── 2. co-mention pairs -> weighted graph ────────────────────────────────────
print("Counting co-mentions ...")
pair_w = Counter()
for ks in per_blurb.values():
    if len(ks) >= 2:
        for a, b in combinations(sorted(ks), 2):
            pair_w[(a, b)] += 1

with open(EDGES_OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["author_a", "author_b", "comentions"])
    for (a, b), wt in sorted(pair_w.items(), key=lambda kv: kv[1], reverse=True):
        w.writerow([name[a], name[b], wt])
print(f"Saved: {EDGES_OUT}  ({len(pair_w)} pairs)")

G = nx.Graph()
for (a, b), wt in pair_w.items():
    if wt >= MIN_EDGE_W:
        G.add_edge(a, b, weight=wt)
print(f"Graph: {G.number_of_nodes()} authors / {G.number_of_edges()} edges "
      f"(pairs co-mentioned >= {MIN_EDGE_W}x)")

# ── 3. Leiden communities, single pass ───────────────────────────────────────
def leiden():
    # Rebuild the graph with sorted int labels and sorted edge insertion so
    # the partition is byte-identical across runs (the same trick that fixed
    # Louvain's per-process hash-randomization leak).
    order = sorted(G)
    idx = {n: i for i, n in enumerate(order)}
    edges = sorted((min(idx[u], idx[v]), max(idx[u], idx[v]))
                   for u, v in G.edges)
    gi = ig.Graph(len(order), edges)
    gi.es["weight"] = [G[order[a]][order[b]]["weight"] for a, b in edges]
    part = la.find_partition(gi, la.RBConfigurationVertexPartition,
                             weights="weight",
                             resolution_parameter=RESOLUTION,
                             seed=SEED, n_iterations=-1)   # run to convergence
    comms = defaultdict(set)
    for i, m in enumerate(part.membership):
        comms[m].add(order[i])
    comms = [c for c in comms.values() if len(c) >= MIN_GROUP]
    return sorted(comms, key=lambda c: (-len(c), min(c)))


print(f"Leiden communities (resolution {RESOLUTION}) ...")
top_groups = leiden()
print(f"  groups (>= {MIN_GROUP} authors): {len(top_groups)}")

group_of = {}
for gid, comm in enumerate(top_groups):
    for n in comm:
        group_of[n] = gid

def wdeg(n):
    """Weighted co-mention degree: the number of blurb co-mentions this author
    takes part in, summed over all their partners. One author, one number --
    it is the node-size channel in the figures, where the *edge* weight is the
    strength of a single pair. Always computed on the full graph, so drawing a
    subset of a group doesn't shrink anyone."""
    return G.degree(n, weight="weight")


def by_degree(nodes):
    """Descending weighted degree with a deterministic tie-break."""
    return sorted(nodes, key=lambda n: (-wdeg(n), n))

def leads(nodes, k=6):
    return [name[n] for n in by_degree(nodes)[:k]]


def community(gid, comm):
    """Documented community title, falling back to the lead authors when the group
    has no entry (the tail groups, and anything after a re-run renumbers)."""
    t = GROUP_TITLES.get(gid)
    return t[0] if t else ", ".join(leads(comm, 3))


drifted = [gid for gid, (_, n) in sorted(GROUP_TITLES.items())
           if gid < len(top_groups) and n != len(top_groups[gid])]
if drifted:
    print(f"  !! {len(drifted)} titled groups changed size since the "
          f"summaries were written (G{', G'.join(map(str, drifted[:8]))}"
          f"{' …' if len(drifted) > 8 else ''}) -- ids have been renumbered, "
          f"re-check author_group_summaries.md before trusting the labels")

with open(GROUPS_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "group_title", "group_size", "author",
                "weighted_degree"])
    for gid, comm in enumerate(top_groups):
        # blank, not the lead-author fallback, so "documented" stays legible
        title = GROUP_TITLES.get(gid, ("",))[0]
        for n in by_degree(comm):
            w.writerow([f"G{gid}", title, len(comm), name[n], wdeg(n)])
print(f"Saved: {GROUPS_CSV}")

with open(GROUPS_TXT, "w") as f:
    for gid, comm in enumerate(top_groups):
        titled = GROUP_TITLES.get(gid)
        f.write(f"==== G{gid}  ({len(comm)} authors)  "
                f"{titled[0] + '  ' if titled else ''}"
                f"lead: {', '.join(leads(comm, 4))} ====\n")
        f.write("  " + ", ".join(name[n] for n in by_degree(comm)) + "\n\n")
print(f"Saved: {GROUPS_TXT}")

print("\nLargest groups:")
for gid, comm in enumerate(top_groups[:12]):
    print(f"  G{gid:<3} {len(comm):>5}  {community(gid, comm):<34}  "
          f"{', '.join(leads(comm, 4))}")
