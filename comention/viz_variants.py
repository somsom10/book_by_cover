"""The two co-mention network figures.

    comention_groups_full.png   the 7 largest communities, every member
    comention_all.png           every author with a tie into a community

Reads comention.py's CSVs, not the raw dumps, so a redraw is seconds.

    python3 viz_variants.py [full|all]"""
import sys
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from viz_common import (FIG_W, FS_LABEL, FS_COMMUNITY, HALO, INK_MUTE,
                        OTHER, SLOTS, SURFACE, area_sizes, blob,
                        caption, cluster_layout, draw_network, induced,
                        load_graph, load_groups, page_in, page_pt, place_labels,
                        relax_overlaps, ring_angles, ring_title, save,
                        community_label, text_box)

ROOT = Path(__file__).parent
EDGES = ROOT / "comention_author_edges.csv"
GROUPS_CSV = ROOT / "author_groups.csv"
FIG_FULL = ROOT / "comention_groups_full.png"
FIG_ALL = ROOT / "comention_all.png"

N_COLORED = len(SLOTS)      # communities that get a hue; the rest are gray

# ── data ─────────────────────────────────────────────────────────────────────
G = load_graph(EDGES)
groups, titles, group_of = load_groups(GROUPS_CSV)
print(f"Graph: {G.number_of_nodes()} authors / {G.number_of_edges()} edges "
      f"· {len(groups)} communities")


def wdeg(n):
    """Weighted co-mention degree on the *full* graph, so both figures share one
    node-size scale."""
    return G.degree(n, weight="weight")


def by_degree(nodes):
    return sorted(nodes, key=lambda n: (-wdeg(n), n))


def community(gid, k=3):
    return community_label(gid, titles, groups[gid], k)


# Cross-community structure, used by the whole-graph figure: how much
# co-mention traffic runs between each pair of communities.
meta_w = Counter()
for u, v, w in G.edges(data="weight"):
    gu, gv = group_of.get(u), group_of.get(v)
    if gu is None or gv is None or gu == gv:
        continue
    meta_w[(min(gu, gv), max(gu, gv))] += w

ORPHANS = [n for n in G if n not in group_of]   # communities under MIN_GROUP


def hue_of(n):
    """Categorical hue for the first N_COLORED communities, gray past them."""
    g = group_of.get(n, -1)
    return SLOTS[g] if 0 <= g < N_COLORED else OTHER


# ── 1. the 7 big communities, every member ────────────────────────────────────────
def fig_full():
    """The 7 largest communities, nobody dropped. Sampling a community down to
    its best-connected members makes it look like a clique; drawing all 1,003
    members shows dense-core-plus-tail (thrillers, poetry) versus several knots
    loosely tied (Christianity, the MFA circuit)."""
    colored = list(range(N_COLORED))
    members = {g: groups[g] for g in colored}
    H = induced(G, [n for g in colored for n in members[g]])
    # Leiden guarantees every community is internally connected, so no member
    # is an isolate here and none has to be dropped (unlike the sampled figure).
    clusters = [members[g] for g in colored]
    radius, scale, lim = 0.86, 0.34, 1.56

    fig, ax = plt.subplots(figsize=(FIG_W, page_in(7.3)))
    fig.patch.set_facecolor(SURFACE)
    pos = cluster_layout(H, clusters, radius=radius, scale=scale, k=0.35)
    sizes = area_sizes(H.nodes, wdeg, 300, 4)
    draw_network(ax, H, pos, hue_of, sizes, lim, edge_width=page_pt(0.35),
                 edge_alpha=0.55, ring=0.5)
    reserved = [ring_title(ax, ang, textwrap.wrap(community(g), 16),
                           f"{len(members[g])} authors", SLOTS[g],
                           radius + scale + 0.03)
                for ang, g in zip(ring_angles(len(colored)), colored)]
    place_labels(ax, pos, by_degree([n for g in colored
                                     for n in members[g][:10]]),
                 sizes, FS_LABEL, str, reserved=reserved)
    caption(ax, "The seven biggest communities",
            ["every author is a circle, the size "
            "depending on the number of mentions"])
    save(fig, FIG_FULL)


# ── 2. everything ────────────────────────────────────────────────────────────
def fig_all():
    """Every author with a tie into a community, laid out two-level.

    A single spring layout of this graph is a ball of wool, so: a spring layout
    of the *community* graph places each community, members are sprung inside
    their own disc, and the discs are pushed apart until they stop overlapping.
    The disc is layout, never a mark.

    The 428 authors in communities under MIN_GROUP have no disc. The 2 with a
    tie into a community are drawn inside it in gray; the other 426 (212
    isolated pairs and 2 authors whose only partners are drawn elsewhere) have
    nothing to be placed next to and are not drawn."""
    gids = sorted(groups)
    meta = nx.Graph()
    meta.add_nodes_from(gids)
    for (a, b), w in sorted(meta_w.items()):
        meta.add_edge(a, b, weight=w)
    # normalise by size: raw weight would let the biggest communities dominate
    # the spring forces and pull everything into one clump
    for a, b, d in meta.edges(data=True):
        d["weight"] = d["weight"] / np.sqrt(len(groups[a]) * len(groups[b]))
    mpos = nx.spring_layout(meta, weight="weight", k=0.55, iterations=400,
                            seed=11)
    size_max = max(len(groups[g]) for g in gids)
    radii = {g: 0.115 * np.sqrt(len(groups[g]) / size_max) + 0.012
             for g in gids}
    mpos = relax_overlaps(mpos, radii, pad=1.02)
    span = max(np.abs(np.array(list(mpos.values()))).max(), 1e-9)
    fit = 1.0 / span                       # scale the whole arrangement to fit

    # an ungrouped author with ties into a community is drawn in that community's disc
    attached = {}
    for n in ORPHANS:
        ties = [(G[n][m]["weight"], group_of[m]) for m in G[n]
                if m in group_of]
        if ties:
            attached[n] = max(ties)[1]
    islands = sorted(set(ORPHANS) - set(attached))
    members = {g: groups[g] + sorted(n for n, gg in attached.items()
                                     if gg == g) for g in gids}

    # The islands are not drawn: they used to sit in a labelled band under the
    # graph, where a reader reasonably took them for authors with no co-mention
    # at all. Every node here has at least one (that is what put it in the
    # graph); these are the isolated pairs too small to be a community, with
    # no tie into one either, so they have nothing to be placed next to.
    H = induced(G, set(G.nodes) - set(islands))
    pos = {}
    for g in gids:
        pos.update(blob(H, members[g], np.array(mpos[g]) * fit,
                        radii[g] * fit, k=0.4, iterations=120))

    fig, ax = plt.subplots(figsize=(FIG_W, page_in(8.1)))
    fig.patch.set_facecolor(SURFACE)
    sizes = area_sizes(H.nodes, wdeg, 150, 1.6)
    # 39k hairlines over one canvas: the wash in the middle is real (that is
    # where the traffic is) but at any more alpha than this it swallows the discs
    draw_network(ax, H, pos, hue_of, sizes, 1.15, edge_width=page_pt(0.15),
                 edge_alpha=0.16, ring=0.25)
    ax.set_ylim(-1.10, 1.18)

    # titles: the 7 hued communities get their name in their hue, the next tier in
    # muted ink, then as many author names as still fit. Community titles are set
    # at a size that survives the page, and at that size the middle of the
    # graph holds fewer of them than it did at 100%: they go on in descending
    # community size and one that lands on a title already placed is dropped,
    # so the largest communities always keep their name.
    reserved = []
    for g in gids[:22]:
        x, y = np.array(mpos[g]) * fit
        col, fs = ((SLOTS[g], FS_COMMUNITY) if g < N_COLORED
                   else (INK_MUTE, FS_LABEL))
        lines = textwrap.wrap(community(g, 2), 20)[:2]
        y_top = y + radii[g] * fit + 0.012
        box = text_box(ax, x, y_top + 0.02, "\n".join(lines), fs,
                       lines=len(lines))
        if any(box[0] < o[2] and box[2] > o[0] and box[1] < o[3]
               and box[3] > o[1] for o in reserved):
            continue
        # haloed, unlike the ring titles of the other two figures: these sit
        # *inside* the drawing, on top of 39k edges and whatever nodes are there
        ax.text(x, y_top, "\n".join(lines), fontsize=fs, color=col,
                ha="center", va="bottom", fontweight="bold", linespacing=1.25,
                path_effects=HALO, zorder=6)
        reserved.append(box)
    place_labels(ax, pos, by_degree(H.nodes)[:70], sizes, FS_LABEL, str,
                 reserved=reserved, limit=34)
    caption(ax, "The co-mention graph", [
        "The seven biggest communities are colored",
        f"{len(islands)} authors with no co-mention tie into any community "
        f"are not drawn"])
    save(fig, FIG_ALL)


FIGURES = {"full": fig_full, "all": fig_all}

if __name__ == "__main__":
    want = sys.argv[1:] or list(FIGURES)
    bad = [k for k in want if k not in FIGURES]
    if bad:
        sys.exit(f"unknown figure(s): {', '.join(bad)}  "
                 f"(choose from {', '.join(FIGURES)})")
    for k in want:
        FIGURES[k]()
