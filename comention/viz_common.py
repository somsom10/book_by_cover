"""Shared figure spec: palette, sizing, layout and label placement.

The three figures must read as one system, so the parts they share live here.
Sizes are given in the points a mark measures *on the page* (page_pt), not on
the canvas, because the PNGs are reproduced at a fraction of their pixel size
and type does not scale with the canvas."""
import csv
import re
import textwrap

import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.transforms import Bbox

# ── palette ──────────────────────────────────────────────────────────────────
SLOTS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
         "#e87ba4"]
OTHER = "#c3c2b7"          # baseline gray for everything past 7 groups
SURFACE = "#fcfcfb"
INK, INK_2, INK_MUTE = "#0b0b0b", "#52514e", "#898781"
EDGE_COL = "#d5d4cb"       # a shade darker than the ink-on-paper hairline:
                           # #e1e0d9 vanishes once the PNG is downscaled
HALO = [pe.withStroke(linewidth=2.4, foreground=SURFACE)]
# Sequential blue, for a color channel that means "more", not "different".
# Never used together with the categorical slots in one panel. Six *discrete*
# bins, so these are the reference ramp's ordinal-safe steps (250, 350, 450,
# 550, 600, 650): the lightest is step 250, the lightest the palette allows to
# carry a mark on a light surface (2.06:1) rather than recede into it.
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"]

DPI = 200
MIN_EDGE_W = 2             # pair must be co-mentioned in >= this many blurbs

GROUP_RE = re.compile(r"G(\d+)")


# ── fitting the page ─────────────────────────────────────────────────────────
# The PNGs are reproduced at a fraction of their pixel size and type does not
# scale with the canvas, so sizes are declared in the points a mark measures
# *on the page*; page_in and page_pt convert. Shared FIG_W keeps all three
# figures at one reproduction scale.
PAGE_W, PAGE_H = 6.5, 8.4  # in -- the text column of the target page, and the
                           # tallest a figure can be there and still leave the
                           # page room for its surrounding caption
FIG_W = 9.0                # in -- shared canvas width
FIT = FIG_W / PAGE_W       # 1.38: canvas inches per page inch


def page_pt(size):
    """Canvas points that reproduce as `size` points on the page."""
    return size * FIT


def page_in(height):
    """Canvas height that reproduces as `height` inches on the page."""
    return height * FIT


FS_LABEL = page_pt(7.0)    # author names -- the floor; nothing goes below it
FS_COMMUNITY = page_pt(9.5)    # community titles, the figure's own legend
FS_SUB = page_pt(7.0)      # the "G3 · 139 authors" line under a community title
FS_TITLE = page_pt(12.5)
FS_CAPTION = page_pt(7.5)
FS_NOTE = page_pt(6.5)     # in-plot annotations
LINESPACE = 1.45           # multiple of the font size, matplotlib's unit


# ── reading the pipeline's outputs ───────────────────────────────────────────
def load_graph(edges_csv, min_w=MIN_EDGE_W):
    """The co-mention graph from comention_author_edges.csv, nodes = canonical
    display names. Same graph comention.py clusters (it writes every pair and
    applies the same threshold), so weighted degrees match author_groups.csv."""
    G = nx.Graph()
    with open(edges_csv, newline="") as f:
        r = csv.reader(f)
        next(r)
        for a, b, w in r:
            w = int(w)
            if w >= min_w:
                G.add_edge(a, b, weight=w)
    return G


def load_groups(groups_csv):
    """(groups, titles, group_of) from author_groups.csv. groups[gid] is the
    member list in descending weighted degree."""
    groups, titles, group_of = {}, {}, {}
    with open(groups_csv, newline="") as f:
        for row in csv.DictReader(f):
            gid = int(GROUP_RE.match(row["group"]).group(1))
            groups.setdefault(gid, []).append(row["author"])
            titles[gid] = row["group_title"]
            group_of[row["author"]] = gid
    return groups, titles, group_of


def community_label(gid, titles, members, k=3):
    """Documented community title, falling back to lead authors for the ~100 tail
    groups that have no entry in author_group_summaries.md."""
    return titles.get(gid) or ", ".join(members[:k])


# ── layout ───────────────────────────────────────────────────────────────────
def induced(G_, nodes):
    """Induced subgraph with a deterministic node order.

    Not G.subgraph(nodes): networkx's filtered view iterates the *set* of
    requested nodes, so the layout it feeds rides on PYTHONHASHSEED. Insert
    nodes and edges in sorted order instead."""
    keep = sorted(nodes)
    ok = set(keep)
    S = nx.Graph()
    S.add_nodes_from(keep)
    # endpoint order also comes out hash-dependent, so normalize it too
    S.add_edges_from(sorted(((min(u, v), max(u, v), {"weight": w})
                             for u, v, w in G_.edges(ok, data="weight")
                             if v in ok), key=lambda e: e[:2]))
    return S


def blob(H, nodes, center, scale, seed=7, k=0.5, iterations=200,
         weight="weight"):
    """Spring-lay `nodes` on their own and drop the result into a disc of the
    given radius around `center`. Communities are dense inside and sparse
    between, so one global spring layout collapses them into a single ball."""
    sub = induced(H, nodes)
    p = nx.spring_layout(sub, k=k, iterations=iterations, seed=seed,
                         weight=weight)
    xy = np.array(list(p.values()))
    xy -= xy.mean(0)
    m = np.abs(xy).max() or 1.0
    center = np.asarray(center, dtype=float)
    return {n: center + scale * q / m for n, q in zip(p, xy)}


def cluster_layout(H, clusters, radius=0.72, scale=0.30, **kw):
    """One blob per cluster, blobs evenly spaced on a circle."""
    pos = {}
    for i, nodes in enumerate(clusters):
        ang = 2 * np.pi * i / len(clusters)
        center = (radius * np.cos(ang), radius * np.sin(ang))
        pos.update(blob(H, nodes, center, scale, **kw))
    return pos


def ring_angles(n):
    """The angles cluster_layout puts n blobs at -- for titling them in place."""
    return [2 * np.pi * i / n for i in range(n)]


# ── marks ────────────────────────────────────────────────────────────────────
def area_sizes(nodes, value_of, node_max, base, vmax=None):
    """Marker areas, area proportional to `value` -- the perceptually correct
    channel for a magnitude on a circle."""
    nodes = list(nodes)
    v = np.array([value_of(n) for n in nodes], dtype=float)
    top = vmax if vmax else (v.max() or 1.0)
    return dict(zip(nodes, base + node_max * v / top))


def data_per_inch(ax):
    """(x, y) data units per inch of axes -- the bridge between font sizes,
    which are absolute, and the layout, which is in data coordinates."""
    fw, fh = ax.figure.get_size_inches()
    bb = ax.get_position()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return (x1 - x0) / (fw * bb.width), (y1 - y0) / (fh * bb.height)


def place_labels(ax, pos, order, sizes, fontsize, label_of, limit=None,
                 reserved=(), color=INK, weight=None):
    """Label as many nodes as fit without overlap, biggest first."""
    dx, dy = data_per_inch(ax)
    placed, n = list(reserved), 0
    limit = len(order) if limit is None else limit
    for node in order:
        if n >= limit:
            break
        t = label_of(node)
        w = 0.60 * fontsize * len(t) / 72 * dx + 0.004    # ~mean glyph width
        h = 1.30 * fontsize / 72 * dy + 0.004
        r = np.sqrt(sizes[node] / np.pi) / 72             # marker radius, in
        px, py = pos[node]
        for cx, cy in ((px, py + r * dy + h / 2), (px, py - r * dy - h / 2),
                       (px + r * dx + w / 2, py), (px - r * dx - w / 2, py)):
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if all(box[0] > o[2] or box[2] < o[0] or box[1] > o[3]
                   or box[3] < o[1] for o in placed):
                placed.append(box)
                ax.text(cx, cy, t, fontsize=fontsize, color=color, ha="center",
                        va="center", path_effects=HALO, zorder=5,
                        fontweight=weight or "normal")
                n += 1
                break
    return n


def ring_title(ax, ang, title_lines, sub, color, d, fontsize=FS_COMMUNITY,
               sub_fontsize=FS_SUB):
    """Title a blob where it sits, just outside the ring, in the blob's own
    hue. A legend in the corner would make the reader ping-pong across the
    figure to decode seven colors. Returns the box it occupies, so it can be
    reserved against author labels."""
    dx, dy = data_per_inch(ax)
    lines = list(title_lines) + [sub]
    hw = 0.60 * fontsize * max(len(s) for s in lines) / 72 * dx / 2
    hh = 1.35 * fontsize * len(lines) / 72 * dy / 2
    ux, uy = np.cos(ang), np.sin(ang)
    x, y = ux * d + ux * hw, uy * d + uy * hh
    ax.text(x, y, "\n".join(title_lines), fontsize=fontsize, color=color,
            ha="center", va="bottom", fontweight="bold", linespacing=1.35,
            zorder=6)
    ax.text(x, y, sub, fontsize=sub_fontsize, color=INK_MUTE, ha="center",
            va="top", zorder=6)
    return (x - hw, y - hh, x + hw, y + hh)


def fit_text(width_in, text, fontsize):
    """`text` hard-wrapped to `width_in` inches at `fontsize`, same ~mean glyph
    width the label placer measures boxes with."""
    cols = max(24, int(width_in * 72 / (0.60 * fontsize)))
    return "\n".join(l for s in text.split("\n")
                     for l in (textwrap.wrap(s, cols) or [""]))


def caption(ax, title, lines, gap=10):
    """Title and subtitle lines, top-left, outside the drawing."""
    fw = ax.figure.get_size_inches()[0]
    sub = fit_text(fw, "\n".join(lines), FS_CAPTION)
    ax.text(0, 1.005, sub, transform=ax.transAxes, fontsize=FS_CAPTION,
            color=INK_2, va="bottom", linespacing=LINESPACE)
    ax.set_title(fit_text(fw, title, FS_TITLE), fontsize=FS_TITLE, color=INK,
                 loc="left",
                 pad=gap + LINESPACE * FS_CAPTION * (sub.count("\n") + 1))


def save(fig, path, pad=0.05):
    """Write the PNG at the shared width so all three figures reproduce at one
    scale on the page."""
    fig.tight_layout()
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    fw, fh = fig.get_size_inches()
    box = Bbox([[0.0, max(0.0, bb.y0 - pad)], [fw, min(fh, bb.y1 + pad)]])
    fig.savefig(path, dpi=DPI, bbox_inches=box, facecolor=SURFACE)
    plt.close(fig)
    w, h = box.width / FIT, box.height / FIT       # as placed on the page
    over = "  !! taller than the page allows" if h > PAGE_H else ""
    print(f"Saved: {path}  ({w:.1f} × {h:.1f} in on the page){over}")


def relax_overlaps(pos, radii, iterations=300, pad=1.04):
    """Push circles apart until they stop overlapping, keeping the arrangement
    the layout found."""
    keys = sorted(pos)
    p = {k: np.asarray(pos[k], dtype=float) for k in keys}
    for _ in range(iterations):
        moved = 0.0
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                d = p[b] - p[a]
                dist = float(np.hypot(*d)) or 1e-9
                need = (radii[a] + radii[b]) * pad
                if dist < need:
                    push = (need - dist) / 2
                    u = d / dist
                    p[a] -= u * push
                    p[b] += u * push
                    moved += push
        if moved < 1e-4:
            break
    return p


def text_box(ax, x, y, text, fontsize, lines=1):
    """The box a piece of centered text will occupy, in data units -- so it can
    be handed to place_labels as reserved space."""
    dx, dy = data_per_inch(ax)
    w = 0.60 * fontsize * max(len(s) for s in text.split("\n")) / 72 * dx
    h = 1.35 * fontsize * lines / 72 * dy
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def draw_network(ax, H, pos, color_of, sizes, lim, label_nodes=(), label_of=str,
                 fontsize=FS_LABEL, edge_width=0.85, edge_alpha=0.7, ring=1.0,
                 reserved=(), label_limit=None):
    """Network with the shared mark spec: hairline edges, area-scaled nodes
    ringed in surface color, haloed ink labels.

    One width for every edge: weight is unreadable as thickness at this size
    (62% of drawn edges are 2-3 co-mentions against a top edge of 32). Weights
    still steer the layout; the counts live in comention_author_edges.csv."""
    nx.draw_networkx_edges(H, pos, ax=ax, edge_color=EDGE_COL, alpha=edge_alpha,
                           width=edge_width)
    nodes = list(H.nodes)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=[sizes[n] for n in nodes],
                           node_color=[color_of(n) for n in nodes],
                           linewidths=ring, edgecolors=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    # limits must be final before labels: their boxes are measured in data units
    x0, x1, y0, y1 = lim if isinstance(lim, tuple) else (-lim, lim, -lim, lim)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    if len(label_nodes):
        place_labels(ax, pos, label_nodes, sizes, fontsize, label_of,
                     limit=label_limit, reserved=reserved)
