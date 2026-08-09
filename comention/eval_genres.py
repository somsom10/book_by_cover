"""Are the co-mention communities also genre communities?

The groups *read* like genres, which is an interpretation, not a result. This
tests it against a signal the clustering never saw: what readers shelve these
authors' books as (build_author_genres.py).

Similarity of two authors is the cosine of their genre profiles. Profiles are
L2-normalised once up front, so the mean over all pairs inside a set of n is
(||sum||^2 - n) / (n(n-1)) -- one vector addition per community, which is what
makes 2,000 permutations cheap.

The yardstick is a random group of the same size: community labels are permuted
with the group sizes held fixed, giving a null for the pooled statistic and a
size-matched null for each community, so a 6-author group is not judged against
the same bar as a 177-author one.

Outputs: community_genres.csv, genre_eval.txt, genre_eval.png"""
import csv
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from author_ident import author_key, signature
from genre_vocab import GENRE_NAMES
from viz_common import (DPI, EDGE_COL, FIG_W, FS_CAPTION, FS_LABEL,
                        FS_NOTE, FS_COMMUNITY, FS_TITLE, INK, INK_2, INK_MUTE,
                        OTHER, SLOTS, SURFACE, load_graph, load_groups,
                        community_label)

ROOT = Path(__file__).parent
EDGES = ROOT / "comention_author_edges.csv"
GROUPS_CSV = ROOT / "author_groups.csv"
GENRES_CSV = ROOT / "author_genres.csv"
OUT_CSV = ROOT / "community_genres.csv"
OUT_TXT = ROOT / "genre_eval.txt"
FIG = ROOT / "genre_eval.png"

MIN_COVERED = 5     # covered members a community needs for its own row
N_PERM = 2000       # label permutations, for both nulls
N_ROWS = 26         # communities in the figure's dot plot
SEED = 42

rng = np.random.default_rng(SEED)
report = []


def say(line=""):
    print(line)
    report.append(line)


# ── 1. profiles, and lining them up with the graph's names ───────────────────
# The graph's nodes are canonical display names, author_genres.csv is keyed by
# author_id. Fold both onto the same token identity (author_ident.py) so the
# two stages agree on who is one person.
prof, books = defaultdict(lambda: np.zeros(len(GENRE_NAMES))), defaultdict(int)
with open(GENRES_CSV, newline="") as f:
    for row in csv.DictReader(f):
        k, n = author_key(row["name"]), int(row["books"])
        if not k:
            continue
        prof[k] += n * np.array([float(row[g]) for g in GENRE_NAMES])
        books[k] += n              # several ids can resolve to one name
for k in prof:
    prof[k] /= books[k]

by_sig = defaultdict(list)
for k in prof:
    if signature(k):
        by_sig[signature(k)].append(k)

G = load_graph(EDGES)
groups, titles, group_of = load_groups(GROUPS_CSV)


def profile_of(node):
    k = author_key(node)
    if k in prof:
        return prof[k]
    forms = by_sig.get(signature(k), [])
    return prof[forms[0]] if len(forms) == 1 else None     # ambiguous -> no


covered = {}
for n in G:
    p = profile_of(n)
    if p is not None and p.sum() > 0:
        covered[n] = p / p.sum()

say(f"Graph:      {G.number_of_nodes()} authors, {len(groups)} communities")
say(f"Profiles:   {len(prof)} authors in {GENRES_CSV.name} "
    f"({len(GENRE_NAMES)} genres)")
say(f"Covered:    {len(covered)} graph authors have one "
    f"({100 * len(covered) / G.number_of_nodes():.1f}%), "
    f"{sum(1 for n in covered if n in group_of)} of them in a community")
say()

# Vectors, ordered by community so a permutation can be sliced with reduceat.
gids = sorted(g for g in groups
              if sum(n in covered for n in groups[g]) >= 2)
members = {g: [n for n in groups[g] if n in covered] for g in gids}
order = [n for g in gids for n in members[g]]
P = np.array([covered[n] for n in order])              # the profiles themselves
# L2-normalised once here, so every later dot product is already a cosine and
# mean_intra's -n self-term (which assumes unit rows) holds
X = P / np.linalg.norm(P, axis=1, keepdims=True)
sizes = np.array([len(members[g]) for g in gids])
starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
N = len(order)


def mean_intra(sums, ns):
    """Mean pairwise similarity inside each set, from the set's vector sum:
    ||sum||^2 counts every ordered pair plus the n unit self-terms."""
    return (np.einsum("ij,ij->i", sums, sums) - ns) / (ns * (ns - 1))


# ── 2. how alike are two authors in one community? ───────────────────────────
# The pooled mean is each community's own mean pairwise similarity, weighted by
# how many pairs it contributes -- i.e. the mean over every same-community pair.
obs_per_group = mean_intra(np.add.reduceat(X, starts), sizes)
pair_counts = sizes * (sizes - 1) / 2
pooled = float((obs_per_group * pair_counts).sum() / pair_counts.sum())

# The median needs the pairs themselves, not just each group's mean.
same_all = np.concatenate([(lambda S: S[np.triu_indices(n, 1)])(
    X[s:s + n] @ X[s:s + n].T) for s, n in zip(starts, sizes)])

say("Genre similarity within a community   (cosine of the genre profiles, 0-1)")
say(f"  {'':<36}{'mean':>6}{'median':>8}{'pairs':>10}")
say(f"  {'two authors in one community':<36}{pooled:>6.3f}"
    f"{np.median(same_all):>8.3f}{int(pair_counts.sum()):>10,}")
say("  (this is the statistic tested below; its mean is the pooled one, "
    "weighted by each community's pair count)")
say()

# ── 3. permutation test ──────────────────────────────────────────────────────
# How genre-coherent would ANY partition of these authors into groups of these
# sizes be? Same draw for every community, so the per-community null comes free
# and is size-matched.
null_pooled = np.empty(N_PERM)
null_group = np.empty((N_PERM, len(gids)))
for i in range(N_PERM):
    S = np.add.reduceat(X[rng.permutation(N)], starts)
    null_group[i] = mean_intra(S, sizes)
    null_pooled[i] = (null_group[i] * pair_counts).sum() / pair_counts.sum()

mu, sd = null_pooled.mean(), null_pooled.std(ddof=1)
p_val = (1 + int((null_pooled >= pooled).sum())) / (N_PERM + 1)
say(f"Permutation test ({N_PERM} shuffles of the community labels, sizes held)")
say(f"  observed {pooled:.3f}   null {mu:.3f} ± {sd:.4f}   "
    f"z = {(pooled - mu) / sd:+.1f}   p < {p_val:.4f}")
say(f"  null range {null_pooled.min():.3f}-{null_pooled.max():.3f}, so no "
    f"shuffle came near the real partition")
say()

g_mu, g_sd = null_group.mean(0), null_group.std(0, ddof=1)
z = (obs_per_group - g_mu) / g_sd
lo, hi = np.percentile(null_group, [5, 95], axis=0)

# ── 4. what each community is *about* ────────────────────────────────────────
mean_prof = np.add.reduceat(P, starts) / sizes[:, None]
top_genre = mean_prof.argmax(1)
top_share = mean_prof.max(1)
own_top = P.argmax(1)
purity = np.array([(own_top[s:s + n] == top_genre[i]).mean()
                   for i, (s, n) in enumerate(zip(starts, sizes))])


def top3(i):
    o = np.argsort(-mean_prof[i])[:3]
    return ", ".join(f"{GENRE_NAMES[j]} {mean_prof[i][j]:.2f}" for j in o)


def title_of(g):
    return community_label(g, titles, groups[g], 3)


with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "group_title", "size", "covered", "genre_similarity",
                "null_mean", "z", "dominant_genre", "dominant_share",
                "purity", "top_genres"])
    for i, g in enumerate(gids):
        w.writerow([f"G{g}", titles.get(g, ""), len(groups[g]), sizes[i],
                    f"{obs_per_group[i]:.4f}", f"{g_mu[i]:.4f}", f"{z[i]:.1f}",
                    GENRE_NAMES[top_genre[i]], f"{top_share[i]:.3f}",
                    f"{purity[i]:.3f}", top3(i)])
print(f"Saved: {OUT_CSV}")

rows = [i for i in range(len(gids)) if sizes[i] >= MIN_COVERED]
say(f"Per community ({len(rows)} with >= {MIN_COVERED} covered members)")
say(f"  above its size-matched null: "
    f"{sum(z[i] > 0 for i in rows)}/{len(rows)}   "
    f"beyond it (z > 2): {sum(z[i] > 2 for i in rows)}/{len(rows)}")
say(f"  median genre similarity {np.median(obs_per_group[rows]):.3f} "
    f"vs. {np.median(g_mu[rows]):.3f} expected · median purity "
    f"{np.median(purity[rows]):.2f}")
say()

# By size. A small community scores higher (less room to drift) but is weaker
# evidence (few pairs -> wide null), so raw score and z are reported together.
BANDS = [(2, 4), (5, 9), (10, 24), (25, 49), (50, 10**9)]
say("By community size (covered members, since that is what is scored)")
say(f"  {'size':<10}{'groups':>7}{'median sim':>12}{'median z':>10}"
    f"{'> null':>8}{'z > 2':>8}{'median purity':>15}")
for lo_n, hi_n in BANDS:
    b = [i for i in range(len(gids)) if lo_n <= sizes[i] <= hi_n]
    if not b:
        continue
    band = f"{lo_n}-{hi_n}" if hi_n < 10**9 else f"{lo_n}+"
    say(f"  {band:<10}{len(b):>7}{np.median(obs_per_group[b]):>12.3f}"
        f"{np.median(z[b]):>10.1f}"
        f"{sum(z[i] > 0 for i in b) / len(b):>7.0%}"
        f"{sum(z[i] > 2 for i in b) / len(b):>8.0%}"
        f"{np.median(purity[b]):>15.2f}")
say(f"  all {len(gids)} scored communities have >= 2 covered members; the "
    f"{len(groups) - len(gids)} others cannot be scored at all")
say()
for label, sel in (("Most genre-coherent",
                    sorted(rows, key=lambda i: -obs_per_group[i])[:10]),
                   ("Least genre-coherent",
                    sorted(rows, key=lambda i: obs_per_group[i])[:8])):
    say(f"{label}")
    for i in sel:
        say(f"  G{gids[i]:<4} {sizes[i]:>3}/{len(groups[gids[i]]):<4} "
            f"sim {obs_per_group[i]:.2f}  z {z[i]:>+5.1f}  "
            f"pure {purity[i]:.2f}  {title_of(gids[i])[:34]:<35} {top3(i)}")
    say()

OUT_TXT.write_text("\n".join(report) + "\n")
print(f"Saved: {OUT_TXT}")


# ── 5. figure ────────────────────────────────────────────────────────────────
# Colour is identity, not magnitude: the seven communities that carry a hue in
# the other figures keep it, so a reader can find G2 here without a legend.
fig = plt.figure(figsize=(FIG_W, 6.8))
fig.patch.set_facecolor(SURFACE)
ax2 = fig.add_axes([0.265, 0.072, 0.495, 0.840])
ax2.set_facecolor(SURFACE)
for side in ("top", "right", "left"):
    ax2.spines[side].set_visible(False)
ax2.spines["bottom"].set_color(EDGE_COL)
ax2.tick_params(colors=INK_MUTE, labelsize=FS_NOTE, length=3, color=EDGE_COL)

# -- per-community coherence against its size-matched null -------------------
rows = sorted(range(len(gids)), key=lambda i: -sizes[i])[:N_ROWS]
rows = sorted(rows, key=lambda i: obs_per_group[i])
y = np.arange(len(rows))
for k, i in enumerate(rows):
    ax2.plot([lo[i], hi[i]], [k, k], color=EDGE_COL, lw=3.5,
             solid_capstyle="round", zorder=1)
ax2.scatter(obs_per_group[rows], y, s=52, zorder=3, linewidths=1.0,
            edgecolors=SURFACE,
            color=[SLOTS[gids[i]] if gids[i] < len(SLOTS) else OTHER
                   for i in rows])
ax2.set_yticks(y)
# shorten on a word boundary: a title clipped mid-word ("private-eye tr")
# costs the reader more than the word it saves
ax2.set_yticklabels(
    [textwrap.shorten(title_of(gids[i]), 36, placeholder="…") for i in rows],
    fontsize=FS_LABEL, color=INK)
ax2.set_ylim(-0.8, len(rows) + 1.1)
ax2.set_xlim(0, 1)
# Grid on the quantitative axis only, recessive by weight (0.7 against the
# bands' 3.5) rather than by colour, which vanishes when the PNG is downscaled.
ax2.vlines(np.arange(0.2, 1.01, 0.2), -0.8, len(rows) - 0.5,
           color=EDGE_COL, lw=0.7, zorder=0)
# Leader per row, stopping at the dot rather than spanning the row: half the
# ink, and length redundantly encodes the value.
ax2.hlines(y, 0, obs_per_group[rows], color=EDGE_COL, lw=0.6, zorder=0)
for k, i in enumerate(rows):
    ax2.text(1.02, k, f"{GENRE_NAMES[top_genre[i]]}  {top_share[i]:.0%}",
             transform=ax2.get_yaxis_transform(), fontsize=FS_NOTE,
             color=INK_MUTE, va="center")
ax2.text(1.02, len(rows) + 0.1, "dominant genre,\nand its share",
         transform=ax2.get_yaxis_transform(), fontsize=FS_NOTE, color=INK_2,
         va="bottom", linespacing=1.3)
ax2.set_xlabel("mean genre similarity within the community",
               fontsize=FS_CAPTION, color=INK_2, labelpad=6)
best = rows[-1]
ax2.annotate("a random group\nof the same size\nwould score here",
             xy=(lo[best], len(rows) - 1), xytext=(0.0, len(rows) + 0.55),
             fontsize=FS_NOTE, color=INK_MUTE, linespacing=1.35, va="center",
             ha="left", arrowprops=dict(arrowstyle="-", color=INK_MUTE,
                                        lw=0.6,
                                        connectionstyle="arc3,rad=-0.2"))
fig.suptitle("Do the co-mention communities hold together by genre?", x=0.012,
             ha="left", fontsize=FS_TITLE, color=INK, y=0.995)
# 0.955, not 0.962: the title's descenders ran into the subtitle. The title
# can't rise (the crop clamps at the canvas top) and the null note starts at
# 0.929, so this splits the pocket evenly -- ~2.4pt of air either side.
fig.text(0.012, 0.955,
         f"Goodreads shelves, cosine · two authors in one community: "
         f"mean {pooled:.2f}, median {np.median(same_all):.2f} · "
         f"{mu:.2f} at random",
         fontsize=FS_CAPTION, color=INK_2, va="top", linespacing=1.45)
# Not viz_common.save(): tight_layout fights the add_axes placement above.
# Crop slack height, never width, so this reproduces at the shared scale.
fig.canvas.draw()
bb = fig.get_tightbbox(fig.canvas.get_renderer())
# The longest community title overhangs the left edge, so slide the window left
# rather than widening it; the width stays exactly FIG_W.
x0 = min(0.0, bb.x0)
box = Bbox([[x0, max(0.0, bb.y0 - 0.05)],
            [x0 + FIG_W, min(fig.get_size_inches()[1], bb.y1 + 0.05)]])
fig.savefig(FIG, dpi=DPI, bbox_inches=box, facecolor=SURFACE)
plt.close(fig)
print(f"Saved: {FIG}  ({box.width:.2f} x {box.height:.2f} in canvas)")
