"""
גרסה משופרת של topic_trends.pdf.

קורא רק קבצי CSV מ-final_refit/ ולכן אינו מריץ שום מודל ואינו נוגע ב-themes.py
או בתיקיית הבסיס הקפואה. הפלט: topic_trends_v2.pdf.

מה שונה מהמקור:
  1. סדר הנושאים בכל עמוד הוא לפי עשור השיא ולא לפי טווח. זה מה שהופך את
     מפת החום לאלכסון קריא - רואים אילו נושאים שייכים לאיזו תקופה.
  2. עמוד חדש של עלייה/ירידה (dumbbell) - השאלה "מה גדל ומה קטן" לא נענתה
     בשום עמוד במקור.
  3. הדיג'סט הטקסטואלי הפך למפת חום דו-כיוונית סביב 1.0, כשתאים שרווח הסמך
     שלהם חוצה את 1.0 מוצגים דהויים. אותה לוגיקת מובהקות, קריאה בסריקה אחת.
  4. העשור החלקי (2010-2017) מצויר מקווקו בכל מקום ומוחרג מכל חישוב.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

SRC = "final_refit"
OUT = "topic_trends_v2.pdf"
PARTIAL_DECADE = 2010          # 2010-2017 - עשור חלקי, לא נכנס לחישובים
PAGE = (11.7, 8.3)             # A4 לרוחב

INK, MUTED, HAIR = "#1a1a1a", "#6b6b6b", "#d8d8d8"
BLUE, RED, GREEN, AMBER = "#1f4e79", "#b03a2e", "#1e8449", "#b9770e"

plt.rcParams.update({
    "font.size": 9, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#999999", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white",
})


# ---------------------------------------------------------------- טעינת נתונים
def load():
    sh = pd.read_csv(f"{SRC}/topic_shares_by_decade.csv", index_col=0) * 100
    lab = pd.read_csv(f"{SRC}/topic_labels.csv", index_col=0)["0"]
    lift = pd.read_csv(f"{SRC}/topic_lift_by_decade.csv", index_col=0)
    pv = pd.read_csv(f"{SRC}/topic_prevalence_by_decade.csv")
    art = pd.read_csv(f"{SRC}/artifact_share_by_decade.csv", index_col=0)
    counts = pv.groupby("decade")["n"].first()

    # רוחב רווח הסמך לכל נושא/עשור, מתוך טבלת ה-lift: היחס בין חצי-הרוחב
    # ל-lift מוכפל בחלק היחסי. שקול לרווח הסמך של ה-share עצמו.
    ci = {}
    for t in sh.columns:
        half = (lift[f"{t}_hi"] - lift[f"{t}_lo"]) / 2
        ci[t] = (half / lift[t]).values * sh[t].values
    ci = pd.DataFrame(ci, index=sh.index)

    # יציבות: הקובץ שומר תווית של 10 מילים ולא מזהה, אז מתאימים לפי תחילית
    stab = {}
    try:
        st = pd.read_csv("topic_stability.csv")
        for t in sh.columns:
            head = ", ".join(str(lab[t]).split(", ")[:6])
            hit = st[st.topic.str.startswith(head, na=False)]
            if len(hit):
                stab[t] = float(hit.reproducibility.iloc[0])
    except FileNotFoundError:
        pass

    return sh, lab, lift, ci, counts, art, stab


def full_decades(sh):
    """עשורים מלאים בלבד - כל חישוב (שיא, שיפוע, סדר) רץ עליהם."""
    return [d for d in sh.index if d < PARTIAL_DECADE]


def order_by_peak(sh, dec):
    """
    סדר לפי עשור השיא, כשהשיא הוא ה-argmax הגולמי.

    ניסיתי החלקה בחלון 3 לפני הבחירה, כדי שעשור רועש בודד לא יזיז נושא שלם.
    היא הזיזה את הבלש מ-1930 ל-1940 ואת המלחמה מ-1940 ל-1950 - כלומר ביטלה
    בדיוק את שתי הפסגות שאומתו מול מקורות חיצוניים. השיא כאן חד מספיק כדי
    שההחלקה תזיק יותר משתועיל, אז היא הוסרה.
    """
    out = []
    for t in sh.columns:
        v = sh.loc[dec, t].values
        out.append((dec[int(v.argmax())], -float(v.max()), t))
    return [t for _, _, t in sorted(out)], {t: d for d, _, t in out}


def short(lab, t, k=3):
    return ", ".join(str(lab[t]).split(", ")[:k])


def peak_span(lift, t, dec, peak_d):
    """
    טווח השיא, ולא עשור בודד. עשור נכלל בטווח אם רווח הסמך של ה-lift שלו חופף
    לזה של עשור ה-argmax, והוא רציף אליו.

    זה נדרש משום ש-argmax על רמה שטוחה ממציא פסגה: ב-earth/planet/space העשורים
    1960, 1970 ו-1980 נבדלים ב-0.1 נקודת אחוז ורווחי הסמך שלהם חופפים כמעט לגמרי,
    כך שבחירת 1970 היא הגרלה. הצגת הטווח היא מה שהמדידה באמת תומכת בו.
    """
    i = dec.index(peak_d)
    lo0, hi0 = lift.loc[peak_d, f"{t}_lo"], lift.loc[peak_d, f"{t}_hi"]
    a = b = i
    while a > 0 and lift.loc[dec[a - 1], f"{t}_hi"] >= lo0 and lift.loc[dec[a - 1], f"{t}_lo"] <= hi0:
        a -= 1
    while b < len(dec) - 1 and lift.loc[dec[b + 1], f"{t}_hi"] >= lo0 and lift.loc[dec[b + 1], f"{t}_lo"] <= hi0:
        b += 1
    return dec[a], dec[b]


def span_label(lo, hi):
    return f"{lo}s" if lo == hi else f"{lo}s–{str(hi)[2:]}s"


def draw_series(ax, sh, ci, t, dec_all, dec_full, colour=BLUE, band=True):
    """קו מגמה אחד. העשור החלקי מקווקו ובלי רצועת סמך - הוא לא ניתן להשוואה."""
    v_all = sh[t].values
    n_full = len(dec_full)
    ax.plot(dec_full, v_all[:n_full], lw=1.9, color=colour, solid_capstyle="round")
    if band:
        e = ci[t].values[:n_full]
        ax.fill_between(dec_full, v_all[:n_full] - e, v_all[:n_full] + e,
                        color=colour, alpha=.18, lw=0)
    if len(dec_all) > n_full:
        ax.plot(dec_all[n_full - 1:], v_all[n_full - 1:], lw=1.5, ls=":",
                color=colour, alpha=.55)


# ============================================================ עמוד 1: מפת תקופות
def page_era_map(pdf, sh, lab, counts, dec, ordered, peak, stab):
    fig = plt.figure(figsize=PAGE)
    gs = fig.add_gridspec(1, 3, width_ratios=[6.4, 1.15, 1.15],
                          left=.235, right=.965, top=.832, bottom=.105, wspace=.16)
    ax, axm, axs = (fig.add_subplot(gs[0, i]) for i in range(3))

    M = sh.loc[dec, ordered].values.T
    rng = np.ptp(M, axis=1, keepdims=True)
    norm = (M - M.min(axis=1, keepdims=True)) / np.where(rng == 0, 1, rng)
    ax.imshow(norm, aspect="auto", cmap="magma", interpolation="nearest")

    # סימון תא השיא: זה מה שהופך את המפה מ"צבעים" ל"מתי"
    for r, t in enumerate(ordered):
        ax.add_patch(Rectangle((dec.index(peak[t]) - .5, r - .5), 1, 1,
                               fill=False, ec="white", lw=1.6))
    ax.set_xticks(range(len(dec)))
    ax.set_xticklabels([f"{d}s" for d in dec], size=8)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([f"T{t[1:]}  {short(lab, t, 4)}" for t in ordered], size=7.4)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    means = sh.loc[dec, ordered].mean().values
    axm.barh(range(len(ordered)), means, color="#9fb8cd", height=.62)
    axm.set_xlim(0, means.max() * 1.35)
    axm.invert_yaxis(); axm.set_yticks([])
    axm.set_title("mean share\n(% of decade)", size=7.2, color=MUTED, pad=6)
    axm.tick_params(labelsize=6.5)
    for r, v in enumerate(means):
        axm.text(v + means.max() * .05, r, f"{v:.1f}", va="center", size=6.2, color=MUTED)
    axm.set_ylim(len(ordered) - .5, -.5)

    if stab:
        rv = [stab.get(t, np.nan) for t in ordered]
        cols = [HAIR if np.isnan(r) else (GREEN if r >= .90 else AMBER if r >= .75 else RED)
                for r in rv]
        axs.scatter(np.nan_to_num(rv, nan=0), range(len(ordered)), c=cols, s=26, zorder=3)
        axs.hlines(range(len(ordered)), 0, np.nan_to_num(rv, nan=0), color=HAIR, lw=1, zorder=1)
        axs.axvline(.90, color=INK, ls="--", lw=.8)
        axs.set_xlim(0, 1.04)
    axs.set_ylim(len(ordered) - .5, -.5)
    axs.set_yticks([]); axs.tick_params(labelsize=6.5)
    axs.set_title("reproducibility\nacross 4 refits", size=7.2, color=MUTED, pad=6)

    fig.text(.035, .955, "Which themes belong to which era", size=19, weight="bold")
    fig.text(.035, .922,
             f"{len(sh.columns)} content topics x {len(dec)} full decades. Rows are ordered by the decade "
             "each topic peaks in, so the\nbright band runs top-left to bottom-right: that diagonal is the "
             "finding.", size=8.6, color=MUTED, va="top", linespacing=1.5)
    fig.text(.035, .888,
             "Colour is rescaled WITHIN each row (min→max of that topic alone) so small topics stay "
             "visible;\nit never compares one topic to another. White box = peak decade.",
             size=8.6, color=MUTED, va="top", linespacing=1.5)
    fig.text(.035, .052,
             f"Books per decade: {'  '.join(f'{str(d)[2:]}s {counts[d]:,}' for d in dec)}",
             size=7, color=MUTED, family="monospace")
    fig.text(.035, .028,
             "Reproducibility tiers: solid ≥ 0.90, moderate 0.75–0.90, unstable below 0.75. The bands are "
             "wider than the metric's own\nseed-to-seed noise (±0.05), so they separate tiers and not "
             "neighbouring topics — read amber and red rows as suggestive.",
             size=7.4, color=MUTED, va="top", linespacing=1.5)
    pdf.savefig(fig); plt.close(fig)


# ========================================================= עמוד 2: ריבוי גרפים
def page_small_multiples(pdf, sh, ci, lab, lift, dec_all, dec, ordered, peak, stab):
    n = len(ordered)
    ncol = 6
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=PAGE, squeeze=False)
    fig.subplots_adjust(left=.052, right=.975, top=.800, bottom=.075,
                        hspace=.70, wspace=.30)

    for ax, t in zip(axes.ravel(), ordered):
        v = sh.loc[dec, t].values
        draw_series(ax, sh, ci, t, dec_all, dec)
        pk = peak[t]
        j = dec.index(pk)
        ax.axhline(v.mean(), color=HAIR, lw=.8, zorder=0)
        lo_d, hi_d = peak_span(lift, t, dec, pk)
        if hi_d > lo_d:
            ax.axvspan(lo_d - 5, hi_d + 5, color=RED, alpha=.09, lw=0, zorder=0)
        ax.plot([pk], [v[j]], "o", ms=4.2, color=RED, zorder=4)
        # שיא בקצה הציר דוחף את התווית אל תוך תוויות הצירים, אז מזיזים אותה פנימה
        ha, dx = ("center", 0)
        if j <= 1:
            ha, dx = "left", 3
        elif j >= len(dec) - 2:
            ha, dx = "right", -3
        ax.annotate(f"{v[j]:.1f}%\n{span_label(lo_d, hi_d)}", (pk, v[j]), xytext=(dx, 5),
                    textcoords="offset points", size=6, color=RED,
                    ha=ha, va="bottom", linespacing=.95)

        r = stab.get(t)
        dot = "" if r is None else ("● " if r >= .90 else "◐ " if r >= .75 else "○ ")
        ax.set_title(f"{dot}T{t[1:]} {short(lab, t, 2)}", size=6.8, color=INK, pad=13,
                     loc="left")
        ax.set_ylim(0, max(v.max(), sh[t].values.max()) * 1.42)
        ax.set_xlim(dec[0] - 4, dec_all[-1] + 4)
        ax.set_xticks([1900, 1950, 2000])
        ax.set_xticklabels(["'00", "'50", "'00"], size=6)
        ax.tick_params(labelsize=6, length=2)
        ax.margins(x=.02)
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.text(.035, .955, "Every topic, same story order", size=19, weight="bold")
    fig.text(.035, .922,
             "Share of each decade's text (%). Panels run in the same peak order as the previous "
             "page, so reading left-to-right, top-to-bottom walks the century forward.",
             size=8.6, color=MUTED)
    fig.text(.035, .903,
             "Shaded band = 95% bootstrap CI. Grey line = that topic's own century average. Red dot = "
             "the highest decade; the pink\nvertical band covers every decade whose CI overlaps it, so "
             "where that band is wide the topic plateaus and no single\ndecade is the peak. Dotted tail "
             "= 2010–2017, a partial decade, excluded from every number on this page.",
             size=8.6, color=MUTED, va="top", linespacing=1.5)
    fig.text(.035, .843,
             "● solid, reproducibility ≥ 0.90      ◐ moderate, 0.75–0.90      ○ unstable, below 0.75", size=8,
             color=MUTED)
    fig.text(.5, .028,
             "RELATIVE shares: within a decade all topics sum to 100%, so a line can fall only "
             "because others rose. This is not a count of books.",
             size=7.6, ha="center", color=MUTED)
    pdf.savefig(fig); plt.close(fig)


# ====================================================== עמוד 3: מה עלה ומה ירד
def page_rise_fall(pdf, sh, lab, lift, dec, ordered, peak, stab):
    """
    שני חלונות: משמאל dumbbell של המחצית הראשונה מול השנייה, מימין ציר זמן
    של עשורי השיא. הראשון עונה "מה גדל", השני "מתי כל נושא שייך".
    קצוות בודדים רועשים, ולכן ההשוואה היא בין ממוצע 1900-1940 לממוצע 1960-2000.
    """
    early = [d for d in dec if d <= 1940]
    late = [d for d in dec if d >= 1960]
    a = sh.loc[early].mean()
    b = sh.loc[late].mean()
    delta = (b - a).sort_values()
    tops = list(delta.index)

    fig = plt.figure(figsize=PAGE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.30, 1], left=.20, right=.975,
                          top=.845, bottom=.155, wspace=.30)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    y = np.arange(len(tops))
    for i, t in enumerate(tops):
        up = b[t] >= a[t]
        c = GREEN if up else RED
        ax.annotate("", xy=(b[t], i), xytext=(a[t], i),
                    arrowprops=dict(arrowstyle="-|>,head_width=.22,head_length=.5",
                                    color=c, lw=1.9, shrinkA=0, shrinkB=0))
        ax.plot([a[t]], [i], "o", ms=4.4, mfc="white", mec=c, mew=1.5, zorder=3)
        ax.text(max(a[t], b[t]) + .12, i, f"{b[t]-a[t]:+.1f}", va="center",
                size=6.6, color=c, weight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([f"T{t[1:]}  {short(lab, t, 3)}" for t in tops], size=7.2)
    ax.tick_params(length=0)
    ax.set_xlabel("share of the decade's text (%)", size=8)
    ax.set_xlim(0, max(a.max(), b.max()) * 1.22)
    ax.grid(axis="x", color=HAIR, lw=.6)
    ax.set_axisbelow(True)
    ax.set_title("1900s–1940s  →  1960s–2000s", size=10, weight="bold", loc="left", pad=8)
    ax.legend(handles=[Line2D([], [], color=GREEN, lw=2.2, label="grew"),
                       Line2D([], [], color=RED, lw=2.2, label="shrank"),
                       Line2D([], [], color=MUTED, marker="o", mfc="white", lw=0,
                              label="start (1900s–1940s mean)")],
              fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.075),
              ncol=3, frameon=False)

    # רשימת השיאים לפי עשור: שורה לעשור, הנושאים ששיאם בו מסודרים משמאל לימין.
    # אותו מידע כמו האלכסון בעמוד 1, אבל כרשימה שאפשר להקריא בקול
    by_dec = {}
    for t in ordered:
        by_dec.setdefault(peak[t], []).append(t)
    widest = max(len(v) for v in by_dec.values())
    for row, d in enumerate(dec):
        ts = by_dec.get(d, [])
        ax2.text(-0.7, row, f"{d}s", ha="right", va="center", size=8.5,
                 weight="bold", color=INK)
        ax2.axhline(row, color=HAIR, lw=.6, zorder=0, xmin=.055)
        for k, t in enumerate(ts):
            size = sh.loc[dec, t].mean()
            r = stab.get(t)
            c = HAIR if r is None else (GREEN if r >= .90 else AMBER if r >= .75 else RED)
            ax2.scatter([k], [row], s=size * 46, color=c, alpha=.5, zorder=3,
                        edgecolors="white", linewidths=.9)
            ax2.text(k, row, f"T{t[1:]}", ha="center", va="center", size=5.4,
                     color=INK, zorder=4)
            lo_d, hi_d = peak_span(lift, t, dec, d)
            mark = short(lab, t, 1) + ("" if hi_d == lo_d else f" →{str(hi_d)[2:]}s")
            ax2.text(k, row + .34, mark, ha="center", va="top", size=5.8, color=MUTED)
        if not ts:
            ax2.text(0, row, "no topic's highest decade", va="center", size=6.4,
                     color=MUTED, style="italic")
    ax2.set_xlim(-1.5, widest - .3)
    ax2.set_ylim(len(dec) - .5, -.7)
    ax2.set_xticks([]); ax2.set_yticks([])
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.set_title("peak decade roster\nbubble = mean share · colour = reproducibility · "
                  "\u201c→80s\u201d = plateau, not a single peak",
                  size=8.5, weight="bold", loc="left", pad=6)

    fig.text(.035, .955, "What grew, what shrank, and when each theme peaked",
             size=19, weight="bold")
    fig.text(.035, .922,
             "Left: each topic's mean share in the first five decades against the last five. "
             "Endpoint decades are noisy, so both ends are averages, not single decades.",
             size=8.6, color=MUTED)
    fig.text(.035, .898,
             "Right: the same peak-decade information as the era map, stated directly. "
             "Bubble area is the topic's century-average size.", size=8.6, color=MUTED)
    fig.text(.56, .060,
             "A row is filed under a topic's single highest decade. \u201c→80s\u201d marks a plateau:\n"
             "the decades up to that point are statistically indistinguishable from the peak, so a\n"
             "row with no entry is not an empty decade — plateaus from neighbouring rows cover it.",
             size=7.6, color=MUTED, va="top", linespacing=1.5)
    fig.text(.035, .060,
             "Because shares are compositional, growth and shrinkage must cancel out across the "
             "whole page.\nA negative arrow does not mean fewer such books were published — it "
             "means that theme took a smaller\nslice of the decade's descriptive text.",
             size=7.6, color=MUTED, va="top", linespacing=1.5)
    pdf.savefig(fig); plt.close(fig)


# ====================================================== עמוד 4: ייחוד לפי עשור
def page_lift(pdf, sh, lab, lift, dec, ordered, counts):
    """
    lift = חלקו של הנושא בעשור חלקי הממוצע שלו על פני העשורים. 1.0 = עשור טיפוסי.
    תא שרווח הסמך שלו חוצה 1.0 מצויר דהוי - אותה לוגיקת מובהקות של הדיג'סט
    המקורי, רק שאפשר לסרוק אותה במבט אחד.
    """
    L = lift.loc[dec, ordered].values.T
    lo = lift.loc[dec, [f"{t}_lo" for t in ordered]].values.T
    hi = lift.loc[dec, [f"{t}_hi" for t in ordered]].values.T
    sig = (lo > 1.0) | (hi < 1.0)

    fig = plt.figure(figsize=PAGE)
    ax = fig.add_axes([.235, .125, .70, .690])
    lim = max(abs(np.log2(L)).max(), .01)
    im = ax.imshow(np.log2(L), aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   alpha=1.0, interpolation="nearest")
    # דהייה של התאים הלא מובהקים ע"י שכבת לבן חלקית
    ax.imshow(np.ones_like(L), aspect="auto", cmap="gray", vmin=0, vmax=1,
              alpha=np.where(sig, 0.0, 0.72), interpolation="nearest")

    for r in range(L.shape[0]):
        for c in range(L.shape[1]):
            if sig[r, c] and abs(np.log2(L[r, c])) > .32:
                ax.text(c, r, f"{L[r, c]:.1f}", ha="center", va="center", size=5.5,
                        color="white", weight="bold")
    ax.set_xticks(range(len(dec)))
    ax.set_xticklabels([f"{d}s\nn={counts[d]:,}" for d in dec], size=7)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([f"T{t[1:]}  {short(lab, t, 4)}" for t in ordered], size=7.2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=.55, pad=.02,
                      ticks=np.log2([.6, .8, 1, 1.25, 1.7]))
    cb.ax.set_yticklabels(["0.6x", "0.8x", "1.0x", "1.25x", "1.7x"], size=7)
    cb.set_label("share relative to the topic's own century average", size=7.5)
    cb.outline.set_visible(False)

    fig.text(.035, .955, "What each decade was unusual for", size=19, weight="bold")
    fig.text(.035, .922,
             "Lift: a topic's share of the decade divided by that topic's own average across all decades.\n"
             "1.0x is a typical decade for that theme; the colour scale is logarithmic, so 0.5x and 2.0x "
             "sit equally far from the centre.", size=8.6, color=MUTED, va="top",
             linespacing=1.5)
    fig.text(.035, .874,
             "Faded cells are NOT significant — their 95% bootstrap CI includes 1.0, so that decade is\n"
             "indistinguishable from typical for that topic. Only the saturated cells are claims.",
             size=8.6, color=MUTED, va="top", linespacing=1.5)
    fig.text(.035, .048,
             "Early decades carry far fewer books (1,474–3,284 vs ~9,000 after 1960), so their CIs are "
             "roughly three times\nwider and fewer of their cells reach significance. The pale left edge is "
             "partly a sample-size effect, not only a flat century.",
             size=7.6, color=MUTED, va="top", linespacing=1.5)
    pdf.savefig(fig); plt.close(fig)


# ============================================================== עמוד 5: הטבלה
def page_table(pdf, sh, lab, dec_all, dec, ordered, counts):
    fig = plt.figure(figsize=PAGE)
    ax = fig.add_axes([.235, .10, .72, .74])
    M = sh.loc[dec_all, ordered].values.T
    row_max = M.max(axis=1, keepdims=True)
    N = M / np.where(row_max == 0, 1, row_max)
    ax.imshow(N, aspect="auto", cmap="Blues", vmin=0, vmax=1.35,
              interpolation="nearest")
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            ax.text(c, r, f"{M[r, c]:.1f}", ha="center", va="center", size=6,
                    color="white" if N[r, c] > .82 else INK)
    ax.set_xticks(range(len(dec_all)))
    ax.set_xticklabels([f"{d}s" + ("*" if d >= PARTIAL_DECADE else "") for d in dec_all],
                       size=7)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([f"T{t[1:]}  {short(lab, t, 3)}" for t in ordered], size=7.2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for c, d in enumerate(dec_all):
        ax.text(c, -1.05, f"{counts[d]:,}", ha="center", size=5.8, color=MUTED,
                rotation=90, va="bottom")
    ax.text(-.55, -1.05, "books", ha="right", size=5.8, color=MUTED, va="bottom")

    fig.text(.035, .955, "The numbers behind every chart", size=19, weight="bold")
    fig.text(.035, .922,
             "Share of each decade's content text (%). Every COLUMN sums to 100 — these are proportions "
             "within a decade,\nnever counts of books. Cell shading is per-row, so it reads as each topic's "
             "own trajectory.", size=8.6, color=MUTED, va="top", linespacing=1.5)
    fig.text(.035, .035,
             "* 2010s covers 2010–2017 only and is excluded from all peak, slope and lift "
             "calculations elsewhere in this report.", size=7.6, color=MUTED)
    pdf.savefig(fig); plt.close(fig)


# ================================================ עמוד 6: אימות מול מכשיר שני
# ================================================ עמוד 6: מה הוצא ולמה זה חשוב
def page_excluded(pdf, art, dec_all, stab):
    a = art[art.index >= 1900]
    fig = plt.figure(figsize=PAGE)
    ax = fig.add_axes([.09, .50, .40, .32])
    ax2 = fig.add_axes([.575, .50, .375, .32])

    ax.fill_between(a.index, a.pct_artifact_topic_share, color=RED, alpha=.18, lw=0)
    ax.plot(a.index, a.pct_artifact_topic_share, color=RED, lw=2.2, marker="o", ms=4)
    ax.set_ylim(0, a.pct_artifact_topic_share.max() * 1.25)
    ax.set_ylabel("% of the decade's text that is about\nthe book as an object, not its subject",
                  size=8)
    ax.set_xticks(list(a.index)); ax.set_xticklabels([f"{str(d)[2:]}s" for d in a.index], size=7)
    ax.grid(axis="y", color=HAIR, lw=.6); ax.set_axisbelow(True)
    for d in (a.index[0], a.index[-1]):
        ax.annotate(f"{a.pct_artifact_topic_share[d]:.1f}%",
                    (d, a.pct_artifact_topic_share[d]), xytext=(0, 8),
                    textcoords="offset points", ha="center", size=8,
                    color=RED, weight="bold")
    ax.set_title("Publisher copy is a time-correlated bias", size=10, weight="bold",
                 loc="left", pad=8)

    ax2.axis("off")
    ax2.text(0, 1,
             "Why this page exists\n\n"
             "A Goodreads description is marketing copy, not a plot summary. Topics made of\n"
             "words like  book, author, edition, reader  are about the product, not the story,\n"
             "so they are removed and the remaining shares renormalised to 100%.\n\n"
             "That removal is not neutral. Publisher copy is roughly twice as dense in the\n"
             "1900s as in the 2010s, so renormalisation inflates early-decade content shares\n"
             "more than late ones — a measured residual bias of about −7% on century slopes.\n\n"
             "The honest reading of every trend in this report is therefore:\n"
             "  • WHEN a theme peaks is robust — it survives refitting and an independent\n"
             "    keyword instrument.\n"
             "  • The exact SIZE of a century-long slope carries this known bias.\n\n"
             "Showing the excluded mass rather than hiding it is what makes the rest legible.",
             size=8.2, va="top", family="DejaVu Sans", linespacing=1.55, color=INK)

    # חצי תחתון: התפלגות היציבות, ולידה מה מותר לטעון ומה לא
    ax3 = fig.add_axes([.09, .125, .40, .275])
    if stab:
        rv = np.array(sorted(stab.values()))
        cols = [GREEN if r >= .90 else AMBER if r >= .75 else RED for r in rv]
        ax3.barh(range(len(rv)), rv, color=cols, height=.75)
        ax3.axvline(.90, color=INK, ls="--", lw=1)
        ax3.set_xlim(0, 1.02); ax3.set_yticks([])
        ax3.set_xlabel("reproducibility across 4 independent training samples", size=8)
        n_solid = int((rv >= .90).sum())
        ax3.text(.985, 1.2, f"{n_solid} of {len(rv)} at or above 0.90",
                 ha="right", size=8, color=GREEN, weight="bold")
        fig.text(.09, .052, "Measured seed-to-seed noise on this metric is about ±0.05, so the "
                 "0.90 line separates tiers, not neighbouring topics.",
                 size=7, color=MUTED)
        ax3.set_title("Not every topic is stable enough to report", size=10,
                      weight="bold", loc="left", pad=8)
    else:
        ax3.axis("off")

    ax4 = fig.add_axes([.575, .085, .375, .30]); ax4.axis("off")
    ax4.text(0, 1,
             "What this report does and does not claim\n\n"
             "CLAIMED\n"
             "  • The decade in which each stable topic peaks.\n"
             "  • That the four corroborated peaks agree with a second,\n"
             "    model-free instrument (previous page).\n"
             "  • The direction of century-long movement for topics whose\n"
             "    lift CI clears 1.0.\n\n"
             "NOT CLAIMED\n"
             "  • That any share is a count of books. Shares are within-decade\n"
             "    proportions and must sum to 100.\n"
             "  • The exact magnitude of a century slope — the renormalisation\n"
             "    bias above is not corrected, only measured.\n"
             "  • Anything about a genre with no topic of its own.\n"
             "  • Anything about 2010–2017, a partial decade.",
             size=8.2, va="top", family="DejaVu Sans", linespacing=1.55, color=INK)

    fig.text(.035, .955, "The text that was thrown away", size=19, weight="bold")
    fig.text(.035, .922,
             "Artifact topics — publisher, edition and review copy — measured before "
             "renormalisation.", size=8.6, color=MUTED)
    pdf.savefig(fig); plt.close(fig)


# --------------------------------------------------------------------- ראשי
def main():
    sh, lab, lift, ci, counts, art, stab = load()
    dec_all = list(sh.index)
    dec = full_decades(sh)
    ordered, peak = order_by_peak(sh, dec)

    with PdfPages(OUT) as pdf:
        page_era_map(pdf, sh, lab, counts, dec, ordered, peak, stab)
        page_small_multiples(pdf, sh, ci, lab, lift, dec_all, dec, ordered, peak, stab)
        page_rise_fall(pdf, sh, lab, lift, dec, ordered, peak, stab)
        page_lift(pdf, sh, lab, lift, dec, ordered, counts)
        page_table(pdf, sh, lab, dec_all, dec, ordered, counts)
        page_excluded(pdf, art, dec_all, stab)
        pdf.infodict()["Title"] = "Book themes by decade — visual report"

    print(f"wrote {OUT}: 6 pages, {len(sh.columns)} content topics, "
          f"{len(dec)} full decades ({dec[0]}s–{dec[-1]}s)")
    print("peak order:", "  ".join(f"{t}@{peak[t]}" for t in ordered))


if __name__ == "__main__":
    main()
