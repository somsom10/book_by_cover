"""
שלושה איורים עצמאיים לדוח המקוצר, בגודל ובגופן שמתאימים למסמך Word.

הכל נקרא מ-final_refit/ - אין כאן שום התאמת מודל, ולכן האיורים תמיד עקביים
עם המספרים שבטקסט.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

SRC = "final_refit"
PARTIAL = 2010                      # 2010-2017, עשור חלקי - מוחרג מכל חישוב
INK, MUTED, HAIR = "#1a1a1a", "#6b6b6b", "#d8d8d8"
BLUE, RED, GREEN = "#1f4e79", "#a01f1f", "#1e8449"

plt.rcParams.update({
    "font.size": 10, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#999999", "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def load():
    sh = pd.read_csv(f"{SRC}/topic_shares_by_decade.csv", index_col=0) * 100
    lab = pd.read_csv(f"{SRC}/topic_labels.csv", index_col=0)["0"]
    lift = pd.read_csv(f"{SRC}/topic_lift_by_decade.csv", index_col=0)
    ci = pd.DataFrame({t: ((lift[f"{t}_hi"] - lift[f"{t}_lo"]) / 2 / lift[t]).values * sh[t].values
                       for t in sh.columns}, index=sh.index)
    return sh, lab, lift, ci


def full_decades(sh):
    return [d for d in sh.index if d < PARTIAL]


def order_by_peak(sh, dec):
    """סדר לפי argmax גולמי. החלקה מזיזה פסגות מאומתות ולכן אינה בשימוש."""
    rows = [(dec[int(sh.loc[dec, t].values.argmax())], -float(sh.loc[dec, t].mean()), t)
            for t in sh.columns]
    return [t for _, _, t in sorted(rows)], {t: d for d, _, t in rows}


def dec_label(d):
    """1900 ו-2000 שניהם "00s" על אותו ציר, ולכן הקצוות נכתבים במלואם."""
    return f"{d}s" if d % 100 == 0 else f"{str(d)[2:]}s"


def short(lab, t, k=4):
    return ", ".join(str(lab[t]).split(", ")[:k])


def peak_span(lift, t, dec, peak_d):
    """טווח העשורים שרווח הסמך שלהם חופף לזה של עשור השיא."""
    i = dec.index(peak_d)
    lo0, hi0 = lift.loc[peak_d, f"{t}_lo"], lift.loc[peak_d, f"{t}_hi"]
    a = b = i
    while a > 0 and lift.loc[dec[a-1], f"{t}_hi"] >= lo0 and lift.loc[dec[a-1], f"{t}_lo"] <= hi0:
        a -= 1
    while b < len(dec)-1 and lift.loc[dec[b+1], f"{t}_hi"] >= lo0 and lift.loc[dec[b+1], f"{t}_lo"] <= hi0:
        b += 1
    return dec[a], dec[b]


# ------------------------------------------------------------- איור 1: מפת תקופות
def fig_era_map(sh, lab, lift, dec, ordered, peak):
    """
    צבע = מינימום-מקסימום בתוך השורה, ולכן הוא עונה על "מתי" בלבד. הוא אינו
    יכול לענות על "כמה": נושא שזז 0.7 נקודת אחוז מקבל בדיוק אותו טווח צבעים
    כמו נושא שזז 5.7. העמודה מימין נותנת את הגודל במספר - היחס בין העשור
    הגבוה לנמוך - כך שאי אפשר לקרוא שורה שטוחה כתקופה חזקה.
    """
    fig = plt.figure(figsize=(9.6, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[8.4, 1.3], left=.285, right=.965,
                          top=.875, bottom=.085, wspace=.035)
    ax, axg = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    M = sh.loc[dec, ordered].values.T
    rng = np.ptp(M, axis=1, keepdims=True)
    norm = (M - M.min(axis=1, keepdims=True)) / np.where(rng == 0, 1, rng)
    ax.imshow(norm, aspect="auto", cmap="magma", interpolation="nearest")
    for r, t in enumerate(ordered):
        ax.add_patch(Rectangle((dec.index(peak[t]) - .5, r - .5), 1, 1,
                               fill=False, ec="white", lw=1.6))
    ax.set_xticks(range(len(dec)))
    ax.set_xticklabels([dec_label(d) for d in dec], size=9)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([short(lab, t, 4) for t in ordered], size=8.4)
    ax.tick_params(length=0)
    ax.set_title("when each theme peaks", size=9, color=MUTED, loc="left", pad=6)

    ratio = np.array([lift.loc[dec, t].max() / lift.loc[dec, t].min() for t in ordered])
    axg.barh(range(len(ordered)), ratio, color="#9fb8cd", height=.62)
    axg.set_xlim(0, ratio.max() * 1.45)
    axg.set_ylim(len(ordered) - .5, -.5)
    axg.set_yticks([]); axg.set_xticks([1, 2, 3])
    axg.set_xticklabels(["1x", "2x", "3x"], size=7.5)
    axg.tick_params(length=0)
    # התווית מפורשת: המספר הוא היחס בין העשור הגבוה לנמוך של אותו נושא, ולא
    # מדד מופשט. "כמה הוא זז" נשמע כמו יחידה שצריך לפענח, וזה בדיוק מה שקרה
    axg.set_title("biggest decade\n÷ smallest", size=8.5, color=MUTED, loc="left",
                  pad=6, linespacing=1.35)
    for r, v in enumerate(ratio):
        axg.text(v + ratio.max() * .05, r, f"{v:.1f}x", va="center", size=7.2, color=MUTED)
    for s in list(ax.spines.values()) + list(axg.spines.values()):
        s.set_visible(False)
    fig.suptitle("Which themes belong to which era", size=12.5, x=.285,
                 ha="left", y=.975)
    fig.savefig("wfig1_era_map.pdf", bbox_inches="tight")
    fig.savefig("wfig1_era_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------- איור 2: ארבע המגמות
def fig_headline(sh, lab, lift, ci, dec, peak):
    """
    העשור החלקי (2010-2017) מצויר מקווקו ולא מושמט: הטקסט מצטט אותו, ולכן
    קורא שאינו רואה אותו בגרף אינו יכול לאמת את הטענה. הוא מסומן במפורש
    ואינו נכנס לחישוב השיא, הטווח או הממוצע.
    """
    # ארבעה נושאים בארבעה עשורי שיא שונים, לפי סדר כרונולוגי. כולם עוברים
    # את בדיקת ההתאמה מחדש ב-r >= 0.98, הגבוה מבין כל הנושאים המדווחים
    picks = [("T03", "Adventure and tales"), ("T04", "Detective fiction"),
             ("T06", "War writing"), ("T13", "Guides and how-to")]
    tail = [d for d in sh.index if d >= PARTIAL]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.4))
    fig.subplots_adjust(left=.075, right=.985, top=.855, bottom=.115,
                        hspace=.62, wspace=.20)
    for ax, (t, name) in zip(axes.ravel(), picks):
        v = sh.loc[dec, t].values
        e = ci.loc[dec, t].values
        ax.plot(dec, v, lw=2.3, color=BLUE, solid_capstyle="round")
        ax.fill_between(dec, v - e, v + e, color=BLUE, alpha=.18, lw=0)
        if tail:
            ax.plot([dec[-1]] + tail, [v[-1]] + list(sh.loc[tail, t].values),
                    lw=1.9, ls=":", color=BLUE, alpha=.75)
            ax.annotate(f"{sh.loc[tail[-1], t]:.1f}%", (tail[-1], sh.loc[tail[-1], t]),
                        xytext=(-2, 4), textcoords="offset points", ha="right",
                        size=8, color=MUTED)
        pk = peak[t]; j = dec.index(pk)
        lo_d, hi_d = peak_span(lift, t, dec, pk)
        if hi_d > lo_d:
            ax.axvspan(lo_d - 5, hi_d + 5, color=RED, alpha=.09, lw=0, zorder=0)
        ax.plot([pk], [v[j]], "o", ms=6, color=RED, zorder=4)
        span = f"{pk}s" if hi_d == lo_d else f"{lo_d}s–{str(hi_d)[2:]}s"
        # שיא בקצה הציר דוחף את התווית אל מחוץ לאיור, אז מזיזים אותה פנימה
        ha, dx = ("center", 0) if j < len(dec) - 2 else ("right", 4)
        ax.annotate(f"{v[j]:.1f}%  {span}", (pk, v[j]), xytext=(dx, 8),
                    textcoords="offset points", ha=ha, size=9,
                    color=RED, weight="bold")
        ax.axhline(v.mean(), color=HAIR, lw=1, zorder=0)
        ax.set_title(f"{name}\n{short(lab, t, 4)}", size=9.5, loc="left",
                     pad=6, linespacing=1.4)
        top = max(v.max(), sh.loc[tail, t].max() if tail else 0)
        ax.set_ylim(0, top * 1.35)
        ticks = dec + tail
        ax.set_xticks(ticks)
        # רק העשור הראשון נכתב במלואו: 12 תוויות בחצי רוחב, ו-"2000s" ליד
        # "10s*" נדבקות זו לזו. "1900s" בקצה השמאלי מספיק כדי לעגן את הציר
        ax.set_xticklabels([(dec_label(d) if d == ticks[0] else f"{str(d)[2:]}s")
                            + ("*" if d >= PARTIAL else "") for d in ticks],
                           size=7.4)
        ax.set_ylabel("% of decade's text", size=8.5)
        ax.tick_params(labelsize=8.5)
        ax.grid(axis="y", alpha=.18)
    fig.text(.075, .022, "* 2010s covers 2010–2017 only (dotted); it is excluded from every "
             "peak, average and interval above.", size=8, color=MUTED)
    fig.suptitle("Four themes, four peak decades", size=12.5, x=.075,
                 ha="left", y=.972)
    fig.savefig("wfig2_headline.pdf", bbox_inches="tight")
    fig.savefig("wfig2_headline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------- איור 3: מה עלה ומה ירד
def fig_rise_fall(sh, lab, dec):
    early = [d for d in dec if d <= 1940]
    late = [d for d in dec if d >= 1960]
    a, b = sh.loc[early].mean(), sh.loc[late].mean()
    order = list((b - a).sort_values().index)
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    fig.subplots_adjust(left=.285, right=.965, top=.925, bottom=.085)
    for i, t in enumerate(order):
        c = GREEN if b[t] >= a[t] else RED
        ax.annotate("", xy=(b[t], i), xytext=(a[t], i),
                    arrowprops=dict(arrowstyle="-|>,head_width=.22,head_length=.5",
                                    color=c, lw=2.1, shrinkA=0, shrinkB=0))
        ax.plot([a[t]], [i], "o", ms=5, mfc="white", mec=c, mew=1.6, zorder=3)
        ax.text(max(a[t], b[t]) + .12, i, f"{b[t]-a[t]:+.1f}", va="center",
                size=8, color=c, weight="bold")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([short(lab, t, 3) for t in order], size=8.4)
    ax.tick_params(length=0, labelsize=9)
    ax.set_xlabel("share of the decade's text (%)", size=9)
    ax.set_xlim(0, max(a.max(), b.max()) * 1.2)
    ax.grid(axis="x", color=HAIR, lw=.7); ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], color=GREEN, lw=2.4, label="grew"),
                       Line2D([], [], color=RED, lw=2.4, label="shrank"),
                       Line2D([], [], color=MUTED, marker="o", mfc="white", lw=0,
                              label="1900s–1940s mean")],
              fontsize=8.5, loc="lower left", frameon=False)
    fig.suptitle("What grew and what shrank: 1900s–1940s mean vs "
                 "1960s–2000s mean", size=12.5, x=.285, ha="left", y=.975)
    fig.savefig("wfig3_rise_fall.pdf", bbox_inches="tight")
    fig.savefig("wfig3_rise_fall.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    sh, lab, lift, ci = load()
    dec = full_decades(sh)
    ordered, peak = order_by_peak(sh, dec)
    fig_era_map(sh, lab, lift, dec, ordered, peak)
    fig_headline(sh, lab, lift, ci, dec, peak)
    fig_rise_fall(sh, lab, dec)
    print("wrote wfig1_era_map, wfig2_headline, wfig3_rise_fall (.pdf and .png)")
    for t, name in [("T03", "adventure"), ("T04", "detective"), ("T06", "war"),
                    ("T13", "guides")]:
        v = sh.loc[dec, t]
        lo_d, hi_d = peak_span(lift, t, dec, peak[t])
        print(f"  {name:9} peak {peak[t]}s {v.max():.1f}%  span {lo_d}-{hi_d}  "
              f"lift {lift.loc[peak[t], t]:.2f} "
              f"[{lift.loc[peak[t], t+'_lo']:.2f}, {lift.loc[peak[t], t+'_hi']:.2f}]")


if __name__ == "__main__":
    main()
