"""
All visualization functions for the TEXTile Factory project.
Each function saves a PNG and also returns the figure.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = "deep"
DPI = 150
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)


def _save(fig, name: str):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved -> {path}")
    return fig


# Data / EDA


def plot_year_distribution(df: pd.DataFrame, year_col: str = "year") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(
        df[year_col].dropna(),
        bins=60,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlabel("Publication Year")
    ax.set_ylabel("Number of Books")
    ax.set_title("Year Distribution of CMU Book Summaries")
    return _save(fig, "year_distribution")


def plot_genre_counts(
    genre_counts: dict, title: str = "Top Genres by Book Count"
) -> plt.Figure:
    genres = list(genre_counts.keys())
    counts = list(genre_counts.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(
        genres[::-1], counts[::-1], color=sns.color_palette(PALETTE, len(genres))
    )
    ax.set_xlabel("Number of Books")
    ax.set_title(title)
    ax.bar_label(bars, padding=3, fmt="%d", fontsize=9)
    return _save(fig, "genre_counts")



def plot_decade_text_stats(decade_stats: pd.DataFrame) -> plt.Figure:
    cols = [c for c in decade_stats.columns if c != "decade"]
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.plot(decade_stats.index, decade_stats[col], marker="o", linewidth=2)
        ax.set_title(col.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("Decade")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Text Statistics per Decade", fontsize=13, y=1.01)
    fig.tight_layout()
    return _save(fig, "decade_text_stats")


# Correlation


def plot_correlation_with_year(corr_df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, kind in zip(axes, ["pearson_r", "spearman_r"]):
        data = corr_df.set_index("feature")[kind].sort_values()
        colors = ["#d62728" if v < 0 else "#1f77b4" for v in data]
        ax.barh(data.index, data.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(-0.5, 0.5)
        ax.set_title(f"{kind.split('_')[0].title()} Correlation with Year")
        ax.set_xlabel("Correlation Coefficient")
    fig.suptitle("Linguistic Features vs. Publication Year", fontsize=13)
    fig.tight_layout()
    return _save(fig, "correlation_with_year")


def plot_lexical_diversity_over_time(decade_stats: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(
        decade_stats.index,
        decade_stats["lexical_diversity"],
        alpha=0.3,
        color="#2196F3",
    )
    ax.plot(
        decade_stats.index,
        decade_stats["lexical_diversity"],
        marker="o",
        color="#2196F3",
        linewidth=2,
    )
    ax.set_xlabel("Decade")
    ax.set_ylabel("Mean Lexical Diversity (unique/total words)")
    ax.set_title("Lexical Diversity of Book Summaries Over Time")
    return _save(fig, "lexical_diversity")


def plot_keyword_trends(keyword_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in keyword_df.columns:
        ax.plot(keyword_df.index, keyword_df[col], marker="o", label=col, linewidth=2)
    ax.set_xlabel("Decade")
    ax.set_ylabel("Occurrences per 10k Words")
    ax.set_title("Keyword Frequency Trends Over Decades")
    ax.legend(fontsize=9, ncol=3)
    return _save(fig, "keyword_trends")


# Genre classification


def plot_genre_f1_comparison(results: dict) -> plt.Figure:
    rows = []
    for model, res in results.items():
        rows.append(
            {"Model": model, "Micro-F1": res["micro_f1"], "Macro-F1": res["macro_f1"]}
        )
    df = pd.DataFrame(rows).melt("Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(
        data=df,
        x="Model",
        y="Score",
        hue="Metric",
        ax=ax,
        palette=["#4C72B0", "#DD8452"],
    )
    ax.set_ylim(0, 1)
    ax.set_title("Genre Classification: Model Comparison")
    ax.set_xlabel("")
    ax.legend(loc="lower right")
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0.01:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    return _save(fig, "genre_model_comparison")


def plot_per_genre_f1_heatmap(results: dict) -> plt.Figure:
    data = {model: res["per_genre_f1"] for model, res in results.items()}
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(max(8, len(results) * 2.5), max(6, len(df) * 0.45)))
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "F1 Score"},
    )
    ax.set_title("Per-Genre F1 Score by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Genre")
    fig.tight_layout()
    return _save(fig, "per_genre_f1_heatmap")


def plot_top_words_per_genre(top_words: dict, n_genres: int = 8) -> plt.Figure:
    genres = list(top_words.keys())[:n_genres]
    cols = 4
    rows = (len(genres) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, genre in zip(axes_flat, genres):
        words = top_words[genre][:15]
        scores = np.linspace(1, 0.3, len(words))
        ax.barh(
            words[::-1], scores[::-1], color=sns.color_palette("Blues_r", len(words))
        )
        ax.set_title(genre, fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=7)

    for ax in axes_flat[len(genres) :]:
        ax.set_visible(False)

    fig.suptitle("Most Predictive Words per Genre", fontsize=13, y=1.01)
    fig.tight_layout()
    return _save(fig, "top_words_per_genre")


# Year prediction


def plot_year_regression_comparison(results: dict) -> plt.Figure:
    rows = [
        {"Model": m, "MAE": r["mae"], "RMSE": r["rmse"], "R²": r["r2"]}
        for m, r in results.items()
    ]
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, color in zip(
        axes, ["MAE", "RMSE", "R²"], ["#4C72B0", "#DD8452", "#2ca02c"]
    ):
        bars = ax.bar(df["Model"], df[metric], color=color, edgecolor="white")
        ax.set_title(metric)
        ax.set_xticklabels(df["Model"], rotation=30, ha="right", fontsize=9)
        ax.bar_label(bars, padding=2, fmt="%.2f", fontsize=8)
        if metric == "R²":
            ax.set_ylim(min(0, df[metric].min() - 0.05), df[metric].max() + 0.1)
    fig.suptitle("Year Regression: Model Comparison", fontsize=13)
    fig.tight_layout()
    return _save(fig, "year_regression_comparison")


def plot_predicted_vs_actual(results: dict, best_name: str) -> plt.Figure:
    res = results[best_name]
    y_test = res["y_test"]
    y_pred = res["y_pred"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.3, s=15, color="#4C72B0", rasterized=True)
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Year")
    ax.set_ylabel("Predicted Year")
    ax.set_title(f"Predicted vs. Actual Year — {best_name}")
    ax.legend()
    return _save(fig, "year_pred_vs_actual")


def plot_year_error_distribution(results: dict, best_name: str) -> plt.Figure:
    res = results[best_name]
    errors = res["y_pred"] - res["y_test"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors, bins=50, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="red", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Prediction Error (years)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Prediction Error Distribution — {best_name}  (MAE={res['mae']:.1f} yrs)"
    )
    return _save(fig, "year_error_distribution")


def plot_decade_classification_matrix(results: dict, best_name: str) -> plt.Figure:
    from sklearn.metrics import confusion_matrix

    res = results[best_name]
    classes = res["classes"]
    cm = confusion_matrix(res["y_test"], res["y_pred"], labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Row-normalized proportion"},
    )
    ax.set_xlabel("Predicted Decade")
    ax.set_ylabel("Actual Decade")
    ax.set_title(f"Decade Classification Confusion Matrix — {best_name}")
    fig.tight_layout()
    return _save(fig, "decade_confusion_matrix")


def plot_temporal_vocabulary(early_words: list, late_words: list) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, words, label, color in zip(
        axes,
        [early_words[:20], late_words[:20]],
        ["Earlier Works (lower year)", "More Recent Works (higher year)"],
        ["#4C72B0", "#DD8452"],
    ):
        scores = np.linspace(1.0, 0.3, len(words))
        ax.barh(words[::-1], scores[::-1], color=color)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=9)
    fig.suptitle("Vocabulary Most Predictive of Publication Era", fontsize=13)
    fig.tight_layout()
    return _save(fig, "temporal_vocabulary")


def plot_decade_top_words_grid(decade_words: dict) -> plt.Figure:
    decades = sorted(decade_words.keys())
    n = len(decades)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, decade in zip(axes_flat, decades):
        words = decade_words[decade][:12]
        scores = np.linspace(1, 0.3, len(words))
        ax.barh(
            words[::-1], scores[::-1], color=sns.color_palette("mako_r", len(words))
        )
        ax.set_title(f"{decade}s", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=7)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Most Distinctive Words per Decade (TF-IDF)", fontsize=13, y=1.01)
    fig.tight_layout()
    return _save(fig, "decade_top_words")


# Genre trends over time


def plot_genre_trends_over_time(trend_df: pd.DataFrame, top_n: int = 8) -> plt.Figure:
    """Line chart of genre fraction per decade — one line per genre, so each
    genre's own trend can be read directly without being confounded by the
    layers stacked below it."""
    top_cols = trend_df.sum().nlargest(top_n).index.tolist()
    data = trend_df[top_cols]
    palette = sns.color_palette("tab10", len(top_cols))
    fig, ax = plt.subplots(figsize=(12, 6))
    for col, color in zip(top_cols, palette):
        ax.plot(
            data.index, data[col], marker="o", linewidth=2.4,
            markersize=6, color=color, label=col,
        )
    ax.set_xlabel("Decade")
    ax.set_ylabel("Fraction of Books in Genre")
    ax.set_title("Genre Popularity Trends Over Time")
    ax.legend(loc="upper left", fontsize=9, ncol=2, frameon=True)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _save(fig, "genre_trends_over_time")


# AUC

def plot_auc_heatmap(results: dict) -> plt.Figure:
    data = {name: res["auc_per_genre"] for name, res in results.items() if "auc_per_genre" in res}
    if not data:
        return None
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(max(8, len(data) * 2.5), max(6, len(df) * 0.5)))
    sns.heatmap(
        df, annot=True, fmt=".2f", cmap="YlGn", vmin=0.5, vmax=1.0,
        linewidths=0.5, ax=ax, cbar_kws={"label": "AUC"},
    )
    ax.set_title("AUC per Genre by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Genre")
    fig.tight_layout()
    return _save(fig, "genre_auc_heatmap")


# Genre confusion & threshold tuning


def plot_genre_confusion_heatmap(
    y_test: np.ndarray, y_pred: np.ndarray, genre_names: list
) -> plt.Figure:
    """For each true genre A, fraction of those books also predicted as genre B."""
    n = len(genre_names)
    matrix = np.zeros((n, n))
    for i in range(n):
        true_i = y_test[:, i] == 1
        if true_i.sum() == 0:
            continue
        for j in range(n):
            matrix[i, j] = y_pred[true_i, j].mean()
    np.fill_diagonal(matrix, np.nan)  # hide diagonal (recall) to focus on confusion
    df = pd.DataFrame(matrix, index=genre_names, columns=genre_names)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        df, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=0.5,
        linewidths=0.4, ax=ax,
        cbar_kws={"label": "Fraction of true-row books also predicted as col"},
    )
    ax.set_xlabel("Predicted Genre")
    ax.set_ylabel("True Genre")
    ax.set_title("Genre Confusion: Among books of genre A, which other genres does the model predict?")
    fig.tight_layout()
    return _save(fig, "genre_confusion_heatmap")


def plot_genre_threshold_tuning(
    scores: np.ndarray, y_test: np.ndarray
) -> plt.Figure:
    """Precision, Recall, micro-F1 vs decision threshold for the genre classifier."""
    from sklearn.metrics import precision_score, recall_score, f1_score as _f1

    lo, hi = np.percentile(scores, 10), np.percentile(scores, 90)
    thresholds = np.linspace(lo, hi, 60)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_pred_t = (scores > t).astype(int)
        precisions.append(precision_score(y_test, y_pred_t, average="micro", zero_division=0))
        recalls.append(recall_score(y_test, y_pred_t, average="micro", zero_division=0))
        f1s.append(_f1(y_test, y_pred_t, average="micro", zero_division=0))

    best_idx = int(np.argmax(f1s))
    best_t = thresholds[best_idx]
    print(f"  Best threshold: {best_t:.3f}  (micro-F1={f1s[best_idx]:.3f})")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, precisions, label="Precision", color="#4C72B0", linewidth=2)
    ax.plot(thresholds, recalls, label="Recall", color="#DD8452", linewidth=2)
    ax.plot(thresholds, f1s, label="F1 (micro)", color="#2ca02c", linewidth=2.5)
    ax.axvline(best_t, color="red", linestyle="--", linewidth=1.5,
               label=f"Best threshold = {best_t:.2f}  (F1={f1s[best_idx]:.3f})")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Genre Classifier: Precision / Recall / F1 vs Threshold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, "genre_threshold_tuning")


# Year error analysis


def plot_year_error_analysis(
    titles: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, n: int = 10
) -> plt.Figure:
    """Bar chart of the n books with the largest year prediction error."""
    errors = np.abs(y_pred - y_test)
    idx = np.argsort(errors)[-n:][::-1]

    def _label(i):
        t = str(titles[i])
        t = t[:32] + "..." if len(t) > 32 else t
        return f"{t}  (actual {int(y_test[i])})"

    labels = [_label(i) for i in idx]
    vals   = errors[idx]
    preds  = [int(round(y_pred[i])) for i in idx]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(labels[::-1], vals[::-1], color="#DD8452")
    for bar, pred in zip(bars, reversed(preds)):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"pred: {pred}", va="center", fontsize=8,
        )
    ax.set_xlabel("Absolute Error (years)")
    ax.set_title(f"Top {n} Hardest-to-Predict Books (Year Regression)")
    fig.tight_layout()
    return _save(fig, "year_error_analysis")


# Clustering

def plot_cluster_scatter(
    X_2d: np.ndarray,
    labels: np.ndarray,
    cluster_info: dict,
    sample_n: int = 4000,
) -> plt.Figure:
    n_clusters = int(labels.max()) + 1
    # "husl" spaces hues evenly around the color wheel, unlike "tab20" which
    # pairs adjacent dark/light shades of the same hue — with 12 categories
    # those near-duplicate pairs become indistinguishable at a glance.
    palette = sns.color_palette("husl", n_clusters)

    rng = np.random.RandomState(42)
    if len(X_2d) > sample_n:
        idx = rng.choice(len(X_2d), sample_n, replace=False)
        X_s, L_s = X_2d[idx], labels[idx]
    else:
        X_s, L_s = X_2d, labels

    fig, ax = plt.subplots(figsize=(13, 9))
    for c in range(n_clusters):
        mask = L_s == c
        dom = cluster_info[c]["dominant"]
        ax.scatter(
            X_s[mask, 0], X_s[mask, 1],
            color=palette[c], alpha=0.45, s=10, rasterized=True,
            edgecolors="none", label=f"C{c}: {dom}",
        )

    # Centroid labels let the cluster identity be read directly off the plot
    # instead of requiring a round-trip through a 12-entry legend.
    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() == 0:
            continue
        cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
        ax.annotate(
            f"C{c}", (cx, cy),
            fontsize=10, fontweight="bold", color="black",
            ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.3", fc="white",
                      ec=palette[c], lw=2, alpha=0.92),
        )

    ax.legend(
        fontsize=8, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0),
        markerscale=2.5, title="Cluster : Dominant Genre", title_fontsize=9,
    )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Book Summaries — KMeans Clusters (TF-IDF + LSA, PCA 2D)")
    fig.tight_layout()
    return _save(fig, "cluster_scatter")


def plot_cluster_genre_heatmap(cluster_info: dict, top_genres: list) -> plt.Figure:
    n_clusters = len(cluster_info)
    data = np.zeros((n_clusters, len(top_genres)))
    for c, info in cluster_info.items():
        for j, g in enumerate(top_genres):
            data[c, j] = info["distribution"].get(g, 0.0)

    row_labels = [
        f"C{c}: {cluster_info[c]['dominant']} (n={cluster_info[c]['size']})"
        for c in range(n_clusters)
    ]
    df = pd.DataFrame(data, index=row_labels, columns=top_genres)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        df, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=0.8,
        linewidths=0.4, ax=ax, cbar_kws={"label": "Genre fraction in cluster"},
    )
    ax.set_title("Cluster-Genre Composition Heatmap")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    return _save(fig, "cluster_genre_heatmap")


def plot_cluster_purity(cluster_info: dict) -> plt.Figure:
    items = sorted(cluster_info.items(), key=lambda x: x[1]["purity"], reverse=True)
    labels_list = [f"C{c}: {info['dominant']}" for c, info in items]
    purities    = [info["purity"] for _, info in items]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels_list[::-1], purities[::-1],
                   color=sns.color_palette("Blues_d", len(items)))
    ax.set_xlim(0, 1)
    ax.set_xlabel("Purity (dominant genre fraction among genre-labeled books)")
    ax.set_title("Cluster Purity — How Genre-Aligned is Each Cluster?")
    ax.bar_label(bars, padding=3, fmt="%.2f", fontsize=9)
    ax.axvline(1 / len(cluster_info), color="red", linestyle="--",
               linewidth=1, label=f"Random baseline (1/{len(cluster_info)})")
    ax.legend(fontsize=9)
    return _save(fig, "cluster_purity")


def plot_cluster_top_terms_grid(top_terms: dict) -> plt.Figure:
    n = len(top_terms)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (c, words) in zip(axes_flat, top_terms.items()):
        words = words[:10]
        scores = np.linspace(1, 0.3, len(words))
        ax.barh(words[::-1], scores[::-1], color=sns.color_palette("Greens_r", len(words)))
        ax.set_title(f"Cluster {c}", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=7)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Top Terms per Cluster (TF-IDF + LSA)", fontsize=13, y=1.01)
    fig.tight_layout()
    return _save(fig, "cluster_top_terms")
