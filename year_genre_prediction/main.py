"""
Main pipeline: loads the CMU dataset, runs correlation analysis, trains the
genre/year/decade models, runs clustering, and saves figures + models.

Usage: python main.py --data data/booksummaries.txt
"""

import argparse
import warnings
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from src.data_loader import (
    assign_decade,
    books_with_genre,
    encode_genres,
    filter_year_range,
    get_top_genres,
    genre_label_matrix,
    load_cmu,
)
from src.preprocessor import preprocess_series, text_statistics
from src.correlation_analysis import (
    correlate_text_stats_with_year,
    decade_text_stats,
    genre_discriminating_words,
    keyword_over_time,
    top_tfidf_words_per_decade,
)
from src.genre_classifier import (
    STAT_COLS,
    best_model as best_genre_model,
    get_top_words_per_genre,
    train_and_evaluate as genre_train,
)
from src.year_predictor import (
    get_temporal_vocabulary,
    train_decade_classification,
    train_regression,
)
from src.clustering import (
    analyze_cluster_genres,
    cluster_summaries,
    get_cluster_top_terms,
    overall_purity,
    reduce_to_2d,
)
import src.visualizations as viz

MODELS_DIR = Path("outputs/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def print_section(title: str):
    print(f"\n{'-' * 60}")
    print(title)
    print("-" * 60)


def main(
    data_path: str, top_n_genres: int = 12, year_lo: int = 1950, year_hi: int = 2008
):
    # Load data
    print_section("1. Loading Data")
    df = load_cmu(data_path)
    print(f"  Total records: {len(df):,}")
    print(f"  Records with year: {df['year'].notna().sum():,}")
    print(f"  Records with genre: {(df['genre_list'].map(len) > 0).sum():,}")

    # Preprocess summaries
    print_section("2. Preprocessing Summaries")
    df["clean_summary"] = preprocess_series(df["summary"], lemmatize=True)
    print(f"  Preprocessing done. Sample: {df['clean_summary'].iloc[0][:80]}...")

    # Text statistics & EDA
    print_section("3. Text Statistics & EDA")
    stats = text_statistics(df["summary"])
    df = pd.concat([df, stats], axis=1)
    print(stats.describe().round(2))

    viz.plot_year_distribution(df)

    # Genre counts
    top_genres = get_top_genres(df, n=top_n_genres)
    print(f"\n  Top {top_n_genres} genres: {top_genres}")
    viz.plot_genre_counts(
        dict(
            list(Counter(g for gl in df["genre_list"] for g in gl).most_common())[
                :top_n_genres
            ]
        )
    )

    # Year-filtered subset
    print_section("4. Year-Filtered Subset")
    df_year = filter_year_range(df, year_lo, year_hi)
    df_year = assign_decade(df_year)
    print(f"  Books in {year_lo}–{year_hi}: {len(df_year):,}")
    print(f"  Decades present: {sorted(df_year['decade'].unique())}")

    # Correlation analysis
    print_section("5. Correlation Analysis")

    corr_df = correlate_text_stats_with_year(df_year[stats.columns], df_year["year"])
    print(corr_df.to_string(index=False))
    viz.plot_correlation_with_year(corr_df)

    dec_stats = decade_text_stats(df_year[stats.columns], df_year["decade"])
    viz.plot_decade_text_stats(
        dec_stats[["lexical_diversity", "avg_word_len", "avg_sent_len"]]
    )
    viz.plot_lexical_diversity_over_time(dec_stats)

    KEYWORDS = [
        "war",
        "love",
        "murder",
        "technology",
        "family",
        "alien",
        "vampire",
        "revolution",
    ]
    kw_df = keyword_over_time(df_year["summary"], df_year["decade"], KEYWORDS)
    viz.plot_keyword_trends(kw_df)

    decade_words = top_tfidf_words_per_decade(
        df_year["clean_summary"], df_year["decade"]
    )
    viz.plot_decade_top_words_grid(decade_words)

    # Genre trends over time
    trend_rows = []
    for decade in sorted(df_year["decade"].unique()):
        sub = df_year[df_year["decade"] == decade]
        row = {"decade": decade}
        for g in top_genres:
            row[g] = sub["genre_list"].apply(lambda gl, _g=g: _g in (gl or [])).mean()
        trend_rows.append(row)
    genre_trend_df = pd.DataFrame(trend_rows).set_index("decade")
    viz.plot_genre_trends_over_time(genre_trend_df)

    # Genre classification
    print_section("6. Genre Classification (Multi-Label)")
    df_genre = books_with_genre(df)
    df_genre = encode_genres(df_genre, top_genres)
    Y = genre_label_matrix(df_genre, top_genres)
    valid_mask = Y.sum(axis=1) > 0
    X_genre = df_genre[["clean_summary"] + STAT_COLS][valid_mask]
    Y_genre = Y[valid_mask]
    print(f"  Books with known genres: {valid_mask.sum():,}")
    print(f"  Label matrix shape: {Y_genre.shape}")

    genre_results, genre_pipelines, genre_splits = genre_train(
        X_genre, Y_genre, top_genres
    )

    best_genre = best_genre_model(genre_results)
    print(
        f"\n  Best genre model: {best_genre}  "
        f"(micro-F1={genre_results[best_genre]['micro_f1']:.3f})"
    )

    viz.plot_genre_f1_comparison(genre_results)
    viz.plot_per_genre_f1_heatmap(genre_results)

    top_words = get_top_words_per_genre(genre_pipelines[best_genre], top_genres)
    viz.plot_top_words_per_genre(top_words)

    joblib.dump(
        {"pipeline": genre_pipelines[best_genre], "genre_names": top_genres},
        MODELS_DIR / "genre_best.pkl",
    )
    print(f"  Best genre model saved -> {MODELS_DIR}/genre_best.pkl")

    # AUC per genre
    if "auc_per_genre" in genre_results[best_genre]:
        print("\n  AUC per genre (best model):")
        for genre, auc in genre_results[best_genre]["auc_per_genre"].items():
            print(f"    {genre:<35} {auc:.3f}")
        viz.plot_auc_heatmap(genre_results)

    # Genre confusion & threshold tuning
    _, X_test_genre, _, y_test_genre = genre_splits
    y_pred_genre = genre_results[best_genre]["y_pred"]
    viz.plot_genre_confusion_heatmap(y_test_genre, y_pred_genre, top_genres)
    scores_genre = genre_pipelines[best_genre].decision_function(X_test_genre)
    viz.plot_genre_threshold_tuning(scores_genre, y_test_genre)

    flat_genre = pd.Series([g for gl in df_genre["genre_list"] for g in (gl or [])])
    flat_text = pd.Series(
        [
            t
            for gl, t in zip(df_genre["genre_list"], df_genre["clean_summary"])
            for _ in (gl or [])
        ]
    )
    if len(flat_genre) > 100:
        disc_words = genre_discriminating_words(flat_text, flat_genre, n=20)
        print("\n  Sample discriminating words:")
        for genre in list(disc_words)[:3]:
            print(f"    {genre}: {', '.join(disc_words[genre][:8])}")

    # Year regression
    print_section("7. Year Regression")
    df_yr = df_year.dropna(subset=["clean_summary", "year"])
    X_yr = df_yr[["clean_summary"] + STAT_COLS]
    years_yr = df_yr["year"].astype(float)

    reg_results, reg_pipelines, _ = train_regression(X_yr, years_yr)

    best_reg = min(reg_results, key=lambda k: reg_results[k]["mae"])
    print(
        f"\n  Best regression model: {best_reg}  "
        f"(MAE={reg_results[best_reg]['mae']:.2f} yrs, "
        f"R²={reg_results[best_reg]['r2']:.4f})"
    )

    viz.plot_year_regression_comparison(reg_results)
    viz.plot_predicted_vs_actual(reg_results, best_reg)
    viz.plot_year_error_distribution(reg_results, best_reg)

    early_words, late_words = get_temporal_vocabulary(reg_pipelines[best_reg])
    viz.plot_temporal_vocabulary(early_words, late_words)

    joblib.dump(reg_pipelines[best_reg], MODELS_DIR / "year_regression_best.pkl")
    print(f"  Best regression model saved -> {MODELS_DIR}/year_regression_best.pkl")

    # Year error analysis — top-10 hardest books to predict
    _, X_test_yr_split, _, _ = train_test_split(X_yr, years_yr, test_size=0.2, random_state=42)
    titles_test = df_yr.loc[X_test_yr_split.index, "title"].values
    viz.plot_year_error_analysis(
        titles_test,
        reg_results[best_reg]["y_test"],
        reg_results[best_reg]["y_pred"],
    )

    # Decade classification
    print_section("8. Decade Classification")
    decades_yr = df_yr["decade"].astype(int)

    decade_counts = decades_yr.value_counts()
    valid_decades = decade_counts[decade_counts >= 5].index
    mask_dec = decades_yr.isin(valid_decades)
    X_dec = X_yr[mask_dec]
    decades_dec = decades_yr[mask_dec]

    dec_results, dec_pipelines, _ = train_decade_classification(
        X_dec, decades_dec
    )

    best_dec = max(dec_results, key=lambda k: dec_results[k]["accuracy"])
    print(
        f"\n  Best decade model: {best_dec}  "
        f"(Acc={dec_results[best_dec]['accuracy']:.3f}, "
        f"macro-F1={dec_results[best_dec]['macro_f1']:.3f})"
    )

    viz.plot_decade_classification_matrix(dec_results, best_dec)

    joblib.dump(dec_pipelines[best_dec], MODELS_DIR / "decade_best.pkl")
    print(f"  Best decade model saved -> {MODELS_DIR}/decade_best.pkl")

    # Clustering
    print_section("9. Unsupervised Clustering (TF-IDF + LSA + KMeans)")
    n_clusters = top_n_genres  # same as number of genres for comparison
    print(f"  Clustering {len(df):,} summaries into {n_clusters} clusters...")
    labels, X_lsa, lsa_pipe, km = cluster_summaries(
        df["clean_summary"], n_clusters=n_clusters
    )

    X_2d = reduce_to_2d(X_lsa)
    cluster_info = analyze_cluster_genres(labels, df["genre_list"], top_genres)
    top_cluster_terms = get_cluster_top_terms(lsa_pipe, km)
    purity = overall_purity(labels, df["genre_list"], top_genres)

    print(f"  Overall cluster purity: {purity:.3f}")
    print(f"\n  {'Cluster':<10} {'Dominant Genre':<30} {'Purity':>8} {'Size':>6}")
    print("  " + "-" * 58)
    for c in sorted(cluster_info, key=lambda x: cluster_info[x]["purity"], reverse=True):
        info = cluster_info[c]
        print(f"  C{c:<9} {info['dominant']:<30} {info['purity']:>8.3f} {info['size']:>6}")

    viz.plot_cluster_scatter(X_2d, labels, cluster_info)
    viz.plot_cluster_genre_heatmap(cluster_info, top_genres)
    viz.plot_cluster_purity(cluster_info)
    viz.plot_cluster_top_terms_grid(top_cluster_terms)

    # Summary
    print_section("10. Results Summary")
    print("\n  GENRE CLASSIFICATION")
    for model, res in sorted(
        genre_results.items(), key=lambda x: x[1]["micro_f1"], reverse=True
    ):
        print(
            f"    {model:<25}  micro-F1={res['micro_f1']:.3f}  macro-F1={res['macro_f1']:.3f}"
        )

    print("\n  YEAR REGRESSION")
    for model, res in sorted(reg_results.items(), key=lambda x: x[1]["mae"]):
        print(
            f"    {model:<25}  MAE={res['mae']:.2f} yrs  RMSE={res['rmse']:.2f}  R²={res['r2']:.4f}"
        )

    print("\n  DECADE CLASSIFICATION")
    for model, res in sorted(
        dec_results.items(), key=lambda x: x[1]["accuracy"], reverse=True
    ):
        print(
            f"    {model:<25}  Accuracy={res['accuracy']:.3f}  macro-F1={res['macro_f1']:.3f}"
        )

    print("\n  All figures saved to: outputs/figures/")
    print("  All models saved to:  outputs/models/")
    print("\nDone.\n")

    return {
        "genre_results": genre_results,
        "reg_results": reg_results,
        "dec_results": dec_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TEXTile Factory – Genre & Year Prediction"
    )
    parser.add_argument("--data", default="data/booksummaries.txt")
    parser.add_argument("--top_genres", type=int, default=12)
    parser.add_argument("--year_lo", type=int, default=1950)
    parser.add_argument("--year_hi", type=int, default=2008)
    args = parser.parse_args()
    main(args.data, args.top_genres, args.year_lo, args.year_hi)
