"""
Year prediction from book summaries.

Two tasks:
  1. Regression - predict publication year (MAE, RMSE, R²)
  2. Decade classification - predict decade label (Accuracy, macro-F1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR

STAT_COLS = [
    "word_count",
    "avg_sent_len",
    "lexical_diversity",
    "avg_word_len",
    "sentence_count",
    "char_count",
    "unique_words",
]

WORD_TFIDF_KWARGS = dict(
    ngram_range=(1, 2), max_features=50_000, sublinear_tf=True, min_df=2,
)
CHAR_TFIDF_KWARGS = dict(
    analyzer="char_wb", ngram_range=(3, 5), max_features=30_000,
    sublinear_tf=True, min_df=5,
)

REGRESSION_MODELS = {
    "Ridge": Ridge(alpha=1.0),
    "Linear SVR": LinearSVR(max_iter=3000, C=0.5),
}


def _make_preprocessor() -> ColumnTransformer:
    stats_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    return ColumnTransformer([
        ("word_tfidf", TfidfVectorizer(**WORD_TFIDF_KWARGS), "clean_summary"),
        ("char_tfidf", TfidfVectorizer(**CHAR_TFIDF_KWARGS), "clean_summary"),
        ("stats",      stats_pipe, STAT_COLS),
    ])


def build_regression_pipeline(reg) -> Pipeline:
    return Pipeline(
        [
            ("prep", _make_preprocessor()),
            ("reg", reg),
        ]
    )


def train_regression(
    X: pd.DataFrame,
    years: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[dict, dict, tuple]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        years,
        test_size=test_size,
        random_state=random_state,
    )

    results = {}
    trained = {}

    for name, reg in REGRESSION_MODELS.items():
        print(f"  Training {name} (regression)...")
        pipe = build_regression_pipeline(reg)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_pred_clipped = np.clip(y_pred, years.min(), years.max())

        mae = mean_absolute_error(y_test, y_pred_clipped)
        rmse = root_mean_squared_error(y_test, y_pred_clipped)
        r2 = r2_score(y_test, y_pred_clipped)

        results[name] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "y_test": y_test.values,
            "y_pred": y_pred_clipped,
        }
        trained[name] = pipe
        print(f"    MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

    return results, trained, (X_train, X_test, y_train, y_test)


def train_decade_classification(
    X: pd.DataFrame,
    decades: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[dict, dict, tuple]:
    CLASSIFIERS = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", class_weight="balanced"
        ),
        "Linear SVC": LinearSVC(max_iter=2000, C=1.0, class_weight="balanced"),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        decades,
        test_size=test_size,
        random_state=random_state,
        stratify=decades,
    )

    results = {}
    trained = {}

    for name, clf in CLASSIFIERS.items():
        print(f"  Training {name} (decade classification)...")
        pipe = Pipeline(
            [
                ("prep", _make_preprocessor()),
                ("clf", clf),
            ]
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        results[name] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "y_test": y_test.values,
            "y_pred": y_pred,
            "classes": sorted(y_test.unique()),
        }
        trained[name] = pipe
        print(f"    Accuracy={acc:.3f}  macro-F1={macro_f1:.3f}")

    # C-tuning for Linear SVC
    print("  Tuning C for Linear SVC...")
    best_C = 1.0
    best_acc = results["Linear SVC"]["accuracy"]
    best_pipe = trained["Linear SVC"]

    for C in [0.1, 0.5, 2.0, 5.0]:
        pipe_c = Pipeline(
            [
                ("prep", _make_preprocessor()),
                ("clf", LinearSVC(max_iter=2000, C=C, class_weight="balanced")),
            ]
        )
        pipe_c.fit(X_train, y_train)
        pred_c = pipe_c.predict(X_test)
        acc_c = accuracy_score(y_test, pred_c)
        if acc_c > best_acc:
            best_acc = acc_c
            best_C = C
            best_pipe = pipe_c
            f1_c = f1_score(y_test, pred_c, average="macro", zero_division=0)
            results["Linear SVC"] = {
                "accuracy": acc_c,
                "macro_f1": f1_c,
                "y_test": y_test.values,
                "y_pred": pred_c,
                "classes": sorted(y_test.unique()),
            }

    trained["Linear SVC"] = best_pipe
    print(f"    Best C={best_C}  Accuracy={best_acc:.3f}")

    return results, trained, (X_train, X_test, y_train, y_test)


def get_temporal_vocabulary(
    pipeline: Pipeline, n: int = 30
) -> tuple[list[str], list[str]]:
    """Words most predictive of early vs. late periods (from regression coefficients)."""
    if "prep" in pipeline.named_steps:
        prep = pipeline.named_steps["prep"]
        tfidf = prep.named_transformers_.get(
            "word_tfidf", prep.named_transformers_.get("tfidf")
        )
        reg = pipeline.named_steps["reg"]
    else:
        tfidf = pipeline.named_steps["tfidf"]
        reg = pipeline.named_steps["reg"]

    vocab = np.array(tfidf.get_feature_names_out())
    n_tfidf = len(vocab)

    if hasattr(reg, "coef_"):
        coef = reg.coef_.ravel()[:n_tfidf]
    elif hasattr(reg, "feature_importances_"):
        coef = reg.feature_importances_[:n_tfidf]
    else:
        return [], []

    early_words = list(vocab[np.argsort(coef)[:n]])
    late_words = list(vocab[np.argsort(coef)[-n:][::-1]])
    return early_words, late_words
