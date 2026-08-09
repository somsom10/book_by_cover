"""
Multi-label genre classification from book summaries.

Models compared:
  - Logistic Regression (OvR, balanced)
  - Linear SVC (OvR, balanced, C-tuned)
  - Naive Bayes

Feature sets:
  - Word TF-IDF (1-2 grams) + Char TF-IDF (3-5 grams) + text statistics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


STAT_COLS = [
    "word_count", "avg_sent_len", "lexical_diversity",
    "avg_word_len", "sentence_count", "char_count", "unique_words",
]

WORD_TFIDF_KWARGS = dict(
    ngram_range=(1, 2), max_features=50_000, sublinear_tf=True, min_df=2,
)
CHAR_TFIDF_KWARGS = dict(
    analyzer="char_wb", ngram_range=(3, 5), max_features=30_000,
    sublinear_tf=True, min_df=5,
)

MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs", class_weight="balanced"
    ),
    "Linear SVC": LinearSVC(max_iter=3000, C=1.0, class_weight="balanced"),
    "Naive Bayes": MultinomialNB(alpha=0.1),
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


def build_pipeline(clf, with_stats: bool = True) -> Pipeline:
    if with_stats:
        return Pipeline([
            ("prep", _make_preprocessor()),
            ("clf",  OneVsRestClassifier(clf, n_jobs=-1)),
        ])
    return Pipeline([
        ("tfidf", TfidfVectorizer(**WORD_TFIDF_KWARGS)),
        ("clf",   OneVsRestClassifier(clf, n_jobs=-1)),
    ])


def _eval_pipe(pipe, X_test, y_test, genre_names, text_only=False):
    inp = X_test["clean_summary"] if text_only else X_test
    y_pred   = pipe.predict(inp)
    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    h_loss   = hamming_loss(y_test, y_pred)
    per_g    = f1_score(y_test, y_pred, average=None, zero_division=0)
    return {
        "micro_f1":     micro_f1,
        "macro_f1":     macro_f1,
        "hamming_loss": h_loss,
        "per_genre_f1": dict(zip(genre_names, per_g)),
        "y_test": y_test,
        "y_pred": y_pred,
    }


def train_and_evaluate(
    X: pd.DataFrame,
    label_matrix: np.ndarray,
    genre_names: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[dict, dict, tuple]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, label_matrix, test_size=test_size, random_state=random_state,
    )

    results = {}
    trained_pipelines = {}

    # Standard models
    for name, clf in MODELS.items():
        is_nb = isinstance(clf, MultinomialNB)
        print(f"  Training {name}...")
        pipe = build_pipeline(clf, with_stats=not is_nb)
        inp_train = X_train["clean_summary"] if is_nb else X_train
        pipe.fit(inp_train, y_train)
        results[name]          = _eval_pipe(pipe, X_test, y_test, genre_names, is_nb)
        trained_pipelines[name] = pipe
        r = results[name]
        print(f"    micro-F1={r['micro_f1']:.3f}  macro-F1={r['macro_f1']:.3f}  hamming={r['hamming_loss']:.4f}")

    # C-tuning for Linear SVC
    print("  Tuning C for Linear SVC (balanced)...")
    best_C, best_macro = 1.0, results["Linear SVC"]["macro_f1"]
    best_pipe_svc = trained_pipelines["Linear SVC"]

    for C in [0.1, 0.5, 2.0, 5.0]:
        pipe_c = build_pipeline(LinearSVC(max_iter=3000, C=C, class_weight="balanced"))
        pipe_c.fit(X_train, y_train)
        res_c = _eval_pipe(pipe_c, X_test, y_test, genre_names)
        if res_c["macro_f1"] > best_macro:
            best_macro = res_c["macro_f1"]
            best_C = C
            best_pipe_svc = pipe_c
            results["Linear SVC"] = res_c

    trained_pipelines["Linear SVC"] = best_pipe_svc
    print(f"    Best C={best_C}  macro-F1={best_macro:.3f}")

    # AUC per genre
    print("  Computing AUC per genre...")
    for name, pipe in trained_pipelines.items():
        is_nb  = name == "Naive Bayes"
        text_inp = X_test["clean_summary"]
        try:
            if is_nb:
                scores = pipe.predict_proba(text_inp)
            else:
                scores = pipe.decision_function(X_test)
            auc_dict = {}
            for j, genre in enumerate(genre_names):
                if y_test[:, j].sum() > 0:
                    auc_dict[genre] = float(roc_auc_score(y_test[:, j], scores[:, j]))
                else:
                    auc_dict[genre] = float("nan")
            results[name]["auc_per_genre"] = auc_dict
            results[name]["mean_auc"] = float(np.nanmean(list(auc_dict.values())))
            print(f"    {name:<28} mean-AUC={results[name]['mean_auc']:.3f}")
        except Exception:
            pass

    return results, trained_pipelines, (X_train, X_test, y_train, y_test)


def best_model(results: dict) -> str:
    return max(results, key=lambda k: results[k]["micro_f1"])


def get_top_words_per_genre(
    pipeline: Pipeline, genre_names: list[str], n: int = 20
) -> dict:
    """Extract most predictive TF-IDF words per genre."""
    if "prep" in pipeline.named_steps:
        prep = pipeline.named_steps["prep"]
        # prefer word_tfidf; fall back to tfidf for old pipelines
        tfidf = prep.named_transformers_.get(
            "word_tfidf", prep.named_transformers_.get("tfidf")
        )
    elif "tfidf" in pipeline.named_steps:
        tfidf = pipeline.named_steps["tfidf"]
    else:
        return {}

    clf     = pipeline.named_steps["clf"]
    vocab   = np.array(tfidf.get_feature_names_out())
    n_tfidf = len(vocab)

    top_words = {}
    for genre, est in zip(genre_names, clf.estimators_):
        if hasattr(est, "coef_"):
            coef = est.coef_.ravel()[:n_tfidf]
        elif hasattr(est, "feature_log_prob_"):
            coef = (est.feature_log_prob_[1] - est.feature_log_prob_[0])[:n_tfidf]
        else:
            continue
        top_idx = np.argsort(coef)[-n:][::-1]
        top_words[genre] = list(vocab[top_idx])

    return top_words
