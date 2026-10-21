from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "cleaned_jobs.csv"
MODEL_DIR = PROJECT_ROOT / "models"
VIZ_DIR = PROJECT_ROOT / "visualizations"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)
TEXT_SIMILARITY_THRESHOLD = 0.95


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def ensure_unique_job_ids(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ordered = df.copy()
    ordered["job_id"] = ordered["job_id"].astype(str).str.strip()
    ordered = ordered.drop_duplicates(subset=["job_id"]).sort_values("job_id").reset_index(drop=True)
    return ordered


class EncodedXGBClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        encoded_labels = self.label_encoder_.fit_transform(y)
        self.classifier_ = XGBClassifier(
            n_estimators=40,
            max_depth=3,
            learning_rate=0.12,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=self.random_state,
            n_jobs=1,
            verbosity=0,
        )
        self.classifier_.fit(X, encoded_labels)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X):
        encoded_predictions = self.classifier_.predict(X).astype(int)
        return self.label_encoder_.inverse_transform(encoded_predictions)


def build_text_features(row: Dict[str, Any] | pd.Series) -> str:
    title = normalize_text(row.get("job_title", "")) if isinstance(row, dict) else normalize_text(row.get("job_title", ""))
    description = normalize_text(row.get("job_description", "")) if isinstance(row, dict) else normalize_text(row.get("job_description", ""))
    skills = normalize_text(row.get("skills", row.get("job_skill_set", ""))) if isinstance(row, dict) else normalize_text(row.get("skills", row.get("job_skill_set", "")))
    parts = [part for part in [title, description, skills] if part]
    return " ".join(parts)


def prepare_dataset(data_file: Path = DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(data_file)
    required = {"job_id", "category", "job_title", "job_description", "job_skill_set", "skills"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    prepared = df.copy()
    prepared["job_id"] = prepared["job_id"].astype(str).str.strip()
    prepared["category"] = prepared["category"].fillna("").astype(str).str.strip()
    prepared["job_title"] = prepared["job_title"].fillna("").astype(str).str.strip()
    prepared["job_description"] = prepared["job_description"].fillna("").astype(str).str.strip()
    prepared["job_skill_set"] = prepared["job_skill_set"].fillna("").astype(str).str.strip()
    prepared["skills"] = prepared["skills"].fillna("").astype(str).str.strip()

    prepared = prepared[(prepared["category"] != "") & ((prepared["job_title"] != "") | (prepared["job_description"] != "") | (prepared["skills"] != ""))].copy()
    prepared["text_features"] = prepared.apply(build_text_features, axis=1)
    prepared = prepared[prepared["text_features"].str.len() > 0].reset_index(drop=True)

    category_counts = prepared["category"].value_counts()
    prepared = prepared[prepared["category"].isin(category_counts[category_counts >= 2].index)].copy()
    prepared = prepared.reset_index(drop=True)
    prepared["leakage_group"] = build_leakage_groups(prepared)
    return prepared


def build_leakage_groups(
    df: pd.DataFrame,
    similarity_threshold: float = TEXT_SIMILARITY_THRESHOLD,
) -> np.ndarray:
    if df.empty:
        return np.array([], dtype=int)

    from sklearn.metrics.pairwise import cosine_similarity

    text_matrix = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=2**18,
        alternate_sign=False,
        norm="l2",
    ).transform(df["text_features"].tolist())
    similarities = cosine_similarity(text_matrix)
    parents = list(range(len(df)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    first_by_job_id: Dict[str, int] = {}
    for index, job_id in enumerate(df["job_id"].astype(str).str.strip()):
        if job_id in first_by_job_id:
            union(index, first_by_job_id[job_id])
        else:
            first_by_job_id[job_id] = index

    for first in range(len(df)):
        matching = np.flatnonzero(similarities[first, first + 1 :] >= similarity_threshold)
        for offset in matching:
            union(first, first + 1 + int(offset))

    _, group_ids = np.unique([find(index) for index in range(len(df))], return_inverse=True)
    return group_ids


def split_train_test_by_group(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Dataset is empty.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    grouped = df.copy().reset_index(drop=True)
    if "text_features" not in grouped:
        grouped["text_features"] = grouped.apply(build_text_features, axis=1)
    if "leakage_group" not in grouped:
        grouped["leakage_group"] = build_leakage_groups(grouped)

    n_splits = max(2, round(1 / test_size))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_indices, test_indices = next(
        splitter.split(
            grouped["text_features"],
            grouped["category"],
            groups=grouped["leakage_group"],
        )
    )
    train_df = grouped.iloc[train_indices].copy().reset_index(drop=True)
    test_df = grouped.iloc[test_indices].copy().reset_index(drop=True)
    return train_df, test_df


def make_vectorizer(kind: str = "word_tfidf") -> TfidfVectorizer:
    if kind == "word_tfidf":
        return TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            strip_accents="unicode",
            lowercase=True,
            sublinear_tf=True,
            max_features=8000,
        )
    if kind == "char_tfidf":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            strip_accents="unicode",
            lowercase=True,
            sublinear_tf=True,
            max_features=6000,
        )
    raise ValueError(f"Unsupported vectorizer: {kind}")


def make_estimator(model_name: str, random_state: int = 42):
    if model_name == "Logistic Regression":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=None,
        )
    if model_name == "Linear SVM":
        return LinearSVC(C=1.0, class_weight="balanced")
    if model_name == "Complement Naive Bayes":
        return ComplementNB(alpha=0.5)
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=80,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=1,
            max_depth=None,
        )
    if model_name == "XGBoost":
        return EncodedXGBClassifier(random_state=random_state)
    raise ValueError(f"Unsupported model name: {model_name}")


def evaluate_model(
    model_name: str,
    vectorizer_name: str,
    estimator: Optional[Any],
    X_train: Iterable[str],
    y_train: Iterable[str],
    X_test: Iterable[str],
    y_test: Iterable[str],
    label_encoder: Optional[Any] = None,
) -> Dict[str, Any]:
    pipeline = Pipeline(
        steps=[
            ("vectorizer", make_vectorizer(vectorizer_name)),
            ("classifier", estimator if estimator is not None else make_estimator(model_name)),
        ]
    )

    pipeline.fit(list(X_train), list(y_train))
    predictions = pipeline.predict(list(X_test))

    accuracy = accuracy_score(list(y_test), predictions)
    macro_f1 = f1_score(list(y_test), predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(list(y_test), predictions, average="weighted", zero_division=0)
    precision, recall, _, _ = precision_recall_fscore_support(
        list(y_test), predictions, average="macro", zero_division=0
    )

    cm = confusion_matrix(list(y_test), predictions, labels=sorted(set(list(y_test))))
    labels = sorted(set(list(y_test)))

    per_class = {}
    for label in labels:
        idx = list(sorted(set(list(y_test)))).index(label)
        class_precision = precision_recall_fscore_support(
            list(y_test), predictions, labels=[label], average=None, zero_division=0
        )[0][0]
        class_recall = precision_recall_fscore_support(
            list(y_test), predictions, labels=[label], average=None, zero_division=0
        )[1][0]
        per_class[str(label)] = {"precision": float(class_precision), "recall": float(class_recall)}

    return {
        "model_name": model_name,
        "vectorizer_name": vectorizer_name,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_labels": labels,
    }


def get_candidate_configs() -> List[Tuple[str, str, Any]]:
    return [
        ("Logistic Regression", "word_tfidf", None),
        ("Linear SVM", "word_tfidf", None),
        ("Complement Naive Bayes", "word_tfidf", None),
        ("Random Forest", "word_tfidf", None),
        ("XGBoost", "word_tfidf", None),
    ]


def select_best_model(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        raise ValueError("No model results available for selection.")
    return sorted(results, key=lambda item: (item["validation_macro_f1"], item["validation_weighted_f1"]), reverse=True)[0]


def run_full_model_selection(data_file: Path = DATA_FILE) -> Dict[str, Any]:
    df = prepare_dataset(data_file)
    if df.empty:
        raise ValueError("Prepared dataset is empty after filtering.")

    train_df, test_df = split_train_test_by_group(df, test_size=0.2, random_state=42)
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced empty partitions.")

    unique_train_ids = train_df["job_id"].drop_duplicates().tolist()
    unique_test_ids = test_df["job_id"].drop_duplicates().tolist()
    if set(unique_train_ids) & set(unique_test_ids):
        raise ValueError("Leakage detected: job IDs overlap between train and test partitions.")
    if set(train_df["leakage_group"]) & set(test_df["leakage_group"]):
        raise ValueError("Leakage detected: related text groups overlap between train and test partitions.")

    labels = sorted(df["category"].unique().tolist())
    model_results = []
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    X_train = train_df["text_features"].tolist()
    y_train = train_df["category"].tolist()
    train_groups = train_df["leakage_group"].to_numpy()

    for model_name, vectorizer_name, estimator in get_candidate_configs():
        fold_scores = []
        fold_weighted = []
        for train_idx, valid_idx in cv.split(X_train, y_train, train_groups):
            pipe = Pipeline([
                ("vectorizer", make_vectorizer(vectorizer_name)),
                ("classifier", estimator if estimator is not None else make_estimator(model_name)),
            ])
            train_labels = [y_train[index] for index in train_idx]
            valid_labels = [y_train[index] for index in valid_idx]
            pipe.fit([X_train[index] for index in train_idx], train_labels)
            valid_predictions = pipe.predict([X_train[index] for index in valid_idx])
            fold_scores.append(f1_score(valid_labels, valid_predictions, average="macro", zero_division=0))
            fold_weighted.append(f1_score(valid_labels, valid_predictions, average="weighted", zero_division=0))

        validation_macro = float(np.mean(fold_scores))
        validation_weighted = float(np.mean(fold_weighted))
        result = {
            "model_name": model_name,
            "vectorizer_name": vectorizer_name,
            "validation_macro_f1": validation_macro,
            "validation_weighted_f1": validation_weighted,
        }
        model_results.append(result)

    best = select_best_model(model_results)
    best_pipeline = Pipeline([
        ("vectorizer", make_vectorizer(best["vectorizer_name"])),
        ("classifier", make_estimator(best["model_name"])),
    ])
    best_pipeline.fit(X_train, y_train)
    test_labels = test_df["category"].tolist()
    test_predictions = best_pipeline.predict(test_df["text_features"].tolist())
    test_accuracy = float(accuracy_score(test_labels, test_predictions))
    test_macro_f1 = float(f1_score(test_labels, test_predictions, average="macro", zero_division=0))
    test_weighted_f1 = float(f1_score(test_labels, test_predictions, average="weighted", zero_division=0))
    confusion = confusion_matrix(test_labels, test_predictions, labels=labels)

    best_pipeline_path = MODEL_DIR / "job_category_best_model.joblib"
    joblib.dump(best_pipeline, best_pipeline_path)

    split_strategy = {
        "holdout": "First fold of 5-fold StratifiedGroupKFold (approximately 20%), random_state=42.",
        "model_selection": "3-fold StratifiedGroupKFold applied only to the remaining training partition.",
        "grouping": "Connected components joining shared job_id values and stateless character-hash cosine similarity >= threshold.",
        "similarity_vectorizer": "HashingVectorizer (char_wb, n-grams 3-5, 2^18 features)",
        "similarity_threshold": TEXT_SIMILARITY_THRESHOLD,
    }
    leakage_group_sizes = df["leakage_group"].value_counts()
    near_duplicate_groups = int((leakage_group_sizes > 1).sum())
    near_duplicate_rows = int(leakage_group_sizes[leakage_group_sizes > 1].sum())

    metadata = {
        "task": "job-category classification",
        "target": "category",
        "feature_columns": ["job_title", "job_description", "skills"],
        "vectorizer": best["vectorizer_name"],
        "model": best["model_name"],
        "validation_macro_f1": float(best["validation_macro_f1"]),
        "validation_weighted_f1": float(best["validation_weighted_f1"]),
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "labels": labels,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "saved_model": str(best_pipeline_path),
        "split_strategy": split_strategy,
        "train_job_ids": int(train_df["job_id"].nunique()),
        "test_job_ids": int(test_df["job_id"].nunique()),
        "train_similarity_groups": int(train_df["leakage_group"].nunique()),
        "test_similarity_groups": int(test_df["leakage_group"].nunique()),
        "near_duplicate_groups": near_duplicate_groups,
        "near_duplicate_rows": near_duplicate_rows,
    }
    metadata_path = MODEL_DIR / "job_category_best_model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary = {
        "task": "job-category classification",
        "baseline_comparison": {
            "baseline_metrics_path": str(MODEL_DIR / "baseline_metrics.json"),
            "baseline_exists": (MODEL_DIR / "baseline_metrics.json").exists(),
        },
        "selected_model": {
            "model_name": best["model_name"],
            "vectorizer_name": best["vectorizer_name"],
            "validation_macro_f1": float(best["validation_macro_f1"]),
            "validation_weighted_f1": float(best["validation_weighted_f1"]),
            "test_accuracy": test_accuracy,
            "test_macro_f1": test_macro_f1,
            "test_weighted_f1": test_weighted_f1,
        },
        "all_model_results": [
            {
                "model_name": item["model_name"],
                "vectorizer_name": item["vectorizer_name"],
                "validation_macro_f1": float(item["validation_macro_f1"]),
                "validation_weighted_f1": float(item["validation_weighted_f1"]),
            }
            for item in model_results
        ],
        "labels": labels,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "split_strategy": split_strategy,
        "leakage_group_count": int(len(leakage_group_sizes)),
        "near_duplicate_groups": near_duplicate_groups,
        "near_duplicate_rows": near_duplicate_rows,
    }
    comparison_path = MODEL_DIR / "job_category_model_comparison.json"
    comparison_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    chart_labels = [f"{item['model_name']}\n({item['vectorizer_name']})" for item in model_results]
    chart_scores = [item["validation_macro_f1"] for item in model_results]
    chart_test_scores = [test_macro_f1 if item["model_name"] == best["model_name"] and item["vectorizer_name"] == best["vectorizer_name"] else np.nan for item in model_results]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(chart_labels))
    ax.bar(x - 0.18, chart_scores, width=0.32, label="Validation Macro-F1")
    ax.bar(x + 0.18, chart_test_scores, width=0.32, label="Held-out Test Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(chart_labels, rotation=18, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Job-Category Classification Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "job_category_model_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=labels).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Best Model Confusion Matrix: {best['model_name']} + {best['vectorizer_name']}")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "job_category_best_confusion_matrix.png", dpi=180)
    plt.close(fig)

    return summary


def load_best_model(model_path: Optional[Path | str] = None):
    if model_path is None:
        model_path = MODEL_DIR / "job_category_best_model.joblib"
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found at {path}")
    return joblib.load(path)


def predict_job_category(model_path: Optional[str] = None, text: str = ""):
    model = load_best_model(model_path)
    predicted = model.predict([text])[0]
    return predicted
