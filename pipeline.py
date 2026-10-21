
"""
SkillGraph-X: Machine Learning Pipeline
Task: Job-category classification using TF-IDF and XGBoost.
"""

import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)
from xgboost import XGBClassifier


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "job_skill_edges.csv"
OUTPUT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load data and identify text and target columns."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Update these names to match your actual dataset.
    text_column = "job_description"
    target_column = "job_category"

    if text_column not in df.columns:
        raise ValueError(
            f"Missing '{text_column}'. Available columns: "
            f"{df.columns.tolist()}"
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Missing '{target_column}'. Available columns: "
            f"{df.columns.tolist()}"
        )

    df = df.dropna(subset=[text_column, target_column])
    df[text_column] = df[text_column].astype(str)

    return df, text_column, target_column


def build_pipeline():
    """Combine text feature extraction and classification."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words="english"
            )
        ),
        (
            "classifier",
            XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1
            )
        )
    ])


def train_and_evaluate():
    """Train, tune, evaluate, and save the model."""
    df, text_column, target_column = load_data()

    X = df[text_column]
    y = df[target_column]

    # Split data before fitting the pipeline.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline = build_pipeline()

    # Small illustrative grid; tuning can take time.
    param_grid = {
        "tfidf__max_features": [5000, 10000],
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.05, 0.1]
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    print("\nStarting hyperparameter tuning...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro"
    )

    print("\nBest parameters:")
    print(search.best_params_)

    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Test macro-F1: {macro_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    # Save trained pipeline
    model_path = OUTPUT_DIR / "skillgraph_classifier.joblib"
    joblib.dump(best_model, model_path)

    # Save experiment metadata and metrics
    results = {
        "experiment": "SkillGraph-X job classification",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "TF-IDF + XGBoost",
        "random_state": 42,
        "cv_folds": 3,
        "best_parameters": search.best_params_,
        "best_cv_macro_f1": float(search.best_score_),
        "test_accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test))
    }

    results_path = OUTPUT_DIR / "experiment_results.json"
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"\nModel saved to: {model_path}")
    print(f"Experiment results saved to: {results_path}")


if __name__ == "__main__":
    train_and_evaluate()
