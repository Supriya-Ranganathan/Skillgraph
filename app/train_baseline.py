from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "cleaned_jobs.csv"
MODEL_DIR = ROOT / "models"
VIZ_DIR = ROOT / "visualizations"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_FILE)

    df = df.dropna(subset=["category", "skills"]).copy()
    df["category"] = df["category"].astype(str).str.strip()
    df["skills"] = df["skills"].astype(str).str.strip()
    df = df[(df["category"] != "") & (df["skills"] != "")]

    # Remove categories with fewer than two examples so stratification works.
    counts = df["category"].value_counts()
    df = df[df["category"].isin(counts[counts >= 2].index)].copy()

    if df["category"].nunique() < 2:
        raise ValueError("At least two categories are needed.")

    X = df["skills"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                token_pattern=r"(?u)\b\w[\w+#.-]*\b",
                ngram_range=(1, 2),
                min_df=2,
                max_features=5000,
            ),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    results = {
        "model": "TF-IDF + Random Forest",
        "target": "job category",
        "test_accuracy": float(accuracy),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_count": int(y.nunique()),
        "classification_report": report,
    }

    results_file = MODEL_DIR / "baseline_metrics.json"
    with open(results_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    # Confusion matrix for the held-out test set
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        xticks_rotation="vertical",
        include_values=False,
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Random Forest Baseline — Job Category")
    fig.tight_layout()

    chart_file = VIZ_DIR / "baseline_confusion_matrix.png"
    fig.savefig(chart_file, dpi=180)
    plt.close(fig)

    print("=" * 55)
    print("SKILLGRAPH-X BASELINE EVALUATION")
    print("=" * 55)
    print("Prediction target: job category")
    print("Number of categories:", y.nunique())
    print("Training records:", len(X_train))
    print("Test records:", len(X_test))
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))
    print("\nSaved metrics:", results_file)
    print("Saved confusion matrix:", chart_file)


if __name__ == "__main__":
    main()
