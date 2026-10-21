from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
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

    counts = df["category"].value_counts()
    df = df[df["category"].isin(counts[counts >= 2].index)]

    X = df["skills"]
    y = df["category"]

    # Encode category labels for XGBoost
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded,
    )

    # Fit TF-IDF only on training data
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w[\w+#.-]*\b",
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=2,
            tree_method="hist",
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_vec, y_train)
        predictions = model.predict(X_test_vec)

        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        results[name] = {
            "accuracy": float(accuracy),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
        }

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Macro F1:  {f1:.4f}")

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=range(len(encoder.classes_)),
                target_names=encoder.classes_,
                zero_division=0,
            )
        )

    # Save comparison metrics
    output_file = MODEL_DIR / "model_comparison.json"
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    # Comparison chart
    names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in names]
    f1_scores = [results[name]["macro_f1"] for name in names]

    positions = list(range(len(names)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        [p - width / 2 for p in positions],
        accuracies,
        width,
        label="Accuracy",
    )
    ax.bar(
        [p + width / 2 for p in positions],
        f1_scores,
        width,
        label="Macro F1",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Random Forest vs XGBoost")
    ax.legend()
    fig.tight_layout()

    chart_file = VIZ_DIR / "model_comparison.png"
    fig.savefig(chart_file, dpi=180)
    plt.close(fig)

    print("\nComparison summary:")
    print(pd.DataFrame(results).T.round(4).to_string())
    print("\nSaved metrics:", output_file)
    print("Saved chart:", chart_file)


if __name__ == "__main__":
    main()
