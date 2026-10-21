from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from app.job_category_model import (
    DATA_FILE,
    PROJECT_ROOT,
    make_estimator,
    make_vectorizer,
    prepare_dataset,
    split_train_test_by_group,
)

DEFAULT_RESULT = PROJECT_ROOT / "experiments" / "classifier_feature_ablation_v1.json"
DEFAULT_PLOT = PROJECT_ROOT / "visualizations" / "classifier_feature_ablation_v1.png"
VARIANTS = {
    "full_title_description_skills": ("job_title", "job_description", "skills"),
    "without_skills": ("job_title", "job_description"),
    "without_description": ("job_title", "skills"),
}
SEED = 42
FOLDS = 3


def build_ablation_features(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Join selected text columns in the same normalized format as the classifier."""
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Ablation columns are missing from the dataset: {sorted(missing)}")
    normalized = frame.loc[:, columns].fillna("").astype(str).apply(
        lambda column: column.str.strip().str.lower().str.split().str.join(" ")
    )
    return normalized.agg(" ".join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip()


def run_ablation(data_file: Path = DATA_FILE) -> dict[str, Any]:
    start = time.perf_counter()
    prepared = prepare_dataset(data_file)
    train_data, _ = split_train_test_by_group(prepared, test_size=0.2, random_state=SEED)
    labels = train_data["category"].to_numpy()
    groups = train_data["leakage_group"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    fold_indices = list(splitter.split(np.zeros(len(train_data)), labels, groups))
    results: dict[str, Any] = {}

    for name, columns in VARIANTS.items():
        features = build_ablation_features(train_data, columns).to_numpy()
        fold_metrics = []
        for train_indices, validation_indices in fold_indices:
            pipeline = Pipeline(
                steps=[
                    ("vectorizer", make_vectorizer("word_tfidf")),
                    ("classifier", make_estimator("XGBoost", random_state=SEED)),
                ]
            )
            pipeline.fit(features[train_indices], labels[train_indices])
            predicted = pipeline.predict(features[validation_indices])
            actual = labels[validation_indices]
            fold_metrics.append(
                {
                    "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
                    "weighted_f1": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
                }
            )
        results[name] = {
            "features": list(columns),
            "validation_fold_metrics": fold_metrics,
            "validation_macro_f1_mean": float(np.mean([item["macro_f1"] for item in fold_metrics])),
            "validation_macro_f1_sample_std": float(np.std([item["macro_f1"] for item in fold_metrics], ddof=1)),
            "validation_weighted_f1_mean": float(np.mean([item["weighted_f1"] for item in fold_metrics])),
            "validation_weighted_f1_sample_std": float(np.std([item["weighted_f1"] for item in fold_metrics], ddof=1)),
        }

    return {
        "experiment": "XGBoost text-feature ablation; validation only",
        "data_file": str(data_file),
        "dataset_rows": int(len(prepared)),
        "training_rows_for_ablation": int(len(train_data)),
        "heldout_test_rows_used": 0,
        "split": {
            "outer_split": "The existing grouped 80/20 split was reproduced; the held-out partition was not read by the experiment.",
            "validation": f"{FOLDS}-fold StratifiedGroupKFold on outer training rows only; shuffle=True; random_state={SEED}.",
            "groups": "Existing job-ID and near-duplicate text connected components from the classifier pipeline.",
        },
        "model_configuration": {
            "model": "XGBoost via EncodedXGBClassifier",
            "vectorizer": "word_tfidf (unigrams/bigrams; current repository implementation)",
            "estimator_parameters": {
                "n_estimators": 40,
                "max_depth": 3,
                "learning_rate": 0.12,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": SEED,
                "n_jobs": 1,
            },
            "selection": "No test-set selection; variants are compared by mean grouped validation macro-F1.",
        },
        "variants": results,
        "environment": {
            package: importlib.metadata.version(package)
            for package in ("scikit-learn", "xgboost", "pandas", "numpy")
        },
        "runtime_seconds": round(time.perf_counter() - start, 3),
        "limitations": [
            "Only three grouped validation folds are used; results have fold variability and are not a new held-out test score.",
            "The ablation is compared within this installed environment and is not directly comparable to training performed under a different scikit-learn version.",
            "This experiment does not retrain, replace, or evaluate the canonical saved classifier on its held-out test set.",
        ],
    }


def write_plot(result: dict[str, Any], output: Path) -> None:
    variants = result["variants"]
    names = list(variants)
    labels = [name.replace("_", " ") for name in names]
    means = [variants[name]["validation_macro_f1_mean"] for name in names]
    stds = [variants[name]["validation_macro_f1_sample_std"] for name in names]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.barh(labels, means, xerr=stds, color=("#19758a", "#91a8b2", "#d57a36"), capsize=4)
    axis.invert_yaxis()
    axis.set_xlim(0, min(1.08, max(means) + max(stds) + 0.08))
    axis.set_xlabel("Grouped validation macro-F1 (mean ± fold standard deviation)")
    axis.set_title("XGBoost Text-Feature Ablation (Validation Only)")
    axis.grid(axis="x", alpha=0.22)
    axis.set_axisbelow(True)
    axis.bar_label(bars, labels=[f"{value:.4f}" for value in means], padding=4)
    figure.text(
        0.5,
        0.015,
        "Three-fold grouped CV on outer training rows only; held-out test rows were not used.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a grouped-validation-only XGBoost feature ablation.")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    plot = args.plot if args.plot.is_absolute() else PROJECT_ROOT / args.plot
    existing = [path for path in (output, plot) if path.exists()]
    if existing:
        parser.error(f"Refusing to overwrite existing experiment outputs: {existing}")
    result = run_ablation()
    output.parent.mkdir(parents=True, exist_ok=True)
    plot.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_plot(result, plot)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Result: {output}")
    print(f"Plot: {plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())