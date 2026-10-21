from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from app.job_category_model import MODEL_DIR, PROJECT_ROOT, prepare_dataset, split_train_test_by_group

EXPECTED_SKLEARN_VERSION = "1.6.1"
DEFAULT_OUTPUT = PROJECT_ROOT / "visualizations" / "job_category_best_confusion_matrix_labeled_v1.png"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify and plot the saved classifier confusion matrix without retraining."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    if output.exists():
        parser.error(f"Refusing to overwrite existing plot: {output}")
    if sklearn.__version__ != EXPECTED_SKLEARN_VERSION:
        parser.error(
            "The saved classifier requires the artifact-compatible scikit-learn "
            f"version {EXPECTED_SKLEARN_VERSION}; found {sklearn.__version__}."
        )

    metadata = json.loads((MODEL_DIR / "job_category_best_model_metadata.json").read_text(encoding="utf-8"))
    comparison = json.loads((MODEL_DIR / "job_category_model_comparison.json").read_text(encoding="utf-8"))
    prepared = prepare_dataset()
    train_data, test_data = split_train_test_by_group(prepared, test_size=0.2, random_state=42)
    if len(train_data) != metadata["train_rows"] or len(test_data) != metadata["test_rows"]:
        parser.error("Recreated grouped split does not match the saved classifier metadata.")

    model = joblib.load(MODEL_DIR / "job_category_best_model.joblib")
    actual = test_data["category"].tolist()
    predicted = model.predict(test_data["text_features"].tolist())
    metrics = {
        "test_accuracy": float(accuracy_score(actual, predicted)),
        "test_macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
    }
    selected = comparison["selected_model"]
    for name, value in metrics.items():
        if value != metadata[name] or value != selected[name]:
            parser.error(f"Inference does not exactly reproduce saved {name}; refusing to plot.")

    labels = metadata["labels"]
    matrix = confusion_matrix(actual, predicted, labels=labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    display_labels = [textwrap.fill(label, width=16, break_long_words=False) for label in labels]
    figure, axis = plt.subplots(figsize=(11, 8))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Held-out job postings")
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=display_labels,
        yticklabels=display_labels,
        xlabel="Predicted category",
        ylabel="True category",
        title="Held-out Job-Category Confusion Matrix\nVerified saved XGBoost + word-TF-IDF model",
    )
    axis.tick_params(axis="x", labelrotation=20)
    for tick in axis.get_xticklabels():
        tick.set_horizontalalignment("right")
    threshold = float(matrix.max()) / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "#17212b",
            )
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(
        json.dumps(
            {
                "scikit_learn": sklearn.__version__,
                "train_rows": len(train_data),
                "test_rows": len(test_data),
                "metrics_reproduced_exactly": metrics,
                "confusion_matrix_labels": labels,
                "confusion_matrix": matrix.tolist(),
                "plot": str(output),
                "model_retrained": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())