from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.job_category_model import MODEL_DIR, PROJECT_ROOT

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "visualizations"
CLASSIFICATION_PLOT = "job_category_validation_scores_readable.png"
PAIRED_PLOT = "paired_graph_benchmark_k10_with_ci.png"
METHODS = ("frequency", "cooccurrence", "graphsage")
METHOD_LABELS = ("Frequency", "Co-occurrence", "GraphSAGE")
COLORS = ("#39789a", "#d57a36", "#4b865c")
METRICS = (
    ("precision", "Precision@10"),
    ("recall", "Recall@10"),
    ("ndcg", "NDCG@10"),
)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_classifier_plot(path: Path, comparison: dict) -> None:
    selected = comparison["selected_model"]
    results = comparison["all_model_results"]
    names = [item["model_name"] for item in results]
    scores = [float(item["validation_macro_f1"]) for item in results]
    colors = ["#19758a" if name == selected["model_name"] else "#91a8b2" for name in names]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    positions = np.arange(len(names))
    bars = axis.barh(positions, scores, color=colors, height=0.66)
    axis.set_yticks(positions, labels=names)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.08)
    axis.set_xlabel("Validation macro-F1")
    axis.set_title("Saved Job-Category Model Comparison")
    axis.grid(axis="x", alpha=0.22)
    axis.set_axisbelow(True)
    axis.bar_label(bars, labels=[f"{score:.4f}" for score in scores], padding=4, fontsize=9)
    figure.text(
        0.5,
        0.015,
        f"Selected on grouped validation: {selected['model_name']} + {selected['vectorizer_name']}; "
        f"saved held-out macro-F1={float(selected['test_macro_f1']):.4f}, "
        f"accuracy={float(selected['test_accuracy']):.4f}.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _write_paired_plot(path: Path, benchmark: dict) -> None:
    aggregated = benchmark["aggregate"]["ranking_metrics"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    for axis, (metric, title) in zip(axes, METRICS):
        means = [aggregated[method]["10"][metric]["mean"] for method in METHODS]
        lower = [
            mean - aggregated[method]["10"][metric]["ci95_lower"]
            for method, mean in zip(METHODS, means)
        ]
        upper = [
            aggregated[method]["10"][metric]["ci95_upper"] - mean
            for method, mean in zip(METHODS, means)
        ]
        bars = axis.bar(
            METHOD_LABELS,
            means,
            color=COLORS,
            yerr=np.asarray([lower, upper]),
            capsize=4,
            width=0.68,
        )
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in means], padding=5, fontsize=9)
        axis.set_title(title)
        axis.set_ylabel("Mean per-job score")
        axis.set_ylim(0, max(0.4, max(means) * 1.35))
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
    seeds = ", ".join(str(seed) for seed in benchmark["seeds"])
    figure.suptitle("Paired Graph Recommendation Benchmark at K=10", fontsize=15)
    figure.text(
        0.5,
        0.015,
        f"Five-seed means; error bars show per-method 95% Student-t intervals. Seeds: {seeds}. Transductive edge completion.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.93))
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write readable audit plots to new filenames without replacing existing visuals."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    paths = (output_dir / CLASSIFICATION_PLOT, output_dir / PAIRED_PLOT)
    existing = [path for path in paths if path.exists()]
    if existing:
        parser.error(f"Refusing to overwrite existing plots: {existing}")

    comparison = _read_json(MODEL_DIR / "job_category_model_comparison.json")
    benchmark = _read_json(MODEL_DIR / "paired_graph_benchmark_evaluation.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_classifier_plot(paths[0], comparison)
    _write_paired_plot(paths[1], benchmark)
    print(
        json.dumps(
            {
                "plots": [str(path) for path in paths],
                "source": "saved job_category_model_comparison.json and paired_graph_benchmark_evaluation.json",
                "selected_classifier": comparison["selected_model"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())