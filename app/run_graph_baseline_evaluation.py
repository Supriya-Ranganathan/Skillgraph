from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.heterogeneous_skill_graph import (
    DATA_FILE,
    build_heterogeneous_graph,
    dataset_to_records,
    evaluate_recommendation_baselines,
    split_job_skill_edges,
)

PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
VISUALIZATION_DIR = ROOT / "visualizations"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    jobs, edges = dataset_to_records(DATA_FILE)
    arrays, mappings = build_heterogeneous_graph(jobs, edges)
    training_edges, heldout_edges = split_job_skill_edges(edges)
    training_arrays, training_mappings = build_heterogeneous_graph(jobs, training_edges)
    evaluation = evaluate_recommendation_baselines(edges)

    graph_path = PROCESSED_DIR / "heterogeneous_skill_graph.npz"
    mapping_path = PROCESSED_DIR / "heterogeneous_skill_graph_mappings.json"
    training_graph_path = PROCESSED_DIR / "graph_baseline_training_graph.npz"
    training_mapping_path = PROCESSED_DIR / "graph_baseline_training_mappings.json"
    results_path = MODEL_DIR / "graph_recommendation_baselines.json"
    plot_path = VISUALIZATION_DIR / "graph_recommendation_baselines.png"

    np.savez_compressed(graph_path, **arrays)
    mapping_path.write_text(json.dumps(mappings, indent=2), encoding="utf-8")
    np.savez_compressed(training_graph_path, **training_arrays)
    training_mapping_path.write_text(json.dumps(training_mappings, indent=2), encoding="utf-8")
    evaluation["graph_artifacts"] = {
        "full_graph": str(graph_path),
        "full_graph_mappings": str(mapping_path),
        "recommendation_training_graph": str(training_graph_path),
        "recommendation_training_mappings": str(training_mapping_path),
        "recommendation_training_edges": len(training_edges),
        "heldout_edges_excluded_from_recommendation_graph": len(heldout_edges),
    }
    results_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    k = "10"
    metrics = ("precision", "recall", "ndcg")
    positions = np.arange(len(metrics))
    width = 0.34
    frequency = [evaluation["metrics"]["frequency"][k][metric] for metric in metrics]
    cooccurrence = [evaluation["metrics"]["cooccurrence"][k][metric] for metric in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(positions - width / 2, frequency, width, label="Frequency")
    ax.bar(positions + width / 2, cooccurrence, width, label="Skill co-occurrence")
    ax.set_xticks(positions, [metric.upper() + "@10" for metric in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean per-job score")
    ax.set_title("Graph Skill Recommendation Baselines")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(json.dumps({
        "dataset": str(DATA_FILE),
        "job_nodes": len(mappings["job_ids_by_index"]),
        "skill_nodes": len(mappings["skill_names_by_index"]),
        "job_skill_edges": mappings["edge_count"],
        "training_edges": evaluation["split_strategy"]["training_graph_edges"],
        "heldout_edges": evaluation["split_strategy"]["heldout_target_edges"],
        "evaluation_jobs": evaluation["split_strategy"]["evaluation_jobs"],
        "saved_graph": str(graph_path),
        "saved_mappings": str(mapping_path),
        "saved_training_graph": str(training_graph_path),
        "saved_training_mappings": str(training_mapping_path),
        "saved_results": str(results_path),
        "saved_plot": str(plot_path),
        "metrics_at_10": {name: evaluation["metrics"][name][k] for name in ("frequency", "cooccurrence")},
    }, indent=2))


if __name__ == "__main__":
    main()
