"""Experimental heterogeneous GraphSAGE job-skill link prediction.

This module is part of the Phase 3.3 GraphSAGE experiment and is not a production
recommendation path for the live application. The saved evaluation artifacts show that
this configuration underperforms the same-split frequency and co-occurrence
baselines at K=10. The current model uses trainable job/skill ID embeddings only;
it does not use validated text, category, or metadata feature matrices. Until a
compatible evaluation demonstrates a meaningful improvement, GraphSAGE should remain
an experimental artifact and must not be integrated into the live endpoint.
"""

from __future__ import annotations

import json
import random
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graphsage_link_prediction import (
    HeterogeneousGraphSAGE,
    build_training_heterodata,
    checkpoint_config,
    decode_edge_index,
    encode_edge_index,
    link_prediction_metrics,
    load_checkpoint,
    ranking_metrics,
    sample_negative_edges,
    save_checkpoint,
    split_link_prediction_edges,
)

PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
VISUALIZATION_DIR = ROOT / "visualizations"
SEED = 43
NEGATIVE_SEED_BASE = 8100
HIDDEN_DIM = 32
DROPOUT = 0.1
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 60
PATIENCE = 8
RANKING_K = (5, 10, 20)


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_graph_edges() -> Tuple[Dict, Dict, List[Tuple[str, str]], List[Tuple[str, str]]]:
    full_mapping = read_json(PROCESSED_DIR / "heterogeneous_skill_graph_mappings.json")
    train_mapping = read_json(PROCESSED_DIR / "graph_baseline_training_mappings.json")
    full_arrays = np.load(PROCESSED_DIR / "heterogeneous_skill_graph.npz")
    train_arrays = np.load(PROCESSED_DIR / "graph_baseline_training_graph.npz")
    all_edges = decode_edge_index(
        full_arrays["job_to_skill_edge_index"],
        full_mapping["job_ids_by_index"],
        full_mapping["skill_names_by_index"],
    )
    training_edges = decode_edge_index(
        train_arrays["job_to_skill_edge_index"],
        train_mapping["job_ids_by_index"],
        train_mapping["skill_names_by_index"],
    )
    if not set(training_edges).issubset(all_edges):
        raise ValueError("Saved training graph contains edges absent from the full graph.")
    return full_mapping, train_mapping, all_edges, training_edges


def train_with_validation(
    model: HeterogeneousGraphSAGE,
    graph,
    training_positive: torch.Tensor,
    training_negative: torch.Tensor,
    validation_positive: torch.Tensor,
    validation_negative: torch.Tensor,
) -> Tuple[HeterogeneousGraphSAGE, Dict[str, float | int]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_ap = -1.0
    best_auc = -1.0
    best_epoch = 0
    best_training_loss = 0.0
    best_state = None
    stale_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        embeddings = model.encode(graph.edge_index_dict)
        positive_logits = model.score_pairs(embeddings, training_positive)
        negative_logits = model.score_pairs(embeddings, training_negative)
        logits = torch.cat((positive_logits, negative_logits))
        labels = torch.cat((torch.ones_like(positive_logits), torch.zeros_like(negative_logits)))
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_embeddings = model.encode(graph.edge_index_dict)
            validation_positive_scores = model.score_pairs(validation_embeddings, validation_positive).cpu().numpy()
            validation_negative_scores = model.score_pairs(validation_embeddings, validation_negative).cpu().numpy()
        validation_metrics = link_prediction_metrics(validation_positive_scores, validation_negative_scores)
        validation_ap = validation_metrics["average_precision"]
        validation_auc = validation_metrics["roc_auc"]

        if (validation_ap, validation_auc) > (best_ap, best_auc):
            best_ap = validation_ap
            best_auc = validation_auc
            best_epoch = epoch
            best_training_loss = float(loss.detach().cpu())
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a validation-selected checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "validation_roc_auc": best_auc,
        "validation_average_precision": best_ap,
        "training_loss_at_selected_epoch": best_training_loss,
    }


def _training_statistics(edges: Iterable[Tuple[str, str]]):
    skills_by_job: Dict[str, set[str]] = defaultdict(set)
    frequency: Counter[str] = Counter()
    for job_id, skill in edges:
        skills_by_job[job_id].add(skill)
        frequency[skill] += 1
    cooccurrence: Counter[Tuple[str, str]] = Counter()
    for skills in skills_by_job.values():
        ordered = sorted(skills)
        for first_index, first in enumerate(ordered):
            for second in ordered[first_index + 1 :]:
                cooccurrence[(first, second)] += 1
    return skills_by_job, frequency, cooccurrence


def _mean_ranking_results(rankings, targets, k_values=RANKING_K):
    totals = {k: Counter() for k in k_values}
    for job_id, relevant in targets.items():
        for k in k_values:
            values = ranking_metrics(rankings.get(job_id, []), relevant, k)
            for name, value in values.items():
                totals[k][name] += value
    if not targets:
        raise ValueError("No test jobs contain held-out positive edges.")
    return {
        str(k): {name: totals[k][name] / len(targets) for name in ("precision", "recall", "ndcg")}
        for k in k_values
    }


def ranking_candidate_skills(
    candidate_skills: Sequence[str],
    training_context: Iterable[str],
    all_known_job_skills: Iterable[str],
    test_targets: Iterable[str],
) -> List[str]:
    known_non_targets = set(all_known_job_skills) - set(test_targets)
    excluded = set(training_context) | known_non_targets
    return [skill for skill in candidate_skills if skill not in excluded]


def build_rankings(
    model: HeterogeneousGraphSAGE,
    graph,
    training_edges: Sequence[Tuple[str, str]],
    test_edges: Sequence[Tuple[str, str]],
    all_positive_edges: Sequence[Tuple[str, str]],
    job_to_index: Mapping[str, int],
    skill_to_index: Mapping[str, int],
) -> Dict[str, Dict[str, List[str]]]:
    skills_by_job, frequency, cooccurrence = _training_statistics(training_edges)
    targets: Dict[str, set[str]] = defaultdict(set)
    for job_id, skill in test_edges:
        targets[job_id].add(skill)
    all_positive_by_job: Dict[str, set[str]] = defaultdict(set)
    for job_id, skill in all_positive_edges:
        all_positive_by_job[job_id].add(skill)
    candidate_skills = sorted(frequency)
    candidate_indices = torch.tensor([skill_to_index[skill] for skill in candidate_skills], dtype=torch.long)
    job_ids = sorted(targets)

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(graph.edge_index_dict)
        job_indices = torch.tensor([job_to_index[job_id] for job_id in job_ids], dtype=torch.long)
        weighted_jobs = embeddings["job"][job_indices] * model.edge_weight
        candidate_vectors = embeddings["skill"][candidate_indices]
        score_matrix = (weighted_jobs @ candidate_vectors.T / model.hidden_dim**0.5 + model.edge_bias).cpu().numpy()

    graphsage_rankings: Dict[str, List[str]] = {}
    frequency_rankings: Dict[str, List[str]] = {}
    cooccurrence_rankings: Dict[str, List[str]] = {}
    for row_index, job_id in enumerate(job_ids):
        known_skills = skills_by_job.get(job_id, set())
        eligible_skills = set(ranking_candidate_skills(
            candidate_skills,
            known_skills,
            all_positive_by_job.get(job_id, set()),
            targets[job_id],
        ))
        candidate_scores = {
            skill: float(score_matrix[row_index, index])
            for index, skill in enumerate(candidate_skills)
            if skill in eligible_skills
        }
        graphsage_rankings[job_id] = sorted(candidate_scores, key=lambda skill: (-candidate_scores[skill], skill))[: max(RANKING_K)]
        frequency_rankings[job_id] = sorted(
            eligible_skills,
            key=lambda skill: (-frequency[skill], skill),
        )[: max(RANKING_K)]
        cooccurrence_scores: Counter[str] = Counter()
        for known_skill in known_skills:
            for candidate in candidate_skills:
                if candidate != known_skill:
                    cooccurrence_scores[candidate] += cooccurrence.get(tuple(sorted((known_skill, candidate))), 0)
        cooccurrence_rankings[job_id] = sorted(
            eligible_skills,
            key=lambda skill: (-cooccurrence_scores[skill], skill),
        )[: max(RANKING_K)]

    return {
        "graphsage": _mean_ranking_results(graphsage_rankings, targets),
        "frequency_same_test_split": _mean_ranking_results(frequency_rankings, targets),
        "cooccurrence_same_test_split": _mean_ranking_results(cooccurrence_rankings, targets),
    }


def _svg_text(parent, x: int, y: int, value: str, size: int = 14, color: str = "#243047"):
    element = ET.SubElement(parent, "text", {"x": str(x), "y": str(y), "font-size": str(size), "fill": color})
    element.text = value


def write_comparison_svg(path: Path, test_metrics: Mapping, ranking: Mapping) -> None:
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    svg = ET.Element(f"{{{namespace}}}svg", {"width": "1040", "height": "600", "viewBox": "0 0 1040 600"})
    ET.SubElement(svg, f"{{{namespace}}}rect", {"width": "1040", "height": "600", "fill": "#f6f8fb"})
    _svg_text(svg, 36, 42, "GraphSAGE Link Prediction", 24)
    _svg_text(svg, 36, 72, "Ranking metrics use the same held-out test edges and train-visible skill candidates", 13, "#596579")

    series = [
        ("GraphSAGE", "#187b8f", "graphsage"),
        ("Frequency", "#cf6a3a", "frequency_same_test_split"),
        ("Co-occurrence", "#4b8c5a", "cooccurrence_same_test_split"),
    ]
    metric_names = (("precision", "Precision@10"), ("recall", "Recall@10"), ("ndcg", "NDCG@10"))
    _svg_text(svg, 36, 116, "Ranking", 18)
    for group_index, (metric, label) in enumerate(metric_names):
        y_base = 170 + group_index * 112
        _svg_text(svg, 36, y_base - 13, label, 14)
        for series_index, (series_name, color, key) in enumerate(series):
            value = ranking[key]["10"][metric]
            y = y_base + series_index * 24
            ET.SubElement(svg, f"{{{namespace}}}rect", {
                "x": "180", "y": str(y - 13), "width": str(max(1, round(value * 380))), "height": "16", "rx": "2", "fill": color,
            })
            _svg_text(svg, 574, y, f"{series_name}: {value:.4f}", 12)

    _svg_text(svg, 700, 116, "Sampled-negative link classification", 17)
    for index, (name, key) in enumerate((("ROC-AUC", "roc_auc"), ("Average precision", "average_precision"))):
        value = test_metrics[key]
        y = 178 + index * 78
        _svg_text(svg, 700, y, name, 14)
        ET.SubElement(svg, f"{{{namespace}}}rect", {
            "x": "700", "y": str(y + 12), "width": str(max(1, round(value * 260))), "height": "20", "rx": "2", "fill": "#187b8f",
        })
        _svg_text(svg, 970, y + 28, f"{value:.4f}", 12)

    _svg_text(svg, 36, 548, "ROC-AUC/AP use 1:1 sampled negatives; all known positive pairs were excluded from negative sampling.", 12, "#596579")
    _svg_text(svg, 36, 572, "The existing Phase 3.2 full-holdout results are reported separately because their evaluation split differs.", 12, "#596579")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg).write(path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    start_time = time.perf_counter()
    set_deterministic_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    full_mapping, training_mapping, all_edges, phase3_training_edges = load_graph_edges()
    train_edges, validation_edges, test_edges = split_link_prediction_edges(
        all_edges,
        phase3_training_edges,
        random_state=SEED,
    )
    job_ids = full_mapping["job_ids_by_index"]
    skill_names = full_mapping["skill_names_by_index"]
    job_to_index = {job_id: index for index, job_id in enumerate(job_ids)}
    skill_to_index = {skill: index for index, skill in enumerate(skill_names)}
    train_positive = encode_edge_index(train_edges, job_to_index, skill_to_index)
    validation_positive = encode_edge_index(validation_edges, job_to_index, skill_to_index)
    test_positive = encode_edge_index(test_edges, job_to_index, skill_to_index)
    all_known_positive = {
        (job_to_index[job_id], skill_to_index[skill])
        for job_id, skill in all_edges
    }
    train_negative = sample_negative_edges(len(train_edges), len(job_ids), len(skill_names), all_known_positive, NEGATIVE_SEED_BASE)
    train_negative_pairs = {tuple(map(int, pair)) for pair in train_negative.T}
    validation_negative = sample_negative_edges(
        len(validation_edges), len(job_ids), len(skill_names), all_known_positive,
        NEGATIVE_SEED_BASE + 1, excluded_negative_edges=train_negative_pairs,
    )
    validation_negative_pairs = {tuple(map(int, pair)) for pair in validation_negative.T}
    test_negative = sample_negative_edges(
        len(test_edges), len(job_ids), len(skill_names), all_known_positive,
        NEGATIVE_SEED_BASE + 2, excluded_negative_edges=train_negative_pairs | validation_negative_pairs,
    )

    graph = build_training_heterodata(len(job_ids), len(skill_names), train_positive)
    model = HeterogeneousGraphSAGE(len(job_ids), len(skill_names), HIDDEN_DIM, DROPOUT)
    model, validation_metrics = train_with_validation(
        model,
        graph,
        train_positive,
        train_negative,
        validation_positive,
        validation_negative,
    )

    model_config = checkpoint_config(len(job_ids), len(skill_names), HIDDEN_DIM, DROPOUT)
    training_config = {
        **model_config,
        "seed": SEED,
        "device": "cpu",
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "negative_sampling_ratio": 1.0,
        "negative_sampling_seed_base": NEGATIVE_SEED_BASE,
        "message_passing_edges": len(train_edges),
        "validation_edges": len(validation_edges),
        "test_edges": len(test_edges),
        "node_mapping_reference": "data/processed/heterogeneous_skill_graph_mappings.json",
        "training_graph_reference": "data/processed/graph_baseline_training_graph.npz",
    }
    checkpoint_path = MODEL_DIR / "graphsage_link_prediction.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        model_config,
        {**validation_metrics, "selected_by": "validation_average_precision", "seed": SEED},
    )
    config_path = MODEL_DIR / "graphsage_link_prediction_config.json"
    config_path.write_text(json.dumps(training_config, indent=2), encoding="utf-8")

    model, _ = load_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(graph.edge_index_dict)
        test_positive_scores = model.score_pairs(embeddings, test_positive).numpy()
        test_negative_scores = model.score_pairs(embeddings, test_negative).numpy()
    test_metrics = link_prediction_metrics(test_positive_scores, test_negative_scores)
    same_split_ranking = build_rankings(
        model,
        graph,
        train_edges,
        test_edges,
        all_edges,
        job_to_index,
        skill_to_index,
    )

    phase3_baseline = read_json(MODEL_DIR / "graph_recommendation_baselines.json")
    phase3_reference = {
        "protocol": "Existing Phase 3.2 single 20% per-job edge holdout; not directly identical to the GraphSAGE test subset.",
        "metrics_at_10": {
            name: phase3_baseline["metrics"][name]["10"]
            for name in ("frequency", "cooccurrence")
        },
    }
    elapsed = time.perf_counter() - start_time
    evaluation = {
        "task": "bipartite job-skill link prediction",
        "model": "heterogeneous GraphSAGE",
        "validation_checkpoint_selection": "Highest validation average precision; ROC-AUC is the deterministic tie-breaker.",
        "split": {
            "source": "Existing deterministic Phase 3.1 training-only graph; its 4,611 held-out edges were split into validation and test.",
            "seed": SEED,
            "training_positive_edges": len(train_edges),
            "validation_positive_edges": len(validation_edges),
            "test_positive_edges": len(test_edges),
            "training_message_passing_edges": len(train_edges),
            "validation_test_edges_in_message_passing_adjacency": 0,
            "train_jobs": len(job_ids),
            "full_skill_nodes": len(skill_names),
            "training_visible_candidate_skills": len(training_mapping["skill_names_by_index"]),
        },
        "negative_sampling": {
            "ratio": "1:1 negatives to positives for training, validation, and test.",
            "rule": "Uniform job-skill pair sampling; every known positive edge from all splits is excluded.",
            "unobserved_pair_caveat": "Unobserved pairs may be undiscovered true relationships and are treated as negatives for this benchmark.",
        },
        "validation_metrics": validation_metrics,
        "test_sampled_negative_metrics": test_metrics,
        "ranking_metrics_same_test_edges": same_split_ranking,
        "phase3_2_reference_different_protocol": phase3_reference,
        "checkpoint": str(checkpoint_path),
        "configuration": str(config_path),
        "node_mapping_reference": "data/processed/heterogeneous_skill_graph_mappings.json",
        "training_graph_reference": "data/processed/graph_baseline_training_graph.npz",
        "runtime_seconds": elapsed,
        "limitations": [
            "Node features are trainable ID embeddings because the graph has no validated numeric or textual node-feature matrix; category and job-title metadata are not model inputs.",
            "The model is transductive and uses the full node universe, including skills without a training edge; ranking candidates are restricted to training-visible skills for baseline comparability.",
            "Sampled unobserved pairs are not guaranteed true negatives.",
            "Phase 3.2 reference scores use a different single 20% edge holdout; same-test-split frequency and co-occurrence metrics are provided for direct ranking comparison.",
        ],
    }
    result_path = MODEL_DIR / "graphsage_link_prediction_evaluation.json"
    result_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    split_path = MODEL_DIR / "graphsage_link_prediction_splits.npz"
    np.savez_compressed(
        split_path,
        training_positive_edge_index=train_positive.numpy(),
        validation_positive_edge_index=validation_positive.numpy(),
        test_positive_edge_index=test_positive.numpy(),
        training_negative_edge_index=train_negative.numpy(),
        validation_negative_edge_index=validation_negative.numpy(),
        test_negative_edge_index=test_negative.numpy(),
        message_passing_edge_index=train_positive.numpy(),
    )
    plot_path = VISUALIZATION_DIR / "graphsage_link_prediction_comparison.svg"
    write_comparison_svg(plot_path, test_metrics, same_split_ranking)

    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "configuration": str(config_path),
        "splits": str(split_path),
        "evaluation": str(result_path),
        "plot": str(plot_path),
        "validation": validation_metrics,
        "test_sampled_negative_metrics": test_metrics,
        "ranking_at_10": {name: metrics["10"] for name, metrics in same_split_ranking.items()},
        "runtime_seconds": round(elapsed, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
