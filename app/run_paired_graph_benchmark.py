"""Paired five-seed edge-completion benchmark for graph recommendations.

This runner uses one per-seed train/validation/test split and one filtered
candidate set per test job for all three rankers. It is separate from the
existing Phase 3.2 and 3.3 runners and refuses to overwrite its output files.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch

from app.graphsage_link_prediction import (
    HeterogeneousGraphSAGE,
    build_training_heterodata,
    encode_edge_index,
    link_prediction_metrics,
    ranking_metrics,
    sample_negative_edges,
)
from app.heterogeneous_skill_graph import RANKING_K, _training_skill_statistics
from app.run_graphsage_link_prediction import (
    DROPOUT,
    HIDDEN_DIM,
    LEARNING_RATE,
    MAX_EPOCHS,
    NEGATIVE_SEED_BASE,
    PATIENCE,
    WEIGHT_DECAY,
    load_graph_edges,
    ranking_candidate_skills,
    set_deterministic_seed,
    train_with_validation,
)


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
SEEDS = (43, 47, 53, 59, 61)
HOLDOUT_FRACTION = 0.20
VALIDATION_FRACTION_OF_HOLDOUT = 0.50
METHODS = ("frequency", "cooccurrence", "graphsage")
BASELINES = ("frequency", "cooccurrence")
METRICS = ("precision", "recall", "ndcg")
K_VALUES = tuple(RANKING_K)
T_95_DF4 = 2.7764451051977987

SUMMARY_PATH = MODEL_DIR / "paired_graph_benchmark_evaluation.json"
SPLITS_PATH = MODEL_DIR / "paired_graph_benchmark_splits.npz"
PAIRED_ROWS_PATH = MODEL_DIR / "paired_graph_benchmark_per_job_differences.csv"

JobSkillEdge = Tuple[str, str]


def split_edges_per_job(
    edges: Iterable[JobSkillEdge],
    random_state: int,
    holdout_fraction: float = HOLDOUT_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION_OF_HOLDOUT,
) -> Dict[str, list[JobSkillEdge]]:
    """Split each job's edges into train plus held-out validation/test edges.

    At least one edge per job stays in training. Low-degree jobs may have a
    held-out edge in only validation or test; the global partitions must both
    be non-empty. Jobs with one observed skill remain training-only.
    """
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    unique_edges = set((str(job), str(skill)) for job, skill in edges)
    by_job: Dict[str, list[str]] = defaultdict(list)
    for job_id, skill in unique_edges:
        by_job[job_id].append(skill)

    rng = random.Random(random_state)
    result: Dict[str, list[JobSkillEdge]] = {
        "training": [],
        "validation": [],
        "test": [],
    }

    for job_id in sorted(by_job):
        skills = sorted(by_job[job_id])
        rng.shuffle(skills)
        if len(skills) < 2:
            result["training"].extend((job_id, skill) for skill in skills)
            continue

        heldout_count = min(
            len(skills) - 1,
            max(1, math.ceil(len(skills) * holdout_fraction)),
        )
        heldout = skills[:heldout_count]
        training = skills[heldout_count:]
        result["training"].extend((job_id, skill) for skill in training)

        if len(heldout) == 1:
            partition = "validation" if rng.random() < validation_fraction else "test"
            result[partition].append((job_id, heldout[0]))
            continue

        validation_count = round(len(heldout) * validation_fraction)
        validation_count = min(len(heldout) - 1, max(1, validation_count))
        result["validation"].extend((job_id, skill) for skill in heldout[:validation_count])
        result["test"].extend((job_id, skill) for skill in heldout[validation_count:])

    for partition in result:
        result[partition].sort()
    if not result["training"] or not result["validation"] or not result["test"]:
        raise ValueError("The split must produce non-empty train, validation, and test edges.")

    train_set = set(result["training"])
    validation_set = set(result["validation"])
    test_set = set(result["test"])
    if train_set & validation_set or train_set & test_set or validation_set & test_set:
        raise RuntimeError("An edge appears in more than one split.")
    if train_set | validation_set | test_set != unique_edges:
        raise RuntimeError("The split does not cover all unique input edges.")

    train_by_job: Dict[str, set[str]] = defaultdict(set)
    for job_id, skill in result["training"]:
        train_by_job[job_id].add(skill)
    evaluated_jobs = {job_id for job_id, _ in result["validation"] + result["test"]}
    if any(not train_by_job[job_id] for job_id in evaluated_jobs):
        raise RuntimeError("Every held-out job must retain at least one training edge.")
    return result


def build_filtered_candidate_sets(
    training_edges: Iterable[JobSkillEdge],
    all_positive_edges: Iterable[JobSkillEdge],
    test_edges: Iterable[JobSkillEdge],
) -> Tuple[Dict[str, Tuple[str, ...]], Dict[str, set[str]], Dict[str, set[str]], Counter[str], Counter[Tuple[str, str]]]:
    """Build one training-vocabulary candidate set per test job.

    The same returned candidate lists are passed to frequency, co-occurrence,
    and GraphSAGE; known non-test positives and the job's training context are
    excluded, while test targets remain eligible.
    """
    training = set((str(job), str(skill)) for job, skill in training_edges)
    all_positive = set((str(job), str(skill)) for job, skill in all_positive_edges)
    targets: Dict[str, set[str]] = defaultdict(set)
    all_skills_by_job: Dict[str, set[str]] = defaultdict(set)
    for job_id, skill in all_positive:
        all_skills_by_job[job_id].add(skill)
    for job_id, skill in test_edges:
        targets[str(job_id)].add(str(skill))

    training_context, frequency, cooccurrence = _training_skill_statistics(training)
    candidate_vocabulary = sorted(frequency)
    candidate_sets: Dict[str, Tuple[str, ...]] = {}
    for job_id in sorted(targets):
        candidate_sets[job_id] = tuple(
            ranking_candidate_skills(
                candidate_vocabulary,
                training_context.get(job_id, set()),
                all_skills_by_job.get(job_id, set()),
                targets[job_id],
            )
        )
    return candidate_sets, dict(targets), dict(training_context), frequency, cooccurrence


def rank_all_methods_for_job(
    candidates: Sequence[str],
    training_context: Iterable[str],
    frequency: Mapping[str, int],
    cooccurrence: Mapping[Tuple[str, str], int],
    graphsage_scores: Mapping[str, float],
    limit: int,
) -> Dict[str, list[str]]:
    """Rank the exact same candidate sequence with all three methods."""
    context = set(training_context)
    available = tuple(candidates)
    cooccurrence_scores: Counter[str] = Counter()
    for known_skill in context:
        for candidate in available:
            if candidate != known_skill:
                cooccurrence_scores[candidate] += cooccurrence.get(
                    tuple(sorted((known_skill, candidate))), 0
                )

    rankings = {
        "frequency": sorted(available, key=lambda skill: (-frequency.get(skill, 0), skill))[:limit],
        "cooccurrence": sorted(available, key=lambda skill: (-cooccurrence_scores[skill], skill))[:limit],
        "graphsage": sorted(available, key=lambda skill: (-graphsage_scores.get(skill, float("-inf")), skill))[:limit],
    }
    candidate_set = set(available)
    if any(not set(ranked).issubset(candidate_set) for ranked in rankings.values()):
        raise RuntimeError("A ranker returned a skill outside the shared candidate set.")
    return rankings


def evaluate_paired_rankings(
    rankings_by_method: Mapping[str, Mapping[str, Sequence[str]]],
    targets_by_job: Mapping[str, Iterable[str]],
    candidate_sets: Mapping[str, Sequence[str]],
    k_values: Sequence[int] = K_VALUES,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], list[dict[str, Any]], Dict[str, Dict[str, Dict[str, float]]]]:
    """Return macro metrics, per-job metrics, and paired per-job deltas."""
    jobs = sorted(targets_by_job)
    if not jobs:
        raise ValueError("No test jobs have held-out positive edges.")
    if set(jobs) != set(candidate_sets):
        raise ValueError("Every evaluated test job must have exactly one candidate set.")
    if any(set(jobs) != set(rankings_by_method.get(method, {})) for method in METHODS):
        raise ValueError("All methods must rank exactly the same test jobs.")

    per_job: list[dict[str, Any]] = []
    totals = {method: {str(k): Counter() for k in k_values} for method in METHODS}
    paired_totals = {
        baseline: {str(k): Counter() for k in k_values}
        for baseline in BASELINES
    }
    for job_id in jobs:
        row: dict[str, Any] = {
            "job_id": job_id,
            "candidate_count": len(candidate_sets[job_id]),
            "target_count": len(set(targets_by_job[job_id])),
            "metrics": {},
            "paired_differences": {},
        }
        for method in METHODS:
            row["metrics"][method] = {}
            for k in k_values:
                values = ranking_metrics(rankings_by_method[method][job_id], targets_by_job[job_id], k)
                row["metrics"][method][str(k)] = values
                for metric, score in values.items():
                    totals[method][str(k)][metric] += score

        for baseline in BASELINES:
            row["paired_differences"][baseline] = {}
            for k in k_values:
                differences = {
                    metric: (
                        row["metrics"]["graphsage"][str(k)][metric]
                        - row["metrics"][baseline][str(k)][metric]
                    )
                    for metric in METRICS
                }
                row["paired_differences"][baseline][str(k)] = differences
                for metric, value in differences.items():
                    paired_totals[baseline][str(k)][metric] += value
        per_job.append(row)

    denominator = len(jobs)
    macro_metrics = {
        method: {
            str(k): {
                metric: totals[method][str(k)][metric] / denominator
                for metric in METRICS
            }
            for k in k_values
        }
        for method in METHODS
    }
    macro_differences = {
        baseline: {
            str(k): {
                metric: paired_totals[baseline][str(k)][metric] / denominator
                for metric in METRICS
            }
            for k in k_values
        }
        for baseline in BASELINES
    }
    return macro_metrics, per_job, macro_differences


def confidence_interval(values: Sequence[float]) -> Dict[str, float | int | str | None]:
    """Compute a two-sided 95% Student-t CI over seed-level observations."""
    observations = np.asarray(values, dtype=float)
    count = int(observations.size)
    mean = float(observations.mean()) if count else float("nan")
    if count < 2:
        return {
            "n_seeds": count,
            "mean": mean,
            "sample_standard_deviation": None,
            "ci95_lower": None,
            "ci95_upper": None,
            "method": "Student-t interval unavailable with fewer than two seeds.",
        }
    standard_deviation = float(observations.std(ddof=1))
    critical = T_95_DF4 if count == 5 else float("nan")
    if count != 5:
        raise ValueError("The predeclared benchmark requires exactly five seed observations.")
    margin = critical * standard_deviation / math.sqrt(count)
    return {
        "n_seeds": count,
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "ci95_lower": mean - margin,
        "ci95_upper": mean + margin,
        "method": "Two-sided Student-t 95% interval across five seed-level values (df=4, t*=2.776445).",
    }


def _rank_seed(
    model: HeterogeneousGraphSAGE,
    graph,
    candidate_sets: Mapping[str, Sequence[str]],
    targets: Mapping[str, Iterable[str]],
    training_context: Mapping[str, Iterable[str]],
    frequency: Mapping[str, int],
    cooccurrence: Mapping[Tuple[str, str], int],
    job_to_index: Mapping[str, int],
    skill_to_index: Mapping[str, int],
) -> Tuple[Dict[str, Dict[str, list[str]]], Dict[str, int]]:
    candidate_vocabulary = sorted(frequency)
    max_k = max(K_VALUES)
    if any(len(candidate_sets[job]) != len(set(candidate_sets[job])) for job in candidate_sets):
        raise RuntimeError("Candidate sets must contain unique skills.")
    if any(not set(candidate_sets[job]).issubset(candidate_vocabulary) for job in candidate_sets):
        raise RuntimeError("Candidate sets must use the training-visible skill vocabulary.")

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(graph.edge_index_dict)
        rankings: Dict[str, Dict[str, list[str]]] = {method: {} for method in METHODS}
        for job_id in sorted(targets):
            candidates = tuple(candidate_sets[job_id])
            job_index = job_to_index[job_id]
            candidate_indices = [skill_to_index[skill] for skill in candidates]
            edge_index = torch.tensor(
                [[job_index] * len(candidates), candidate_indices],
                dtype=torch.long,
            ).reshape(2, -1)
            scores = model.score_pairs(embeddings, edge_index).cpu().numpy().tolist()
            score_by_skill = dict(zip(candidates, scores))
            ranked = rank_all_methods_for_job(
                candidates,
                training_context.get(job_id, set()),
                frequency,
                cooccurrence,
                score_by_skill,
                max_k,
            )
            for method in METHODS:
                rankings[method][job_id] = ranked[method]
    candidate_counts = {job_id: len(skills) for job_id, skills in candidate_sets.items()}
    return rankings, candidate_counts


def _encode_edges(
    edges: Iterable[JobSkillEdge],
    job_to_index: Mapping[str, int],
    skill_to_index: Mapping[str, int],
) -> torch.Tensor:
    return encode_edge_index(edges, job_to_index, skill_to_index)


def _paired_csv_row(seed: int, job: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "seed": seed,
        "job_id": job["job_id"],
        "candidate_count": job["candidate_count"],
        "target_count": job["target_count"],
    }
    for method in METHODS:
        for k in K_VALUES:
            for metric in METRICS:
                row[f"{method}_{metric}@{k}"] = job["metrics"][method][str(k)][metric]
    for baseline in BASELINES:
        for k in K_VALUES:
            for metric in METRICS:
                row[f"graphsage_minus_{baseline}_{metric}@{k}"] = job["paired_differences"][baseline][str(k)][metric]
    return row


def _validate_output_paths() -> None:
    existing = [path for path in (SUMMARY_PATH, SPLITS_PATH, PAIRED_ROWS_PATH) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite paired benchmark outputs: "
            + ", ".join(str(path) for path in existing)
        )


def run_paired_benchmark() -> Dict[str, Any]:
    """Train/evaluate the five predeclared seeds and write new output files."""
    if len(SEEDS) != 5 or len(set(SEEDS)) != 5:
        raise RuntimeError("SEEDS must contain five unique, predeclared values.")
    _validate_output_paths()

    full_mapping, _, all_positive_edges, _ = load_graph_edges()
    job_ids = list(full_mapping["job_ids_by_index"])
    skill_names = list(full_mapping["skill_names_by_index"])
    job_to_index = {job_id: index for index, job_id in enumerate(job_ids)}
    skill_to_index = {skill: index for index, skill in enumerate(skill_names)}
    all_known_positive_edges = {
        (job_to_index[job], skill_to_index[skill])
        for job, skill in all_positive_edges
    }

    output_arrays: Dict[str, np.ndarray] = {}
    per_seed: list[Dict[str, Any]] = []
    all_paired_rows: list[Dict[str, Any]] = []

    for seed in SEEDS:
        set_deterministic_seed(seed)
        splits = split_edges_per_job(all_positive_edges, random_state=seed)
        training_edges = splits["training"]
        validation_edges = splits["validation"]
        test_edges = splits["test"]
        train_positive = _encode_edges(training_edges, job_to_index, skill_to_index)
        validation_positive = _encode_edges(validation_edges, job_to_index, skill_to_index)
        test_positive = _encode_edges(test_edges, job_to_index, skill_to_index)

        negative_seed = NEGATIVE_SEED_BASE + seed * 10
        training_negative = sample_negative_edges(
            len(training_edges), len(job_ids), len(skill_names),
            all_known_positive_edges, negative_seed,
        )
        training_negative_pairs = {tuple(map(int, pair)) for pair in training_negative.T}
        validation_negative = sample_negative_edges(
            len(validation_edges), len(job_ids), len(skill_names),
            all_known_positive_edges, negative_seed + 1,
            excluded_negative_edges=training_negative_pairs,
        )
        validation_negative_pairs = {tuple(map(int, pair)) for pair in validation_negative.T}
        test_negative = sample_negative_edges(
            len(test_edges), len(job_ids), len(skill_names),
            all_known_positive_edges, negative_seed + 2,
            excluded_negative_edges=training_negative_pairs | validation_negative_pairs,
        )

        message_passing = set(training_edges)
        if message_passing & set(validation_edges) or message_passing & set(test_edges):
            raise RuntimeError("Validation or test positives entered message passing.")
        graph = build_training_heterodata(len(job_ids), len(skill_names), train_positive)
        model = HeterogeneousGraphSAGE(len(job_ids), len(skill_names), HIDDEN_DIM, DROPOUT)
        model, validation_metrics = train_with_validation(
            model,
            graph,
            train_positive,
            training_negative,
            validation_positive,
            validation_negative,
        )

        candidate_sets, targets, training_context, frequency, cooccurrence = build_filtered_candidate_sets(
            training_edges,
            all_positive_edges,
            test_edges,
        )
        rankings, candidate_counts = _rank_seed(
            model,
            graph,
            candidate_sets,
            targets,
            training_context,
            frequency,
            cooccurrence,
            job_to_index,
            skill_to_index,
        )
        macro_metrics, paired_jobs, macro_differences = evaluate_paired_rankings(
            rankings,
            targets,
            candidate_sets,
        )

        model.eval()
        with torch.no_grad():
            embeddings = model.encode(graph.edge_index_dict)
            test_positive_scores = model.score_pairs(embeddings, test_positive).cpu().numpy()
            test_negative_scores = model.score_pairs(embeddings, test_negative).cpu().numpy()
        sampled_negative_metrics = link_prediction_metrics(test_positive_scores, test_negative_scores)

        one_skill_jobs = sum(
            len(skills) == 1
            for skills in _skills_by_job(all_positive_edges).values()
        )
        split_metadata = {
            "seed": seed,
            "method": "Per-job random edge split; each job with at least two edges retains at least one training edge.",
            "edge_holdout_fraction_per_job": HOLDOUT_FRACTION,
            "validation_fraction_of_heldout_edges": VALIDATION_FRACTION_OF_HOLDOUT,
            "training_positive_edges": len(training_edges),
            "validation_positive_edges": len(validation_edges),
            "test_positive_edges": len(test_edges),
            "training_message_passing_edges": int(train_positive.shape[1]),
            "validation_test_edges_in_message_passing": 0,
            "training_jobs": len({job for job, _ in training_edges}),
            "validation_jobs": len({job for job, _ in validation_edges}),
            "test_jobs": len(targets),
            "jobs_with_one_skill_training_only": one_skill_jobs,
            "candidate_vocabulary_size": len(frequency),
            "candidate_count_min": min(candidate_counts.values()),
            "candidate_count_max": max(candidate_counts.values()),
            "candidate_set_parity": "The same filtered per-job candidate sequence is passed to frequency, co-occurrence, and GraphSAGE.",
        }
        per_seed.append({
            "seed": seed,
            "split": split_metadata,
            "negative_sampling": {
                "ratio": "1:1 negatives to positives in train, validation, and test.",
                "seed_base": negative_seed,
                "known_positive_pairs_excluded": len(all_known_positive_edges),
                "negative_pools_disjoint": True,
                "unobserved_pair_caveat": "An unobserved job-skill pair may be a true but undiscovered relationship.",
            },
            "validation_checkpoint_selection": "Highest validation average precision; ROC-AUC is the deterministic tie-breaker.",
            "validation_metrics": validation_metrics,
            "sampled_negative_test_metrics": sampled_negative_metrics,
            "ranking_metrics": macro_metrics,
            "paired_macro_differences": macro_differences,
        })
        all_paired_rows.extend(_paired_csv_row(seed, job) for job in paired_jobs)

        prefix = f"seed_{seed}_"
        output_arrays[prefix + "training_positive"] = train_positive.cpu().numpy()
        output_arrays[prefix + "validation_positive"] = validation_positive.cpu().numpy()
        output_arrays[prefix + "test_positive"] = test_positive.cpu().numpy()
        output_arrays[prefix + "training_negative"] = training_negative.cpu().numpy()
        output_arrays[prefix + "validation_negative"] = validation_negative.cpu().numpy()
        output_arrays[prefix + "test_negative"] = test_negative.cpu().numpy()
        output_arrays[prefix + "message_passing"] = train_positive.cpu().numpy()

        del model, graph, embeddings

    aggregate: Dict[str, Any] = {"ranking_metrics": {}, "paired_macro_differences": {}, "sampled_negative_test_metrics": {}}
    for method in METHODS:
        aggregate["ranking_metrics"][method] = {
            str(k): {
                metric: confidence_interval([
                    run["ranking_metrics"][method][str(k)][metric] for run in per_seed
                ])
                for metric in METRICS
            }
            for k in K_VALUES
        }
    for baseline in BASELINES:
        aggregate["paired_macro_differences"][baseline] = {
            str(k): {
                metric: confidence_interval([
                    run["paired_macro_differences"][baseline][str(k)][metric]
                    for run in per_seed
                ])
                for metric in METRICS
            }
            for k in K_VALUES
        }
    for metric in ("roc_auc", "average_precision"):
        aggregate["sampled_negative_test_metrics"][metric] = confidence_interval([
            run["sampled_negative_test_metrics"][metric] for run in per_seed
        ])

    summary = {
        "experiment": "paired five-seed graph recommendation benchmark",
        "seeds": list(SEEDS),
        "data_source": "Existing heterogeneous graph and mappings; no artifacts regenerated.",
        "protocol": {
            "task": "Transductive held-out job-skill edge completion; not cold-start or temporal prediction.",
            "split": "Per-job seeded edge holdout; validation and test held-out edges never enter message-passing adjacency or baseline statistics.",
            "baselines": "Frequency counts and unweighted skill co-occurrence are computed from training edges only.",
            "candidate_policy": "Training-visible skills only; exclude each job's training context and known non-test positives; all methods share the same filtered candidate sequence.",
            "ranking_metrics": "Macro-averaged per test job; Precision@K divides by K, Recall@K by all held-out target skills, and binary NDCG@K uses the ideal ranking.",
            "sampled_negative_metrics": "GraphSAGE ROC-AUC/AP are secondary 1:1 sampled-negative link-classification metrics, separate from full-candidate ranking results.",
            "uncertainty": "Two-sided Student-t 95% interval across five seed-level macro scores (df=4, t*=2.776445); small n means intervals are approximate and describe seed/split variability.",
            "limitations": [
                "Transductive: test jobs contribute their training edges to the message-passing graph; this is not unseen-job generalization.",
                "Skills absent from training are not candidates; held-out targets outside that vocabulary remain in the recall denominator.",
                "Unobserved pairs can be undiscovered positives and are treated as negatives for sampled-negative metrics.",
                "Per-job edge splits do not protect against semantically near-duplicate postings with different job IDs crossing partitions.",
                "No temporal split or temporal prediction claim is made.",
            ],
        },
        "model_configuration": {
            "architecture": "two-layer heterogeneous GraphSAGE with trainable job/skill ID embeddings",
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "device": "cpu",
            "message_passing_edges": "training positive edges only",
        },
        "aggregate": aggregate,
        "per_seed": per_seed,
        "per_job_differences_csv": str(PAIRED_ROWS_PATH.relative_to(ROOT)),
        "split_arrays_npz": str(SPLITS_PATH.relative_to(ROOT)),
        "mapping_reference": "data/processed/heterogeneous_skill_graph_mappings.json",
    }

    _validate_output_paths()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SPLITS_PATH, **output_arrays)
    with PAIRED_ROWS_PATH.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(all_paired_rows[0]) if all_paired_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_paired_rows)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _skills_by_job(edges: Iterable[JobSkillEdge]) -> Dict[str, set[str]]:
    by_job: Dict[str, set[str]] = defaultdict(set)
    for job_id, skill in edges:
        by_job[str(job_id)].add(str(skill))
    return by_job


if __name__ == "__main__":
    result = run_paired_benchmark()
    print(json.dumps({
        "summary": str(SUMMARY_PATH),
        "splits": str(SPLITS_PATH),
        "paired_per_job_differences": str(PAIRED_ROWS_PATH),
        "seeds": result["seeds"],
    }, indent=2))