from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "cleaned_jobs.csv"
HOLDOUT_FRACTION = 0.2
HOLDOUT_SEED = 42
RANKING_K = (5, 10, 20)

JobSkillEdge = Tuple[str, str]


def _normalize_skill_list(value: Any) -> List[str]:
    if value is None or pd.isna(value):
        return []
    return sorted({skill.strip().lower() for skill in str(value).split("|") if skill.strip()})


def dataset_to_records(data_file: Path = DATA_FILE) -> Tuple[Dict[str, Dict[str, str]], List[JobSkillEdge]]:
    """Read cleaned jobs into deterministic metadata and deduplicated job-skill edges."""
    df = pd.read_csv(data_file).fillna("")
    required = {"job_id", "category", "job_title", "skills"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    jobs: Dict[str, Dict[str, str]] = {}
    edge_set = set()
    for row in df.itertuples(index=False):
        job_id = str(getattr(row, "job_id")).strip()
        if not job_id:
            continue
        metadata = {
            "job_title": str(getattr(row, "job_title")).strip(),
            "category": str(getattr(row, "category")).strip(),
        }
        if job_id in jobs and jobs[job_id] != metadata:
            raise ValueError(f"Conflicting metadata found for job_id {job_id!r}.")
        jobs[job_id] = metadata
        for skill in _normalize_skill_list(getattr(row, "skills")):
            edge_set.add((job_id, skill))

    return dict(sorted(jobs.items())), sorted(edge_set)


def build_heterogeneous_graph(
    jobs: Mapping[str, Mapping[str, str]],
    edges: Iterable[JobSkillEdge],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Build portable typed edge-index arrays and stable original-ID mappings."""
    unique_edges = sorted({(str(job_id), str(skill)) for job_id, skill in edges})
    job_ids = sorted(str(job_id) for job_id in jobs)
    skill_names = sorted({skill for _, skill in unique_edges})
    job_to_index = {job_id: index for index, job_id in enumerate(job_ids)}
    skill_to_index = {skill: index for index, skill in enumerate(skill_names)}

    invalid_jobs = sorted({job_id for job_id, _ in unique_edges} - set(job_to_index))
    if invalid_jobs:
        raise ValueError(f"Edges reference unknown jobs: {invalid_jobs[:5]}")

    job_to_skill = np.asarray(
        [[job_to_index[job_id] for job_id, _ in unique_edges],
         [skill_to_index[skill] for _, skill in unique_edges]],
        dtype=np.int64,
    ).reshape(2, -1)
    skill_to_job = job_to_skill[::-1].copy()
    arrays = {
        "job_to_skill_edge_index": job_to_skill,
        "skill_to_job_edge_index": skill_to_job,
        "num_job_nodes": np.asarray([len(job_ids)], dtype=np.int64),
        "num_skill_nodes": np.asarray([len(skill_names)], dtype=np.int64),
    }
    mappings = {
        "schema_version": 1,
        "graph_schema": "heterogeneous job-skill graph",
        "edge_types": [
            {"source_type": "job", "relation": "requires", "target_type": "skill"},
            {"source_type": "skill", "relation": "required_by", "target_type": "job"},
        ],
        "node_features": "No predictive node features are stored. Job category and title are descriptive metadata only.",
        "job_ids_by_index": job_ids,
        "skill_names_by_index": skill_names,
        "job_metadata_by_index": [
            {
                "job_id": job_id,
                "job_title": str(jobs[job_id].get("job_title", "")),
                "category": str(jobs[job_id].get("category", "")),
            }
            for job_id in job_ids
        ],
        "edge_count": len(unique_edges),
    }
    return arrays, mappings


def edge_index_from_edges(
    edges: Iterable[JobSkillEdge],
    job_ids_by_index: Sequence[str],
    skill_names_by_index: Sequence[str],
) -> np.ndarray:
    """Encode a subset of known edges against a supplied deterministic mapping."""
    job_to_index = {job_id: index for index, job_id in enumerate(job_ids_by_index)}
    skill_to_index = {skill: index for index, skill in enumerate(skill_names_by_index)}
    unique_edges = sorted({(str(job_id), str(skill)) for job_id, skill in edges})
    try:
        return np.asarray(
            [[job_to_index[job_id] for job_id, _ in unique_edges],
             [skill_to_index[skill] for _, skill in unique_edges]],
            dtype=np.int64,
        ).reshape(2, -1)
    except KeyError as error:
        raise ValueError(f"Edge references an ID missing from the supplied mapping: {error.args[0]!r}") from error


def split_job_skill_edges(
    edges: Iterable[JobSkillEdge],
    test_fraction: float = HOLDOUT_FRACTION,
    random_state: int = HOLDOUT_SEED,
) -> Tuple[List[JobSkillEdge], List[JobSkillEdge]]:
    """Hold out edges per job, always retaining at least one observed skill per job."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    by_job: Dict[str, List[str]] = defaultdict(list)
    for job_id, skill in set((str(job_id), str(skill)) for job_id, skill in edges):
        by_job[job_id].append(skill)

    rng = random.Random(random_state)
    training_edges: List[JobSkillEdge] = []
    heldout_edges: List[JobSkillEdge] = []
    for job_id in sorted(by_job):
        skills = sorted(by_job[job_id])
        if len(skills) < 2:
            training_edges.extend((job_id, skill) for skill in skills)
            continue
        rng.shuffle(skills)
        holdout_count = min(len(skills) - 1, max(1, math.ceil(len(skills) * test_fraction)))
        heldout = set(skills[:holdout_count])
        training_edges.extend((job_id, skill) for skill in skills if skill not in heldout)
        heldout_edges.extend((job_id, skill) for skill in skills if skill in heldout)

    return sorted(training_edges), sorted(heldout_edges)


def _training_skill_statistics(
    training_edges: Iterable[JobSkillEdge],
) -> Tuple[Dict[str, set[str]], Counter[str], Counter[Tuple[str, str]]]:
    skills_by_job: Dict[str, set[str]] = defaultdict(set)
    frequency: Counter[str] = Counter()
    for job_id, skill in training_edges:
        skills_by_job[job_id].add(skill)
        frequency[skill] += 1

    cooccurrence: Counter[Tuple[str, str]] = Counter()
    for skills in skills_by_job.values():
        for first, second in combinations(sorted(skills), 2):
            cooccurrence[(first, second)] += 1
    return skills_by_job, frequency, cooccurrence


def _ordered_recommendations(
    context: Iterable[str],
    candidate_skills: Iterable[str],
    scores: Mapping[str, float],
    limit: int,
) -> List[str]:
    already_known = set(context)
    candidates = (skill for skill in candidate_skills if skill not in already_known)
    return sorted(candidates, key=lambda skill: (-scores.get(skill, 0.0), skill))[:limit]


def recommend_frequency(
    context: Iterable[str],
    frequency: Mapping[str, int],
    limit: int,
) -> List[str]:
    return _ordered_recommendations(context, frequency.keys(), frequency, limit)


def recommend_cooccurrence(
    context: Iterable[str],
    candidate_skills: Iterable[str],
    cooccurrence: Mapping[Tuple[str, str], int],
    limit: int,
) -> List[str]:
    candidate_list = tuple(candidate_skills)
    scores: Counter[str] = Counter()
    for known_skill in set(context):
        for candidate in candidate_list:
            if candidate == known_skill:
                continue
            pair = tuple(sorted((known_skill, candidate)))
            scores[candidate] += cooccurrence.get(pair, 0)
    return _ordered_recommendations(context, candidate_list, scores, limit)


def ranking_metrics(recommendations: Sequence[str], relevant: Iterable[str], k: int) -> Dict[str, float]:
    if k < 1:
        raise ValueError("k must be positive.")
    relevant_set = set(relevant)
    if not relevant_set:
        return {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}

    top_k = list(recommendations[:k])
    hits = [1 if skill in relevant_set else 0 for skill in top_k]
    hit_count = sum(hits)
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1 / math.log2(rank + 2) for rank in range(ideal_hits))
    return {
        "precision": hit_count / k,
        "recall": hit_count / len(relevant_set),
        "ndcg": dcg / idcg if idcg else 0.0,
    }


def evaluate_recommendation_baselines(
    edges: Iterable[JobSkillEdge],
    test_fraction: float = HOLDOUT_FRACTION,
    random_state: int = HOLDOUT_SEED,
    k_values: Sequence[int] = RANKING_K,
) -> Dict[str, Any]:
    unique_edges = sorted(set((str(job_id), str(skill)) for job_id, skill in edges))
    training_edges, heldout_edges = split_job_skill_edges(unique_edges, test_fraction, random_state)
    if not heldout_edges:
        raise ValueError("No held-out edges available; jobs need at least two skills.")

    skills_by_job, frequency, cooccurrence = _training_skill_statistics(training_edges)
    heldout_by_job: Dict[str, set[str]] = defaultdict(set)
    for job_id, skill in heldout_edges:
        heldout_by_job[job_id].add(skill)

    candidates = sorted(frequency)
    maximum_k = max(k_values)
    metric_sums = {
        name: {k: Counter() for k in k_values}
        for name in ("frequency", "cooccurrence")
    }
    evaluation_jobs = 0
    for job_id in sorted(heldout_by_job):
        context = skills_by_job.get(job_id, set())
        relevant = heldout_by_job[job_id]
        if not context:
            continue
        evaluation_jobs += 1
        ranked = {
            "frequency": recommend_frequency(context, frequency, maximum_k),
            "cooccurrence": recommend_cooccurrence(context, candidates, cooccurrence, maximum_k),
        }
        for name, recommendations in ranked.items():
            for k in k_values:
                values = ranking_metrics(recommendations, relevant, k)
                for metric, value in values.items():
                    metric_sums[name][k][metric] += value

    if not evaluation_jobs:
        raise ValueError("No eligible evaluation jobs have both observed and held-out skills.")

    metrics = {
        name: {
            str(k): {metric: metric_sums[name][k][metric] / evaluation_jobs for metric in ("precision", "recall", "ndcg")}
            for k in k_values
        }
        for name in metric_sums
    }
    return {
        "task": "held-out job-skill edge recommendation",
        "split_strategy": {
            "method": "Deterministic per-job edge holdout; each eligible job retains at least one observed skill.",
            "heldout_fraction": test_fraction,
            "random_state": random_state,
            "training_graph_edges": len(training_edges),
            "heldout_target_edges": len(heldout_edges),
            "evaluation_jobs": evaluation_jobs,
            "jobs_with_one_skill_retained_in_training_only": sum(len(skills) < 2 for skills in _skills_by_job(unique_edges).values()),
            "candidate_skill_vocabulary": "Skills observed in training edges only.",
            "leakage_control": "Held-out edges are omitted before frequency and co-occurrence statistics are computed.",
        },
        "k_values": list(k_values),
        "metrics": metrics,
        "limitations": [
            "Transductive edge completion: evaluation jobs contribute their non-held-out skills to the training graph.",
            "Skills absent from all training edges are outside the candidate vocabulary and cannot be recommended.",
            "The co-occurrence baseline uses unweighted counts from two-hop job-skill neighborhoods; it is not a trained GNN.",
        ],
    }


def _skills_by_job(edges: Iterable[JobSkillEdge]) -> Dict[str, set[str]]:
    by_job: Dict[str, set[str]] = defaultdict(set)
    for job_id, skill in edges:
        by_job[str(job_id)].add(str(skill))
    return by_job


def build_from_dataset(
    data_file: Path = DATA_FILE,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[JobSkillEdge]]:
    jobs, edges = dataset_to_records(data_file)
    arrays, mappings = build_heterogeneous_graph(jobs, edges)
    evaluation = evaluate_recommendation_baselines(edges)
    training_edges, _ = split_job_skill_edges(edges)
    mappings["graph_statistics"] = {
        "job_nodes": len(jobs),
        "skill_nodes": len(mappings["skill_names_by_index"]),
        "unique_skills": len(mappings["skill_names_by_index"]),
        "job_skill_edges": len(edges),
    }
    return arrays, mappings, evaluation, training_edges
