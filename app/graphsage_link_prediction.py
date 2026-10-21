from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

JobSkillEdge = Tuple[str, str]

EdgeIndex = torch.Tensor
EdgeType = Tuple[str, str, str]
JOB_TO_SKILL: EdgeType = ("job", "requires", "skill")
SKILL_TO_JOB: EdgeType = ("skill", "required_by", "job")


def split_link_prediction_edges(
    all_positive_edges: Iterable[JobSkillEdge],
    training_edges: Iterable[JobSkillEdge],
    random_state: int = 43,
) -> Tuple[List[JobSkillEdge], List[JobSkillEdge], List[JobSkillEdge]]:
    """Reuse the Phase 3.1 training graph and split its complement into validation/test."""
    all_edges = set((str(job), str(skill)) for job, skill in all_positive_edges)
    train_edges = set((str(job), str(skill)) for job, skill in training_edges)
    if not train_edges or not train_edges.issubset(all_edges):
        raise ValueError("Training graph edges must be a non-empty subset of known positive edges.")

    heldout = sorted(all_edges - train_edges)
    if len(heldout) < 2:
        raise ValueError("At least two held-out positive edges are required for validation and test.")
    random.Random(random_state).shuffle(heldout)
    validation_count = len(heldout) // 2
    validation_edges = sorted(heldout[:validation_count])
    test_edges = sorted(heldout[validation_count:])
    return sorted(train_edges), validation_edges, test_edges


def decode_edge_index(
    edge_index: np.ndarray,
    job_ids_by_index: Sequence[str],
    skill_names_by_index: Sequence[str],
) -> List[JobSkillEdge]:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges).")
    return [
        (job_ids_by_index[int(job_index)], skill_names_by_index[int(skill_index)])
        for job_index, skill_index in edge_index.T
    ]


def encode_edge_index(
    edges: Iterable[JobSkillEdge],
    job_to_index: Mapping[str, int],
    skill_to_index: Mapping[str, int],
) -> EdgeIndex:
    pairs = sorted(set((str(job), str(skill)) for job, skill in edges))
    missing_jobs = {job for job, _ in pairs} - set(job_to_index)
    missing_skills = {skill for _, skill in pairs} - set(skill_to_index)
    if missing_jobs or missing_skills:
        raise ValueError(
            f"Unknown graph IDs: jobs={sorted(missing_jobs)[:5]}, skills={sorted(missing_skills)[:5]}"
        )
    return torch.tensor(
        [[job_to_index[job] for job, _ in pairs], [skill_to_index[skill] for _, skill in pairs]],
        dtype=torch.long,
    ).reshape(2, -1)


def build_training_heterodata(
    num_jobs: int,
    num_skills: int,
    training_edge_index: EdgeIndex,
) -> HeteroData:
    if training_edge_index.ndim != 2 or training_edge_index.shape[0] != 2:
        raise ValueError("training_edge_index must have shape (2, num_edges).")
    if training_edge_index.numel():
        if int(training_edge_index[0].min()) < 0 or int(training_edge_index[0].max()) >= num_jobs:
            raise ValueError("Training adjacency contains an out-of-range job index.")
        if int(training_edge_index[1].min()) < 0 or int(training_edge_index[1].max()) >= num_skills:
            raise ValueError("Training adjacency contains an out-of-range skill index.")

    graph = HeteroData()
    graph["job"].num_nodes = num_jobs
    graph["skill"].num_nodes = num_skills
    graph[JOB_TO_SKILL].edge_index = training_edge_index.clone()
    graph[SKILL_TO_JOB].edge_index = training_edge_index.flip(0).contiguous()
    return graph


def sample_negative_edges(
    count: int,
    num_jobs: int,
    num_skills: int,
    all_known_positive_edges: Iterable[Tuple[int, int]],
    random_state: int,
    excluded_negative_edges: Iterable[Tuple[int, int]] = (),
) -> EdgeIndex:
    """Sample unique unknown job-skill pairs, excluding positives from every split."""
    if count < 0 or num_jobs < 1 or num_skills < 1:
        raise ValueError("count must be non-negative and both node counts must be positive.")
    known_positive = set((int(job), int(skill)) for job, skill in all_known_positive_edges)
    excluded = known_positive | set((int(job), int(skill)) for job, skill in excluded_negative_edges)
    possible_negatives = num_jobs * num_skills - len(excluded)
    if count > possible_negatives:
        raise ValueError("Requested more unique negatives than unknown job-skill pairs.")
    rng = random.Random(random_state)
    negatives = set()
    max_attempts = max(1000, count * 30)
    attempts = 0
    while len(negatives) < count and attempts < max_attempts:
        pair = (rng.randrange(num_jobs), rng.randrange(num_skills))
        if pair not in excluded:
            negatives.add(pair)
        attempts += 1

    if len(negatives) < count:
        for job_index in range(num_jobs):
            for skill_index in range(num_skills):
                pair = (job_index, skill_index)
                if pair not in excluded:
                    negatives.add(pair)
                    if len(negatives) == count:
                        break
            if len(negatives) == count:
                break
    ordered = sorted(negatives)
    if len(ordered) != count:
        raise RuntimeError("Negative sampler could not produce the requested number of pairs.")
    return torch.tensor(ordered, dtype=torch.long).T.contiguous().reshape(2, -1)


class HeterogeneousGraphSAGE(nn.Module):
    """Two-layer heterogeneous GraphSAGE with learned per-node ID features."""

    def __init__(
        self,
        num_jobs: int,
        num_skills: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_jobs < 1 or num_skills < 1 or hidden_dim < 2:
            raise ValueError("Node counts must be positive and hidden_dim must be at least 2.")
        self.num_jobs = num_jobs
        self.num_skills = num_skills
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.job_embedding = nn.Embedding(num_jobs, hidden_dim)
        self.skill_embedding = nn.Embedding(num_skills, hidden_dim)
        self.conv1 = self._make_convolution(hidden_dim)
        self.conv2 = self._make_convolution(hidden_dim)
        self.edge_weight = nn.Parameter(torch.ones(hidden_dim))
        self.edge_bias = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _make_convolution(hidden_dim: int) -> HeteroConv:
        return HeteroConv(
            {
                JOB_TO_SKILL: SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                SKILL_TO_JOB: SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            },
            aggr="sum",
        )

    def encode(self, edge_index_dict: Mapping[EdgeType, EdgeIndex]) -> Dict[str, torch.Tensor]:
        device = self.job_embedding.weight.device
        x_dict = {
            "job": self.job_embedding(torch.arange(self.num_jobs, device=device)),
            "skill": self.skill_embedding(torch.arange(self.num_skills, device=device)),
        }
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {node_type: F.dropout(F.relu(values), p=self.dropout, training=self.training) for node_type, values in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return {node_type: F.relu(values) for node_type, values in x_dict.items()}

    def score_pairs(
        self,
        embeddings: Mapping[str, torch.Tensor],
        edge_index: EdgeIndex,
    ) -> torch.Tensor:
        job_vectors = embeddings["job"][edge_index[0]]
        skill_vectors = embeddings["skill"][edge_index[1]]
        return (job_vectors * skill_vectors * self.edge_weight).sum(dim=-1) / self.hidden_dim**0.5 + self.edge_bias

    def forward(self, graph: HeteroData, edge_index: EdgeIndex) -> torch.Tensor:
        embeddings = self.encode(graph.edge_index_dict)
        return self.score_pairs(embeddings, edge_index)


def link_prediction_metrics(
    positive_scores: Sequence[float] | np.ndarray,
    negative_scores: Sequence[float] | np.ndarray,
) -> Dict[str, float]:
    positive = np.asarray(positive_scores, dtype=float).reshape(-1)
    negative = np.asarray(negative_scores, dtype=float).reshape(-1)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("Both positive and negative scores are required.")
    scores = np.concatenate([positive, negative])
    positive_flags = np.concatenate([np.ones(positive.size, dtype=bool), np.zeros(negative.size, dtype=bool)])
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_positive = positive_flags[order]

    ranks = np.empty(scores.size, dtype=float)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2
        start = end
    positive_rank_sum = ranks[sorted_positive].sum()
    roc_auc = (positive_rank_sum - positive.size * (positive.size + 1) / 2) / (positive.size * negative.size)

    descending_order = np.argsort(-scores, kind="mergesort")
    descending_positive = positive_flags[descending_order].astype(float)
    cumulative_positive = np.cumsum(descending_positive)
    positive_ranks = np.flatnonzero(descending_positive)
    average_precision = float(np.mean(cumulative_positive[positive_ranks] / (positive_ranks + 1)))
    return {
        "roc_auc": float(roc_auc),
        "average_precision": average_precision,
    }


def ranking_metrics(recommendations: Sequence[str], relevant: Iterable[str], k: int) -> Dict[str, float]:
    if k < 1:
        raise ValueError("k must be positive.")
    relevant_set = set(relevant)
    if not relevant_set:
        return {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}
    hits = [1 if skill in relevant_set else 0 for skill in recommendations[:k]]
    hit_count = sum(hits)
    dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1 / np.log2(rank + 2) for rank in range(ideal_hits))
    return {
        "precision": float(hit_count / k),
        "recall": float(hit_count / len(relevant_set)),
        "ndcg": float(dcg / idcg) if idcg else 0.0,
    }


def checkpoint_config(
    num_jobs: int,
    num_skills: int,
    hidden_dim: int,
    dropout: float,
) -> Dict[str, Any]:
    return {
        "architecture": "two-layer heterogeneous GraphSAGE",
        "num_jobs": num_jobs,
        "num_skills": num_skills,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "edge_scorer": "learned diagonal bilinear score",
        "initial_node_features": "trainable job/skill ID embeddings; no category/title inputs",
        "edge_types": [list(JOB_TO_SKILL), list(SKILL_TO_JOB)],
    }


def save_checkpoint(
    path: Path,
    model: HeterogeneousGraphSAGE,
    config: Mapping[str, Any],
    training_metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": dict(config),
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "training_metadata": dict(training_metadata),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    map_location: str | torch.device = "cpu",
) -> Tuple[HeterogeneousGraphSAGE, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    config = checkpoint["model_config"]
    model = HeterogeneousGraphSAGE(
        num_jobs=int(config["num_jobs"]),
        num_skills=int(config["num_skills"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(map_location)
    model.eval()
    return model, checkpoint
