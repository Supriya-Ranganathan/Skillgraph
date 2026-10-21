import numpy as np
import pytest
import torch

from app.graphsage_link_prediction import (
    HeterogeneousGraphSAGE,
    build_training_heterodata,
    checkpoint_config,
    encode_edge_index,
    link_prediction_metrics,
    load_checkpoint,
    ranking_metrics,
    sample_negative_edges,
    save_checkpoint,
    split_link_prediction_edges,
)
from app.run_graphsage_link_prediction import ranking_candidate_skills


SMALL_POSITIVE_EDGES = [
    ("j1", "python"), ("j1", "sql"), ("j1", "docker"), ("j1", "git"),
    ("j2", "python"), ("j2", "sql"), ("j2", "linux"), ("j2", "bash"),
    ("j3", "excel"), ("j3", "finance"), ("j3", "reporting"), ("j3", "sql"),
    ("j4", "excel"), ("j4", "finance"), ("j4", "powerpoint"), ("j4", "analysis"),
]


def _small_split():
    training_edges = [edge for edge in SMALL_POSITIVE_EDGES if edge[1] in {"python", "sql", "excel", "finance"}]
    return split_link_prediction_edges(SMALL_POSITIVE_EDGES, training_edges, random_state=7)


def test_validation_and_test_edges_never_enter_message_passing_adjacency():
    train_edges, validation_edges, test_edges = _small_split()
    jobs = sorted({job for job, _ in SMALL_POSITIVE_EDGES})
    skills = sorted({skill for _, skill in SMALL_POSITIVE_EDGES})
    job_to_index = {job: index for index, job in enumerate(jobs)}
    skill_to_index = {skill: index for index, skill in enumerate(skills)}
    train_index = encode_edge_index(train_edges, job_to_index, skill_to_index)
    graph = build_training_heterodata(len(jobs), len(skills), train_index)
    adjacency = {
        (jobs[int(job_index)], skills[int(skill_index)])
        for job_index, skill_index in graph["job", "requires", "skill"].edge_index.T
    }

    assert adjacency == set(train_edges)
    assert adjacency.isdisjoint(validation_edges)
    assert adjacency.isdisjoint(test_edges)
    assert not set(train_edges) & set(validation_edges)
    assert not set(train_edges) & set(test_edges)
    assert not set(validation_edges) & set(test_edges)


def test_negative_sampling_excludes_all_known_positives_and_is_reproducible():
    known = {(0, 0), (0, 1), (1, 1), (2, 2)}
    first = sample_negative_edges(12, 4, 4, known, random_state=123)
    second = sample_negative_edges(12, 4, 4, known, random_state=123)
    first_pairs = [tuple(map(int, pair)) for pair in first.T]

    assert torch.equal(first, second)
    assert len(first_pairs) == len(set(first_pairs))
    assert set(first_pairs).isdisjoint(known)
    with pytest.raises(ValueError, match="more unique negatives"):
        sample_negative_edges(13, 2, 2, {(0, 0), (0, 1), (1, 0)}, random_state=1)


def test_negative_examples_do_not_repeat_across_splits():
    known = {(0, 0), (1, 1), (2, 2)}
    training = sample_negative_edges(4, 4, 4, known, random_state=11)
    training_pairs = {tuple(map(int, pair)) for pair in training.T}
    validation = sample_negative_edges(4, 4, 4, known, random_state=12, excluded_negative_edges=training_pairs)
    validation_pairs = {tuple(map(int, pair)) for pair in validation.T}
    test = sample_negative_edges(
        4, 4, 4, known, random_state=13,
        excluded_negative_edges=training_pairs | validation_pairs,
    )
    test_pairs = {tuple(map(int, pair)) for pair in test.T}

    assert training_pairs.isdisjoint(validation_pairs)
    assert training_pairs.isdisjoint(test_pairs)
    assert validation_pairs.isdisjoint(test_pairs)
    assert not (training_pairs | validation_pairs | test_pairs) & known


def test_split_is_deterministic_and_preserves_the_existing_training_edges():
    train_a, validation_a, test_a = _small_split()
    train_b, validation_b, test_b = _small_split()

    assert (train_a, validation_a, test_a) == (train_b, validation_b, test_b)
    assert train_a == sorted(edge for edge in SMALL_POSITIVE_EDGES if edge[1] in {"python", "sql", "excel", "finance"})
    assert set(train_a).isdisjoint(validation_a)
    assert set(train_a).isdisjoint(test_a)
    assert set(validation_a).isdisjoint(test_a)


def test_graphsage_output_shapes_and_deterministic_initialization():
    train_edges, _, _ = _small_split()
    jobs = sorted({job for job, _ in SMALL_POSITIVE_EDGES})
    skills = sorted({skill for _, skill in SMALL_POSITIVE_EDGES})
    train_index = encode_edge_index(train_edges, {v: i for i, v in enumerate(jobs)}, {v: i for i, v in enumerate(skills)})
    graph = build_training_heterodata(len(jobs), len(skills), train_index)
    torch.manual_seed(21)
    first_model = HeterogeneousGraphSAGE(len(jobs), len(skills), hidden_dim=8, dropout=0.0)
    torch.manual_seed(21)
    second_model = HeterogeneousGraphSAGE(len(jobs), len(skills), hidden_dim=8, dropout=0.0)
    first_model.eval()
    second_model.eval()

    first_embeddings = first_model.encode(graph.edge_index_dict)
    second_embeddings = second_model.encode(graph.edge_index_dict)
    edge_scores = first_model(graph, train_index[:, :3])

    assert first_embeddings["job"].shape == (len(jobs), 8)
    assert first_embeddings["skill"].shape == (len(skills), 8)
    assert edge_scores.shape == (3,)
    assert torch.equal(first_embeddings["job"], second_embeddings["job"])
    assert torch.equal(first_embeddings["skill"], second_embeddings["skill"])


def test_checkpoint_round_trip_preserves_link_scores(tmp_path):
    train_edges, _, _ = _small_split()
    jobs = sorted({job for job, _ in SMALL_POSITIVE_EDGES})
    skills = sorted({skill for _, skill in SMALL_POSITIVE_EDGES})
    train_index = encode_edge_index(train_edges, {v: i for i, v in enumerate(jobs)}, {v: i for i, v in enumerate(skills)})
    graph = build_training_heterodata(len(jobs), len(skills), train_index)
    model = HeterogeneousGraphSAGE(len(jobs), len(skills), hidden_dim=8, dropout=0.0)
    model.eval()
    expected = model(graph, train_index[:, :4]).detach()
    path = tmp_path / "graphsage.pt"
    config = checkpoint_config(len(jobs), len(skills), 8, 0.0)

    save_checkpoint(path, model, config, {"selected_by": "validation_average_precision"})
    loaded, checkpoint = load_checkpoint(path)
    actual = loaded(graph, train_index[:, :4]).detach()

    assert torch.equal(expected, actual)
    assert checkpoint["training_metadata"]["selected_by"] == "validation_average_precision"
    assert checkpoint["model_config"]["num_jobs"] == len(jobs)


def test_link_prediction_metrics_match_hand_calculated_examples():
    perfect = link_prediction_metrics([0.9, 0.8], [0.2, 0.1])
    mixed = link_prediction_metrics([0.8, 0.4], [0.5, 0.1])

    assert perfect == {"roc_auc": 1.0, "average_precision": 1.0}
    assert mixed["roc_auc"] == pytest.approx(0.75)
    assert mixed["average_precision"] == pytest.approx(5 / 6)
    with pytest.raises(ValueError, match="Both positive and negative"):
        link_prediction_metrics([0.2], [])


def test_ranking_metrics_and_no_candidate_case():
    metrics = ranking_metrics(["wrong", "target-a", "target-b"], {"target-a", "target-b"}, 2)
    empty = ranking_metrics([], {"target-a"}, 5)

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["ndcg"] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))
    assert empty == {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}


def test_test_ranking_excludes_other_known_positive_edges():
    candidates = ranking_candidate_skills(
        candidate_skills=["a", "b", "c", "d", "unobserved"],
        training_context={"a", "b"},
        all_known_job_skills={"a", "b", "c", "d"},
        test_targets={"d"},
    )

    assert candidates == ["d", "unobserved"]
