import numpy as np
import pandas as pd
import pytest

from app.heterogeneous_skill_graph import (
    DATA_FILE,
    PROJECT_ROOT,
    _training_skill_statistics,
    build_heterogeneous_graph,
    dataset_to_records,
    edge_index_from_edges,
    evaluate_recommendation_baselines,
    ranking_metrics,
    recommend_cooccurrence,
    recommend_frequency,
    split_job_skill_edges,
)


def test_graph_mappings_edges_and_metadata_are_deterministic():
    jobs = {
        "job-b": {"job_title": "Analyst", "category": "finance"},
        "job-a": {"job_title": "Engineer", "category": "technology"},
    }
    edges = iter([("job-b", "sql"), ("job-a", "python"), ("job-a", "sql"), ("job-a", "python")])

    first_arrays, first_mapping = build_heterogeneous_graph(jobs, edges)
    second_arrays, second_mapping = build_heterogeneous_graph(jobs, [
        ("job-a", "sql"), ("job-b", "sql"), ("job-a", "python")
    ])

    assert first_mapping == second_mapping
    np.testing.assert_array_equal(first_arrays["job_to_skill_edge_index"], second_arrays["job_to_skill_edge_index"])
    assert first_mapping["job_ids_by_index"] == ["job-a", "job-b"]
    assert first_mapping["skill_names_by_index"] == ["python", "sql"]
    assert first_arrays["job_to_skill_edge_index"].shape == (2, 3)
    np.testing.assert_array_equal(
        first_arrays["skill_to_job_edge_index"],
        first_arrays["job_to_skill_edge_index"][::-1],
    )
    reconstructed = {
        (
            first_mapping["job_ids_by_index"][int(job_index)],
            first_mapping["skill_names_by_index"][int(skill_index)],
        )
        for job_index, skill_index in first_arrays["job_to_skill_edge_index"].T
    }
    assert reconstructed == {("job-a", "python"), ("job-a", "sql"), ("job-b", "sql")}
    assert "category" not in first_arrays
    assert first_mapping["job_metadata_by_index"][0]["category"] == "technology"


def test_graph_builder_rejects_edges_with_unknown_jobs():
    with pytest.raises(ValueError, match="unknown jobs"):
        build_heterogeneous_graph({}, [("missing", "python")])


def test_production_graph_edges_match_processed_dataset_and_edge_csv():
    jobs, edges = dataset_to_records(DATA_FILE)
    arrays, mapping = build_heterogeneous_graph(jobs, edges)
    edge_csv = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "job_skill_edges.csv").fillna("")
    source_edges = {
        (str(row.job_id).strip(), str(row.skill).strip().lower())
        for row in edge_csv.itertuples(index=False)
    }

    assert set(edges) == source_edges
    assert len(edges) == len(set(edges))
    assert arrays["job_to_skill_edge_index"].shape[1] == len(source_edges)
    assert len(mapping["job_ids_by_index"]) == len(jobs)
    assert len(mapping["skill_names_by_index"]) == edge_csv["skill"].nunique()
    assert all(
        0 <= int(job_index) < len(jobs)
        and 0 <= int(skill_index) < len(mapping["skill_names_by_index"])
        for job_index, skill_index in arrays["job_to_skill_edge_index"].T
    )
    reconstructed = {
        (
            mapping["job_ids_by_index"][int(job_index)],
            mapping["skill_names_by_index"][int(skill_index)],
        )
        for job_index, skill_index in arrays["job_to_skill_edge_index"].T
    }
    assert reconstructed == set(edges)


def test_edge_split_is_reproducible_and_has_no_edge_or_job_leakage():
    edges = [
        ("j1", "python"), ("j1", "sql"), ("j1", "linux"),
        ("j2", "python"), ("j2", "excel"), ("j2", "finance"),
        ("j3", "sql"), ("j3", "excel"),
        ("j4", "singleton"),
    ]
    train_a, heldout_a = split_job_skill_edges(edges, random_state=17)
    train_b, heldout_b = split_job_skill_edges(edges, random_state=17)

    assert (train_a, heldout_a) == (train_b, heldout_b)
    assert set(train_a).isdisjoint(set(heldout_a))
    job_mapping = ["j1", "j2", "j3", "j4"]
    skill_mapping = sorted({skill for _, skill in edges})
    train_index = edge_index_from_edges(train_a, job_mapping, skill_mapping)
    reconstructed_training = {
        (job_mapping[int(job_index)], skill_mapping[int(skill_index)])
        for job_index, skill_index in train_index.T
    }
    assert reconstructed_training == set(train_a)
    assert reconstructed_training.isdisjoint(set(heldout_a))
    train_by_job = {}
    for job_id, skill in train_a:
        train_by_job.setdefault(job_id, set()).add(skill)
    for job_id, _ in heldout_a:
        assert train_by_job[job_id]
    assert ("j4", "singleton") in train_a
    assert not any(job_id == "j4" for job_id, _ in heldout_a)


def test_heldout_skills_are_absent_from_training_statistics():
    edges = [
        ("j1", "python"), ("j1", "sql"), ("j1", "kubernetes"),
        ("j2", "python"), ("j2", "sql"), ("j2", "git"),
        ("j3", "python"), ("j3", "sql"), ("j3", "docker"),
    ]
    train_edges, heldout_edges = split_job_skill_edges(edges, random_state=9)
    _, frequencies, cooccurrence = _training_skill_statistics(train_edges)

    assert set(train_edges).isdisjoint(set(heldout_edges))
    assert all(skill in frequencies for _, skill in train_edges)
    assert all(set(pair).issubset(frequencies) for pair in cooccurrence)
    heldout_only = {skill for _, skill in heldout_edges} - set(frequencies)
    assert all(skill not in frequencies for skill in heldout_only)

    unique_skill_edges = [("only-job", "alpha"), ("only-job", "beta"), ("only-job", "gamma")]
    isolated_train, isolated_test = split_job_skill_edges(unique_skill_edges, random_state=13)
    _, isolated_frequency, isolated_cooccurrence = _training_skill_statistics(isolated_train)
    hidden_skills = {skill for _, skill in isolated_test}
    assert hidden_skills.isdisjoint(isolated_frequency)
    assert all(hidden_skills.isdisjoint(pair) for pair in isolated_cooccurrence)
    assert set(isolated_train).isdisjoint(isolated_test)

    result = evaluate_recommendation_baselines(edges, random_state=9, k_values=(5, 10))
    assert result["split_strategy"]["training_graph_edges"] == len(train_edges)
    assert result["split_strategy"]["heldout_target_edges"] == len(heldout_edges)


def test_cooccurrence_recommendation_excludes_known_skills():
    recommendations = recommend_cooccurrence(
        context=["python", "sql"],
        candidate_skills=["python", "sql", "docker", "pandas"],
        cooccurrence={("docker", "python"): 4, ("pandas", "sql"): 2},
        limit=4,
    )
    assert recommendations == ["docker", "pandas"]
    assert not set(recommendations).intersection({"python", "sql"})


def test_no_eligible_recommendations_returns_empty_list():
    assert recommend_frequency(["python"], {"python": 3}, limit=5) == []
    assert recommend_cooccurrence(["python"], ["python"], {}, limit=5) == []


def test_ranking_metrics_calculate_precision_recall_and_ndcg():
    metrics = ranking_metrics(["wrong", "relevant-a", "relevant-b"], {"relevant-a", "relevant-b"}, 2)

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["ndcg"] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))


def test_empty_recommendations_have_zero_ranking_scores():
    metrics = ranking_metrics([], {"unseen-skill"}, 5)

    assert metrics == {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}


def test_baselines_use_same_cases_and_return_all_ranking_metrics():
    edges = [
        ("j1", "python"), ("j1", "sql"), ("j1", "docker"),
        ("j2", "python"), ("j2", "sql"), ("j2", "git"),
        ("j3", "sql"), ("j3", "docker"), ("j3", "linux"),
        ("j4", "excel"), ("j4", "finance"), ("j4", "reporting"),
    ]
    result = evaluate_recommendation_baselines(edges, random_state=3, k_values=(1, 2))

    assert result["split_strategy"]["evaluation_jobs"] == 4
    assert set(result["metrics"]) == {"frequency", "cooccurrence"}
    for baseline in result["metrics"].values():
        for metrics in baseline.values():
            assert set(metrics) == {"precision", "recall", "ndcg"}
            assert all(0.0 <= score <= 1.0 for score in metrics.values())
    assert any("not a trained GNN" in limitation for limitation in result["limitations"])
