import pytest

from app.run_paired_graph_benchmark import (
    BASELINES,
    METHODS,
    build_filtered_candidate_sets,
    confidence_interval,
    evaluate_paired_rankings,
    rank_all_methods_for_job,
    split_edges_per_job,
)


def _split_fixture_edges():
    return [
        (f"job-{job}", f"skill-{skill}")
        for job in range(4)
        for skill in range(6)
    ]


def test_three_way_split_isolated_and_retains_training_context():
    edges = _split_fixture_edges() + [("job-0", "skill-0")]
    splits = split_edges_per_job(iter(edges), random_state=43)

    training = set(splits["training"])
    validation = set(splits["validation"])
    test = set(splits["test"])
    unique_edges = set(edges)

    assert training.isdisjoint(validation)
    assert training.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert training | validation | test == unique_edges

    training_jobs = {job_id for job_id, _ in training}
    heldout_jobs = {job_id for job_id, _ in validation | test}
    assert heldout_jobs.issubset(training_jobs)
    assert validation
    assert test


def test_three_way_split_is_reproducible_for_predeclared_seed():
    edges = _split_fixture_edges()

    first = split_edges_per_job(edges, random_state=47)
    second = split_edges_per_job(edges, random_state=47)

    assert first == second


def test_all_rankers_use_the_same_filtered_candidate_set():
    training = [
        ("job-a", "alpha"),
        ("job-a", "beta"),
        ("job-b", "alpha"),
        ("job-b", "gamma"),
        ("job-c", "beta"),
        ("job-c", "gamma"),
    ]
    all_edges = training + [
        ("job-a", "gamma"),
        ("job-a", "outside-vocabulary"),
        ("job-b", "beta"),
        ("job-c", "alpha"),
    ]
    test_edges = [
        ("job-a", "gamma"),
        ("job-b", "beta"),
        ("job-c", "alpha"),
    ]

    candidate_sets, targets, training_context, frequency, cooccurrence = build_filtered_candidate_sets(
        training,
        all_edges,
        test_edges,
    )

    assert candidate_sets == {
        "job-a": ("gamma",),
        "job-b": ("beta",),
        "job-c": ("alpha",),
    }
    assert targets == {
        "job-a": {"gamma"},
        "job-b": {"beta"},
        "job-c": {"alpha"},
    }

    rankings = rank_all_methods_for_job(
        candidates=candidate_sets["job-a"],
        training_context=training_context["job-a"],
        frequency=frequency,
        cooccurrence=cooccurrence,
        graphsage_scores={"gamma": 0.75},
        limit=5,
    )

    assert set(rankings) == set(METHODS)
    assert all(ranking == ["gamma"] for ranking in rankings.values())


def test_paired_metrics_aggregate_per_job_differences():
    rankings = {
        "frequency": {"job-a": ["alpha"], "job-b": ["alpha"]},
        "cooccurrence": {"job-a": ["beta"], "job-b": ["beta"]},
        "graphsage": {"job-a": ["alpha"], "job-b": ["beta"]},
    }
    targets = {"job-a": {"alpha"}, "job-b": {"beta"}}
    candidates = {"job-a": ("alpha", "beta"), "job-b": ("alpha", "beta")}

    macro, per_job, differences = evaluate_paired_rankings(
        rankings,
        targets,
        candidates,
        k_values=(1,),
    )

    assert macro["graphsage"]["1"]["precision"] == pytest.approx(1.0)
    assert macro["frequency"]["1"]["precision"] == pytest.approx(0.5)
    assert macro["cooccurrence"]["1"]["recall"] == pytest.approx(0.5)
    assert differences["frequency"]["1"]["precision"] == pytest.approx(0.5)
    assert differences["cooccurrence"]["1"]["precision"] == pytest.approx(0.5)
    assert per_job[0]["paired_differences"]["frequency"]["1"]["precision"] == pytest.approx(0.0)
    assert per_job[1]["paired_differences"]["frequency"]["1"]["precision"] == pytest.approx(1.0)
    assert set(differences) == set(BASELINES)


def test_confidence_interval_uses_seed_level_student_t_summary():
    interval = confidence_interval([1, 2, 3, 4, 5])

    assert interval["n_seeds"] == 5
    assert interval["mean"] == pytest.approx(3.0)
    assert interval["ci95_lower"] < interval["mean"] < interval["ci95_upper"]
    assert "Student-t" in interval["method"]