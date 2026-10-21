import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client():
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def edges(monkeypatch):
    rows = [
        ("f1", "Data Analyst", "Finance", "Python"),
        ("f1", "Data Analyst", "Finance", "SQL"),
        ("f2", "Data Analyst", "Finance", "Python"),
        ("f2", "Data Analyst", "Finance", "SQL"),
        ("f3", "Data Analyst", "Finance", "Python"),
        ("f3", "Data Analyst", "Finance", "Excel"),
        ("s1", "Sales Associate", "Sales", "CRM"),
    ]
    dataset = pd.DataFrame(
        rows,
        columns=["job_id", "job_title", "category", "skill"],
    )
    dataset["normalized_skill"] = dataset["skill"].map(main.normalize)
    dataset["normalized_title"] = dataset["job_title"].map(main.normalize)
    dataset["normalized_category"] = dataset["category"].map(main.normalize)
    monkeypatch.setattr(main, "EDGES", dataset)
    return dataset


def test_root_returns_dashboard_html(client, monkeypatch, tmp_path):
    index = tmp_path / "index.html"
    index.write_text("<html><body>Test dashboard</body></html>", encoding="utf-8")
    monkeypatch.setattr(main, "STATIC_DIR", tmp_path)

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Test dashboard" in response.text


def test_root_returns_server_error_when_dashboard_is_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "STATIC_DIR", tmp_path)

    response = client.get("/")

    assert response.status_code == 500
    assert response.json()["detail"] == "Dashboard index.html not found."


def test_health_reports_service_and_dataset_state(client, edges):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "SkillGraph-X"
    assert response.json()["dataset_loaded"] is True
    assert response.json()["mode"] == "dataset-frequency-analysis"


def test_health_reports_unavailable_dataset(client, monkeypatch):
    monkeypatch.setattr(main, "EDGES", None)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["dataset_loaded"] is False


def test_roles_returns_categories_and_job_titles(client, edges):
    response = client.get("/api/roles")

    assert response.status_code == 200
    payload = response.json()
    assert payload["category_count"] == 2
    assert payload["title_count"] == 2
    assert "Finance" in payload["roles"]
    assert "Data Analyst" in payload["roles"]


def test_roles_returns_service_unavailable_without_dataset(client, monkeypatch):
    monkeypatch.setattr(main, "EDGES", None)

    response = client.get("/api/roles")

    assert response.status_code == 503
    assert "dataset unavailable" in response.json()["detail"].lower()


def test_top_skills_returns_dataset_counts(client, edges):
    response = client.get("/api/top-skills?limit=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_jobs"] == 4
    assert payload["total_skills"] == 4
    assert payload["total_relationships"] == 7
    assert len(payload["results"]) == 2
    assert payload["results"][0]["skill"] == "Python"


@pytest.mark.parametrize("limit", [0, 101])
def test_top_skills_rejects_out_of_range_limit(client, limit):
    response = client.get(f"/api/top-skills?limit={limit}")

    assert response.status_code == 422


def test_top_skills_returns_service_unavailable_without_dataset(client, monkeypatch):
    monkeypatch.setattr(main, "EDGES", None)

    response = client.get("/api/top-skills")

    assert response.status_code == 503


def test_skill_graph_returns_bipartite_nodes_and_edges(client, edges):
    response = client.get("/api/skill-graph?limit=2&max_jobs=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_nodes"] == len(payload["nodes"])
    assert payload["total_edges"] == len(payload["edges"])
    assert payload["selected_jobs"] <= 2
    assert {node["type"] for node in payload["nodes"]} == {"job", "skill"}
    node_ids = {node["id"] for node in payload["nodes"]}
    assert all(
        link["source"] in node_ids and link["target"] in node_ids
        for link in payload["edges"]
    )
    assert "not a trained GNN" in payload["method_note"]


@pytest.mark.parametrize("parameter", ["limit=0", "max_jobs=301"])
def test_skill_graph_rejects_out_of_range_parameters(client, parameter):
    response = client.get(f"/api/skill-graph?{parameter}")

    assert response.status_code == 422


def test_skill_graph_returns_service_unavailable_without_dataset(client, monkeypatch):
    monkeypatch.setattr(main, "EDGES", None)

    response = client.get("/api/skill-graph")

    assert response.status_code == 503


def test_skill_gaps_returns_frequency_based_recommendations(client, edges):
    response = client.post(
        "/api/skill-gaps",
        json={"role": "Finance", "skills": ["python"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["match_type"] == "category"
    assert payload["matching_jobs"] == 3
    assert payload["required_skills"] == [
        {"skill": "Python", "jobs_requiring_skill": 3, "job_coverage_percent": 100.0, "frequency": 1.0},
        {"skill": "SQL", "jobs_requiring_skill": 2, "job_coverage_percent": 66.7, "frequency": 2 / 3},
        {"skill": "Excel", "jobs_requiring_skill": 1, "job_coverage_percent": 33.3, "frequency": 1 / 3},
    ]
    assert payload["matched_skills"] == ["Python"]
    assert payload["missing_skills"] == ["SQL", "Excel"]
    assert payload["coverage_percent"] == 33
    assert payload["recommendations"][0]["skill"] == "SQL"
    assert "does not use a trained graph neural network" in payload["method_note"]


@pytest.mark.parametrize(
    "payload",
    [
        {"role": "   ", "skills": []},
        {"role": "Finance", "skills": ["x" * 101]},
        {"role": "Finance", "skills": ["Python"] * 101},
    ],
)
def test_skill_gaps_rejects_invalid_request_data(client, payload):
    response = client.post("/api/skill-gaps", json=payload)

    assert response.status_code == 422


def test_skill_gaps_returns_not_found_for_unknown_role(client, edges):
    response = client.post(
        "/api/skill-gaps",
        json={"role": "Unknown Role", "skills": []},
    )

    assert response.status_code == 404


def test_skill_gaps_returns_service_unavailable_without_dataset(client, monkeypatch):
    monkeypatch.setattr(main, "EDGES", None)

    response = client.post(
        "/api/skill-gaps",
        json={"role": "Finance", "skills": []},
    )

    assert response.status_code == 503


def test_model_summary_returns_comparison_and_metadata(
    client, monkeypatch, tmp_path
):
    comparison_path = tmp_path / "comparison.json"
    metadata_path = tmp_path / "metadata.json"
    comparison = {
        "selected_model": {"model_name": "Test model"},
        "all_model_results": [{"model_name": "Test model"}],
    }
    metadata = {"model": "Test model"}
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(main, "MODEL_SUMMARY_FILE", comparison_path)
    monkeypatch.setattr(main, "MODEL_METADATA_FILE", metadata_path)

    response = client.get("/api/model-summary")

    assert response.status_code == 200
    assert response.json()["selected_model"] == comparison["selected_model"]
    assert response.json()["all_model_results"] == comparison["all_model_results"]
    assert response.json()["metadata"] == metadata


def test_model_summary_returns_not_found_when_comparison_is_missing(
    client, monkeypatch, tmp_path
):
    monkeypatch.setattr(main, "MODEL_SUMMARY_FILE", tmp_path / "missing.json")

    response = client.get("/api/model-summary")

    assert response.status_code == 404


def test_graph_benchmark_summary_returns_saved_aggregate_only(
    client, monkeypatch, tmp_path
):
    benchmark_path = tmp_path / "paired.json"
    benchmark = {
        "experiment": "paired benchmark",
        "seeds": [43, 47],
        "protocol": {"task": "edge completion"},
        "aggregate": {
            "ranking_metrics": {"graphsage": {"10": {"precision": {"mean": 0.05}}}},
            "paired_macro_differences": {"frequency": {"10": {"precision": {"mean": -0.01}}}},
        },
        "per_seed": [{"private_detail": "not exposed"}],
    }
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    monkeypatch.setattr(main, "GRAPH_BENCHMARK_FILE", benchmark_path)

    response = client.get("/api/graph-benchmark-summary")

    assert response.status_code == 200
    assert response.json() == {
        "experiment": "paired benchmark",
        "seeds": [43, 47],
        "protocol": {"task": "edge completion"},
        "ranking_metrics": benchmark["aggregate"]["ranking_metrics"],
        "paired_macro_differences": benchmark["aggregate"]["paired_macro_differences"],
    }


def test_graph_benchmark_summary_returns_not_found_when_missing(
    client, monkeypatch, tmp_path
):
    monkeypatch.setattr(main, "GRAPH_BENCHMARK_FILE", tmp_path / "missing.json")

    response = client.get("/api/graph-benchmark-summary")

    assert response.status_code == 404


def test_docs_returns_swagger_ui(client):
    response = client.get("/docs")

    assert response.status_code == 200
    assert "swagger-ui" in response.text.lower()