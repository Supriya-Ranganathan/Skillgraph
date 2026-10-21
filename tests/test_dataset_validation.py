from pathlib import Path

import pandas as pd

from app.dataset_validation import validate_processed_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _valid_frames():
    jobs = pd.DataFrame(
        [
            {
                "job_id": 1,
                "category": "Finance",
                "job_title": "Data Analyst",
                "job_description": "Analyze data",
                "job_skill_set": "['Python', 'SQL']",
                "skills": "python | sql",
            }
        ]
    )
    edges = pd.DataFrame(
        [
            {"job_id": 1, "job_title": "Data Analyst", "category": "Finance", "skill": "Python"},
            {"job_id": 1, "job_title": "Data Analyst", "category": "Finance", "skill": "SQL"},
        ]
    )
    return jobs, edges


def test_real_processed_dataset_passes_integrity_checks():
    jobs = pd.read_csv(PROJECT_ROOT / "data/processed/cleaned_jobs.csv")
    edges = pd.read_csv(PROJECT_ROOT / "data/processed/job_skill_edges.csv")

    result = validate_processed_data(jobs, edges)

    assert result["valid"] is True
    assert result["errors"] == []
    assert result["summary"]["job_rows"] == 1167
    assert result["summary"]["edge_rows"] == 20833
    assert result["summary"]["missing_graph_edges"] == 0
    assert result["summary"]["unexpected_graph_edges"] == 0


def test_missing_required_columns_are_reported_without_crashing():
    jobs, edges = _valid_frames()
    result = validate_processed_data(jobs.drop(columns="skills"), edges)

    assert result["valid"] is False
    assert "skills" in result["summary"]["missing_job_columns"]
    assert "missing required columns" in result["errors"][0].lower()


def test_missing_values_and_duplicate_job_ids_are_reported():
    jobs, edges = _valid_frames()
    jobs.loc[0, "job_title"] = None
    jobs = pd.concat([jobs, jobs], ignore_index=True)

    result = validate_processed_data(jobs, edges)

    assert result["valid"] is False
    assert result["summary"]["missing_values"]["jobs"]["job_title"] == 2
    assert result["summary"]["duplicate_job_ids"] == 1


def test_duplicate_and_inconsistent_graph_edges_are_reported():
    jobs, edges = _valid_frames()
    edges.loc[1, "skill"] = "Python"

    result = validate_processed_data(jobs, edges)

    assert result["valid"] is False
    assert result["summary"]["duplicate_job_skill_edges"] == 1
    assert result["summary"]["missing_graph_edges"] == 1
    assert result["summary"]["unexpected_graph_edges"] == 0


def test_edge_metadata_mismatch_and_unknown_job_are_reported():
    jobs, edges = _valid_frames()
    edges.loc[0, "job_title"] = "Different title"
    edges.loc[1, "job_id"] = 99

    result = validate_processed_data(jobs, edges)

    assert result["valid"] is False
    assert result["summary"]["edge_metadata_mismatch_rows"] == 1
    assert result["summary"]["unknown_edge_job_rows"] == 1


def test_blank_job_ids_are_rejected_and_category_blanks_are_reported():
    jobs, edges = _valid_frames()
    jobs["job_id"] = jobs["job_id"].astype(str)
    jobs.loc[0, "job_id"] = "  "

    result = validate_processed_data(jobs, edges)

    assert result["valid"] is False
    assert result["summary"]["blank_job_ids"]["jobs"] == 1


def test_blank_category_is_rejected_and_reported():
    jobs, edges = _valid_frames()
    jobs.loc[0, "category"] = ""

    result = validate_processed_data(jobs, edges)

    assert result["valid"] is False
    assert result["summary"]["blank_text_values"]["jobs"]["category"] == 1


def test_non_text_required_values_are_rejected():
    jobs, edges = _valid_frames()
    jobs["job_title"] = jobs["job_title"].astype(object)
    jobs.loc[0, "job_title"] = 123

    result = validate_processed_data(jobs, edges)

    assert result["valid"] is False
    assert result["summary"]["text_type_errors"]["jobs"]["job_title"] == 1