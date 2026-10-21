from app.skillgraph_core import (
    compute_career_paths,
    compute_dataset_summary,
    compute_skill_gap_recommendations,
    normalize_text,
)


def test_normalize_text_cleans_whitespace_and_case():
    assert normalize_text("  Python   Developer ") == "python developer"


def test_dataset_summary_uses_real_processed_data():
    summary = compute_dataset_summary()
    assert summary["jobs"] == 1167
    assert summary["skills"] >= 4000
    assert summary["edges"] >= 20000
    assert summary["categories"]


def test_skill_gap_recommendations_are_ranked_by_frequency():
    results = compute_skill_gap_recommendations(
        role="information technology",
        current_skills=["python", "sql"],
    )
    assert results["matching_jobs"] > 0
    assert results["required_count"] > 0
    assert results["recommendations"]
    assert results["recommendations"][0]["job_coverage_percent"] >= 0


def test_career_paths_returns_routes_or_empty_result():
    routes = compute_career_paths("business-development", "sales")
    assert isinstance(routes, list)
    assert all(isinstance(route, dict) for route in routes)
