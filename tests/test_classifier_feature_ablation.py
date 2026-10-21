import pandas as pd
import pytest

from app.run_classifier_feature_ablation import build_ablation_features


def test_feature_ablation_uses_only_selected_normalized_columns():
    data = pd.DataFrame(
        {
            "job_title": ["  Data   Analyst "],
            "job_description": ["Build reports"],
            "skills": ["Python | SQL"],
        }
    )

    assert build_ablation_features(data, ("job_title", "skills")).tolist() == [
        "data analyst python | sql"
    ]


def test_feature_ablation_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing from the dataset"):
        build_ablation_features(pd.DataFrame({"job_title": ["Analyst"]}), ("skills",))