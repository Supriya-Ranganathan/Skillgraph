import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from app.job_category_model import (
    PROJECT_ROOT,
    EncodedXGBClassifier,
    ensure_unique_job_ids,
    split_train_test_by_group,
    build_text_features,
    build_leakage_groups,
    evaluate_model,
    prepare_dataset,
)


def test_duplicate_job_ids_are_removed_and_sorted():
    df = pd.DataFrame(
        {
            "job_id": ["j1", "j1", "j2", "j3"],
            "category": ["a", "a", "b", "a"],
            "job_title": ["Title A", "Title A", "Title B", "Title C"],
            "job_description": ["text", "text", "copy", "more"],
            "job_skill_set": ["python", "python", "sql", "x"],
            "skills": ["python", "python", "sql", "x"],
        }
    )
    deduped = ensure_unique_job_ids(df)
    assert deduped["job_id"].is_unique
    assert len(deduped) == 3


def test_grouped_split_keeps_job_ids_out_of_both_sets():
    df = pd.DataFrame(
        {
            "job_id": ["j1", "j1", "j2", "j3", "j4", "j4"],
            "category": ["a", "a", "a", "b", "b", "b"],
            "job_title": ["A", "A", "A", "B", "B", "B"],
            "job_description": ["alpha", "alpha", "beta", "gamma", "gamma", "delta"],
            "job_skill_set": ["x", "x", "y", "z", "z", "q"],
            "skills": ["x", "x", "y", "z", "z", "q"],
        }
    )
    train_df, test_df = split_train_test_by_group(df, test_size=0.5, random_state=42)
    assert train_df["job_id"].isin(test_df["job_id"]).sum() == 0
    assert set(train_df["leakage_group"]).isdisjoint(set(test_df["leakage_group"]))
    assert len(train_df) + len(test_df) == len(df)
    assert set(train_df["category"]).union(set(test_df["category"])) == {"a", "b"}


def test_near_duplicate_text_records_share_a_leakage_group():
    df = pd.DataFrame(
        {
            "job_id": ["j1", "j2", "j3"],
            "category": ["a", "a", "a"],
            "job_title": ["Data analyst", "Data analyst", "Network engineer"],
            "job_description": ["Analyze customer data", "Analyze customer data", "Maintain network hardware"],
            "skills": ["sql python", "sql python", "routing hardware"],
        }
    )
    df["text_features"] = df.apply(build_text_features, axis=1)
    groups = build_leakage_groups(df)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_feature_builder_combines_title_description_and_skills():
    text = build_text_features(
        {
            "job_title": "Data Scientist",
            "job_description": "Build ML pipelines.",
            "skills": "python sql",
        }
    )
    assert "data scientist" in text.lower()
    assert "ml pipelines" in text.lower()
    assert "python sql" in text.lower()


def test_evaluate_model_returns_standard_metrics():
    df = pd.DataFrame(
        {
            "job_id": ["j1", "j2", "j3", "j4", "j5", "j6"],
            "category": ["a", "a", "b", "b", "a", "b"],
            "job_title": ["A", "A", "B", "B", "A", "B"],
            "job_description": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"],
            "job_skill_set": ["x", "x", "y", "y", "x", "y"],
            "skills": ["x", "x", "y", "y", "x", "y"],
        }
    )
    train_df, test_df = split_train_test_by_group(df, test_size=0.5, random_state=42)
    metrics = evaluate_model(
        model_name="Logistic Regression",
        vectorizer_name="word_tfidf",
        estimator=None,
        X_train=train_df["text_features"],
        y_train=train_df["category"],
        X_test=test_df["text_features"],
        y_test=test_df["category"],
        label_encoder=None,
    )
    assert set(metrics.keys()) >= {"accuracy", "macro_f1", "weighted_f1", "confusion_matrix"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_saved_model_matches_metadata_and_benchmark_artifact():
    models_dir = PROJECT_ROOT / "models"
    comparison = json.loads((models_dir / "job_category_model_comparison.json").read_text(encoding="utf-8"))
    metadata = json.loads((models_dir / "job_category_best_model_metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(models_dir / "job_category_best_model.joblib")
    winner = comparison["selected_model"]

    assert metadata["model"] == winner["model_name"]
    assert metadata["vectorizer"] == winner["vectorizer_name"]
    for key in ("validation_macro_f1", "validation_weighted_f1", "test_accuracy", "test_macro_f1", "test_weighted_f1"):
        assert metadata[key] == winner[key]

    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["classifier"]
    assert isinstance(vectorizer, TfidfVectorizer)
    assert metadata["vectorizer"] == "word_tfidf"
    if metadata["model"] == "XGBoost":
        assert isinstance(classifier, EncodedXGBClassifier)
        assert isinstance(classifier.classifier_, XGBClassifier)
        assert set(classifier.classes_) == set(metadata["labels"])
    else:
        expected_estimators = {
            "Logistic Regression": "LogisticRegression",
            "Linear SVM": "LinearSVC",
            "Complement Naive Bayes": "ComplementNB",
            "Random Forest": "RandomForestClassifier",
        }
        assert classifier.__class__.__name__ == expected_estimators[metadata["model"]]

    sample_text = prepare_dataset().iloc[0]["text_features"]
    prediction = model.predict([sample_text])[0]
    assert isinstance(prediction, str)
    assert prediction in metadata["labels"]
