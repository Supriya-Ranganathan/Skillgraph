
"""
SkillGraph-X: Benchmarking and Comparative Analysis

Evaluates:
1. Frequency baseline
2. Skill co-occurrence
3. GraphSAGE

Ground truth and predictions must use the same skill identifiers.
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "artifacts" / "benchmarking"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def precision_at_k(actual, predicted, k=10):
    """Fraction of top-k recommendations that are relevant."""
    predicted_k = list(predicted)[:k]

    if not predicted_k:
        return 0.0

    relevant = len(set(actual) & set(predicted_k))
    return relevant / len(predicted_k)


def recall_at_k(actual, predicted, k=10):
    """Fraction of relevant skills recovered in top-k."""
    actual = set(actual)
    predicted_k = list(predicted)[:k]

    if not actual:
        return 0.0

    return len(actual & set(predicted_k)) / len(actual)


def ndcg_at_k(actual, predicted, k=10):
    """Normalized discounted cumulative gain for binary relevance."""
    actual = set(actual)
    predicted_k = list(predicted)[:k]

    dcg = sum(
        1.0 / __import__("math").log2(rank + 2)
        for rank, skill in enumerate(predicted_k)
        if skill in actual
    )

    ideal_hits = min(len(actual), k)
    idcg = sum(
        1.0 / __import__("math").log2(rank + 2)
        for rank in range(ideal_hits)
    )

    return dcg / idcg if idcg else 0.0


def evaluate_recommender(ground_truth, predictions, k=10):
    """
    ground_truth:
        {job_id: set_of_relevant_skills}

    predictions:
        {job_id: ranked_list_of_recommended_skills}
    """
    precisions = []
    recalls = []
    ndcgs = []
    errors = []

    for job_id, actual_skills in ground_truth.items():
        predicted_skills = predictions.get(job_id, [])

        p = precision_at_k(actual_skills, predicted_skills, k)
        r = recall_at_k(actual_skills, predicted_skills, k)
        n = ndcg_at_k(actual_skills, predicted_skills, k)

        precisions.append(p)
        recalls.append(r)
        ndcgs.append(n)

        actual_set = set(actual_skills)
        predicted_set = set(predicted_skills[:k])

        errors.append({
            "job_id": str(job_id),
            "actual_skill_count": len(actual_set),
            "recommended_skill_count": len(predicted_set),
            "correct_recommendations": len(
                actual_set & predicted_set
            ),
            "irrelevant_recommendations": sorted(
                predicted_set - actual_set
            ),
            "missed_relevant_skills": sorted(
                actual_set - predicted_set
            ),
            "precision_at_10": p,
            "recall_at_10": r,
            "ndcg_at_10": n,
        })

    if not precisions:
        raise ValueError("No ground-truth examples were supplied.")

    metrics = {
        "Precision@10": sum(precisions) / len(precisions),
        "Recall@10": sum(recalls) / len(recalls),
        "NDCG@10": sum(ndcgs) / len(ndcgs),
        "evaluated_jobs": len(precisions),
    }

    return metrics, errors


def compare_models(ground_truth, model_predictions):
    """Evaluate each model using identical test examples."""
    results = []
    all_errors = []

    for model_name, predictions in model_predictions.items():
        metrics, errors = evaluate_recommender(
            ground_truth,
            predictions,
            k=10
        )

        results.append({
            "Model": model_name,
            **metrics
        })

        for error in errors:
            all_errors.append({
                "Model": model_name,
                **error
            })

    results_df = pd.DataFrame(results)
    errors_df = pd.DataFrame(all_errors)

    results_df.to_csv(
        RESULTS_DIR / "benchmark_results.csv",
        index=False
    )

    errors_df.to_csv(
        RESULTS_DIR / "error_analysis.csv",
        index=False
    )

    with open(
        RESULTS_DIR / "benchmark_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)

    # Comparative visualization
    metric_columns = [
        "Precision@10",
        "Recall@10",
        "NDCG@10"
    ]

    ax = results_df.set_index("Model")[metric_columns].plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title("SkillGraph-X: Recommender Benchmark")
    ax.set_ylabel("Score")
    ax.set_xlabel("Recommendation method")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "benchmark_comparison.png",
        dpi=200
    )
    plt.close()

    print("\nBENCHMARK RESULTS")
    print(results_df.to_string(index=False))

    print("\nERROR ANALYSIS")
    print("Total evaluated predictions:", len(errors_df))
    print(
        "Total irrelevant recommendations:",
        errors_df["irrelevant_recommendations"].map(len).sum()
    )
    print(
        "Total missed relevant skills:",
        errors_df["missed_relevant_skills"].map(len).sum()
    )

    print("\nFiles saved to:", RESULTS_DIR)


if __name__ == "__main__":
    # DEMONSTRATION DATA ONLY.
    # Replace this with predictions and labels from your actual test set.

    ground_truth = {
        "job_1": {"Python", "SQL", "Pandas"},
        "job_2": {"React", "JavaScript", "CSS"},
        "job_3": {"Python", "Machine Learning", "NumPy"},
    }

    model_predictions = {
        "Frequency baseline": {
            "job_1": ["Python", "Java", "SQL"],
            "job_2": ["Python", "React", "CSS"],
            "job_3": ["Python", "SQL", "NumPy"],
        },
        "Skill co-occurrence": {
            "job_1": ["Python", "SQL", "Pandas"],
            "job_2": ["React", "JavaScript", "CSS"],
            "job_3": ["Python", "Machine Learning", "NumPy"],
        },
        "GraphSAGE": {
            "job_1": ["Python", "Java", "C++"],
            "job_2": ["React", "CSS", "Python"],
            "job_3": ["Python", "NumPy", "SQL"],
        },
    }

    compare_models(ground_truth, model_predictions)
