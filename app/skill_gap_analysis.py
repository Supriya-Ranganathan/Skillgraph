from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "job_skill_edges.csv"
VIZ_DIR = ROOT / "visualizations"
OUT_DIR = ROOT / "data" / "processed"

VIZ_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_skill_gap(target_role, current_skills):
    edges = pd.read_csv(DATA_FILE)

    # Normalize the user's skill input
    current = {
        skill.strip().lower()
        for skill in current_skills
        if skill.strip()
    }

    # Find jobs matching the selected role
    role_jobs = edges[
        edges["job_title"].str.contains(
            target_role, case=False, na=False, regex=False
        )
    ]

    if role_jobs.empty:
        role_jobs = edges[
            edges["category"].str.contains(
                target_role, case=False, na=False, regex=False
            )
        ]

    if role_jobs.empty:
        return None, (
            f"No matching jobs found for '{target_role}'. "
            "Try a job title or category from the dataset."
        )

    # Count how often each skill occurs in matching jobs
    required_counts = role_jobs["skill"].value_counts()
    total_jobs = role_jobs["job_id"].nunique()

    results = []
    for skill, count in required_counts.items():
        required_fraction = count / total_jobs

        # Only recommend skills occurring in at least 20% of matching jobs
        if required_fraction < 0.20:
            continue

        results.append({
            "skill": skill,
            "jobs_requiring_skill": int(count),
            "job_coverage": round(float(required_fraction), 3),
            "already_have": skill.lower() in current,
            "gap": skill.lower() not in current,
        })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return None, "Not enough matching job data to identify common skills."

    result_df = result_df.sort_values(
        ["gap", "job_coverage"],
        ascending=[False, False],
    )

    output_file = OUT_DIR / "skill_gap_results.csv"
    result_df.to_csv(output_file, index=False)

    # Plot the most frequently required skills
    chart_df = result_df.head(12).sort_values("job_coverage")

    plt.figure(figsize=(10, 6))
    plt.barh(
        chart_df["skill"],
        chart_df["job_coverage"] * 100,
    )
    plt.xlabel("Percentage of matching jobs requiring skill")
    plt.title(f"Common Skills: {target_role}")
    plt.tight_layout()

    chart_file = VIZ_DIR / "skill_gap_analysis.png"
    plt.savefig(chart_file, dpi=180)
    plt.close()

    return result_df, (
        f"Analysis complete. Matching jobs: {total_jobs}. "
        f"Results saved to {output_file}"
    )


if __name__ == "__main__":
    role = input("Enter target job title or category: ")
    raw_skills = input(
        "Enter your current skills separated by commas: "
    )
    current_skills = raw_skills.split(",")

    results, message = analyze_skill_gap(role, current_skills)
    print("\n" + message)

    if results is not None:
        print("\nIdentified skill gaps:")
        gaps = results[results["gap"]]
        print(gaps.head(15).to_string(index=False))
        print("\nVisualization saved to visualizations/skill_gap_analysis.png")
