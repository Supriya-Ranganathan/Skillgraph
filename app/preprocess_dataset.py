from pathlib import Path
import ast
import re
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "train-00000-of-00001.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "cleaned_jobs.csv"
SKILLS_FILE = OUTPUT_DIR / "job_skill_edges.csv"

def clean_text(value):
    """Normalize text while preserving useful words and punctuation."""
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()

def parse_skills(value):
    """Convert a stored skill list into a Python list."""
    if isinstance(value, list):
        skills = value
    elif pd.isna(value):
        return []
    else:
        try:
            parsed = ast.literal_eval(str(value))
            skills = parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            skills = []

    result = []
    seen = set()

    for skill in skills:
        skill = re.sub(r"\s+", " ", str(skill)).strip().lower()
        if skill and skill not in seen:
            result.append(skill)
            seen.add(skill)

    return result

def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")

    df = pd.read_parquet(RAW_FILE)

    required = [
        "job_id", "category", "job_title",
        "job_description", "job_skill_set"
    ]

    missing_columns = set(required) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    print("Original dataset shape:", df.shape)

    # Clean text columns
    for column in ["category", "job_title", "job_description"]:
        df[column] = df[column].map(clean_text)

    # Remove records without useful job information
    df = df.dropna(subset=["job_id"])
    df = df[
        (df["job_title"] != "") &
        (df["job_description"] != "")
    ].copy()

    # One record per job ID
    df = df.drop_duplicates(subset=["job_id"]).copy()

    # Parse and normalize the existing extracted skills
    df["skills"] = df["job_skill_set"].map(parse_skills)

    # Keep records with at least one skill
    df = df[df["skills"].map(len) > 0].copy()

    # Save cleaned jobs
    df["skills"] = df["skills"].map(
        lambda skills: " | ".join(skills)
    )
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    # Create job-skill edges for graph construction
    edges = []
    for _, row in df.iterrows():
        for skill in row["skills"].split(" | "):
            edges.append({
                "job_id": row["job_id"],
                "job_title": row["job_title"],
                "category": row["category"],
                "skill": skill
            })

    edges_df = pd.DataFrame(edges).drop_duplicates()
    edges_df.to_csv(SKILLS_FILE, index=False, encoding="utf-8")

    print("\nPreprocessing completed!")
    print("Cleaned jobs:", len(df))
    print("Unique job IDs:", df["job_id"].nunique())
    print("Unique skills:", edges_df["skill"].nunique())
    print("Job-skill relationships:", len(edges_df))
    print("\nSaved files:")
    print(OUTPUT_FILE)
    print(SKILLS_FILE)
    print("\nTop 15 skills:")
    print(edges_df["skill"].value_counts().head(15).to_string())

if __name__ == "__main__":
    main()
