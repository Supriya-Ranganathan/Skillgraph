from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return " ".join(text.split())


def load_processed_edges() -> pd.DataFrame:
    path = DATA_DIR / "job_skill_edges.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    required = {"job_id", "job_title", "category", "skill"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required dataset columns: {sorted(missing)}")

    df = df.copy()
    df["job_id"] = df["job_id"].astype(str)
    df["job_title"] = df["job_title"].fillna("").astype(str).str.strip()
    df["category"] = df["category"].fillna("").astype(str).str.strip()
    df["skill"] = df["skill"].fillna("").astype(str).str.strip()
    df = df[(df["job_title"] != "") & (df["skill"] != "")].copy()
    df["normalized_skill"] = df["skill"].map(normalize_text)
    df["normalized_title"] = df["job_title"].map(normalize_text)
    df["normalized_category"] = df["category"].map(normalize_text)
    df = df.drop_duplicates(subset=["job_id", "normalized_skill"]).reset_index(drop=True)
    return df


def compute_dataset_summary() -> Dict[str, Any]:
    df = load_processed_edges()
    categories = sorted(
        df["category"].fillna("").astype(str).str.strip().str.lower().unique().tolist()
    )
    return {
        "jobs": int(df["job_id"].nunique()),
        "skills": int(df["normalized_skill"].nunique()),
        "edges": int(len(df)),
        "categories": categories,
        "top_skills": [
            {"skill": skill, "job_count": int(count)}
            for skill, count in (
                df.groupby("normalized_skill")["job_id"].nunique().sort_values(ascending=False).head(10).items()
            )
        ],
    }


def find_role_jobs(role: str, dataset: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, str]:
    edges = dataset if dataset is not None else load_processed_edges()
    key = normalize_text(role)
    if not key:
        return edges.iloc[0:0].copy(), "empty"

    category_match = edges[edges["normalized_category"] == key]
    if not category_match.empty:
        return category_match.copy(), "category"

    title_match = edges[edges["normalized_title"] == key]
    if not title_match.empty:
        return title_match.copy(), "job_title"

    partial_match = edges[edges["normalized_title"].str.contains(key, regex=False, na=False)]
    if not partial_match.empty:
        return partial_match.copy(), "job_title_partial"

    return edges.iloc[0:0].copy(), "not_found"


def compute_skill_gap_recommendations(
    role: str,
    current_skills: Iterable[str],
    dataset: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    edges = dataset if dataset is not None else load_processed_edges()
    job_matches, match_type = find_role_jobs(role, edges)
    if job_matches.empty:
        return {
            "role": role,
            "match_type": match_type,
            "matching_jobs": 0,
            "required_count": 0,
            "matched_count": 0,
            "missing_count": 0,
            "coverage_percent": 0,
            "matched_skills": [],
            "missing_skills": [],
            "recommendations": [],
            "method_note": "No matching role was found in the processed dataset.",
        }

    job_count = int(job_matches["job_id"].nunique())
    unique = job_matches.drop_duplicates(subset=["job_id", "normalized_skill"])
    counts = unique.groupby("normalized_skill")["job_id"].nunique().sort_values(ascending=False)
    threshold = 0.20
    user_skills = {normalize_text(skill) for skill in current_skills if normalize_text(skill)}

    required = []
    for skill_key, count in counts.items():
        frequency = count / job_count if job_count else 0.0
        if frequency < threshold:
            continue
        display = unique.loc[unique["normalized_skill"] == skill_key, "skill"].mode()
        display_skill = str(display.iloc[0]) if not display.empty else skill_key
        required.append(
            {
                "skill": display_skill,
                "key": skill_key,
                "job_count": int(count),
                "frequency": float(frequency),
            }
        )

    matched = [item for item in required if item["key"] in user_skills]
    missing = [item for item in required if item["key"] not in user_skills]

    recommendations = []
    for item in missing:
        percent = round(item["frequency"] * 100, 1)
        priority = (
            "High" if percent >= 50 else "Medium" if percent >= 30 else "Explore"
        )
        recommendations.append(
            {
                "skill": item["skill"],
                "reason": (
                    f"Observed in {item['job_count']} of {job_count} matching job postings."
                ),
                "jobs_requiring_skill": item["job_count"],
                "job_coverage_percent": percent,
                "priority": priority,
            }
        )

    recommendations.sort(key=lambda item: item["job_coverage_percent"], reverse=True)
    coverage = round(100 * len(matched) / len(required), 1) if required else 0.0

    return {
        "role": role,
        "match_type": match_type,
        "matching_jobs": job_count,
        "required_count": len(required),
        "matched_count": len(matched),
        "missing_count": len(missing),
        "coverage_percent": round(float(coverage), 1),
        "matched_skills": [item["skill"] for item in matched],
        "missing_skills": [item["skill"] for item in missing],
        "recommendations": recommendations,
        "method_note": (
            "Dataset-frequency baseline: a skill is included when it occurs in at least 20% of matching job postings. "
            "These recommendations are based on observed job data, not a learned causal prediction."
        ),
    }


def compute_career_paths(start_role: str, target_role: str, dataset: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    edges = dataset if dataset is not None else load_processed_edges()
    if not start_role.strip() or not target_role.strip():
        return []

    start_jobs = find_role_jobs(start_role, edges)[0]
    target_jobs = find_role_jobs(target_role, edges)[0]

    if start_jobs.empty or target_jobs.empty:
        return []

    start_skills = set(start_jobs["normalized_skill"].unique())
    target_skills = set(target_jobs["normalized_skill"].unique())

    role_profiles: List[Tuple[str, set[str]]] = []
    for normalized_title, role_df in edges.groupby("normalized_title", sort=False):
        if not normalized_title or role_df.empty:
            continue
        role_name = str(role_df["job_title"].iloc[0]).strip()
        role_profiles.append((role_name, set(role_df["normalized_skill"].unique())))

    candidate_paths: List[Dict[str, Any]] = []
    excluded_roles = {normalize_text(start_role), normalize_text(target_role)}
    for role_name, profile in role_profiles:
        if normalize_text(role_name) in excluded_roles:
            continue
        overlap = len(start_skills & profile)
        forward_overlap = len(target_skills & profile)
        score = overlap * 0.6 + forward_overlap * 0.4 + min(2.0, len(profile & target_skills) / max(1, len(target_skills)))
        if score <= 0:
            continue
        candidate_paths.append(
            {
                "path": [start_role, role_name, target_role],
                "score": round(float(score), 3),
                "overlap_with_start": overlap,
                "overlap_with_target": forward_overlap,
                "skills_needed": sorted(list(target_skills - profile))[:8],
                "explanation": (
                    f"This intermediate role shares {overlap} skills with {start_role} and {forward_overlap} skills with {target_role}. "
                    "The route is inferred from the observed job-skill graph and is not a validated transition probability."
                ),
            }
        )

    candidate_paths.sort(key=lambda item: item["score"], reverse=True)
    return candidate_paths[:5]


def compute_what_if_scenario(
    role: str,
    current_skills: Iterable[str],
    additions: Iterable[str],
    removals: Iterable[str],
    dataset: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    base = compute_skill_gap_recommendations(role, current_skills, dataset)
    updated_skills = [skill for skill in current_skills if normalize_text(skill) not in {normalize_text(s) for s in removals}]
    updated_skills = list(dict.fromkeys([*updated_skills, *additions]))
    updated = compute_skill_gap_recommendations(role, updated_skills, dataset)

    delta = []
    for item in updated["recommendations"]:
        if item["skill"] not in {entry["skill"] for entry in base["recommendations"]}:
            delta.append({"skill": item["skill"], "change": "new"})

    return {
        "role": role,
        "before": base,
        "after": updated,
        "delta": delta,
        "summary": (
            "This is a scenario simulation based on the current dataset. It helps explore how a changed skill profile would alter recommendations, "
            "but it does not claim causal labor-market outcomes."
        ),
    }


def load_saved_metrics(path: Optional[str] = None) -> Dict[str, Any]:
    file_path = Path(path) if path else MODEL_DIR / "classification_results.json"
    if not file_path.exists():
        return {"classification": [], "graph": []}
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_metrics(payload: Dict[str, Any], file_name: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_DIR / file_name
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return target
