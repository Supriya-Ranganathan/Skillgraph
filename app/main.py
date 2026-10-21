
from pathlib import Path
import json
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

if __package__:
    from .skillgraph_core import (
        compute_career_paths,
        compute_what_if_scenario,
        find_role_jobs,
    )
else:
    from skillgraph_core import (
        compute_career_paths,
        compute_what_if_scenario,
        find_role_jobs,
    )


# ============================================================
# PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
EDGES_FILE = PROJECT_ROOT / "data" / "processed" / "job_skill_edges.csv"
MODEL_SUMMARY_FILE = PROJECT_ROOT / "models" / "job_category_model_comparison.json"
MODEL_METADATA_FILE = PROJECT_ROOT / "models" / "job_category_best_model_metadata.json"
GRAPH_BENCHMARK_FILE = PROJECT_ROOT / "models" / "paired_graph_benchmark_evaluation.json"


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="SkillGraph-X API",
    version="1.0.0",
    description="Dataset-based workforce skill gap analysis",
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


# ============================================================
# DATASET
# ============================================================

def normalize(value):
    return " ".join(str(value).strip().lower().split())


def load_dataset():
    if not EDGES_FILE.exists():
        print(f"Dataset file not found: {EDGES_FILE}")
        return None

    try:
        df = pd.read_csv(EDGES_FILE)

        required = {"job_id", "job_title", "category", "skill"}
        if not required.issubset(df.columns):
            print(f"Missing columns: {required - set(df.columns)}")
            return None

        for column in ["job_title", "category", "skill"]:
            df[column] = df[column].fillna("").astype(str).str.strip()

        df = df[(df["job_title"] != "") & (df["skill"] != "")].copy()

        df["normalized_skill"] = df["skill"].map(normalize)
        df["normalized_title"] = df["job_title"].map(normalize)
        df["normalized_category"] = df["category"].map(normalize)

        df = df[df["normalized_skill"] != ""].drop_duplicates(
            subset=["job_id", "normalized_skill"]
        ).reset_index(drop=True)

        print(f"Loaded {len(df)} job-skill relationships")
        print(f"Jobs: {df['job_id'].nunique()}")
        print(f"Skills: {df['normalized_skill'].nunique()}")

        return df

    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"Dataset loading error: {exc}")
        return None


EDGES = load_dataset()


# ============================================================
# REQUEST MODEL
# ============================================================

class GapRequest(BaseModel):
    role: str = Field(min_length=1, max_length=150)
    skills: List[str] = Field(default_factory=list, max_length=100)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Role cannot be empty.")
        return value

    @field_validator("skills")
    @classmethod
    def validate_skills(cls, values):
        result = []

        for skill in values:
            if not isinstance(skill, str):
                raise ValueError("Skills must be text.")

            skill = skill.strip()

            if len(skill) > 100:
                raise ValueError("Each skill must be at most 100 characters.")

            if skill:
                result.append(skill)

        return result


class WhatIfRequest(BaseModel):
    role: str = Field(min_length=1, max_length=150)
    current_skills: List[str] = Field(max_length=100)
    additions: List[str] = Field(max_length=100)
    removals: List[str] = Field(max_length=100)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Role cannot be empty.")
        return value

    @field_validator("current_skills", "additions", "removals")
    @classmethod
    def validate_skill_lists(cls, values):
        result = []

        for skill in values:
            skill = skill.strip()

            if len(skill) > 100:
                raise ValueError("Each skill must be at most 100 characters.")

            if skill:
                result.append(skill)

        return result


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():
    index = STATIC_DIR / "index.html"

    if not index.exists():
        raise HTTPException(500, "Dashboard index.html not found.")

    return index.read_text(encoding="utf-8")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "SkillGraph-X",
        "dataset_loaded": EDGES is not None,
        "dataset_path": str(EDGES_FILE),
        "mode": "dataset-frequency-analysis",
    }


# ============================================================
# DATASET ROLES
# ============================================================

@app.get("/api/roles")
def get_roles():
    if EDGES is None:
        raise HTTPException(
            503,
            "Processed dataset unavailable. Run preprocessing first.",
        )

    categories = sorted({
        str(value).strip()
        for value in EDGES["category"].unique()
        if str(value).strip()
    })

    titles = (
        EDGES.loc[EDGES["job_title"] != "", "job_title"]
        .value_counts()
        .head(100)
        .index.tolist()
    )

    return {
        "roles": list(dict.fromkeys(categories + titles)),
        "category_count": len(categories),
        "title_count": len(titles),
    }


@app.get("/api/career-paths")
def career_paths(
    start_role: str = Query(..., min_length=1, max_length=150),
    target_role: str = Query(..., min_length=1, max_length=150),
):
    if EDGES is None:
        raise HTTPException(503, "Processed dataset unavailable.")

    start_role = start_role.strip()
    target_role = target_role.strip()
    if not start_role or not target_role:
        raise HTTPException(422, "Start role and target role cannot be empty.")

    for role in (start_role, target_role):
        _, match_type = find_role_jobs(role, EDGES)
        if match_type == "not_found":
            raise HTTPException(404, f"No dataset jobs found for '{role}'.")

    return compute_career_paths(start_role, target_role, EDGES)


# ============================================================
# ROLE MATCHING
# ============================================================

def matching_jobs(role):
    if EDGES is None:
        raise HTTPException(503, "Processed dataset unavailable.")

    key = normalize(role)

    # Exact category match
    result = EDGES[EDGES["normalized_category"] == key]
    if not result.empty:
        return result.copy(), "category"

    # Exact job title match
    result = EDGES[EDGES["normalized_title"] == key]
    if not result.empty:
        return result.copy(), "job_title"

    # Partial title match
    result = EDGES[
        EDGES["normalized_title"].str.contains(
            key, regex=False, na=False
        )
    ]

    if not result.empty:
        return result.copy(), "job_title_partial"

    raise HTTPException(
        404,
        f"No dataset jobs found for '{role}'. Select a role from the dropdown.",
    )


# ============================================================
# SKILL-GAP ANALYSIS
# ============================================================

@app.post("/api/skill-gaps")
def skill_gaps(payload: GapRequest):
    matches, match_type = matching_jobs(payload.role)

    job_count = int(matches["job_id"].nunique())

    if job_count == 0:
        raise HTTPException(404, "No matching job postings found.")

    # Count each skill at most once per job.
    counts = (
        matches.drop_duplicates(["job_id", "normalized_skill"])
        .groupby("normalized_skill")["job_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    # Explicit research parameter: skill must occur in >=20% of
    # the matched jobs to count as a requirement.
    threshold = 0.20
    user_skills = {normalize(s) for s in payload.skills if normalize(s)}

    required = []

    for skill_key, count in counts.items():
        frequency = count / job_count

        if frequency < threshold:
            continue

        names = matches.loc[
            matches["normalized_skill"] == skill_key, "skill"
        ].mode()

        display = str(names.iloc[0]) if not names.empty else skill_key

        required.append({
            "skill": display,
            "key": skill_key,
            "job_count": int(count),
            "frequency": frequency,
        })

    matched = [
        item["skill"] for item in required
        if item["key"] in user_skills
    ]

    missing = [
        item for item in required
        if item["key"] not in user_skills
    ]

    recommendations = []

    for item in missing:
        percent = round(item["frequency"] * 100, 1)

        priority = (
            "High" if percent >= 50
            else "Medium" if percent >= 30
            else "Explore"
        )

        recommendations.append({
            "skill": item["skill"],
            "reason": (
                f"Observed in {item['job_count']} of "
                f"{job_count} matching job postings."
            ),
            "jobs_requiring_skill": item["job_count"],
            "job_coverage_percent": percent,
            "priority": priority,
        })

    recommendations.sort(
        key=lambda item: item["job_coverage_percent"],
        reverse=True,
    )

    coverage = (
        round(100 * len(matched) / len(required))
        if required else 0
    )

    return {
        "role": payload.role,
        "match_type": match_type,
        "matching_jobs": job_count,
        "required_count": len(required),
        "required_skills": [
            {
                "skill": item["skill"],
                "jobs_requiring_skill": item["job_count"],
                "job_coverage_percent": round(item["frequency"] * 100, 1),
                "frequency": float(item["frequency"]),
            }
            for item in required
        ],
        "matched_count": len(matched),
        "missing_count": len(missing),
        "coverage_percent": coverage,
        "matched_skills": matched,
        "missing_skills": [item["skill"] for item in missing],
        "recommendations": recommendations,
        "method_note": (
            "Dataset-frequency baseline: a skill is included when "
            "it occurs in at least 20% of matching job postings. "
            "Priorities are frequency-based, not learned predictions. "
            "This endpoint does not use a trained graph neural network."
        ),
    }


@app.post("/api/what-if")
def what_if(payload: WhatIfRequest):
    if EDGES is None:
        raise HTTPException(503, "Processed dataset unavailable.")

    _, match_type = find_role_jobs(payload.role, EDGES)
    if match_type == "not_found":
        raise HTTPException(404, f"No dataset jobs found for '{payload.role}'.")

    return compute_what_if_scenario(
        payload.role,
        payload.current_skills,
        payload.additions,
        payload.removals,
        EDGES,
    )


# ============================================================
# TOP SKILLS
# ============================================================

@app.get("/api/top-skills")
def top_skills(limit: int = Query(default=10, ge=1, le=100)):
    if EDGES is None:
        raise HTTPException(503, "Processed dataset unavailable.")

    unique = EDGES.drop_duplicates(["job_id", "normalized_skill"])

    counts = (
        unique.groupby("normalized_skill")["job_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(limit)
    )

    results = []

    for skill, count in counts.items():
        names = unique.loc[
            unique["normalized_skill"] == skill, "skill"
        ].mode()

        results.append({
            "skill": str(names.iloc[0]) if not names.empty else skill,
            "job_count": int(count),
            "job_percentage": round(
                100 * count / max(unique["job_id"].nunique(), 1), 1
            ),
        })

    return {
        "total_jobs": int(unique["job_id"].nunique()),
        "total_skills": int(unique["normalized_skill"].nunique()),
        "total_relationships": int(len(unique)),
        "results": results,
    }


# ============================================================
# SKILL GRAPH
# ============================================================

@app.get("/api/skill-graph")
def skill_graph(
    limit: int = Query(default=15, ge=1, le=100),
    max_jobs: int = Query(default=30, ge=1, le=300),
):
    if EDGES is None:
        raise HTTPException(503, "Processed dataset unavailable.")

    unique = EDGES.drop_duplicates(["job_id", "normalized_skill"])

    counts = (
        unique.groupby("normalized_skill")["job_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    selected_skills = set(counts.head(limit).index)

    selected = unique[
        unique["normalized_skill"].isin(selected_skills)
    ].copy()

    selected_jobs = selected["job_id"].drop_duplicates().head(max_jobs)
    selected = selected[selected["job_id"].isin(selected_jobs)]

    nodes = []
    links = []
    seen = set()

    for row in selected.itertuples(index=False):
        job_id = str(row.job_id)
        skill_key = str(row.normalized_skill)

        job_node = f"job:{job_id}"
        skill_node = f"skill:{skill_key}"

        if job_node not in seen:
            nodes.append({
                "id": job_node,
                "label": str(row.job_title),
                "type": "job",
            })
            seen.add(job_node)

        if skill_node not in seen:
            nodes.append({
                "id": skill_node,
                "label": str(row.skill),
                "type": "skill",
                "job_count": int(counts.get(skill_key, 0)),
            })
            seen.add(skill_node)

        links.append({
            "source": job_node,
            "target": skill_node,
        })

    return {
        "nodes": nodes,
        "edges": links,
        "total_nodes": len(nodes),
        "total_edges": len(links),
        "selected_jobs": int(selected["job_id"].nunique()),
        "selected_skills": len(selected_skills),
        "method_note": (
            "Bipartite graph of observed job-skill relationships. "
            "This is a graph visualization, not a trained GNN."
        ),
    }


# ============================================================
# MODEL LAB
# ============================================================

@app.get("/api/model-summary")
def model_summary():
    if not MODEL_SUMMARY_FILE.exists():
        raise HTTPException(404, "Job-category classification results are not available yet.")

    with MODEL_SUMMARY_FILE.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if MODEL_METADATA_FILE.exists():
        with MODEL_METADATA_FILE.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        payload["metadata"] = metadata

    return payload


@app.get("/api/graph-benchmark-summary")
def graph_benchmark_summary():
    if not GRAPH_BENCHMARK_FILE.exists():
        raise HTTPException(404, "Paired graph benchmark results are not available.")

    try:
        with GRAPH_BENCHMARK_FILE.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(503, "Paired graph benchmark results could not be read.") from exc

    aggregate = payload.get("aggregate", {})
    if not isinstance(aggregate, dict):
        raise HTTPException(503, "Paired graph benchmark results have an invalid schema.")

    return {
        "experiment": payload.get("experiment"),
        "seeds": payload.get("seeds", []),
        "protocol": payload.get("protocol", {}),
        "ranking_metrics": aggregate.get("ranking_metrics", {}),
        "paired_macro_differences": aggregate.get("paired_macro_differences", {}),
    }
