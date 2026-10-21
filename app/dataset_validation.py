from __future__ import annotations

import argparse
import json
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JOBS_FILE = PROJECT_ROOT / "data" / "processed" / "cleaned_jobs.csv"
EDGES_FILE = PROJECT_ROOT / "data" / "processed" / "job_skill_edges.csv"
REQUIRED_JOB_COLUMNS = (
    "job_id",
    "category",
    "job_title",
    "job_description",
    "job_skill_set",
    "skills",
)
REQUIRED_EDGE_COLUMNS = ("job_id", "job_title", "category", "skill")
JOB_TEXT_COLUMNS = REQUIRED_JOB_COLUMNS[1:]
EDGE_TEXT_COLUMNS = REQUIRED_EDGE_COLUMNS[1:]


def _normalize(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def _is_missing(value: Any) -> bool:
    if not pd.api.types.is_scalar(value):
        return False
    return bool(pd.isna(value))


def _text_errors(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, int]:
    return {
        column: int(
            sum(
                not _is_missing(value) and not isinstance(value, str)
                for value in frame[column]
            )
        )
        for column in columns
        if column in frame.columns
    }


def _blank_counts(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, int]:
    return {
        column: int(
            frame[column]
            .map(lambda value: isinstance(value, str) and not value.strip())
            .sum()
        )
        for column in columns
        if column in frame.columns
    }


def _edge_keys(jobs: pd.DataFrame) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in jobs.itertuples(index=False):
        job_id = str(getattr(row, "job_id")).strip()
        skill_list = getattr(row, "skills")
        if _is_missing(skill_list):
            continue
        keys.update(
            (job_id, _normalize(skill))
            for skill in str(skill_list).split("|")
            if _normalize(skill)
        )
    return keys


def validate_processed_data(
    jobs: pd.DataFrame,
    edges: pd.DataFrame,
) -> dict[str, Any]:
    """Check processed dataset schema, row quality, and job-skill edge integrity."""
    missing_job_columns = sorted(set(REQUIRED_JOB_COLUMNS) - set(jobs.columns))
    missing_edge_columns = sorted(set(REQUIRED_EDGE_COLUMNS) - set(edges.columns))
    summary: dict[str, Any] = {
        "job_rows": int(len(jobs)),
        "edge_rows": int(len(edges)),
        "job_columns": {column: str(dtype) for column, dtype in jobs.dtypes.items()},
        "edge_columns": {column: str(dtype) for column, dtype in edges.dtypes.items()},
        "missing_job_columns": missing_job_columns,
        "missing_edge_columns": missing_edge_columns,
    }
    errors: list[str] = []
    if missing_job_columns:
        errors.append(f"Jobs CSV is missing required columns: {missing_job_columns}")
    if missing_edge_columns:
        errors.append(f"Edges CSV is missing required columns: {missing_edge_columns}")
    if errors:
        return {"valid": False, "summary": summary, "errors": errors}

    jobs_missing = {column: int(jobs[column].isna().sum()) for column in REQUIRED_JOB_COLUMNS}
    edges_missing = {column: int(edges[column].isna().sum()) for column in REQUIRED_EDGE_COLUMNS}
    jobs_blanks = _blank_counts(jobs, JOB_TEXT_COLUMNS)
    edges_blanks = _blank_counts(edges, EDGE_TEXT_COLUMNS)
    job_type_errors = _text_errors(jobs, JOB_TEXT_COLUMNS)
    edge_type_errors = _text_errors(edges, EDGE_TEXT_COLUMNS)
    job_id_type_errors = int(
        sum(
            not _is_missing(value)
            and not isinstance(value, (str, Integral))
            for value in jobs["job_id"]
        )
    )
    edge_id_type_errors = int(
        sum(
            not _is_missing(value)
            and not isinstance(value, (str, Integral))
            for value in edges["job_id"]
        )
    )
    job_id_blanks = int(
        jobs["job_id"].map(lambda value: isinstance(value, str) and not value.strip()).sum()
    )
    edge_job_id_blanks = int(
        edges["job_id"].map(lambda value: isinstance(value, str) and not value.strip()).sum()
    )

    job_ids = jobs["job_id"].map(lambda value: str(value).strip())
    edge_job_ids = edges["job_id"].map(lambda value: str(value).strip())
    duplicate_job_ids = int(job_ids.duplicated().sum())
    normalized_edges = pd.DataFrame(
        {
            "job_id": edge_job_ids,
            "skill": edges["skill"].map(_normalize),
        }
    )
    duplicate_edges = int(normalized_edges.duplicated(["job_id", "skill"]).sum())
    expected_edges = _edge_keys(jobs)
    actual_edges = set(zip(normalized_edges["job_id"], normalized_edges["skill"]))
    missing_edges = expected_edges - actual_edges
    unexpected_edges = actual_edges - expected_edges
    known_jobs = set(job_ids)
    unknown_edge_jobs = int((~edge_job_ids.isin(known_jobs)).sum())

    metadata_by_job = {
        job_id: (_normalize(title), _normalize(category))
        for job_id, title, category in zip(job_ids, jobs["job_title"], jobs["category"])
    }
    metadata_mismatches = 0
    for job_id, title, category in zip(edge_job_ids, edges["job_title"], edges["category"]):
        expected = metadata_by_job.get(job_id)
        if expected is not None and expected != (_normalize(title), _normalize(category)):
            metadata_mismatches += 1

    required_job_blanks = sum(
        jobs_blanks[column]
        for column in JOB_TEXT_COLUMNS
    )
    required_edge_blanks = sum(edges_blanks[column] for column in ("job_title", "skill"))
    summary.update(
        {
            "unique_jobs": int(job_ids.nunique()),
            "unique_categories": int(jobs["category"].nunique(dropna=True)),
            "unique_skills": int(normalized_edges["skill"].nunique()),
            "expected_unique_edges": len(expected_edges),
            "missing_values": {"jobs": jobs_missing, "edges": edges_missing},
            "blank_text_values": {"jobs": jobs_blanks, "edges": edges_blanks},
            "text_type_errors": {"jobs": job_type_errors, "edges": edge_type_errors},
            "job_id_type_errors": {"jobs": job_id_type_errors, "edges": edge_id_type_errors},
            "blank_job_ids": {"jobs": job_id_blanks, "edges": edge_job_id_blanks},
            "duplicate_job_ids": duplicate_job_ids,
            "duplicate_job_skill_edges": duplicate_edges,
            "missing_graph_edges": len(missing_edges),
            "unexpected_graph_edges": len(unexpected_edges),
            "unknown_edge_job_rows": unknown_edge_jobs,
            "edge_metadata_mismatch_rows": metadata_mismatches,
        }
    )

    if any(jobs_missing.values()) or any(edges_missing.values()):
        errors.append("Missing values found in required columns.")
    if any(job_type_errors.values()) or any(edge_type_errors.values()):
        errors.append("Non-text values found in text columns.")
    if job_id_type_errors or edge_id_type_errors:
        errors.append("Job IDs must be strings or integers.")
    if job_id_blanks or edge_job_id_blanks or required_job_blanks:
        errors.append("Blank job IDs or required job text values found.")
    if required_edge_blanks:
        errors.append("Blank edge job titles or skill values found.")
    if duplicate_job_ids:
        errors.append("Duplicate job IDs found in cleaned jobs.")
    if duplicate_edges:
        errors.append("Duplicate normalized job-skill edges found.")
    if missing_edges or unexpected_edges:
        errors.append("Processed graph edges do not match skills in the cleaned jobs CSV.")
    if unknown_edge_jobs:
        errors.append("Graph edges reference job IDs absent from the cleaned jobs CSV.")
    if metadata_mismatches:
        errors.append("Graph edge title/category metadata disagrees with cleaned jobs.")

    return {"valid": not errors, "summary": summary, "errors": errors}


def validate_project_files(
    jobs_file: Path = JOBS_FILE,
    edges_file: Path = EDGES_FILE,
) -> dict[str, Any]:
    jobs = pd.read_csv(jobs_file)
    edges = pd.read_csv(edges_file)
    result = validate_processed_data(jobs, edges)
    result["inputs"] = {
        "jobs": str(jobs_file),
        "edges": str(edges_file),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate processed SkillGraph-X data and graph edges.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "dataset_validation_report.json",
        help="New JSON report path; an existing file is never overwritten.",
    )
    args = parser.parse_args(argv)
    result = validate_project_files()
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    if output.exists():
        parser.error(f"Refusing to overwrite existing report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Validation report: {output}")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())