from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.job_category_model import run_full_model_selection


if __name__ == "__main__":
    results = run_full_model_selection()
    print("\n=== JOB-CATEGORY CLASSIFICATION MODEL SELECTION ===")
    print(json.dumps(results["selected_model"], indent=2))
    print("\nSaved comparison metrics:", "models/job_category_model_comparison.json")
    print("Saved best model:", "models/job_category_best_model.joblib")
    print("Saved comparison plot:", "visualizations/job_category_model_comparison.png")
    print("Saved confusion matrix:", "visualizations/job_category_best_confusion_matrix.png")
