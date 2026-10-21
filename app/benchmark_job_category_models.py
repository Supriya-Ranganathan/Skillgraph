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
    print(json.dumps(results["selected_model"], indent=2))
    print("Saved comparison metrics: models/job_category_model_comparison.json")
    print("Saved metadata: models/job_category_best_model_metadata.json")
    print("Saved model: models/job_category_best_model.joblib")
