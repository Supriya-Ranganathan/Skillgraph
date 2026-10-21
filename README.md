---
title: SkillGraph-X
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---


# SkillGraph-X

**Graph-Based Machine Learning for Dynamic Workforce Skill Gap Prediction**

A starter project with a polished responsive dashboard and a FastAPI backend. It is structured to grow into the rubric-aligned AML CAT II project.

## Project status

SkillGraph-X is a dataset-backed workforce skill analysis prototype. The processed dataset contains 1,167 job postings and 20,833 job-skill relationships. The existing FastAPI dashboard remains operational. Phase 2 includes a leakage-aware job-category classifier and saved benchmark artifacts. Phase 3.1/3.2 adds a heterogeneous job-skill graph and evaluated graph-neighborhood recommendation baselines. Phase 3.3 adds an experimental heterogeneous GraphSAGE link-prediction implementation and saved evaluation artifacts.

The current GraphSAGE model is an experimental artifact only. It is not yet integrated into the live recommendation endpoint, and it has not demonstrated superiority over the simpler frequency and co-occurrence baselines under the current compatible evaluation protocol. The current `/api/skill-gaps` endpoint remains frequency-based and is not replaced by the GraphSAGE experiment.

The final rubric report is in `FINAL_PROJECT_REPORT.md`. The current source UI adds a pastel-accent intelligence workspace, role profiles, a category heatmap, bounded-sample skill co-occurrence, local personal plans, a data story, and a paired recommender comparison. A read-only benchmark-summary API projects saved JSON. The separate grouped classifier ablation is validation-only and does not replace the protected model. On 2026-10-14, image `skillgraph-x:0.2` was built locally but not rolled out: the Docker Desktop Kubernetes cluster could not see the local image, and no supported image-import command was available. The existing deployment remains on `skillgraph-x:0.1`; see the verification record below.

## Features in this starter

- Responsive dashboard and navigation
- Target-role selector and current-skill input
- Skill coverage and missing-skill results
- Dataset-derived bipartite job-skill relationship graph
- FastAPI API with Pydantic request validation
- `/api/health`, `/api/roles`, `/api/top-skills`, `/api/skill-graph`, `/api/skill-gaps`, `/api/career-paths`, `/api/what-if`, and `/api/model-summary` endpoints
- `/api/graph-benchmark-summary` returns the saved paired five-seed ranking aggregates, paired differences, and protocol; it does not load model artifacts or recompute metrics.
- `/api/skill-gaps` additively returns `required_skills` with observed job counts and frequency for local checklist recalculation.
- Dockerfile for containerization
- Processed jobs/edge schema and graph-integrity validator
- Grouped validation-only XGBoost feature ablation, isolated from protected model outputs

## Skill Intelligence UI

The seven dashboard views are Overview, Skill Graph, Skill Intelligence, Skill Gaps, Career Paths, What-if Analysis, and Model Evaluation. The Intelligence workspace uses current API data to show a selected role's threshold-qualified skills, category-versus-skill overlap, and related categories ranked by Jaccard overlap. The relationships panel counts skills sharing returned job nodes in one bounded graph sample; it explicitly is not causation or a population estimate. A browser-local learning plan stores only user-selected skill goals and progress in local storage and can export JSON or CSV. The data story uses the dataset totals and saved classifier metrics. No salary, demand, certification, employment probability, account, LLM, or cloud-sync data is fabricated.

The Skill Gap checklist toggles the API-returned required skills and recomputes coverage locally; it does not add new model predictions. Career comparisons call the existing route endpoint independently for up to three targets and show an explicit empty state when no route is returned. What-if examples choose a skill from the selected role's API recommendations.

For a local source preview from the project root, use a separate port:

```powershell
python -m uvicorn --app-dir . app.main:app --host 127.0.0.1 --port 8004
```

This source preview is not the Kubernetes deployment. The existing deployment remains on `skillgraph-x:0.1` until a separately approved image build and rollout.

## Phase 3 graph artifacts and evaluation

The graph schema is heterogeneous: `job` nodes connect to `skill` nodes through `requires` edges, with the reverse `required_by` edge index included for traversal. Job category and title are preserved in the mapping JSON as descriptive metadata; neither is stored as a predictive node feature. The stable job and skill index ordering is lexicographic over original job IDs and normalized skill names.

The saved GraphSAGE experiment is present in `app/graphsage_link_prediction.py` and `app/run_graphsage_link_prediction.py`, along with checkpoint and evaluation outputs in `models/` and `visualizations/`. The current model uses trainable job and skill ID embeddings only; no validated job-title, category, or text feature matrix is incorporated. This is a deliberately simple representation and not yet a production feature pipeline.

Saved GraphSAGE evaluation results in `models/graphsage_link_prediction_evaluation.json` show that the model underperforms the same-split frequency and co-occurrence baselines at K=10 under the current protocol. For the same test split, the saved metrics are:

- GraphSAGE: Precision@10 = 0.0549, Recall@10 = 0.2650, NDCG@10 = 0.2025
- Frequency: Precision@10 = 0.0601, Recall@10 = 0.2909, NDCG@10 = 0.2331
- Co-occurrence: Precision@10 = 0.0618, Recall@10 = 0.2993, NDCG@10 = 0.2420

These values are evidence that the current GraphSAGE setup is experimental and inferior to the simpler baselines, not that it is ready for endpoint integration.

The independent five-seed paired results are in `models/paired_graph_benchmark_evaluation.json`. At K=10, GraphSAGE means are Precision 0.0508, Recall 0.2727, NDCG 0.2190; frequency scores 0.0550, 0.2970, 0.2390; co-occurrence scores 0.0570, 0.3067, 0.2493. This is transductive edge completion, not temporal forecasting or cold-start evaluation. The readable comparison plot is `visualizations/paired_graph_benchmark_k10_with_ci.png`.

## Processed data validation

Run from the project root:

```powershell
python -m app.dataset_validation --output reports/dataset_validation_report.json
```

The validator checks required columns, missing and blank values, field types, duplicate job IDs and normalized edges, graph edge consistency with the `skills` field, unknown job IDs, and edge title/category consistency. It does not rewrite input data and refuses to overwrite an existing report. The checked dataset report records 1,167 jobs, 4,984 unique skills, 20,833 edges, and zero detected missing values, duplicates, unknown jobs, or graph mismatches.

The raw directory contains `data/raw/train-00000-of-00001.parquet`, but no source, license, or citation metadata. Neither existing project Python environment has a Parquet reader available for footer inspection. Do not infer the provider or usage rights from the shard filename.

## Isolated classifier ablation

Run from the project root in an environment with the model-training dependencies:

```powershell
python -m app.run_classifier_feature_ablation
```

The experiment fits new XGBoost models only within three grouped folds of the outer training partition and writes `experiments/classifier_feature_ablation_v1.json` plus `visualizations/classifier_feature_ablation_v1.png`. It does not use held-out test rows for model selection and refuses to overwrite existing outputs. In the recorded scikit-learn 1.9.1 environment, validation macro-F1 means were 0.9542 with title+description+skills, 0.9701 without skills, and 0.9649 without description. These are validation-only results, not test metrics and not directly comparable with the canonical saved run created under scikit-learn 1.6.1. The saved model's test results were independently reproduced in the existing 1.6.1 environment without retraining.

Run the graph construction and recommendation benchmark from the project root with:

```powershell
python app/run_graph_baseline_evaluation.py
```

Generated artifacts:

- `data/processed/heterogeneous_skill_graph.npz` and `heterogeneous_skill_graph_mappings.json`: full graph edge indices and original-ID mappings.
- `data/processed/graph_baseline_training_graph.npz` and `graph_baseline_training_mappings.json`: training-only graph used for recommendation statistics.
- `models/graph_recommendation_baselines.json`: split description, metrics, and limitations.
- `visualizations/graph_recommendation_baselines.png`: Precision, Recall, and NDCG comparison at 10.

Evaluation holds out 20% of each job's unique skill edges with a fixed seed, retaining at least one training skill for every eligible job. Jobs with only one skill remain in training and are not evaluation cases. Frequency and two-hop skill co-occurrence baselines use the same cases and only training edges; candidates are skills observed in training. Known skills are excluded from recommendations. Metrics are macro-averaged per evaluation job: Precision@K divides by K, Recall@K divides by the number of held-out relevant skills, and NDCG@K uses binary relevance. K values are 5, 10, and 20.

This is a transductive edge-completion protocol: each evaluation job contributes its non-held-out skills to the training graph. Skills unseen in training cannot be recommended. Near-duplicate job postings are not grouped for this edge-level protocol, so related postings can still make evaluation optimistic. The active Python 3.11 environment has PyTorch installed and the project includes a PyTorch Geometric implementation for a heterogeneous GraphSAGE experiment, but the current model remains an experimental artifact rather than a production recommendation engine. The saved GraphSAGE evaluation demonstrates that it currently underperforms the same-split baselines, so it must not be integrated into the live recommendation endpoint until the evaluation protocol and feature design justify a meaningful improvement.

## Run locally (Windows)

1. Install Python 3.11 or later.
2. Open a terminal in the `app` directory.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```

4. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

5. Start the application:

   ```powershell
   uvicorn main:app --reload
   ```

6. Open:
   - Dashboard: http://127.0.0.1:8000
   - API docs: http://127.0.0.1:8000/docs
   - Health check: http://127.0.0.1:8000/api/health

## Run with Docker

From the project root (the Dockerfile copies files from the root build context):

```powershell
docker build -f app/Dockerfile -t skillgraph-x:0.2 .
docker run --rm -p 8000:8000 skillgraph-x:0.2
```

Then open http://127.0.0.1:8000.

## Docker Desktop Kubernetes verification

The checked-in deployment still uses image `skillgraph-x:0.1`, two replicas, container/service port 8000, and NodePort 30082. The service remains NodePort 30082. On 2026-10-14, the existing deployment was 2/2 Ready with two service endpoints. The new image was not deployed.

The project-root build completed successfully:

```powershell
docker build -f app/Dockerfile -t skillgraph-x:0.2 .
```

The local image ID is `sha256:557871b9cccac63c64f28c7479113438938672c47e53aea3af393f9989444830`. Docker Desktop reports this cluster in `kind` mode on Kubernetes 1.36.1 with a `containerd` runtime. A temporary pod using `imagePullPolicy: Never` failed with `ErrImageNeverPull`, proving the built image was not loaded into the cluster; that probe pod was deleted. The installed `docker desktop kubernetes` CLI only lists cluster images, and the `kind` CLI is not installed. The Deployment was not changed, so no rollback was needed. Do not apply `0.2` until it is available to the cluster runtime.

The existing service port-forward was active in a separate terminal at `127.0.0.1:8003`. Against the still-running `0.1` deployment, `GET /api/health` returned HTTP 200 with `status: ok` and `dataset_loaded: true`; `GET /` returned HTTP 200 HTML. These checks do not verify image `0.2`. Browser navigation on the local source preview at `127.0.0.1:8004` confirmed Overview, Skill Graph, Skill Intelligence, Skill Gaps, Career Paths, What-if Analysis, and Model Evaluation render with dataset-backed content.

The Phase 2 integrity manifest matched all five recorded artifact hashes. The GraphSAGE model/evaluation and paired benchmark evaluation, split, and per-job result hashes also matched their pre-build values.

Read-only status checks:

```powershell
kubectl config current-context
kubectl get nodes
kubectl get deployments,pods,services
kubectl get endpoints skillgraph-x-service
```

For local HTTP verification, run the port-forward in a separate terminal and leave it active:

```powershell
kubectl port-forward service/skillgraph-x-service 8003:8000
```

Then request `http://127.0.0.1:8003/api/health` and `http://127.0.0.1:8003/`. Do not reset Docker Desktop or delete cluster resources as a troubleshooting step. For rollback only after a failed rollout, inspect `kubectl rollout history deployment/skillgraph-x` and, when a previous revision exists, use `kubectl rollout undo deployment/skillgraph-x`.

## Project structure

```text
SkillGraph-X/
├── app/
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── static/
│       ├── index.html
│       ├── styles.css
│       └── app.js
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── experiments/
├── visualizations/
└── README.md
```

## Remaining work and limitations

1. Recover and document the dataset's authoritative source, license, collection date, and citation; they are not present in the workspace.
2. The feature ablation is limited to three grouped validation folds and ran under a different scikit-learn version from the protected model's original environment. Re-run only as a separately named experiment in a pinned, compatible environment if needed; do not overwrite canonical artifacts.
3. Add genuinely separated temporal and unseen-job evaluations only if the source data contains valid timestamps and unseen-job cases.
4. GraphSAGE remains experimental because its paired ranking metrics trail both simple baselines; its ID-only transductive representation is not evidence of inductive/cold-start performance.
5. Current test and deployment/browser evidence are recorded in `FINAL_PROJECT_REPORT.md`; preserve those outputs and do not claim a Docker image build was run unless a build log is captured.

## API example

```powershell
$body = @{
  role = "Machine Learning Engineer"
  skills = @("Python", "SQL", "Statistics")
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/api/skill-gaps `
  -Method Post -ContentType "application/json" -Body $body
```
