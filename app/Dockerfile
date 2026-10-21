FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /workspace/app
COPY app/requirements-container.txt ./requirements-container.txt
RUN pip install --no-cache-dir -r requirements-container.txt

COPY app/main.py ./main.py
COPY app/skillgraph_core.py ./skillgraph_core.py
COPY app/static/ ./static/
COPY data/processed/job_skill_edges.csv /workspace/data/processed/job_skill_edges.csv
COPY models/job_category_model_comparison.json models/job_category_best_model_metadata.json models/paired_graph_benchmark_evaluation.json /workspace/models/

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
