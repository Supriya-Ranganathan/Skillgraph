
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /workspace/app

COPY app/requirements-container.txt ./requirements-container.txt
RUN pip install --no-cache-dir -r requirements-container.txt

COPY app/main.py ./main.py
COPY app/skillgraph_core.py ./skillgraph_core.py
COPY app/skill_gap_analysis.py ./skill_gap_analysis.py
COPY app/static/ ./static/

WORKDIR /workspace

COPY data/processed/job_skill_edges.csv ./data/processed/job_skill_edges.csv
COPY models/ ./models/

WORKDIR /workspace/app

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
