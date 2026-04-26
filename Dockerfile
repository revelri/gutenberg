# Stage 1: Build SvelteKit frontend
FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python image hosting BOTH api and worker.
# Services share the image; docker-compose sets the per-service command
# and working directory.
FROM python:3.13-slim
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gcc \
      # ocrmypdf + tesseract for scanned-PDF preprocessing (worker);
      # skipped if absent, so harmless for digital-only corpora.
      ocrmypdf tesseract-ocr tesseract-ocr-eng unpaper \
      libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install API deps (superset — includes sentence-transformers/chromadb
# used by both services)
COPY services/api/requirements.txt /tmp/api-requirements.txt
COPY services/worker/requirements.txt /tmp/worker-requirements.txt
RUN pip install --no-cache-dir -r /tmp/api-requirements.txt \
    && pip install --no-cache-dir python-multipart watchdog docling \
    && python -m spacy download en_core_web_sm

# Layout:
#   /app/api/      ← services/api/       (uvicorn main:app)
#   /app/worker/   ← services/worker/    (python -m pipeline.watcher)
#   /app/shared/   ← services/shared/    (importable as `shared.*`)
COPY services/api/ /app/api/
COPY services/worker/ /app/worker/
COPY services/shared/ /app/shared/
COPY --from=frontend /app/frontend/build /app/api/static

RUN mkdir -p /data/inbox /data/processing /data/processed /data/failed /data/state

# PYTHONPATH lets both services import `shared.*` and `core.*` / `pipeline.*`
ENV PYTHONPATH=/app:/app/api:/app/worker

EXPOSE 8000
# Default command targets the API. Override in docker-compose for the worker.
WORKDIR /app/api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
