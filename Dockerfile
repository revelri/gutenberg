# Stage 1: Build SvelteKit frontend
FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python API + static frontend
FROM python:3.13-slim
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY services/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/api/ .
COPY services/shared/ ./shared/
COPY --from=frontend /app/frontend/build ./static

RUN mkdir -p /data/inbox /data/processing /data/processed /data/failed /data/state

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
