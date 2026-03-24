"""Gutenberg API — OpenAI-compatible RAG endpoint."""

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import chat, models, documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

app = FastAPI(title="Gutenberg API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(models.router)
app.include_router(documents.router)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    import httpx
    from core.config import settings

    checks = {}

    # Check Ollama
    try:
        r = httpx.get(f"{settings.ollama_host}/api/version", timeout=5)
        checks["ollama"] = "ok" if r.status_code == 200 else "error"
    except Exception:
        checks["ollama"] = "error"

    # Check ChromaDB
    try:
        r = httpx.get(f"{settings.chroma_host}/api/v1/heartbeat", timeout=5)
        checks["chromadb"] = "ok" if r.status_code == 200 else "error"
    except Exception:
        checks["chromadb"] = "error"

    status = "healthy" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": status, "checks": checks}
