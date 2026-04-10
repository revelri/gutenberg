"""Gutenborg API — RAG citation retrieval with multi-corpus support."""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.database import init_db, set_db_path, auto_populate_exemplar
from routers import chat, models, documents, corpus, conversations, pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("gutenberg.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and auto-populate exemplar corpus on startup."""
    set_db_path(settings.database_path)
    await init_db()
    await auto_populate_exemplar(settings.chroma_host)
    log.info("Gutenborg API ready")
    yield


app = FastAPI(title="Gutenborg API", version="0.2.0", lifespan=lifespan)

_cors_origins = os.environ.get("CORS_ORIGINS", "").strip()
_allowed_origins = (
    [o.strip() for o in _cors_origins.split(",") if o.strip()]
    if _cors_origins
    else ["http://localhost:*", "http://127.0.0.1:*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Existing routers (backward-compatible)
app.include_router(chat.router)
app.include_router(models.router)
app.include_router(documents.router)

# New beta routers
app.include_router(corpus.router)
app.include_router(conversations.router)
app.include_router(pdf.router)


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
        r = httpx.get(f"{settings.chroma_host}/api/v2/heartbeat", timeout=5)
        checks["chromadb"] = "ok" if r.status_code == 200 else "error"
    except Exception:
        checks["chromadb"] = "error"

    status = "healthy" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": status, "checks": checks}


# Serve built SvelteKit frontend (must be LAST — catches all unmatched routes)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")
