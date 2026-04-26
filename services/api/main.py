"""Gutenberg API — RAG citation retrieval with multi-corpus support."""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.database import init_db, set_db_path, auto_populate_exemplar
from core.auth import auth_backend, fastapi_users
from core.schemas import UserCreate, UserRead, UserUpdate
from routers import chat, models, documents, corpus, conversations, pdf, search

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

    # P11: index manifest consistency check (opt-in, guarded by flag)
    if settings.enforce_index_manifest:
        try:
            import json as _json
            from pathlib import Path as _Path
            _path = _Path(settings.index_manifest_path)
            if not _path.exists():
                raise RuntimeError(f"index manifest missing: {_path}")
            _manifest = _json.loads(_path.read_text())
            _saved = _manifest.get("flags", {})
            _flags = [
                "feature_contextual_chunking", "feature_entity_gazetteer",
                "feature_modal_chunks", "feature_colbert_retrieval",
                "feature_raptor", "feature_graph_boost",
            ]
            _diff = {
                k: (_saved.get(k), bool(getattr(settings, k)))
                for k in _flags
                if _saved.get(k) != bool(getattr(settings, k))
            }
            if _diff:
                raise RuntimeError(f"index/flag mismatch: {_diff} — run scripts/reindex.py")
        except RuntimeError:
            raise
        except Exception as _e:
            log.warning(f"manifest check failed (non-fatal): {_e}")

    log.info("Gutenberg API ready")
    yield


app = FastAPI(title="Gutenberg API", version="0.2.0", lifespan=lifespan)

_cors_origins = os.environ.get("CORS_ORIGINS", "").strip()
_allowed_origins = (
    [o.strip() for o in _cors_origins.split(",") if o.strip()]
    if _cors_origins
    else ["http://localhost:*", "http://127.0.0.1:*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept", "X-Requested-With"],
)

# Auth routes (Phase 2 — JWT bearer + email/password registration)
app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_reset_password_router(),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_verify_router(UserRead),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)

# Existing routers (backward-compatible)
app.include_router(chat.router)
app.include_router(models.router)
app.include_router(documents.router)

# New beta routers
app.include_router(corpus.router)
app.include_router(conversations.router)
app.include_router(pdf.router)
app.include_router(search.router)


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
