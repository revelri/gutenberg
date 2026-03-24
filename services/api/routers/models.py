"""OpenAI-compatible /v1/models endpoint."""

import time

from fastapi import APIRouter

from core.config import collection_routes

router = APIRouter()


@router.get("/v1/models")
async def list_models():
    """Return available models (OpenAI API compatible).

    Lists the default gutenberg-rag model plus one model per
    configured collection route (e.g., gutenberg-rag/macy).
    """
    now = int(time.time())
    models = [
        {
            "id": "gutenberg-rag",
            "object": "model",
            "created": now,
            "owned_by": "gutenberg",
        }
    ]

    for slug in sorted(collection_routes):
        models.append({
            "id": f"gutenberg-rag/{slug}",
            "object": "model",
            "created": now,
            "owned_by": "gutenberg",
        })

    return {"object": "list", "data": models}
