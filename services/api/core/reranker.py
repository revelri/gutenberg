"""BGE Reranker wrapper — GPU-accelerated cross-encoder."""

import logging

import torch
from sentence_transformers import CrossEncoder

from core.config import settings

log = logging.getLogger("gutenberg.reranker")

_model = None
_model_loaded: bool = False
_load_error: str | None = None


def _get_device() -> str:
    # Reranker stays on CPU to leave GPU VRAM for Ollama LLM inference.
    # BGE reranker uses ~3GB VRAM which starves the 8B LLM on an 8GB GPU.
    return "cpu"


def _get_model() -> CrossEncoder | None:
    """Load the reranker model with error handling.

    Returns None if model loading fails, allowing graceful fallback.
    """
    global _model, _model_loaded, _load_error

    if _model is None and not _model_loaded:
        device = _get_device()
        try:
            log.info(f"Loading reranker model: {settings.reranker_model} on {device}")
            _model = CrossEncoder(settings.reranker_model, device=device)
            _model_loaded = True
            log.info("Reranker model loaded")
        except (OSError, RuntimeError, Exception) as e:
            _load_error = str(e)
            log.warning(f"Failed to load reranker model: {e}")
            _model_loaded = True  # Mark as attempted (even if failed)

    return _model


def preload_model() -> None:
    """Eagerly load the reranker model. Call at startup."""
    _get_model()


def is_available() -> bool:
    """Check if the reranker model is available and loaded."""
    return _model_loaded and _load_error is None and _model is not None


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank chunks by relevance to query using cross-encoder.

    Each chunk dict must have a 'text' key.
    Returns top_k chunks sorted by relevance score.

    If the reranker model is unavailable, falls back to returning chunks
    sorted by their existing rerank_score (from RRF), or as-is if no scores.
    """
    if not chunks:
        return []

    model = _get_model()
    if model is None:
        log.warning("Reranker model unavailable, returning RRF-ranked results")
        # Fall back to existing rerank_score from RRF, or return as-is
        if all("rerank_score" in c for c in chunks):
            ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
            return ranked[:top_k]
        return chunks[:top_k]

    pairs = [[query, c["text"]] for c in chunks]
    scores = model.predict(pairs)

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
