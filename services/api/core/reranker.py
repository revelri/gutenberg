"""BGE Reranker wrapper — GPU-accelerated cross-encoder."""

import logging

import torch
from sentence_transformers import CrossEncoder

from core.config import settings

log = logging.getLogger("gutenberg.reranker")

_model = None


def _get_device() -> str:
    # Reranker stays on CPU to leave GPU VRAM for Ollama LLM inference.
    # BGE reranker uses ~3GB VRAM which starves the 8B LLM on an 8GB GPU.
    return "cpu"


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        device = _get_device()
        log.info(f"Loading reranker model: {settings.reranker_model} on {device}")
        _model = CrossEncoder(settings.reranker_model, device=device)
        log.info("Reranker model loaded")
    return _model


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank chunks by relevance to query using cross-encoder.

    Each chunk dict must have a 'text' key.
    Returns top_k chunks sorted by relevance score.
    """
    if not chunks:
        return []

    model = _get_model()
    pairs = [[query, c["text"]] for c in chunks]
    scores = model.predict(pairs)

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
