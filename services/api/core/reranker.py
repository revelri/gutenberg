"""BGE Reranker wrapper — CPU-only cross-encoder."""

import logging

from sentence_transformers import CrossEncoder

from core.config import settings

log = logging.getLogger("gutenberg.reranker")

_model = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        log.info(f"Loading reranker model: {settings.reranker_model}")
        _model = CrossEncoder(settings.reranker_model, device="cpu")
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
