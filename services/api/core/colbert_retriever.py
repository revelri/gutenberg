"""ColBERTv2 late-interaction retriever (via RAGatouille).

Runs in parallel with the dense + BM25 retrievers; its ranked list is fused
with the others via RRF. Typically lifts recall on rare/long-tail terminology
that mean-pooled dense embeddings flatten (German/French philosophical
vocabulary, proper names with low corpus frequency).

Index lifecycle:
  * Built by ``scripts/reindex.py`` (P11) from the chunk corpus, rooted at
    ``settings.colbert_index_root``.
  * Loaded lazily on first retrieve call (threadsafe via module-level lock).
  * One index per Chroma collection name.

Flag-gated via ``settings.feature_colbert_retrieval``. If RAGatouille is not
installed, returns []. Caller (``rag.retrieve``) treats missing ColBERT
results as a no-op and continues with dense + BM25.
"""

from __future__ import annotations

import logging
import os
import threading

from core.config import settings

log = logging.getLogger("gutenberg.colbert")

_INDICES: dict[str, object] = {}
_LOCK = threading.Lock()


def _index_path_for(collection: str | None) -> str:
    col_key = collection or settings.chroma_collection
    return os.path.join(settings.colbert_index_root, col_key)


def _load(collection: str | None):
    if not settings.feature_colbert_retrieval:
        return None
    col_key = collection or settings.chroma_collection
    if col_key in _INDICES:
        return _INDICES[col_key]

    with _LOCK:
        if col_key in _INDICES:
            return _INDICES[col_key]
        path = _index_path_for(collection)
        if not os.path.isdir(path):
            log.info(f"ColBERT index not built for '{col_key}' at {path}")
            _INDICES[col_key] = None
            return None
        try:
            from ragatouille import RAGPretrainedModel
            idx = RAGPretrainedModel.from_index(path)
            _INDICES[col_key] = idx
            log.info(f"ColBERT index loaded for '{col_key}'")
            return idx
        except Exception as e:
            log.warning(f"ColBERT index load failed for '{col_key}': {e}")
            _INDICES[col_key] = None
            return None


def colbert_search(
    query: str, top_k: int, collection: str | None = None
) -> list[dict]:
    """Return ColBERT top-k results or [] if disabled/unavailable.

    Each result dict mirrors the shape returned by ``_dense_search`` /
    ``_bm25_search``: ``{"id", "text", "metadata", "colbert_score"}``.
    """
    idx = _load(collection)
    if idx is None:
        return []
    try:
        hits = idx.search(query=query, k=top_k)
    except Exception as e:
        log.warning(f"ColBERT search failed: {e}")
        return []

    out: list[dict] = []
    for h in hits:
        out.append(
            {
                "id": h.get("document_id") or h.get("id") or "",
                "text": h.get("content") or h.get("document_text") or "",
                "metadata": h.get("document_metadata") or {},
                "colbert_score": float(h.get("score", 0.0)),
            }
        )
    return out


def build_index(
    corpus: list[dict],
    collection: str | None = None,
) -> bool:
    """Build a ColBERT index from a chunk corpus.

    ``corpus`` items: ``{"id", "text", "metadata", "contextual_text"?}``.
    Contextual text (P0) is preferred so the ColBERT embeddings see the same
    enriched text as the dense index.
    """
    if not settings.feature_colbert_retrieval:
        return False
    try:
        from ragatouille import RAGPretrainedModel
    except ImportError:
        log.warning("RAGatouille not installed; skipping ColBERT index build")
        return False

    docs = [c.get("contextual_text") or c["text"] for c in corpus]
    ids = [c["id"] for c in corpus]
    metas = [c.get("metadata") or {} for c in corpus]

    path = _index_path_for(collection)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    model = RAGPretrainedModel.from_pretrained(settings.colbert_retriever_model)
    model.index(
        collection=docs,
        document_ids=ids,
        document_metadatas=metas,
        index_name=os.path.basename(path),
        max_document_length=384,
        split_documents=False,
    )
    log.info(f"ColBERT index built for '{collection}' → {path}")
    return True
