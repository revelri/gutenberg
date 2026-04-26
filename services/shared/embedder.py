"""In-process embedding via sentence-transformers.

Replaces Ollama HTTP calls for embedding. Loads the model once at first use
and keeps it in memory.

Set EMBED_MODEL env var to match the model used for your ChromaDB collection.
Default: Alibaba-NLP/gte-large-en-v1.5 (1024-dim, strong retrieval, no prefix needed).
"""

import logging
import os

log = logging.getLogger("gutenberg.embedder")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "Alibaba-NLP/gte-large-en-v1.5")
# Pin model snapshot when using trust_remote_code to prevent upstream code changes
# from executing unvetted. Set EMBED_MODEL_REVISION to a commit SHA or tag.
EMBED_MODEL_REVISION = os.environ.get("EMBED_MODEL_REVISION", "").strip() or None
# Models with custom architecture (e.g. gte-large) require trust_remote_code.
# Default off for safety; opt in explicitly per deployment.
EMBED_TRUST_REMOTE_CODE = os.environ.get("EMBED_TRUST_REMOTE_CODE", "true").lower() in ("1", "true", "yes")
BATCH_SIZE = 32

# Models that require task-specific prefixes
_PREFIX_MODELS = {
    "nomic-ai/nomic-embed-text-v1.5": ("search_document: ", "search_query: "),
    "intfloat/e5-large-v2": ("passage: ", "query: "),
    "intfloat/e5-base-v2": ("passage: ", "query: "),
    "intfloat/multilingual-e5-large": ("passage: ", "query: "),
}

# Lazy-loaded model singleton
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        kwargs = {"trust_remote_code": EMBED_TRUST_REMOTE_CODE}
        if EMBED_MODEL_REVISION:
            kwargs["revision"] = EMBED_MODEL_REVISION
        if EMBED_TRUST_REMOTE_CODE and not EMBED_MODEL_REVISION:
            log.warning(
                "EMBED_TRUST_REMOTE_CODE=true without EMBED_MODEL_REVISION — "
                "upstream HF repo changes will execute unvetted. "
                "Pin EMBED_MODEL_REVISION to a commit SHA before shipping."
            )
        _model = SentenceTransformer(EMBED_MODEL, **kwargs)
        log.info(
            f"Loaded embedding model: {EMBED_MODEL} "
            f"(revision={EMBED_MODEL_REVISION or 'HEAD'}, "
            f"trust_remote_code={EMBED_TRUST_REMOTE_CODE}, "
            f"dim={_model.get_sentence_embedding_dimension()})"
        )
    return _model


def _get_prefixes() -> tuple[str, str]:
    """Get (document_prefix, query_prefix) for the current model."""
    return _PREFIX_MODELS.get(EMBED_MODEL, ("", ""))


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, returning vectors.

    Handles batching internally. Returns list of float lists.
    """
    if not texts:
        return []

    model = _get_model()
    doc_prefix, _ = _get_prefixes()

    if doc_prefix:
        texts = [f"{doc_prefix}{t}" for t in texts]

    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    model = _get_model()
    _, query_prefix = _get_prefixes()

    if query_prefix:
        text = f"{query_prefix}{text}"

    embedding = model.encode(text, show_progress_bar=False)
    return embedding.tolist()
