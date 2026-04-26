"""Embedding via sentence-transformers (in-process, no Ollama dependency)."""

import logging

from shared.embedder import embed_texts

log = logging.getLogger("gutenberg.embedder")


def embed_chunks(texts: list[str]) -> list[list[float]]:
    """Embed a list of text chunks.

    Returns list of embedding vectors.
    """
    embeddings = embed_texts(texts)
    log.info(f"Embedded {len(embeddings)} chunks")
    return embeddings
