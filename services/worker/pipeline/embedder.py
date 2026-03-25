"""Embedding via Ollama API."""

import logging
import os
import re

import httpx

log = logging.getLogger("gutenberg.embedder")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
BATCH_SIZE = 8

# Model-specific context limits (chars, conservative ~4 chars/token estimate).
# Models with short context windows need aggressive truncation.
_MODEL_MAX_CHARS = {
    "nomic-embed-text": 6000,
    "nomic-embed-text-v2-moe": 6000,
    "mxbai-embed-large": 1800,       # 512-token context
    "all-minilm": 1800,              # 512-token context
    "snowflake-arctic-embed": 1800,
    "qwen3-embedding": 32000,        # 32K-token context
    "qwen3-embedding:4b": 32000,
    "qwen3-embedding:0.6b": 32000,
    "qwen3-embedding:8b": 32000,
}
MAX_CHARS = _MODEL_MAX_CHARS.get(EMBED_MODEL, 6000)

# Detected embedding dimensions (populated on first successful call)
_embed_dim: int | None = None


def _clean_for_embedding(text: str) -> str:
    """Clean text to reduce token count for embedding models.

    BERT-based tokenizers tokenize each punctuation char separately,
    so repeated dots/dashes in table-of-contents formatting explode
    token counts.
    """
    text = text.strip()
    if not text:
        return "[empty]"
    # Collapse repeated dots (table of contents: "Chapter 1 ......... 5")
    text = re.sub(r'\.{3,}', '...', text)
    # Collapse repeated dashes
    text = re.sub(r'-{3,}', '---', text)
    # Collapse repeated underscores
    text = re.sub(r'_{3,}', '___', text)
    # Collapse repeated spaces
    text = re.sub(r' {3,}', '  ', text)
    # Truncate to safe length
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    return text


def embed_chunks(texts: list[str]) -> list[list[float]]:
    """Embed a list of text chunks via Ollama.

    Returns list of embedding vectors.
    """
    global _embed_dim
    cleaned = [_clean_for_embedding(t) for t in texts]
    all_embeddings = []

    for i in range(0, len(cleaned), BATCH_SIZE):
        batch = cleaned[i : i + BATCH_SIZE]
        resp = httpx.post(
            f"{OLLAMA_HOST}/api/embed",
            json={"model": EMBED_MODEL, "input": batch},
            timeout=120,
        )
        if resp.status_code != 200:
            log.error(f"Embed failed (batch {i // BATCH_SIZE + 1}): {resp.status_code} {resp.text[:500]}")
            # Fallback: try one at a time with aggressive truncation
            for j, item in enumerate(batch):
                single_resp = httpx.post(
                    f"{OLLAMA_HOST}/api/embed",
                    json={"model": EMBED_MODEL, "input": [item[:min(MAX_CHARS, 1500)]]},
                    timeout=120,
                )
                if single_resp.status_code != 200:
                    log.error(f"Single embed also failed for chunk {i + j}, using zero vector")
                    dim = _embed_dim or 768
                    all_embeddings.append([0.0] * dim)
                else:
                    embs = single_resp.json()["embeddings"]
                    if _embed_dim is None and embs:
                        _embed_dim = len(embs[0])
                    all_embeddings.extend(embs)
            continue
        data = resp.json()
        embs = data["embeddings"]
        if _embed_dim is None and embs:
            _embed_dim = len(embs[0])
        all_embeddings.extend(embs)

    return all_embeddings
