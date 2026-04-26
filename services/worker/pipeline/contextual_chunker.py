"""Anthropic-style contextual chunking with prompt caching.

For each document, we generate a short (~60–100 token) context prefix per chunk
that grounds it in the document's rhetorical flow. The prefix is prepended to
the chunk text for embedding AND BM25 tokenization. The original chunk text is
preserved verbatim on every chunk so exact-match citation verification still
works (this is a hard invariant).

Cost is controlled by Anthropic ephemeral prompt caching: the full document
body is cached once, and each per-chunk call pays only the marginal
(instruction + chunk) tokens.

Falls back to Ollama when Anthropic is unavailable. Fails open (returns
original chunks unchanged) if both providers fail — retrieval still works,
we just lose the contextual-prefix lift.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

try:
    from core.config import settings  # API-process import
except Exception:  # worker uses a different sys.path
    from services.api.core.config import settings  # type: ignore

log = logging.getLogger("gutenberg.contextual_chunker")

_SYSTEM_PROMPT = (
    "You help make scholarly text chunks more retrievable. For each CHUNK from "
    "the DOCUMENT, write ONE short sentence (max 35 words) that places the chunk "
    "in the document's argument. Preserve proper names, foreign terms, and page "
    "anchors verbatim. Do not summarize the chunk. Do not quote it. Do not add "
    "editorial commentary. Output ONLY the single context sentence."
)


def _cache_key(source: str, chunk_index: int, chunk_text: str) -> str:
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(f"|{chunk_index}|".encode("utf-8"))
    h.update(chunk_text.encode("utf-8"))
    h.update(f"|{settings.anthropic_model}|".encode("utf-8"))
    return h.hexdigest()


def _read_cache(key: str) -> str | None:
    cache_dir = Path(settings.contextual_cache_dir)
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("prefix")
    except Exception:
        return None


def _write_cache(key: str, prefix: str) -> None:
    cache_dir = Path(settings.contextual_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(json.dumps({"prefix": prefix}))


def _anthropic_describe(document: str, chunks: list[dict]) -> list[str | None]:
    """Describe every chunk via Anthropic with ephemeral cache on the document."""
    api_key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return [None] * len(chunks)

    try:
        import anthropic
    except ImportError:
        log.warning("anthropic SDK not installed; contextual enrichment disabled")
        return [None] * len(chunks)

    client = anthropic.Anthropic(api_key=api_key)
    prefixes: list[str | None] = []

    system = [
        {"type": "text", "text": _SYSTEM_PROMPT},
        {
            "type": "text",
            "text": f"<document>\n{document}\n</document>",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    for chunk in chunks:
        try:
            msg = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=settings.contextual_prefix_max_tokens,
                system=system,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"<chunk>\n{chunk['text']}\n</chunk>\n"
                            "Respond with the single context sentence only."
                        ),
                    }
                ],
            )
            parts = getattr(msg, "content", [])
            text = "".join(p.text for p in parts if getattr(p, "type", "") == "text").strip()
            prefixes.append(text or None)
        except Exception as e:
            log.warning(f"Anthropic contextual call failed: {e}")
            prefixes.append(None)

    return prefixes


def _ollama_describe(document: str, chunk_text: str) -> str | None:
    import httpx

    truncated_doc = document[:20000]  # Ollama has no prompt cache
    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"<document>\n{truncated_doc}\n</document>\n\n"
        f"<chunk>\n{chunk_text}\n</chunk>\n"
        "Respond with the single context sentence only."
    )
    try:
        r = httpx.post(
            f"{settings.ollama_host}/api/generate",
            json={
                "model": settings.ollama_llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": settings.contextual_prefix_max_tokens,
                },
            },
            timeout=60,
        )
        r.raise_for_status()
        return (r.json().get("response") or "").strip() or None
    except Exception as e:
        log.warning(f"Ollama contextual fallback failed: {e}")
        return None


def enrich_chunks(document: str, chunks: list[dict]) -> list[dict]:
    """Add ``contextual_text`` and ``context_prefix`` to every chunk.

    Invariants:
      * chunk["text"] is unchanged (verification source of truth)
      * chunk["contextual_text"] = prefix + "\n\n" + chunk["text"] (embed/BM25 source)
      * chunk["metadata"]["context_prefix"] stored for audit
      * On any failure, ``contextual_text`` falls back to ``chunk["text"]``
    """
    if not chunks or not settings.feature_contextual_chunking:
        for c in chunks:
            c.setdefault("contextual_text", c["text"])
        return chunks

    # Cache lookup first (path-independent, content-hashed).
    source = (chunks[0].get("metadata") or {}).get("source", "")
    keys = [_cache_key(source, i, c["text"]) for i, c in enumerate(chunks)]
    cached = [_read_cache(k) for k in keys]
    needs = [i for i, v in enumerate(cached) if v is None]

    if needs:
        # Try Anthropic first for the uncached ones.
        to_describe = [chunks[i] for i in needs]
        prefixes = _anthropic_describe(document, to_describe)
        for local_idx, prefix in enumerate(prefixes):
            i = needs[local_idx]
            if prefix:
                cached[i] = prefix
                _write_cache(keys[i], prefix)

        # Ollama fallback for anything still missing.
        if settings.contextual_fallback == "ollama":
            for i, v in enumerate(cached):
                if v is None:
                    fb = _ollama_describe(document, chunks[i]["text"])
                    if fb:
                        cached[i] = fb
                        _write_cache(keys[i], fb)

    for c, prefix in zip(chunks, cached):
        ctx = prefix or ""
        c["context_prefix"] = ctx
        c["contextual_text"] = f"{ctx}\n\n{c['text']}" if ctx else c["text"]
        if "metadata" in c and ctx:
            c["metadata"]["context_prefix"] = ctx[:500]  # bounded for Chroma metadata

    enriched = sum(1 for p in cached if p)
    log.info(f"Contextual enrichment: {enriched}/{len(chunks)} chunks enriched")
    return chunks
