"""RAPTOR — recursive clustering + abstractive summaries.

Builds a hierarchical tree over the leaf chunks of a document and emits each
interior node as a synthetic chunk with ``metadata.level > 0`` and
``metadata.children_ids`` populated. Summary nodes are indexed in the same
ChromaDB + BM25 infrastructure as leaves; the retriever treats them as
first-class candidates.

Abstract/conceptual queries (queries without proper names) retrieve summaries
better than leaves because leaves rarely contain the abstract framing that
summaries express.

Flag-gated via ``settings.feature_raptor``. Falls back to skip-build on any
failure (clusterer missing, LLM unavailable, etc.).

Provider dispatch:

  * ``settings.raptor_provider="ollama"`` (default) — serial ``/api/generate``
    calls against the local model. Free, slow (~1–2 s per cluster).
  * ``settings.raptor_provider="openrouter"`` — async batched OpenRouter calls
    with ``settings.raptor_concurrency`` concurrent requests; requires
    ``OPENROUTER_API_KEY`` or ``OPENROUTER_KEY`` in env. ~30× faster on a
    large corpus.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import uuid

import httpx

try:
    from core.config import settings
except Exception:
    from services.api.core.config import settings  # type: ignore

log = logging.getLogger("gutenberg.raptor")


_PROMPT = (
    "Summarize the passage below in 3 sentences. Preserve proper names, "
    "foreign terms, and the core argument. Do not add commentary.\n\n"
)


def _kmeans(vectors: list[list[float]], k: int) -> list[int]:
    """Tiny NumPy-free k-means. Deterministic init, 20 iter max."""
    if not vectors:
        return []
    import random

    dim = len(vectors[0])
    random.seed(42)
    centroids = [vectors[i] for i in random.sample(range(len(vectors)), min(k, len(vectors)))]
    labels = [0] * len(vectors)

    def _dist(a: list[float], b: list[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b))

    for _ in range(20):
        changed = False
        for i, v in enumerate(vectors):
            best = min(range(len(centroids)), key=lambda c: _dist(v, centroids[c]))
            if labels[i] != best:
                labels[i] = best
                changed = True
        sums = [[0.0] * dim for _ in centroids]
        counts = [0] * len(centroids)
        for v, lbl in zip(vectors, labels):
            for j in range(dim):
                sums[lbl][j] += v[j]
            counts[lbl] += 1
        for c in range(len(centroids)):
            if counts[c] > 0:
                centroids[c] = [s / counts[c] for s in sums[c]]
        if not changed:
            break
    return labels


# ── Summarizer backends ────────────────────────────────────────────────

def _ollama_summarize(text: str) -> str:
    prompt = f"{_PROMPT}{text[:6000]}"
    try:
        r = httpx.post(
            f"{settings.ollama_host}/api/generate",
            json={
                "model": settings.ollama_llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 240},
            },
            timeout=90,
        )
        r.raise_for_status()
        return (r.json().get("response") or "").strip()
    except Exception as e:
        log.warning(f"raptor ollama summarize failed: {e}")
        return ""


async def _openrouter_one(client, sem, text: str, model: str, api_key: str) -> str:
    import re as _re

    prompt = f"{_PROMPT}{text[:6000]}"
    async with sem:
        for attempt in range(3):
            try:
                r = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "HTTP-Referer": "https://github.com/revelri/gutenborg",
                        "X-Title": "gutenborg-raptor",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 240,
                        "temperature": 0.1,
                    },
                    timeout=120,
                )
                r.raise_for_status()
                out = (
                    r.json()
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    or ""
                ).strip()
                out = _re.sub(r"<think>.*?</think>", "", out, flags=_re.DOTALL).strip()
                out = out.split("<think>")[0].strip()
                return out
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                log.warning(f"raptor OR http {e.response.status_code}: {e}")
                return ""
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                log.warning(f"raptor OR call failed: {e}")
                return ""
    return ""


def _openrouter_batch(
    texts: list[str],
    model: str,
    api_key: str,
    concurrency: int,
) -> list[str]:
    """Synchronous wrapper — dispatches all summaries concurrently."""
    async def _runner() -> list[str]:
        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(http2=False) as client:
            tasks = [
                _openrouter_one(client, sem, t, model, api_key) for t in texts
            ]
            return await asyncio.gather(*tasks, return_exceptions=False)

    try:
        return asyncio.run(_runner())
    except RuntimeError:
        # Already inside an event loop — create a new one in a thread.
        import threading

        result: list[list[str]] = [[]]
        def _worker() -> None:
            result[0] = asyncio.new_event_loop().run_until_complete(_runner())
        t = threading.Thread(target=_worker)
        t.start()
        t.join()
        return result[0]


def _summarize_level(texts: list[str]) -> list[str]:
    """Summarize a list of cluster texts. Order-preserving."""
    if not texts:
        return []

    provider = (getattr(settings, "raptor_provider", None) or "ollama").lower()
    if provider == "openrouter":
        api_key = (
            getattr(settings, "openrouter_api_key", "")
            or os.environ.get("OPENROUTER_API_KEY", "")
            or os.environ.get("OPENROUTER_KEY", "")
        )
        model = (
            getattr(settings, "raptor_openrouter_model", None)
            or "google/gemini-2.5-flash-lite"
        )
        concurrency = int(getattr(settings, "raptor_concurrency", 40) or 40)
        if not api_key:
            log.warning(
                "raptor_provider=openrouter but no OPENROUTER_API_KEY; "
                "falling back to Ollama"
            )
        else:
            log.info(
                f"RAPTOR level summarize via OpenRouter: "
                f"{len(texts)} clusters @ concurrency={concurrency}, model={model}"
            )
            return _openrouter_batch(texts, model, api_key, concurrency)

    # Fallback: serial Ollama.
    return [_ollama_summarize(t) for t in texts]


# ── Tree build ─────────────────────────────────────────────────────────

def build_tree(
    leaves: list[dict],
    leaf_embeddings: list[list[float]],
) -> list[dict]:
    """Return summary chunks (level>=1) built from ``leaves``.

    Only the new summary chunks are returned — the caller is responsible for
    concatenating them with the leaf list. Each summary carries a fresh UUID
    in ``metadata.chunk_id`` and a ``children_ids`` list (comma-joined for
    ChromaDB metadata compatibility).
    """
    if not settings.feature_raptor or not leaves:
        return []

    if len(leaves) != len(leaf_embeddings):
        log.warning("raptor: leaves/embeddings length mismatch, skipping")
        return []

    for c in leaves:
        meta = c.setdefault("metadata", {})
        meta.setdefault("chunk_id", str(uuid.uuid4()))
        meta.setdefault("level", 0)

    summaries: list[dict] = []
    current = list(zip(leaves, leaf_embeddings))

    for level in range(1, settings.raptor_max_levels + 1):
        if len(current) <= settings.raptor_cluster_branch:
            break
        k = max(2, math.ceil(len(current) / settings.raptor_cluster_branch))
        labels = _kmeans([v for _, v in current], k)

        groups: dict[int, list[tuple[dict, list[float]]]] = {}
        for label, item in zip(labels, current):
            groups.setdefault(label, []).append(item)

        # Build per-cluster combined texts + metadata up front, so we can
        # summarize them all in a single parallel batch.
        group_order: list[list[tuple[dict, list[float]]]] = list(groups.values())
        texts_to_summarize = [
            "\n\n".join(c["text"] for c, _ in g) for g in group_order
        ]
        summary_texts = _summarize_level(texts_to_summarize)

        next_level: list[tuple[dict, list[float]]] = []
        for group, summary_text in zip(group_order, summary_texts):
            if not group or not summary_text:
                continue
            children_ids = [c["metadata"]["chunk_id"] for c, _ in group]
            source = group[0][0].get("metadata", {}).get("source", "")
            pages = [
                c["metadata"].get("page_start", 0) for c, _ in group
                if c.get("metadata", {}).get("page_start")
            ]
            page_start = min(pages) if pages else 0
            pages_end = [
                c["metadata"].get("page_end", 0) for c, _ in group
                if c.get("metadata", {}).get("page_end")
            ]
            page_end = max(pages_end) if pages_end else 0

            summary_id = str(uuid.uuid4())
            chunk = {
                "text": summary_text,
                "contextual_text": summary_text,
                "metadata": {
                    "source": source,
                    "chunk_id": summary_id,
                    "level": level,
                    "children_ids": ",".join(children_ids),
                    "page_start": page_start,
                    "page_end": page_end,
                    "modality": "raptor_summary",
                },
            }
            summaries.append(chunk)

            dim = len(group[0][1])
            avg = [
                sum(v[j] for _, v in group) / len(group)
                for j in range(dim)
            ]
            next_level.append((chunk, avg))

        if not next_level:
            break
        current = next_level

    log.info(f"RAPTOR built {len(summaries)} summary nodes")
    return summaries
