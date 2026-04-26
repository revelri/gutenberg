"""Retroactive P0/P1/P5/P7 upgrade of an existing Chroma collection.

Avoids re-ingesting source PDFs — reads current chunks, applies new-feature
transforms in place, writes back.

Passes (each opt-in via CLI):

  * ``--tag``        P1 gazetteer tagging:    compute canonical_ids from chunk text
                     and write into chunk metadata.
  * ``--contextual`` P0 contextual prefixes: group chunks per source, call Ollama
                     to generate a ~80-token context sentence per chunk, store
                     ``context_prefix`` in metadata and re-embed the
                     contextualized text so dense search sees the enriched form.
  * ``--raptor``     P5 summary tree:         cluster chunks per source, summarize,
                     and insert level>=1 summary chunks alongside leaves.
  * ``--graph``      P7 build entity co-occurrence graph from tagged chunks.

Always rewrites the BM25 disk cache for the target collection when chunk
content is re-embedded (so BM25 sees the same text). Updates the index
manifest (P11) when finished.

Example:
    python scripts/retro_upgrade.py --collection gutenberg-anti-oedipus \\
        --tag --contextual --raptor --graph
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services"))
sys.path.insert(0, str(ROOT / "services" / "api"))
sys.path.insert(0, str(ROOT / "services" / "worker"))

log = logging.getLogger("retro_upgrade")


def _meta_sanitize(meta: dict) -> dict:
    """Chroma metadata rejects None — strip them out."""
    return {k: v for k, v in meta.items() if v is not None}


def load_chunks(collection_name: str) -> list[dict]:
    from core.chroma import get_collection

    col = get_collection(collection_name)
    result = col.get(include=["documents", "metadatas", "embeddings"])
    chunks = []
    embs_raw = result.get("embeddings")
    if embs_raw is None or (hasattr(embs_raw, "__len__") and len(embs_raw) == 0):
        embs = [None] * len(result["ids"])
    else:
        embs = list(embs_raw)
    for id_, doc, meta, emb in zip(result["ids"], result["documents"], result["metadatas"], embs):
        chunks.append({"id": id_, "text": doc, "metadata": meta or {}, "embedding": emb})
    log.info(f"loaded {len(chunks)} chunks from '{collection_name}'")
    return chunks


def update_chunks(
    collection_name: str,
    chunks: list[dict],
    *,
    update_metadata: bool = True,
    update_documents: bool = False,
    update_embeddings: bool = False,
) -> None:
    from core.chroma import get_collection

    col = get_collection(collection_name)
    batch = 200
    for i in range(0, len(chunks), batch):
        group = chunks[i : i + batch]
        kwargs = {"ids": [c["id"] for c in group]}
        if update_metadata:
            kwargs["metadatas"] = [_meta_sanitize(c["metadata"]) for c in group]
        if update_documents:
            kwargs["documents"] = [c["text"] for c in group]
        if update_embeddings:
            kwargs["embeddings"] = [c["embedding"] for c in group]
        col.update(**kwargs)
    log.info(f"updated {len(chunks)} chunks in '{collection_name}'")


# ── Pass: P1 gazetteer tagging ──────────────────────────────────────────

def pass_tag(chunks: list[dict]) -> int:
    from shared.gazetteer import resolve

    tagged = 0
    for c in chunks:
        cids = resolve(c["text"])
        if cids:
            c["metadata"]["canonical_ids"] = ",".join(cids)
            tagged += 1
    log.info(f"P1 tag: {tagged}/{len(chunks)} chunks now carry canonical_ids")
    return tagged


# ── Pass: P0 contextual enrichment via Ollama ──────────────────────────

_SYSTEM = (
    "You help make scholarly text chunks more retrievable. For each chunk from "
    "the document, write ONE short sentence (max 35 words) that places the chunk "
    "in the document's argument. Preserve proper names, foreign terms, and page "
    "anchors verbatim. Do not summarize the chunk. Do not quote it. Do not add "
    "editorial commentary. Output ONLY the single context sentence."
)


async def _openrouter_prefix_async(
    client,  # httpx.AsyncClient
    semaphore,  # asyncio.Semaphore
    document_sample: str,
    chunk_text: str,
    model: str,
    api_key: str,
) -> str:
    import re as _re

    prompt = (
        f"{_SYSTEM}\n\n<document>\n{document_sample[:8000]}\n</document>\n\n"
        f"<chunk>\n{chunk_text}\n</chunk>\n"
        "Respond with the single context sentence only."
    )
    async with semaphore:
        try:
            r = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/revelri/gutenborg",
                    "X-Title": "gutenborg-retro-upgrade",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 120,
                    "temperature": 0.1,
                },
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                or ""
            ).strip()
            text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
            return text.split("<think>")[0].strip()
        except Exception as e:
            log.warning(f"openrouter prefix failed: {e}")
            return ""


async def _run_openrouter_batch(
    chunks: list[dict],
    by_source: dict[str, list[dict]],
    model: str,
    api_key: str,
    concurrency: int,
    cache_dir: Path,
) -> int:
    """Dispatch all Ollama-equivalent calls concurrently against OpenRouter."""
    import asyncio
    from hashlib import sha256
    import httpx

    tasks: list = []
    task_targets: list[tuple[dict, Path]] = []
    started = time.perf_counter()

    async with httpx.AsyncClient(http2=False) as client:
        sem = asyncio.Semaphore(concurrency)

        # Build task list from cache misses.
        for src, group in by_source.items():
            doc_sample = "\n\n".join(c["text"] for c in group[:15])
            for i, c in enumerate(group):
                key = sha256(
                    f"{src}|{i}|{c['text']}|{model}".encode()
                ).hexdigest()
                cache_path = cache_dir / f"{key}.json"
                prefix = ""
                if cache_path.exists():
                    try:
                        prefix = json.loads(cache_path.read_text()).get("prefix", "")
                    except Exception:
                        prefix = ""
                if prefix:
                    c["metadata"]["context_prefix"] = prefix[:500]
                    c["_contextual_text"] = f"{prefix}\n\n{c['text']}"
                    continue
                tasks.append(
                    _openrouter_prefix_async(
                        client, sem, doc_sample, c["text"], model, api_key
                    )
                )
                task_targets.append((c, cache_path))

        if not tasks:
            return 0

        total = len(tasks)
        log.info(f"P0 contextual (OR): {total} tasks @ concurrency={concurrency}")

        # Chunk dispatch into batches of 200 so we can log progress — gather
        # preserves order across (task, target) pairs.
        enriched = 0
        BATCH = 200
        for batch_start in range(0, total, BATCH):
            batch_end = min(batch_start + BATCH, total)
            batch_tasks = tasks[batch_start:batch_end]
            batch_targets = task_targets[batch_start:batch_end]
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for (c, cache_path), prefix in zip(batch_targets, results):
                if isinstance(prefix, Exception) or not prefix:
                    c["_contextual_text"] = c["text"]
                    continue
                cache_path.write_text(json.dumps({"prefix": prefix}))
                c["metadata"]["context_prefix"] = prefix[:500]
                c["_contextual_text"] = f"{prefix}\n\n{c['text']}"
                enriched += 1
            elapsed = time.perf_counter() - started
            rate = enriched / elapsed if elapsed else 0
            remaining = (total - batch_end) / rate if rate else 0
            log.info(
                f"P0 contextual (OR): {batch_end}/{total} dispatched, "
                f"{enriched} enriched ({rate:.1f}/s, ~{remaining/60:.1f}m left)"
            )
        return enriched


def _ollama_prefix(document_sample: str, chunk_text: str, model: str, host: str) -> str:
    import httpx
    import re as _re

    prompt = (
        f"{_SYSTEM}\n\n<document>\n{document_sample[:8000]}\n</document>\n\n"
        f"<chunk>\n{chunk_text}\n</chunk>\n"
        "Respond with the single context sentence only."
    )
    try:
        r = httpx.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                # num_predict bumped — qwen-class models eat ~300 tokens of
                # chain-of-thought before emitting the final sentence; we
                # strip the <think> block below.
                "options": {"temperature": 0.1, "num_predict": 400},
            },
            timeout=180,
        )
        r.raise_for_status()
        text = (r.json().get("response") or "").strip()
        # Strip <think>…</think> chain-of-thought blocks (qwen3-family).
        text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
        # Some models leave an unclosed <think> when num_predict truncates —
        # drop anything from the first <think> onward.
        text = text.split("<think>")[0].strip()
        return text
    except Exception as e:
        log.warning(f"ollama prefix failed: {e}")
        return ""


def pass_contextual(
    chunks: list[dict],
    model: str,
    host: str,
    cache_dir: Path,
    *,
    provider: str = "ollama",
    openrouter_key: str | None = None,
    openrouter_model: str = "google/gemini-2.5-flash-lite",
    concurrency: int = 30,
) -> int:
    """Generate contextual prefixes per chunk. Cached by (source, chunk_index, text_hash).

    ``provider`` ∈ {"ollama","openrouter"}. OpenRouter runs concurrent async
    requests (~30× wall-time speedup vs local Ollama). Cache is shared across
    providers when the model name matches.
    """
    from hashlib import sha256

    import httpx  # noqa: F401 — ensures module present

    cache_dir.mkdir(parents=True, exist_ok=True)

    by_source: dict[str, list[dict]] = {}
    for c in chunks:
        src = c["metadata"].get("source", "_unknown_")
        by_source.setdefault(src, []).append(c)

    if provider == "openrouter":
        if not openrouter_key:
            raise RuntimeError(
                "provider=openrouter requires OPENROUTER_KEY / OPENROUTER_API_KEY"
            )
        import asyncio

        enriched = asyncio.run(
            _run_openrouter_batch(
                chunks,
                by_source,
                openrouter_model,
                openrouter_key,
                concurrency,
                cache_dir,
            )
        )
        log.info(f"P0 contextual (OR): {enriched}/{len(chunks)} chunks got prefixes")
        return enriched

    enriched = 0
    started = time.perf_counter()
    for src, group in by_source.items():
        # Document sample = concatenation of the first 15 chunks (~6-10k chars).
        # Ollama has no prompt cache; we truncate to keep prompt cost bounded.
        doc_sample = "\n\n".join(c["text"] for c in group[:15])
        for i, c in enumerate(group):
            key = sha256(f"{src}|{i}|{c['text']}|{model}".encode()).hexdigest()
            cache_path = cache_dir / f"{key}.json"
            prefix = ""
            if cache_path.exists():
                try:
                    prefix = json.loads(cache_path.read_text()).get("prefix", "")
                except Exception:
                    prefix = ""
            # Regenerate when cached prefix is empty (previous run hit the
            # <think>-only qwen3 failure mode).
            if not prefix:
                prefix = _ollama_prefix(doc_sample, c["text"], model, host)
                if prefix:
                    cache_path.write_text(json.dumps({"prefix": prefix}))
            if prefix:
                c["metadata"]["context_prefix"] = prefix[:500]
                c["_contextual_text"] = f"{prefix}\n\n{c['text']}"
                enriched += 1
            else:
                c["_contextual_text"] = c["text"]
            if enriched and enriched % 25 == 0:
                elapsed = time.perf_counter() - started
                rate = enriched / elapsed
                remaining = (len(chunks) - enriched) / rate if rate else 0
                log.info(
                    f"P0 contextual: {enriched}/{len(chunks)} enriched "
                    f"({rate:.1f}/s, ~{remaining/60:.1f}m left)"
                )
    log.info(f"P0 contextual: {enriched}/{len(chunks)} chunks got prefixes")
    return enriched


def reembed(chunks: list[dict]) -> None:
    """Re-embed chunks using contextual_text if present."""
    from shared.embedder import embed_texts

    texts = [c.get("_contextual_text") or c["text"] for c in chunks]
    # Batch embed
    embeddings = embed_texts(texts)
    for c, emb in zip(chunks, embeddings):
        c["embedding"] = list(emb) if not isinstance(emb, list) else emb
    log.info(f"re-embedded {len(chunks)} chunks on contextualized text")


# ── Pass: P7 graph build ───────────────────────────────────────────────

def pass_graph(chunks: list[dict]) -> int:
    from core.graph import build_from_chunks

    return build_from_chunks(chunks)


# ── Pass: P5 RAPTOR ────────────────────────────────────────────────────

def pass_raptor(chunks: list[dict], collection_name: str) -> int:
    from pipeline.raptor import build_tree
    from shared.embedder import embed_texts
    from core.chroma import get_collection

    # Build per-source so summaries stay within a single work's argument.
    by_source: dict[str, list[dict]] = {}
    for c in chunks:
        by_source.setdefault(c["metadata"].get("source", "_u_"), []).append(c)

    all_new: list[dict] = []
    for src, leaves in by_source.items():
        if len(leaves) < 20:
            continue
        # Ensure each leaf has an embedding present (use existing if loaded).
        def _has_emb(c):
            e = c.get("embedding")
            if e is None:
                return False
            if hasattr(e, "__len__"):
                return len(e) > 0
            return bool(e)

        missing = [i for i, c in enumerate(leaves) if not _has_emb(c)]
        if missing:
            texts = [leaves[i]["text"] for i in missing]
            fresh = embed_texts(texts)
            for k, i in enumerate(missing):
                leaves[i]["embedding"] = list(fresh[k])
        leaf_embs = [c["embedding"] for c in leaves]
        summaries = build_tree(leaves, leaf_embs)
        all_new.extend(summaries)

    if not all_new:
        log.info("P5: no RAPTOR summaries produced (corpus too small)")
        return 0

    summary_embs = embed_texts(
        [s.get("contextual_text") or s["text"] for s in all_new]
    )
    import uuid

    col = get_collection(collection_name)
    ids = [s["metadata"].get("chunk_id") or str(uuid.uuid4()) for s in all_new]
    col.add(
        ids=ids,
        embeddings=[list(e) for e in summary_embs],
        documents=[s["text"] for s in all_new],
        metadatas=[_meta_sanitize(s["metadata"]) for s in all_new],
    )
    log.info(f"P5: added {len(all_new)} RAPTOR summary chunks")
    return len(all_new)


# ── Main ────────────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--tag", action="store_true")
    ap.add_argument("--contextual", action="store_true")
    ap.add_argument("--raptor", action="store_true")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--ollama-model", default=None)
    ap.add_argument("--ollama-host", default=None)
    ap.add_argument("--cache-dir", default="data/cache/contextual_retro")
    ap.add_argument(
        "--provider",
        choices=["ollama", "openrouter"],
        default="ollama",
        help="Contextual-prefix backend. openrouter is ~30x faster but paid.",
    )
    ap.add_argument(
        "--openrouter-model",
        default="google/gemini-2.5-flash-lite",
        help="OpenRouter model id (e.g., google/gemini-2.5-flash-lite, "
        "openai/gpt-4o-mini, anthropic/claude-haiku-4.5)",
    )
    ap.add_argument("--concurrency", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None, help="cap number of chunks (debug)")
    args = ap.parse_args()

    from core.config import settings

    chunks = load_chunks(args.collection)
    if args.limit:
        chunks = chunks[: args.limit]

    touched_meta = False
    touched_docs_and_embs = False

    if args.tag:
        pass_tag(chunks)
        touched_meta = True

    if args.contextual:
        import os as _os

        model = args.ollama_model or settings.ollama_llm_model
        host = args.ollama_host or settings.ollama_host
        or_key = (
            _os.environ.get("OPENROUTER_API_KEY")
            or _os.environ.get("OPENROUTER_KEY")
        )
        pass_contextual(
            chunks,
            model,
            host,
            Path(args.cache_dir),
            provider=args.provider,
            openrouter_key=or_key,
            openrouter_model=args.openrouter_model,
            concurrency=args.concurrency,
        )
        reembed(chunks)
        touched_meta = True
        touched_docs_and_embs = True

    if touched_meta or touched_docs_and_embs:
        update_chunks(
            args.collection,
            chunks,
            update_metadata=touched_meta,
            update_documents=False,  # original text preserved — verification invariant
            update_embeddings=touched_docs_and_embs,
        )

    # Rebuild BM25 so sparse retrieval sees the updated metadata (and, via
    # the BM25 builder's context_prefix prepend, the contextual text).
    if touched_meta or touched_docs_and_embs:
        from core.rag import refresh_bm25_index
        refresh_bm25_index(args.collection)
        log.info("BM25 rebuilt for updated collection")

    if args.graph:
        # Re-read so we operate on the post-tag canonical_ids.
        fresh = load_chunks(args.collection)
        edges = pass_graph(fresh)
        log.info(f"P7: graph rebuilt ({edges} edges)")

    if args.raptor:
        pass_raptor(chunks, args.collection)

    log.info("retro upgrade complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
