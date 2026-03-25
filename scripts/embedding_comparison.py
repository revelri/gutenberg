#!/usr/bin/env python3
"""Comparative embedding model benchmark.

Re-embeds the same corpus with different embedding models into separate
ChromaDB collections, then compares retrieval quality via the citation eval.

Models tested:
  - qwen3-embedding:4b (local Ollama, 32K context)
  - openai/text-embedding-3-large (via OpenRouter, 8K context, 3072 dims)

Usage:
    python scripts/embedding_comparison.py --ground-truth data/eval/ao_ground_truth.json
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import chromadb
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("embed_compare")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "http://localhost:8200")
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
API_URL = os.environ.get("GUTENBERG_API_URL", "http://localhost:8002")


# ── Embedding backends ──────────────────────────────────────────────

def embed_ollama(texts: list[str], model: str = "qwen3-embedding:4b") -> list[list[float]]:
    """Embed via local Ollama."""
    all_embs = []
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = httpx.post(
            f"{OLLAMA_HOST}/api/embed",
            json={"model": model, "input": batch},
            timeout=120,
        )
        resp.raise_for_status()
        all_embs.extend(resp.json()["embeddings"])
    return all_embs


def embed_openrouter(texts: list[str], model: str = "openai/text-embedding-3-large") -> list[list[float]]:
    """Embed via OpenRouter (OpenAI-compatible)."""
    if not OPENROUTER_KEY:
        raise ValueError("OPENROUTER_KEY not set")
    all_embs = []
    batch_size = 10  # Conservative for large-dim models
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(3):
            try:
                resp = httpx.post(
                    "https://openrouter.ai/api/v1/embeddings",
                    headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                    json={"model": model, "input": batch},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                all_embs.extend([d["embedding"] for d in sorted_data])
                break
            except (httpx.ReadTimeout, httpx.HTTPStatusError) as e:
                if attempt < 2:
                    log.warning(f"Retry {attempt+1}/3 for batch {i//batch_size}: {e}")
                    time.sleep(2 ** attempt)
                else:
                    raise
    return all_embs


def embed_query_ollama(text: str, model: str = "qwen3-embedding:4b") -> list[float]:
    resp = httpx.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": model, "input": [text]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def embed_query_openrouter(text: str, model: str = "openai/text-embedding-3-large") -> list[float]:
    resp = httpx.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
        json={"model": model, "input": [text]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


# ── ChromaDB helpers ────────────────────────────────────────────────

def get_chroma_client():
    host = CHROMA_HOST.replace("http://", "").replace("https://", "")
    parts = host.split(":")
    return chromadb.HttpClient(host=parts[0], port=int(parts[1]) if len(parts) > 1 else 8000)


def get_or_create_collection(name: str):
    client = get_chroma_client()
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def clear_collection(name: str):
    client = get_chroma_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass


# ── Main comparison logic ───────────────────────────────────────────

MODELS = {
    "qwen3-4b": {
        "embed_fn": embed_openrouter,
        "query_fn": embed_query_openrouter,
        "model_id": "qwen/qwen3-embedding-4b",
        "collection": "embed-compare-qwen3-4b",
    },
    "qwen3-8b": {
        "embed_fn": embed_openrouter,
        "query_fn": embed_query_openrouter,
        "model_id": "qwen/qwen3-embedding-8b",
        "collection": "embed-compare-qwen3-8b",
    },
    "openai-3-large": {
        "embed_fn": embed_openrouter,
        "query_fn": embed_query_openrouter,
        "model_id": "openai/text-embedding-3-large",
        "collection": "embed-compare-openai-3l",
    },
}


def get_existing_chunks() -> list[dict]:
    """Get chunks from the current default collection."""
    col = get_or_create_collection(os.environ.get("CHROMA_COLLECTION", "gutenberg-qwen3"))
    result = col.get(include=["documents", "metadatas"])
    chunks = []
    for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
        chunks.append({"id": id_, "text": doc, "metadata": meta})
    log.info(f"Loaded {len(chunks)} chunks from default collection")
    return chunks


def ingest_with_model(chunks: list[dict], model_key: str):
    """Re-embed chunks with a specific model into its collection."""
    config = MODELS[model_key]
    collection_name = config["collection"]
    model_id = config["model_id"]
    embed_fn = config["embed_fn"]

    clear_collection(collection_name)
    col = get_or_create_collection(collection_name)

    texts = [c["text"] for c in chunks]
    log.info(f"Embedding {len(texts)} chunks with {model_key} ({model_id})...")

    t0 = time.time()
    embeddings = embed_fn(texts, model=model_id)
    embed_time = time.time() - t0
    log.info(f"Embedded in {embed_time:.1f}s ({len(texts)/embed_time:.1f} chunks/s)")

    # Store in batches
    import uuid
    ids = [str(uuid.uuid4()) for _ in chunks]
    documents = texts
    metadatas = [c["metadata"] for c in chunks]

    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = i + batch_size
        col.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )

    log.info(f"Stored {len(chunks)} chunks in {collection_name}")
    return embed_time


def query_collection(query: str, model_key: str, top_k: int = 5) -> list[dict]:
    """Query a collection using its model's embedding."""
    config = MODELS[model_key]
    query_fn = config["query_fn"]
    model_id = config["model_id"]
    collection_name = config["collection"]

    q_emb = query_fn(query, model=model_id)
    col = get_or_create_collection(collection_name)
    results = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results["ids"] and results["ids"][0]:
        for id_, doc, meta, dist in zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0],
        ):
            chunks.append({
                "id": id_, "text": doc, "metadata": meta,
                "score": 1 - dist,
            })
    return chunks


def run_retrieval_comparison(ground_truth: list[dict], models_to_test: list[str]):
    """Compare retrieval quality across models.

    Uses the ground truth passage as a substring match against retrieved chunks.
    This is model-agnostic and page-number-agnostic — it only checks whether
    the correct passage was retrieved, regardless of metadata.
    """
    import re
    from difflib import SequenceMatcher

    def normalize(t):
        return re.sub(r"\s+", " ", t.lower()).strip()

    results = {}

    for model_key in models_to_test:
        log.info(f"\n{'='*60}")
        log.info(f"Testing {model_key}")
        log.info(f"{'='*60}")

        model_results = []
        for i, tc in enumerate(ground_truth):
            query = tc["query"]
            gt_norm = normalize(tc["ground_truth"])[:150]  # first 150 chars of ground truth

            log.info(f"  [{i+1}/{len(ground_truth)}] {query[:60]}...")

            chunks = query_collection(query, model_key, top_k=5)

            # Check if ground truth passage appears in ANY of the top-5 chunks
            best_overlap = 0.0
            best_rank = -1
            for rank, chunk in enumerate(chunks):
                chunk_norm = normalize(chunk["text"])

                # Exact substring match (strongest signal)
                if gt_norm[:80] in chunk_norm:
                    best_overlap = 1.0
                    best_rank = rank + 1
                    break

                # Fuzzy window match for near-verbatim
                gt_len = len(gt_norm)
                if gt_len < len(chunk_norm):
                    step = max(1, gt_len // 4)
                    for start in range(0, len(chunk_norm) - gt_len + 1, step):
                        window = chunk_norm[start:start + gt_len]
                        ratio = SequenceMatcher(None, gt_norm, window).ratio()
                        if ratio > best_overlap:
                            best_overlap = ratio
                            best_rank = rank + 1
                            if ratio >= 0.95:
                                break
                    if best_overlap >= 0.95:
                        break
                else:
                    ratio = SequenceMatcher(None, gt_norm, chunk_norm).ratio()
                    if ratio > best_overlap:
                        best_overlap = ratio
                        best_rank = rank + 1

            model_results.append({
                "query": query,
                "gt_in_top5": best_overlap >= 0.7,
                "best_overlap": round(best_overlap, 3),
                "best_rank": best_rank,
                "top_chunk_preview": chunks[0]["text"][:80] if chunks else "",
            })

            icon = "✓" if best_overlap >= 0.7 else "✗"
            log.info(f"    {icon} overlap={best_overlap:.0%} rank={best_rank}")

        results[model_key] = model_results

    return results


def print_comparison(results: dict, embed_times: dict):
    """Print comparison table."""
    print(f"\n{'='*80}")
    print(f"  EMBEDDING MODEL COMPARISON")
    print(f"{'='*80}")

    for model_key, model_results in results.items():
        n = len(model_results)
        hits = sum(1 for r in model_results if r["gt_in_top5"])
        avg_overlap = sum(r["best_overlap"] for r in model_results) / n
        avg_rank = sum(r["best_rank"] for r in model_results if r["best_rank"] > 0) / max(1, hits)
        embed_time = embed_times.get(model_key, 0)

        print(f"\n  {model_key}:")
        print(f"    Embed time:          {embed_time:.1f}s")
        print(f"    GT in top-5:         {hits}/{n} ({hits/n:.0%})")
        print(f"    Mean overlap:        {avg_overlap:.1%}")
        print(f"    Mean rank (hits):    {avg_rank:.1f}")

    # Side-by-side per-query
    model_keys = list(results.keys())
    print(f"\n{'='*80}")
    print(f"  PER-QUERY DETAIL")
    print(f"{'='*80}")
    header = f"  {'#':<3} {'Query':<35}"
    for mk in model_keys:
        header += f" {mk:>15}"
    print(header)
    print(f"  {'─'*(38 + 16*len(model_keys))}")

    n = len(list(results.values())[0])
    for i in range(n):
        q = list(results.values())[0][i]["query"][:32] + "..."
        row = f"  {i+1:<3} {q:<35}"
        for mk in model_keys:
            r = results[mk][i]
            icon = "✓" if r["gt_in_top5"] else "✗"
            row += f" {icon} {r['best_overlap']:.0%} r{r['best_rank']:>2}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Compare embedding models")
    parser.add_argument("--ground-truth", default="data/eval/ao_ground_truth.json")
    parser.add_argument("--models", default="qwen3-4b,openai-3-large")
    args = parser.parse_args()

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        log.error(f"Ground truth not found: {gt_path}")
        sys.exit(1)

    with open(gt_path) as f:
        ground_truth = json.load(f)
    log.info(f"Loaded {len(ground_truth)} test cases")

    models_to_test = [m.strip() for m in args.models.split(",")]

    # Get existing chunks from default collection
    chunks = get_existing_chunks()
    if not chunks:
        log.error("No chunks in default collection")
        sys.exit(1)

    # Re-embed with each model
    embed_times = {}
    for model_key in models_to_test:
        if model_key not in MODELS:
            log.warning(f"Unknown model: {model_key}, skipping")
            continue
        embed_times[model_key] = ingest_with_model(chunks, model_key)

    # Compare retrieval quality
    results = run_retrieval_comparison(ground_truth, models_to_test)
    print_comparison(results, embed_times)


if __name__ == "__main__":
    main()
