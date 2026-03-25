#!/usr/bin/env python3
"""Pipeline component ablation test.

Isolates each retrieval component's contribution to accuracy:
  1. Dense search only (embedding similarity, no BM25, no reranker)
  2. BM25 only (keyword search, no embeddings, no reranker)
  3. Dense + BM25 (hybrid, no reranker)
  4. Dense + reranker (no BM25)
  5. BM25 + reranker (no dense)
  6. Full pipeline (dense + BM25 + reranker) — current default

For each configuration, runs the 20-query ground truth test and reports
how many ground truth passages appear in the top-5 retrieved chunks.

Usage:
    python scripts/ablation_test.py --ground-truth data/eval/ao_ground_truth.json
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import chromadb
import httpx
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ablation")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "http://localhost:8200")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "gutenberg-qwen3")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b")
RERANKER_AVAILABLE = True

try:
    from sentence_transformers import CrossEncoder
    RERANKER_MODEL = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
except Exception:
    RERANKER_AVAILABLE = False
    log.warning("Reranker not available — skipping reranker configurations")


# ── Retrieval components ────────────────────────────────────────────

def get_collection():
    host = CHROMA_HOST.replace("http://", "").replace("https://", "")
    parts = host.split(":")
    client = chromadb.HttpClient(host=parts[0], port=int(parts[1]) if len(parts) > 1 else 8000)
    return client.get_or_create_collection(name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})


def embed_query(text: str) -> list[float]:
    resp = httpx.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBED_MODEL, "input": [text]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "qwen3:8b")

def hyde_expand(query: str) -> str:
    """Generate a hypothetical answer passage for the query."""
    try:
        resp = httpx.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": (
                    f"Write a short paragraph (3-4 sentences) that might appear in a "
                    f"philosophy book as an answer to this question. Write in an academic "
                    f"style as if quoting from the source text. Do not add commentary.\n\n"
                    f"Question: {query}"
                ),
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 200},
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip() or query
    except Exception:
        return query


def dense_search(query_embedding: list[float], top_k: int = 20) -> list[dict]:
    col = get_collection()
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    if results["ids"] and results["ids"][0]:
        for id_, doc, meta, dist in zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0],
        ):
            chunks.append({"id": id_, "text": doc, "metadata": meta, "dense_score": 1 - dist})
    return chunks


_bm25_cache = None

def build_bm25():
    global _bm25_cache
    if _bm25_cache is not None:
        return _bm25_cache

    col = get_collection()
    result = col.get(include=["documents", "metadatas"])
    corpus = [
        {"id": id_, "text": doc, "metadata": meta}
        for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
    ]
    tokenized = [doc["text"].lower().split() for doc in corpus]
    index = BM25Okapi(tokenized)
    _bm25_cache = (corpus, index)
    log.info(f"BM25 index built: {len(corpus)} docs")
    return _bm25_cache


def bm25_search(query: str, top_k: int = 20) -> list[dict]:
    corpus, index = build_bm25()
    tokenized_query = query.lower().split()
    scores = index.get_scores(tokenized_query)
    scored = []
    for i, score in enumerate(scores):
        if score > 0:
            scored.append({
                "id": corpus[i]["id"],
                "text": corpus[i]["text"],
                "metadata": corpus[i]["metadata"],
                "bm25_score": float(score),
            })
    scored.sort(key=lambda x: x["bm25_score"], reverse=True)
    return scored[:top_k]


def rrf_fusion(dense: list[dict], sparse: list[dict], k: int = 60,
               dense_weight: float = 1.0, sparse_weight: float = 1.0) -> list[dict]:
    scores = {}
    chunk_map = {}
    for rank, chunk in enumerate(dense):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + dense_weight / (k + rank + 1)
        chunk_map[cid] = chunk
    for rank, chunk in enumerate(sparse):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + sparse_weight / (k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = chunk
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[cid] for cid, _ in ranked]


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    if not RERANKER_AVAILABLE or not chunks:
        return chunks[:top_k]
    pairs = [[query, c["text"]] for c in chunks]
    scores = RERANKER_MODEL.predict(pairs)
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])
    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]


# ── Ablation configurations ────────────────────────────────────────

CONFIGS = {
    "dense_only": {
        "description": "Dense vector search only (no BM25, no reranker)",
        "use_dense": True, "use_bm25": False, "use_reranker": False,
    },
    "bm25_only": {
        "description": "BM25 keyword search only (no dense, no reranker)",
        "use_dense": False, "use_bm25": True, "use_reranker": False,
    },
    "hybrid_no_rerank": {
        "description": "Dense + BM25 hybrid (RRF fusion, no reranker)",
        "use_dense": True, "use_bm25": True, "use_reranker": False,
    },
    "dense_rerank": {
        "description": "Dense + reranker (no BM25)",
        "use_dense": True, "use_bm25": False, "use_reranker": True,
    },
    "bm25_rerank": {
        "description": "BM25 + reranker (no dense)",
        "use_dense": False, "use_bm25": True, "use_reranker": True,
    },
    "full_pipeline": {
        "description": "Full pipeline: Dense + BM25 + reranker (current default)",
        "use_dense": True, "use_bm25": True, "use_reranker": True,
    },
    "hyde_dense_only": {
        "description": "HyDE + Dense only (hypothetical answer embedding, no BM25/reranker)",
        "use_dense": True, "use_bm25": False, "use_reranker": False, "use_hyde": True,
    },
    "hyde_full": {
        "description": "HyDE + Dense + BM25 + reranker (full pipeline with HyDE)",
        "use_dense": True, "use_bm25": True, "use_reranker": True, "use_hyde": True,
    },
}


def retrieve(query: str, config: dict, top_k: int = 5) -> list[dict]:
    """Run retrieval with a specific ablation configuration."""
    retrieve_k = 50  # retrieve more, then cut to top_k after reranking

    dense_results = []
    sparse_results = []

    if config["use_dense"]:
        if config.get("use_hyde"):
            hyde_text = hyde_expand(query)
            q_emb = embed_query(hyde_text)
        else:
            q_emb = embed_query(query)
        dense_results = dense_search(q_emb, top_k=retrieve_k)

    if config["use_bm25"]:
        sparse_results = bm25_search(query, top_k=retrieve_k)

    # Combine
    if dense_results and sparse_results:
        merged = rrf_fusion(dense_results, sparse_results, dense_weight=0.6, sparse_weight=0.4)
    elif dense_results:
        merged = dense_results
    elif sparse_results:
        merged = sparse_results
    else:
        merged = []

    # Rerank
    if config["use_reranker"] and RERANKER_AVAILABLE:
        return rerank(query, merged[:retrieve_k], top_k=top_k)
    else:
        return merged[:top_k]


# ── Evaluation ──────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def check_gt_in_results(gt_text: str, chunks: list[dict]) -> tuple[float, int]:
    """Check if ground truth passage appears in retrieved chunks.
    Returns (best_overlap, best_rank).

    Uses multiple substring lengths (50, 40, 30 chars) for robustness
    against OCR whitespace/punctuation variations.
    """
    gt_norm = normalize(gt_text)
    best_overlap = 0.0
    best_rank = -1

    for rank, chunk in enumerate(chunks):
        chunk_norm = normalize(chunk["text"])

        # Try exact substring match at decreasing lengths
        for substr_len in [80, 50, 40, 30]:
            if len(gt_norm) >= substr_len and gt_norm[:substr_len] in chunk_norm:
                return 1.0, rank + 1

        # Fuzzy window match for near-verbatim
        gt_prefix = gt_norm[:150]
        gt_len = len(gt_prefix)
        if gt_len < len(chunk_norm):
            step = max(1, gt_len // 4)
            for start in range(0, len(chunk_norm) - gt_len + 1, step):
                window = chunk_norm[start:start + gt_len]
                ratio = SequenceMatcher(None, gt_prefix, window).ratio()
                if ratio > best_overlap:
                    best_overlap = ratio
                    best_rank = rank + 1
                    if ratio >= 0.95:
                        return best_overlap, best_rank
        else:
            ratio = SequenceMatcher(None, gt_norm, chunk_norm).ratio()
            if ratio > best_overlap:
                best_overlap = ratio
                best_rank = rank + 1

    return best_overlap, best_rank


def run_ablation(ground_truth: list[dict], configs_to_run: list[str]):
    """Run all ablation configurations and compare."""
    results = {}

    # Pre-build BM25 index (shared across configs)
    build_bm25()

    for config_name in configs_to_run:
        config = CONFIGS[config_name]
        if config["use_reranker"] and not RERANKER_AVAILABLE:
            log.warning(f"Skipping {config_name} — reranker not available")
            continue

        log.info(f"\n{'='*60}")
        log.info(f"Config: {config_name}")
        log.info(f"  {config['description']}")
        log.info(f"{'='*60}")

        config_results = []
        t0 = time.time()

        for i, tc in enumerate(ground_truth):
            query = tc["query"]
            gt_text = tc["ground_truth"]

            chunks = retrieve(query, config, top_k=5)
            overlap, rank = check_gt_in_results(gt_text, chunks)

            hit = overlap >= 0.7
            config_results.append({
                "query": query,
                "hit": hit,
                "overlap": round(overlap, 3),
                "rank": rank,
            })

            icon = "✓" if hit else "✗"
            log.info(f"  [{i+1}/{len(ground_truth)}] {icon} overlap={overlap:.0%} rank={rank} | {query[:50]}...")

        elapsed = time.time() - t0
        results[config_name] = {
            "config": config,
            "results": config_results,
            "elapsed": round(elapsed, 1),
        }

    return results


def print_report(results: dict):
    print(f"\n{'='*80}")
    print(f"  PIPELINE COMPONENT ABLATION TEST")
    print(f"{'='*80}")

    # Summary table
    print(f"\n  {'Configuration':<25} {'Dense':>6} {'BM25':>6} {'Rerank':>7} {'GT@5':>6} {'Overlap':>8} {'Time':>6}")
    print(f"  {'─'*68}")

    for name, data in results.items():
        cfg = data["config"]
        res = data["results"]
        hits = sum(1 for r in res if r["hit"])
        avg_overlap = sum(r["overlap"] for r in res) / len(res)
        n = len(res)
        d = "✓" if cfg["use_dense"] else "✗"
        b = "✓" if cfg["use_bm25"] else "✗"
        r = "✓" if cfg["use_reranker"] else "✗"
        print(f"  {name:<25} {d:>6} {b:>6} {r:>7} {hits:>3}/{n:<2} {avg_overlap:>7.0%} {data['elapsed']:>5.0f}s")

    # Per-query breakdown
    config_names = list(results.keys())
    print(f"\n{'='*80}")
    print(f"  PER-QUERY BREAKDOWN")
    print(f"{'='*80}")

    header = f"  {'#':<3} {'Query':<30}"
    for name in config_names:
        short = name[:8]
        header += f" {short:>9}"
    print(header)
    print(f"  {'─'*(34 + 10*len(config_names))}")

    n_queries = len(list(results.values())[0]["results"])
    for i in range(n_queries):
        q = list(results.values())[0]["results"][i]["query"][:27] + "..."
        row = f"  {i+1:<3} {q:<30}"
        for name in config_names:
            r = results[name]["results"][i]
            icon = "✓" if r["hit"] else "✗"
            row += f" {icon} {r['overlap']:>5.0%} r{r['rank']:>1}"
        print(row)

    # Component contribution analysis
    print(f"\n{'='*80}")
    print(f"  COMPONENT CONTRIBUTION ANALYSIS")
    print(f"{'='*80}")

    if "dense_only" in results and "bm25_only" in results:
        d_hits = sum(1 for r in results["dense_only"]["results"] if r["hit"])
        b_hits = sum(1 for r in results["bm25_only"]["results"] if r["hit"])
        print(f"  Dense alone:    {d_hits}/20 — embedding similarity contribution")
        print(f"  BM25 alone:     {b_hits}/20 — keyword matching contribution")

    if "hybrid_no_rerank" in results:
        h_hits = sum(1 for r in results["hybrid_no_rerank"]["results"] if r["hit"])
        print(f"  Hybrid (no RR): {h_hits}/20 — fusion adds value over best individual")

    if "full_pipeline" in results and "hybrid_no_rerank" in results:
        f_hits = sum(1 for r in results["full_pipeline"]["results"] if r["hit"])
        h_hits = sum(1 for r in results["hybrid_no_rerank"]["results"] if r["hit"])
        rr_delta = f_hits - h_hits
        print(f"  + Reranker:     {f_hits}/20 — reranker adds {rr_delta:+d} over hybrid alone")

    if "dense_rerank" in results and "dense_only" in results:
        dr_hits = sum(1 for r in results["dense_rerank"]["results"] if r["hit"])
        d_hits = sum(1 for r in results["dense_only"]["results"] if r["hit"])
        print(f"  Dense+Rerank:   {dr_hits}/20 — reranker adds {dr_hits - d_hits:+d} to dense alone")

    if "bm25_rerank" in results and "bm25_only" in results:
        br_hits = sum(1 for r in results["bm25_rerank"]["results"] if r["hit"])
        b_hits = sum(1 for r in results["bm25_only"]["results"] if r["hit"])
        print(f"  BM25+Rerank:    {br_hits}/20 — reranker adds {br_hits - b_hits:+d} to BM25 alone")


def main():
    parser = argparse.ArgumentParser(description="Pipeline component ablation test")
    parser.add_argument("--ground-truth", default="data/eval/ao_ground_truth.json")
    parser.add_argument("--configs", default="all", help="Comma-separated config names or 'all'")
    args = parser.parse_args()

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        log.error(f"Ground truth not found: {gt_path}")
        sys.exit(1)

    with open(gt_path) as f:
        ground_truth = json.load(f)
    log.info(f"Loaded {len(ground_truth)} test cases")

    if args.configs == "all":
        configs_to_run = list(CONFIGS.keys())
    else:
        configs_to_run = [c.strip() for c in args.configs.split(",")]

    results = run_ablation(ground_truth, configs_to_run)
    print_report(results)


if __name__ == "__main__":
    main()
