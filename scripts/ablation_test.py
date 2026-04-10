#!/usr/bin/env python3
"""Pipeline component ablation test (v3 — no reranker).

Configurations:
  1. Dense only (chunk collection)
  2. BM25 only
  3. Dense + BM25 hybrid (RRF)
  4. Windows only (sentence-window collection)
  5. Dense + Windows (multi-collection)
  6. Dense + BM25 + Windows (full hybrid)
  7. Dense + BM25 + Windows + Passage Scoring (full pipeline)
  8. Full + Phrase search (for queries with quoted text)

Usage:
    CHROMA_COLLECTION=gutenberg-qwen3-v3 uv run scripts/ablation_test.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))
from shared.text_normalize import normalize_for_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ablation")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "http://localhost:8200")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "gutenberg-qwen3-v3")
WINDOW_COLLECTION = CHROMA_COLLECTION + "-windows"
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b")


# ── ChromaDB ─────────────────────────────────────────────────��────
_client = None


def _get_client():
    global _client
    if _client is None:
        host = CHROMA_HOST.replace("http://", "").replace("https://", "")
        parts = host.split(":")
        _client = chromadb.HttpClient(host=parts[0], port=int(parts[1]) if len(parts) > 1 else 8000)
    return _client


def get_collection(name=None):
    return _get_client().get_or_create_collection(
        name=name or CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )


# ── Retrieval components ────────────────────────────────────────────

def embed_query(text: str) -> list[float]:
    resp = httpx.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBED_MODEL, "input": [text]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def dense_search(query_embedding: list[float], top_k: int = 200, collection_name=None) -> list[dict]:
    col = get_collection(collection_name)
    if col.count() == 0:
        return []
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, col.count()),
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


def bm25_search(query: str, top_k: int = 200) -> list[dict]:
    corpus, index = build_bm25()
    scores = index.get_scores(query.lower().split())
    scored = [
        {"id": corpus[i]["id"], "text": corpus[i]["text"],
         "metadata": corpus[i]["metadata"], "bm25_score": float(score)}
        for i, score in enumerate(scores) if score > 0
    ]
    scored.sort(key=lambda x: x["bm25_score"], reverse=True)
    return scored[:top_k]


def rrf_fusion(list_a: list[dict], list_b: list[dict], k: int = 60,
               weight_a: float = 1.0, weight_b: float = 1.0) -> list[dict]:
    scores = {}
    chunk_map = {}
    for rank, chunk in enumerate(list_a):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + weight_a / (k + rank + 1)
        chunk_map[cid] = chunk
    for rank, chunk in enumerate(list_b):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + weight_b / (k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = chunk
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[cid] for cid, _ in ranked]


def passage_score(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Score chunks by token-level overlap with query text."""
    if not chunks:
        return []

    # Extract quoted phrase
    phrase = None
    m = re.search(r'"([^"]{10,})"', query)
    if m:
        phrase = m.group(1)

    query_lower = re.sub(r"\s+", " ", query.lower()).strip()
    query_words = set(w for w in query_lower.split() if len(w) > 2)

    for chunk in chunks:
        chunk_lower = re.sub(r"\s+", " ", chunk["text"].lower()).strip()
        chunk_words = set(w for w in chunk_lower.split() if len(w) > 2)
        score = 0.0

        if phrase:
            phrase_lower = re.sub(r"\s+", " ", phrase.lower()).strip()
            if phrase_lower[:50] in chunk_lower:
                score += 100.0
            elif phrase_lower[:30] in chunk_lower:
                score += 80.0
            elif phrase_lower[:20] in chunk_lower:
                score += 50.0

        if query_words and chunk_words:
            score += len(query_words & chunk_words) * 2.0

        # Bigram overlap
        q_words_list = query_lower.split()
        for j in range(len(q_words_list) - 1):
            bg = f"{q_words_list[j]} {q_words_list[j+1]}"
            if len(bg) > 5 and bg in chunk_lower:
                score += 5.0

        chunk["passage_score"] = score

    ranked = sorted(chunks, key=lambda c: c.get("passage_score", 0), reverse=True)
    return ranked[:top_k]


def phrase_search(query: str, top_k: int = 10) -> list[dict]:
    """Find chunks containing quoted phrase via substring match."""
    m = re.search(r'"([^"]{10,})"', query)
    if not m:
        return []
    phrase = re.sub(r"\s+", " ", m.group(1).lower()).strip()
    corpus, _ = build_bm25()
    matches = []
    for chunk in corpus:
        chunk_norm = re.sub(r"\s+", " ", chunk["text"].lower())
        if phrase[:30] in chunk_norm:
            matches.append(dict(chunk, phrase_score=100.0))
    return matches[:top_k]


# ── Ablation configurations ────────────────────────────────────────

CONFIGS = {
    "dense_only": {
        "description": "Dense vector search only (chunk collection)",
        "use_dense": True, "use_bm25": False, "use_windows": False,
        "use_passage_score": False, "use_phrase": False,
    },
    "bm25_only": {
        "description": "BM25 keyword search only",
        "use_dense": False, "use_bm25": True, "use_windows": False,
        "use_passage_score": False, "use_phrase": False,
    },
    "hybrid": {
        "description": "Dense + BM25 hybrid (RRF fusion)",
        "use_dense": True, "use_bm25": True, "use_windows": False,
        "use_passage_score": False, "use_phrase": False,
    },
    "windows_only": {
        "description": "Sentence-window collection only (dense)",
        "use_dense": False, "use_bm25": False, "use_windows": True,
        "use_passage_score": False, "use_phrase": False,
    },
    "dense_windows": {
        "description": "Dense chunks + sentence windows (multi-collection)",
        "use_dense": True, "use_bm25": False, "use_windows": True,
        "use_passage_score": False, "use_phrase": False,
    },
    "full_hybrid": {
        "description": "Dense + BM25 + Windows (full hybrid, no scoring)",
        "use_dense": True, "use_bm25": True, "use_windows": True,
        "use_passage_score": False, "use_phrase": False,
    },
    "full_pipeline": {
        "description": "Dense + BM25 + Windows + Passage Scoring",
        "use_dense": True, "use_bm25": True, "use_windows": True,
        "use_passage_score": True, "use_phrase": False,
    },
    "full_phrase": {
        "description": "Full pipeline + Phrase search",
        "use_dense": True, "use_bm25": True, "use_windows": True,
        "use_passage_score": True, "use_phrase": True,
    },
}


def retrieve(query: str, config: dict, top_k: int = 5) -> list[dict]:
    retrieve_k = 200

    dense_results = []
    sparse_results = []
    window_results = []

    q_emb = embed_query(query)

    if config["use_dense"]:
        dense_results = dense_search(q_emb, top_k=retrieve_k)

    if config["use_bm25"]:
        sparse_results = bm25_search(query, top_k=retrieve_k)

    if config["use_windows"]:
        window_results = dense_search(q_emb, top_k=retrieve_k, collection_name=WINDOW_COLLECTION)

    # Combine
    if dense_results and sparse_results:
        merged = rrf_fusion(dense_results, sparse_results, weight_a=0.6, weight_b=0.4)
    elif dense_results:
        merged = dense_results
    elif sparse_results:
        merged = sparse_results
    else:
        merged = []

    if window_results:
        merged = rrf_fusion(merged, window_results, weight_a=1.0, weight_b=0.7) if merged else window_results

    # Phrase search
    if config.get("use_phrase"):
        phrase_results = phrase_search(query)
        if phrase_results:
            existing_ids = {c.get("id") for c in merged}
            for pr in phrase_results:
                if pr.get("id") not in existing_ids:
                    merged.insert(0, pr)

    # Passage scoring
    if config.get("use_passage_score"):
        return passage_score(query, merged[:retrieve_k], top_k=top_k)

    return merged[:top_k]


# ── Evaluation ──────────────────────────────────────────────────────

def check_gt(gt_text: str, chunks: list[dict], top_k: int = 5) -> tuple[float, int]:
    gt_norm = normalize_for_comparison(gt_text)
    best_overlap = 0.0
    best_rank = -1

    for rank, chunk in enumerate(chunks[:top_k]):
        chunk_norm = normalize_for_comparison(chunk["text"])

        for substr_len in [80, 50, 40, 30]:
            if len(gt_norm) >= substr_len and gt_norm[:substr_len] in chunk_norm:
                return 1.0, rank + 1

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


def check_gt_extended(gt_text: str, chunks: list[dict]) -> tuple[float, int, int]:
    """Check GT in top-5 and top-10."""
    o5, r5 = check_gt(gt_text, chunks, top_k=5)
    o10, r10 = check_gt(gt_text, chunks, top_k=10)
    return o5, r5, r10


def run_ablation(ground_truth: list[dict], configs_to_run: list[str]):
    results = {}
    build_bm25()

    for config_name in configs_to_run:
        config = CONFIGS[config_name]
        log.info(f"\n{'='*60}")
        log.info(f"Config: {config_name} — {config['description']}")
        log.info(f"{'='*60}")

        config_results = []
        t0 = time.time()

        for i, tc in enumerate(ground_truth):
            query = tc["query"]
            gt_text = tc["ground_truth"]

            # Retrieve top-10 to measure both GT@5 and GT@10
            chunks = retrieve(query, config, top_k=10)
            o5, r5 = check_gt(gt_text, chunks, top_k=5)
            _, r10 = check_gt(gt_text, chunks, top_k=10)

            hit5 = o5 >= 0.7
            config_results.append({
                "query": query, "hit5": hit5, "overlap5": round(o5, 3),
                "rank5": r5, "rank10": r10,
            })

            icon = "+" if hit5 else "-"
            log.info(f"  [{i+1}/{len(ground_truth)}] {icon} overlap={o5:.0%} r5={r5} r10={r10} | {query[:50]}...")

        elapsed = time.time() - t0
        results[config_name] = {"config": config, "results": config_results, "elapsed": round(elapsed, 1)}

    return results


def print_report(results: dict, n_queries: int):
    print(f"\n{'='*80}")
    print(f"  ABLATION TEST (v3 — no reranker, 384-token chunks + sentence windows)")
    print(f"{'='*80}")

    print(f"\n  {'Config':<20} {'Dense':>6} {'BM25':>6} {'Win':>5} {'PScr':>5} {'GT@5':>6} {'GT@10':>7} {'Avg':>6} {'Time':>6}")
    print(f"  {'─'*66}")

    for name, data in results.items():
        cfg = data["config"]
        res = data["results"]
        h5 = sum(1 for r in res if r["hit5"])
        h10 = sum(1 for r in res if r.get("rank10", -1) > 0 and r.get("rank10", -1) <= 10)
        avg = sum(r["overlap5"] for r in res) / len(res)
        d = "Y" if cfg["use_dense"] else "."
        b = "Y" if cfg["use_bm25"] else "."
        w = "Y" if cfg["use_windows"] else "."
        p = "Y" if cfg["use_passage_score"] else "."
        print(f"  {name:<20} {d:>6} {b:>6} {w:>5} {p:>5} {h5:>3}/{n_queries:<2} {h10:>4}/{n_queries:<2} {avg:>5.0%} {data['elapsed']:>5.0f}s")

    # Per-query breakdown
    config_names = list(results.keys())
    print(f"\n  {'#':<3} {'Query':<30}", end="")
    for name in config_names:
        print(f" {name[:8]:>9}", end="")
    print()
    print(f"  {'─'*(34 + 10*len(config_names))}")

    for i in range(n_queries):
        q = list(results.values())[0]["results"][i]["query"][:27] + "..."
        row = f"  {i+1:<3} {q:<30}"
        for name in config_names:
            r = results[name]["results"][i]
            icon = "+" if r["hit5"] else "-"
            row += f" {icon} {r['overlap5']:>5.0%} r{r['rank5']:>1}"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", default="data/eval/ao_ground_truth_cleaned.json")
    parser.add_argument("--configs", default="all")
    args = parser.parse_args()

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        log.error(f"Ground truth not found: {gt_path}")
        sys.exit(1)

    with open(gt_path) as f:
        ground_truth = json.load(f)
    log.info(f"Loaded {len(ground_truth)} test cases")
    log.info(f"Chunk collection: {CHROMA_COLLECTION}")
    log.info(f"Window collection: {WINDOW_COLLECTION}")

    if args.configs == "all":
        configs_to_run = list(CONFIGS.keys())
    else:
        configs_to_run = [c.strip() for c in args.configs.split(",")]

    results = run_ablation(ground_truth, configs_to_run)
    print_report(results, len(ground_truth))


if __name__ == "__main__":
    main()
