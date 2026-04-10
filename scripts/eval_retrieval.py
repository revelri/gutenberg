#!/usr/bin/env python3
"""Retrieval evaluation framework for Gutenberg.

Runs gold-standard queries against ChromaDB, measures retrieval quality
with precision@k, MRR, and NDCG@k, and outputs a comparison table.

Usage:
    # Evaluate current embedding model
    python scripts/eval_retrieval.py

    # Evaluate with a specific model (requires re-embedding)
    python scripts/eval_retrieval.py --model mxbai-embed-large

    # Use custom query set
    python scripts/eval_retrieval.py --queries data/eval/queries.jsonl

    # Adjust retrieval depth
    python scripts/eval_retrieval.py --top-k 10
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "http://localhost:8200")
COLLECTION = os.environ.get("CHROMA_COLLECTION", "gutenberg")
DEFAULT_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def embed_query(text: str, model: str) -> list[float]:
    """Embed a query string via Ollama."""
    resp = httpx.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": model, "input": [text]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def search_chroma(query_embedding: list[float], top_k: int) -> list[dict]:
    """Query ChromaDB directly via REST API."""
    import chromadb

    host = CHROMA_HOST.replace("http://", "").replace("https://", "")
    parts = host.split(":")
    hostname = parts[0]
    port = int(parts[1]) if len(parts) > 1 else 8000

    client = chromadb.HttpClient(host=hostname, port=port)
    collection = client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results["ids"] and results["ids"][0]:
        for id_, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "id": id_,
                "text": doc,
                "metadata": meta,
                "score": 1 - dist,
            })
    return chunks


def load_queries(path: Path) -> list[dict]:
    """Load gold-standard queries from JSONL.

    Each line should be a JSON object with:
    - query: str — the search query
    - expected_source: str — expected filename (partial match)
    - expected_terms: list[str] — terms that should appear in retrieved text
    - expected_page_range: [int, int] (optional) — expected page range
    """
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def evaluate_result(result: list[dict], gold: dict) -> dict:
    """Score a single query's retrieval results against gold standard."""
    expected_source = gold.get("expected_source", "").lower()
    expected_terms = [t.lower() for t in gold.get("expected_terms", [])]
    expected_pages = gold.get("expected_page_range", [0, 0])

    scores = {
        "source_hits": [],
        "term_hits": [],
        "page_hits": [],
    }

    for i, chunk in enumerate(result):
        text_lower = chunk["text"].lower()
        meta = chunk.get("metadata", {})
        source = meta.get("source", "").lower()

        # Source match
        source_match = expected_source in source if expected_source else True
        scores["source_hits"].append(source_match)

        # Term match — what fraction of expected terms appear in this chunk?
        if expected_terms:
            found = sum(1 for t in expected_terms if t in text_lower)
            scores["term_hits"].append(found / len(expected_terms))
        else:
            scores["term_hits"].append(1.0)

        # Page match
        if expected_pages and expected_pages != [0, 0]:
            page_start = meta.get("page_start", 0)
            page_end = meta.get("page_end", 0)
            page_match = (
                page_start >= expected_pages[0] and page_end <= expected_pages[1]
            ) if page_start and page_end else False
            scores["page_hits"].append(page_match)

    return scores


def precision_at_k(hits: list[bool], k: int) -> float:
    """Precision@K: fraction of top-k results that are relevant."""
    top_k = hits[:k]
    if not top_k:
        return 0.0
    return sum(1 for h in top_k if h) / len(top_k)


def mrr(hits: list[bool]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant result."""
    for i, hit in enumerate(hits):
        if hit:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(relevance_scores: list[float], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    top_k = relevance_scores[:k]
    if not top_k:
        return 0.0

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(top_k))
    ideal = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))

    return dcg / idcg if idcg > 0 else 0.0


def run_eval(queries: list[dict], model: str, top_k: int) -> dict:
    """Run full evaluation suite."""
    results = {
        "model": model,
        "top_k": top_k,
        "num_queries": len(queries),
        "per_query": [],
        "aggregate": {},
    }

    all_precision = []
    all_mrr = []
    all_ndcg = []
    all_term_coverage = []

    for i, gold in enumerate(queries):
        query = gold["query"]
        log.info(f"[{i+1}/{len(queries)}] {query[:80]}...")

        # Embed and search
        embedding = embed_query(query, model)
        chunks = search_chroma(embedding, top_k)

        # Evaluate
        scores = evaluate_result(chunks, gold)

        # Compute metrics
        # Relevance = source_match AND term_coverage > 0.5
        relevance = [
            (1.0 if scores["source_hits"][j] and scores["term_hits"][j] > 0.3 else 0.0)
            for j in range(len(chunks))
        ]
        bool_relevance = [r > 0 for r in relevance]

        p_at_k = precision_at_k(bool_relevance, top_k)
        query_mrr = mrr(bool_relevance)
        query_ndcg = ndcg_at_k(relevance, top_k)
        avg_term_cov = sum(scores["term_hits"]) / len(scores["term_hits"]) if scores["term_hits"] else 0

        all_precision.append(p_at_k)
        all_mrr.append(query_mrr)
        all_ndcg.append(query_ndcg)
        all_term_coverage.append(avg_term_cov)

        query_result = {
            "query": query,
            "precision_at_k": round(p_at_k, 3),
            "mrr": round(query_mrr, 3),
            "ndcg_at_k": round(query_ndcg, 3),
            "term_coverage": round(avg_term_cov, 3),
            "top_chunk_preview": chunks[0]["text"][:150] if chunks else "",
            "top_chunk_source": chunks[0]["metadata"].get("source", "") if chunks else "",
            "top_chunk_pages": f"{chunks[0]['metadata'].get('page_start', '?')}-{chunks[0]['metadata'].get('page_end', '?')}" if chunks else "",
        }
        results["per_query"].append(query_result)

    # Aggregate
    n = len(queries)
    results["aggregate"] = {
        "mean_precision_at_k": round(sum(all_precision) / n, 3) if n else 0,
        "mean_mrr": round(sum(all_mrr) / n, 3) if n else 0,
        "mean_ndcg_at_k": round(sum(all_ndcg) / n, 3) if n else 0,
        "mean_term_coverage": round(sum(all_term_coverage) / n, 3) if n else 0,
    }

    return results


def print_results(results: dict):
    """Print results as a formatted table."""
    agg = results["aggregate"]
    print(f"\n{'='*70}")
    print(f"RETRIEVAL EVALUATION — {results['model']}")
    print(f"{'='*70}")
    print(f"Queries: {results['num_queries']}  |  Top-K: {results['top_k']}")
    print(f"\n{'Metric':<25} {'Score':>10}")
    print(f"{'-'*35}")
    print(f"{'Precision@K':<25} {agg['mean_precision_at_k']:>10.3f}")
    print(f"{'MRR':<25} {agg['mean_mrr']:>10.3f}")
    print(f"{'NDCG@K':<25} {agg['mean_ndcg_at_k']:>10.3f}")
    print(f"{'Term Coverage':<25} {agg['mean_term_coverage']:>10.3f}")

    print(f"\n{'='*70}")
    print("PER-QUERY BREAKDOWN")
    print(f"{'='*70}")
    print(f"{'Query':<45} {'P@K':>6} {'MRR':>6} {'NDCG':>6} {'Terms':>6}")
    print(f"{'-'*75}")
    for q in results["per_query"]:
        short = q["query"][:42] + "..." if len(q["query"]) > 45 else q["query"]
        print(f"{short:<45} {q['precision_at_k']:>6.3f} {q['mrr']:>6.3f} {q['ndcg_at_k']:>6.3f} {q['term_coverage']:>6.3f}")

    print(f"\n{'='*70}")
    print("TOP RETRIEVALS")
    print(f"{'='*70}")
    for q in results["per_query"]:
        print(f"\nQ: {q['query'][:80]}")
        print(f"   Source: {q['top_chunk_source']}, pp. {q['top_chunk_pages']}")
        print(f"   Preview: {q['top_chunk_preview'][:120]}...")


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL, help="Embedding model to use")
    parser.add_argument("--queries", default="data/eval/queries.jsonl", help="Path to gold-standard queries")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--output", default=None, help="Save results JSON to file")
    args = parser.parse_args()

    queries_path = Path(args.queries)
    if not queries_path.exists():
        log.error(f"Queries file not found: {queries_path}")
        sys.exit(1)

    queries = load_queries(queries_path)
    log.info(f"Loaded {len(queries)} queries from {queries_path}")

    results = run_eval(queries, args.model, args.top_k)
    print_results(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
