#!/usr/bin/env python3
"""RunPod evaluation harness — full pipeline matrix test.

Runs on RunPod A6000. Tests embedding models, rerankers, and retrieval
strategies against Anti-Oedipus ground truth.

Usage: python3 runpod_eval_harness.py
"""

import json
import logging
import os
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))
from shared.text_normalize import normalize_for_comparison

import fitz  # PyMuPDF
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("runpod_eval")

WORKSPACE = Path("/workspace")
PDF_PATH = WORKSPACE / "anti-oedipus.pdf"
GT_PATH = WORKSPACE / "ao_ground_truth.json"
RESULTS_PATH = WORKSPACE / "eval_results.json"

# ── Text extraction + chunking ──────────────────────────────────────

def extract_pdf_text(pdf_path: Path) -> tuple[str, list[dict]]:
    """Extract per-page text from PDF."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text")
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    doc.close()
    full_text = "\n\n".join(p["text"] for p in pages)
    return full_text, pages


def simple_chunk(text: str, chunk_size: int = 768, overlap: int = 100) -> list[dict]:
    """Simple token-based chunking with overlap."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_text = enc.decode(tokens[start:end])
        chunks.append({"text": chunk_text, "index": len(chunks)})
        start = end - overlap if overlap > 0 else end
        if start >= end:
            break
    return [c for c in chunks if c["text"].strip()]


# ── Embedding ───────────────────────────────────────────────────────

_embed_models = {}

def get_embed_model(model_name: str) -> SentenceTransformer:
    if model_name not in _embed_models:
        log.info(f"Loading embedding model: {model_name} (flash_attention_2, device_map=auto)")
        _embed_models[model_name] = SentenceTransformer(
            model_name,
            model_kwargs={
                "attn_implementation": "flash_attention_2",
                "device_map": "auto",
            },
            tokenizer_kwargs={"padding_side": "left"},
        )
        log.info(f"  Loaded, dim={_embed_models[model_name].get_sentence_embedding_dimension()}")
    return _embed_models[model_name]


def embed_texts(texts: list[str], model_name: str, batch_size: int = 32) -> list[list[float]]:
    model = get_embed_model(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(query: str, model_name: str) -> list[float]:
    model = get_embed_model(model_name)
    return model.encode(query, convert_to_numpy=True).tolist()


# ── Reranking ───────────────────────────────────────────────────────

_rerank_models = {}

def get_rerank_model(model_name: str) -> CrossEncoder:
    if model_name not in _rerank_models:
        log.info(f"Loading reranker model: {model_name} (FP16)")
        _rerank_models[model_name] = CrossEncoder(
            model_name, device="cuda",
            automodel_args={"torch_dtype": torch.float16},
        )
        log.info(f"  Loaded on GPU")
    return _rerank_models[model_name]


def rerank(query: str, chunks: list[dict], model_name: str, top_k: int = 5) -> list[dict]:
    model = get_rerank_model(model_name)
    pairs = [[query, c["text"]] for c in chunks]
    scores = model.predict(pairs, batch_size=32)
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])
    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]


# ── BM25 ────────────────────────────────────────────────────────────

_bm25_cache = None

def build_bm25(chunks: list[dict]):
    global _bm25_cache
    tokenized = [c["text"].lower().split() for c in chunks]
    _bm25_cache = (chunks, BM25Okapi(tokenized))
    log.info(f"BM25 index built: {len(chunks)} docs")


def bm25_search(query: str, top_k: int = 50) -> list[dict]:
    chunks, index = _bm25_cache
    scores = index.get_scores(query.lower().split())
    scored = [(chunks[i], float(scores[i])) for i in range(len(scores)) if scores[i] > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, s in scored[:top_k]]


# ── Dense search via numpy ──────────────────────────────────────────

import numpy as np

_corpus_embeddings = {}

def build_dense_index(chunks: list[dict], embeddings: list[list[float]], model_name: str):
    _corpus_embeddings[model_name] = {
        "chunks": chunks,
        "embeddings": np.array(embeddings, dtype=np.float32),
    }
    # Normalize for cosine similarity
    norms = np.linalg.norm(_corpus_embeddings[model_name]["embeddings"], axis=1, keepdims=True)
    _corpus_embeddings[model_name]["embeddings"] /= (norms + 1e-8)


def dense_search(query_embedding: list[float], model_name: str, top_k: int = 50) -> list[dict]:
    data = _corpus_embeddings[model_name]
    q = np.array(query_embedding, dtype=np.float32)
    q /= (np.linalg.norm(q) + 1e-8)
    scores = data["embeddings"] @ q
    top_idx = np.argsort(-scores)[:top_k]
    return [dict(data["chunks"][i], dense_score=float(scores[i])) for i in top_idx]


# ── RRF fusion ──────────────────────────────────────────────────────

def rrf_fusion(dense: list[dict], sparse: list[dict], k: int = 60,
               dense_weight: float = 0.5, sparse_weight: float = 0.5) -> list[dict]:
    scores = {}
    chunk_map = {}
    for rank, c in enumerate(dense):
        cid = c["index"]
        scores[cid] = scores.get(cid, 0) + dense_weight / (k + rank + 1)
        chunk_map[cid] = c
    for rank, c in enumerate(sparse):
        cid = c["index"]
        scores[cid] = scores.get(cid, 0) + sparse_weight / (k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = c
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[cid] for cid, _ in ranked]


# ── Evaluation ──────────────────────────────────────────────────────

def check_gt(gt_text: str, chunks: list[dict]) -> tuple[float, int]:
    gt_norm = normalize_for_comparison(gt_text)
    for rank, chunk in enumerate(chunks):
        chunk_norm = normalize_for_comparison(chunk["text"])
        for substr_len in [80, 50, 40, 30]:
            if len(gt_norm) >= substr_len and gt_norm[:substr_len] in chunk_norm:
                return 1.0, rank + 1
        gt_prefix = gt_norm[:150]
        if len(gt_prefix) < len(chunk_norm):
            step = max(1, len(gt_prefix) // 4)
            for start in range(0, len(chunk_norm) - len(gt_prefix) + 1, step):
                window = chunk_norm[start:start + len(gt_prefix)]
                ratio = SequenceMatcher(None, gt_prefix, window).ratio()
                if ratio >= 0.7:
                    return ratio, rank + 1
    return 0.0, -1


# ── Main evaluation matrix ─────────────────────────────────────────

EMBED_MODELS = {
    "qwen3-8b": "Qwen/Qwen3-Embedding-8B",
}

RERANKER_MODELS = {
    "qwen3-reranker-0.6b": "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
    "bge-v2-m3": "BAAI/bge-reranker-v2-m3",
}

STRATEGIES = {
    "dense_only": {"use_dense": True, "use_bm25": False, "use_reranker": False},
    "bm25_only": {"use_dense": False, "use_bm25": True, "use_reranker": False},
    "bm25_rerank": {"use_dense": False, "use_bm25": True, "use_reranker": True},
    "hybrid": {"use_dense": True, "use_bm25": True, "use_reranker": False},
    "hybrid_rerank": {"use_dense": True, "use_bm25": True, "use_reranker": True},
}


def run_config(ground_truth, embed_model_key, reranker_key, strategy, chunks):
    cfg = STRATEGIES[strategy]
    results = []

    for i, tc in enumerate(ground_truth):
        query = tc["query"]

        retrieved = []
        if cfg["use_dense"]:
            q_emb = embed_query(query, EMBED_MODELS[embed_model_key])
            retrieved = dense_search(q_emb, embed_model_key, top_k=50)

        if cfg["use_bm25"]:
            bm25_results = bm25_search(query, top_k=50)
            if retrieved:
                retrieved = rrf_fusion(retrieved, bm25_results)
            else:
                retrieved = bm25_results

        if cfg["use_reranker"] and reranker_key:
            retrieved = rerank(query, retrieved[:50], RERANKER_MODELS[reranker_key], top_k=5)
        else:
            retrieved = retrieved[:5]

        overlap, rank = check_gt(tc["ground_truth"], retrieved)
        results.append({"query": query, "hit": overlap >= 0.7, "overlap": round(overlap, 3), "rank": rank})

    hits = sum(1 for r in results if r["hit"])
    return {"hits": hits, "total": len(results), "pct": round(hits / len(results) * 100, 1), "details": results}


def main():
    log.info("=== RunPod Eval Harness ===")
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load ground truth
    with open(GT_PATH) as f:
        ground_truth = json.load(f)
    log.info(f"Ground truth: {len(ground_truth)} queries")

    # Extract and chunk the PDF
    log.info("Extracting PDF text...")
    full_text, pages = extract_pdf_text(PDF_PATH)
    log.info(f"Extracted {len(pages)} pages, {len(full_text)} chars")

    chunks = simple_chunk(full_text, chunk_size=768, overlap=100)
    log.info(f"Chunked into {len(chunks)} pieces")

    # Build BM25 index
    build_bm25(chunks)

    # Embed with each model
    for key, model_name in EMBED_MODELS.items():
        log.info(f"\n{'='*60}")
        log.info(f"Embedding with {key} ({model_name})")
        t0 = time.time()
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts, model_name)
        embed_time = time.time() - t0
        log.info(f"Embedded {len(chunks)} chunks in {embed_time:.1f}s")
        build_dense_index(chunks, embeddings, key)

        # Free GPU memory for reranker
        del _embed_models[key]
        torch.cuda.empty_cache()

    # Run matrix
    all_results = {}
    for embed_key in EMBED_MODELS:
        for strategy in STRATEGIES:
            cfg = STRATEGIES[strategy]
            reranker_keys = list(RERANKER_MODELS.keys()) if cfg["use_reranker"] else [None]

            for reranker_key in reranker_keys:
                config_name = f"{embed_key}_{strategy}"
                if reranker_key:
                    config_name += f"_{reranker_key}"

                log.info(f"\n--- {config_name} ---")
                t0 = time.time()
                result = run_config(ground_truth, embed_key, reranker_key, strategy, chunks)
                elapsed = time.time() - t0
                result["elapsed"] = round(elapsed, 1)
                result["config"] = config_name
                all_results[config_name] = result
                log.info(f"  GT@5: {result['hits']}/{result['total']} ({result['pct']}%) in {elapsed:.1f}s")

                # Free reranker after use
                if reranker_key and reranker_key in _rerank_models:
                    del _rerank_models[reranker_key]
                    torch.cuda.empty_cache()

    # Print leaderboard
    print(f"\n{'='*70}")
    print(f"  LEADERBOARD")
    print(f"{'='*70}")
    print(f"  {'Config':<50} {'GT@5':>6} {'Time':>6}")
    print(f"  {'─'*64}")
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]["hits"]):
        print(f"  {name:<50} {r['hits']:>2}/{r['total']:<2} {r['elapsed']:>5.1f}s")

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
