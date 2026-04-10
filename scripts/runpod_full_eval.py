#!/usr/bin/env python3
"""Full pipeline eval on RunPod — phrase search + reranker + LLM citation generation.

Runs entirely on the pod: extracts PDF, chunks, embeds, builds BM25,
then for each query: retrieves with phrase search + BM25 + dense + reranker,
generates LLM response, checks quote authenticity against source PDF.

No Docker, no external services — everything runs in-process on GPU.
"""

import json, logging, os, re, sys, time
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))
from shared.text_normalize import normalize_for_comparison

import fitz  # PyMuPDF
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
import httpx
import tiktoken

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval")

WS = Path("/workspace")
PDF = WS / "1972 Anti-Oedipus Capitalism and Schizophr - Deleuze, Gilles.pdf"
GT = WS / "ao_ground_truth.json"

# ── Extract + Chunk ─────────────────────────────────────────────────

def extract_and_chunk(pdf_path, chunk_size=768, overlap=100):
    doc = fitz.open(str(pdf_path))
    pages = []
    for i in range(len(doc)):
        t = doc[i].get_text("text")
        if t.strip():
            pages.append({"page": i+1, "text": t})
    doc.close()
    full = "\n\n".join(p["text"] for p in pages)
    log.info(f"Extracted {len(pages)} pages, {len(full)} chars")

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(full)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        text = enc.decode(tokens[start:end])
        chunks.append({"text": text, "index": len(chunks)})
        start = end - overlap if overlap > 0 else end
        if start >= end: break
    chunks = [c for c in chunks if c["text"].strip()]
    log.info(f"Chunked into {len(chunks)} pieces")
    return full, pages, chunks

# ── Embed ───────────────────────────────────────────────────────────

def embed_corpus(chunks, model_name="Qwen/Qwen3-Embedding-8B"):
    log.info(f"Loading {model_name} (flash_attention_2, device_map=auto)...")
    model = SentenceTransformer(model_name,
                                model_kwargs={
                                    "attn_implementation": "flash_attention_2",
                                    "device_map": "auto",
                                },
                                tokenizer_kwargs={"padding_side": "left"})
    log.info(f"Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embs = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    log.info(f"Embedded, dim={embs.shape[1]}")

    def query_fn(q):
        e = model.encode(q, convert_to_numpy=True)
        return e / (np.linalg.norm(e) + 1e-8)

    return embs, query_fn

# ── BM25 ────────────────────────────────────────────────────────────

def build_bm25(chunks):
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized)

def bm25_search(bm25, chunks, query, top_k=50):
    scores = bm25.get_scores(query.lower().split())
    idx = np.argsort(-scores)[:top_k]
    return [dict(chunks[i], bm25_score=float(scores[i])) for i in idx if scores[i] > 0]

# ── Dense search ────────────────────────────────────────────────────

def dense_search(embs, chunks, q_emb, top_k=50):
    scores = embs @ q_emb
    idx = np.argsort(-scores)[:top_k]
    return [dict(chunks[i], dense_score=float(scores[i])) for i in idx]

# ── Phrase search ───────────────────────────────────────────────────

def phrase_search(chunks, query, top_k=10):
    match = re.search(r'"([^"]{10,})"', query)
    if not match: return []
    phrase = re.sub(r"\s+", " ", match.group(1).lower()).strip()
    matches = []
    for c in chunks:
        if phrase[:30] in re.sub(r"\s+", " ", c["text"].lower()):
            matches.append(dict(c, phrase_score=100.0))
    return matches[:top_k]

# ── RRF fusion ──────────────────────────────────────────────────────

def rrf(dense, sparse, k=60, dw=0.5, sw=0.5):
    scores, cmap = {}, {}
    for r, c in enumerate(dense):
        scores[c["index"]] = scores.get(c["index"], 0) + dw/(k+r+1)
        cmap[c["index"]] = c
    for r, c in enumerate(sparse):
        scores[c["index"]] = scores.get(c["index"], 0) + sw/(k+r+1)
        if c["index"] not in cmap: cmap[c["index"]] = c
    return [cmap[i] for i, _ in sorted(scores.items(), key=lambda x: -x[1])]

# ── Reranker ────────────────────────────────────────────────────────

def load_reranker(name="tomaarsen/Qwen3-Reranker-0.6B-seq-cls"):
    log.info(f"Loading reranker {name}...")
    return CrossEncoder(name, device="cuda",
                        automodel_args={"torch_dtype": torch.float16})

def rerank(model, query, chunks, top_k=5):
    if not chunks: return []
    pairs = [[query, c["text"]] for c in chunks]
    scores = model.predict(pairs, batch_size=64)
    for i, c in enumerate(chunks): c["rerank_score"] = float(scores[i])
    return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)[:top_k]

# ── GT check ────────────────────────────────────────────────────────

def check_gt(gt_text, chunks):
    gt_n = normalize_for_comparison(gt_text)
    for rank, c in enumerate(chunks):
        cn = normalize_for_comparison(c["text"])
        for sl in [80, 50, 40, 30]:
            if len(gt_n) >= sl and gt_n[:sl] in cn:
                return 1.0, rank+1
        prefix = gt_n[:150]
        if len(prefix) < len(cn):
            step = max(1, len(prefix)//4)
            for s in range(0, len(cn)-len(prefix)+1, step):
                if SequenceMatcher(None, prefix, cn[s:s+len(prefix)]).ratio() >= 0.7:
                    return 0.8, rank+1
    return 0.0, -1

# ── Quote authenticity (check against full PDF) ─────────────────────

def check_quote_in_pdf(quote, full_text):
    qn = normalize_for_comparison(quote)
    fn = normalize_for_comparison(full_text)
    if qn[:80] in fn: return 1.0
    # Sliding window
    qlen = len(qn)
    if qlen >= len(fn): return SequenceMatcher(None, qn, fn).ratio()
    best = 0.0
    step = max(1, qlen // 6)
    for s in range(0, len(fn) - qlen + 1, step):
        r = SequenceMatcher(None, qn, fn[s:s+qlen]).ratio()
        if r > best:
            best = r
            if best >= 0.95: return best
    return best

# ── LLM generation (uses Ollama on host if available, else skips) ──

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "")

def generate_response(query, context_chunks):
    """Build RAG prompt and generate LLM response. Returns empty if no Ollama."""
    if not OLLAMA_HOST:
        return ""
    context = "\n\n---\n\n".join(
        f"[Source: Anti-Oedipus, p. {c.get('index','?')}]\n{c['text']}"
        for c in context_chunks
    )
    prompt = f"""You are a scholarly citation assistant. Quote verbatim from the context. Cite with page numbers.

## Context
{context}

## Question
{query}"""
    try:
        resp = httpx.post(f"{OLLAMA_HOST}/api/chat", json={
            "model": "qwen3:8b", "stream": False,
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": query}],
            "options": {"temperature": 0.1, "num_predict": 512},
        }, timeout=120)
        return resp.json().get("message", {}).get("content", "")
    except:
        return ""

def extract_quotes(text):
    quotes = []
    for m in re.finditer(r'"([^"]{15,})"', text): quotes.append(m.group(1))
    for m in re.finditer(r'\u201c([^\u201d]{15,})\u201d', text): quotes.append(m.group(1))
    return list(dict.fromkeys(quotes))

# ── Main ────────────────────────────────────────────────────────────

CONFIGS = {
    "bm25_rerank": {"dense": False, "bm25": True, "phrase": False, "rerank": True},
    "bm25_phrase_rerank": {"dense": False, "bm25": True, "phrase": True, "rerank": True},
    "full_phrase": {"dense": True, "bm25": True, "phrase": True, "rerank": True},
}

def main():
    log.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")

    with open(GT) as f: ground_truth = json.load(f)
    log.info(f"Ground truth: {len(ground_truth)} queries")

    full_text, pages, chunks = extract_and_chunk(PDF)
    corpus_embs, query_embed_fn = embed_corpus(chunks)

    # Free embedding model VRAM for reranker
    torch.cuda.empty_cache()

    bm25 = build_bm25(chunks)
    reranker = load_reranker()

    all_results = {}
    for cfg_name, cfg in CONFIGS.items():
        log.info(f"\n{'='*60}\n  {cfg_name}\n{'='*60}")
        results = []
        for i, tc in enumerate(ground_truth):
            q = tc["query"]
            merged = []

            if cfg["dense"]:
                qe = query_embed_fn(q)
                merged = dense_search(corpus_embs, chunks, qe, top_k=50)

            if cfg["bm25"]:
                bm25_res = bm25_search(bm25, chunks, q, top_k=50)
                merged = rrf(merged, bm25_res) if merged else bm25_res

            if cfg["phrase"]:
                pr = phrase_search(chunks, q)
                ids = {c["index"] for c in merged}
                for p in pr:
                    if p["index"] not in ids:
                        merged.insert(0, p)

            if cfg["rerank"]:
                top5 = rerank(reranker, q, merged[:50], top_k=5)
            else:
                top5 = merged[:5]

            overlap, rank = check_gt(tc["ground_truth"], top5)
            hit = overlap >= 0.7
            results.append({"query": q, "hit": hit, "overlap": round(overlap, 3), "rank": rank})
            icon = "✓" if hit else "✗"
            log.info(f"  [{i+1}/{len(ground_truth)}] {icon} overlap={overlap:.0%} rank={rank} | {q[:50]}...")

        hits = sum(r["hit"] for r in results)
        all_results[cfg_name] = {"hits": hits, "total": len(results), "pct": round(100*hits/len(results), 1), "details": results}
        log.info(f"  → GT@5: {hits}/{len(results)} ({100*hits/len(results):.0f}%)")

    # End-to-end citation eval with LLM (if Ollama available)
    if OLLAMA_HOST:
        log.info(f"\n{'='*60}\n  END-TO-END CITATION EVAL (LLM)\n{'='*60}")
        best_cfg = max(all_results, key=lambda k: all_results[k]["hits"])
        cfg = CONFIGS[best_cfg]
        log.info(f"Using best config: {best_cfg}")

        citation_results = []
        for i, tc in enumerate(ground_truth):
            q = tc["query"]
            merged = []
            if cfg["dense"]:
                merged = dense_search(corpus_embs, chunks, query_embed_fn(q), top_k=50)
            if cfg["bm25"]:
                bm25_res = bm25_search(bm25, chunks, q, top_k=50)
                merged = rrf(merged, bm25_res) if merged else bm25_res
            if cfg["phrase"]:
                pr = phrase_search(chunks, q)
                ids = {c["index"] for c in merged}
                for p in pr:
                    if p["index"] not in ids: merged.insert(0, p)
            if cfg["rerank"]:
                top5 = rerank(reranker, q, merged[:50], top_k=5)
            else:
                top5 = merged[:5]

            response = generate_response(q, top5)
            quotes = extract_quotes(response)
            best_pdf = max((check_quote_in_pdf(qq, full_text) for qq in quotes), default=0.0) if quotes else 0.0

            icon = "✓" if best_pdf >= 0.9 else "≈" if best_pdf >= 0.7 else "✗"
            log.info(f"  [{i+1}/{len(ground_truth)}] {icon} pdf={best_pdf:.0%} quotes={len(quotes)} | {q[:50]}...")
            citation_results.append({"query": q, "pdf_overlap": round(best_pdf, 3), "num_quotes": len(quotes)})

        pdf_scores = [r["pdf_overlap"] for r in citation_results]
        verified = sum(1 for s in pdf_scores if s >= 0.9)
        log.info(f"\n  Citation accuracy: {sum(pdf_scores)/len(pdf_scores):.1%} mean, {verified}/{len(pdf_scores)} verified")
        all_results["citation_eval"] = {"mean": round(sum(pdf_scores)/len(pdf_scores), 3), "verified": verified, "total": len(pdf_scores), "details": citation_results}

    # Leaderboard
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    for name, r in sorted(all_results.items(), key=lambda x: -x[1].get("hits", 0)):
        if "hits" in r:
            print(f"  {name:<30} GT@5: {r['hits']}/{r['total']} ({r['pct']}%)")
        elif "mean" in r:
            print(f"  {name:<30} Citation: {r['mean']:.1%} mean, {r['verified']}/{r['total']} verified")

    with open(WS / "eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nResults saved to {WS / 'eval_results.json'}")

if __name__ == "__main__":
    main()
