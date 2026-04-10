#!/usr/bin/env python3
"""Phrase search eval — all inference via OpenRouter, local CPU only does string matching.

Tests: Does adding exact phrase search to BM25+reranker reach 90%+ GT@5?
"""

import json, logging, os, re, sys, time
from difflib import SequenceMatcher
from pathlib import Path

import fitz
import httpx
import tiktoken
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))
from shared.text_normalize import normalize_for_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval")

OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
PDF_PATH = Path("data/processed/1972 Anti-Oedipus Capitalism and Schizophr - Deleuze, Gilles.pdf")
GT_PATH = Path("data/eval/ao_ground_truth.json")

# ── Extract + chunk (CPU, ~1s) ──────────────────────────────────────

def extract_and_chunk(pdf_path, chunk_size=768, overlap=100):
    doc = fitz.open(str(pdf_path))
    pages = [{"page": i+1, "text": doc[i].get_text("text")} for i in range(len(doc)) if doc[i].get_text("text").strip()]
    doc.close()
    full = "\n\n".join(p["text"] for p in pages)
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(full)
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append({"text": enc.decode(tokens[start:end]), "index": len(chunks)})
        start = end - overlap if overlap else end
        if start >= end: break
    return full, [c for c in chunks if c["text"].strip()]

# ── BM25 (CPU, instant) ────────────────────────────────────────────

def build_bm25(chunks):
    return BM25Okapi([c["text"].lower().split() for c in chunks])

def bm25_search(bm25, chunks, query, top_k=50):
    scores = bm25.get_scores(query.lower().split())
    idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [dict(chunks[i], bm25_score=float(scores[i])) for i in idx if scores[i] > 0]

# ── Phrase search (CPU, instant) ────────────────────────────────────

def phrase_search(chunks, query, top_k=10):
    match = re.search(r'"([^"]{10,})"', query)
    if not match: return []
    phrase = re.sub(r"\s+", " ", match.group(1).lower()).strip()
    return [dict(c, phrase_score=100.0) for c in chunks if phrase[:30] in re.sub(r"\s+", " ", c["text"].lower())][:top_k]

# ── Rerank via OpenRouter (API call) ───────────────────────────────

def rerank_openrouter(query, chunks, top_k=5):
    """Rerank using Qwen3 as a cross-encoder via OpenRouter chat completion.

    Since OpenRouter doesn't have a native rerank endpoint, we use the LLM
    to score relevance of each chunk to the query.
    """
    if not chunks: return []
    # For efficiency, build a single prompt that scores all chunks
    chunk_texts = "\n\n".join(f"[DOC {i}]: {c['text'][:300]}" for i, c in enumerate(chunks[:20]))
    prompt = f"""Rate each document's relevance to the query on a scale of 0-10. Return ONLY a JSON array of scores, e.g. [8, 3, 9, ...]. No explanation.

Query: {query}

{chunk_texts}"""

    try:
        resp = httpx.post("https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
            json={"model": "qwen/qwen3-8b", "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0, "max_tokens": 200},
            timeout=30)
        text = resp.json()["choices"][0]["message"]["content"]
        # Strip thinking tags if present
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        scores = json.loads(re.search(r"\[[\d,\s.]+\]", text).group())
        for i, c in enumerate(chunks[:len(scores)]):
            c["rerank_score"] = float(scores[i])
        return sorted(chunks[:len(scores)], key=lambda c: c.get("rerank_score", 0), reverse=True)[:top_k]
    except Exception as e:
        log.warning(f"Rerank failed: {e}, returning BM25 order")
        return chunks[:top_k]

# ── GT check (CPU, instant) ─────────────────────────────────────────

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

# ── Main ────────────────────────────────────────────────────────────

def main():
    with open(GT_PATH) as f: ground_truth = json.load(f)
    log.info(f"{len(ground_truth)} queries loaded")

    full_text, chunks = extract_and_chunk(PDF_PATH)
    log.info(f"{len(chunks)} chunks")
    bm25 = build_bm25(chunks)

    CONFIGS = {
        "bm25_only": {"bm25": True, "phrase": False, "rerank": False},
        "bm25_rerank": {"bm25": True, "phrase": False, "rerank": True},
        "bm25_phrase": {"bm25": True, "phrase": True, "rerank": False},
        "bm25_phrase_rerank": {"bm25": True, "phrase": True, "rerank": True},
    }

    for cfg_name, cfg in CONFIGS.items():
        log.info(f"\n{'='*50}\n  {cfg_name}\n{'='*50}")
        results = []
        for i, tc in enumerate(ground_truth):
            q = tc["query"]
            merged = bm25_search(bm25, chunks, q, top_k=50) if cfg["bm25"] else []

            if cfg["phrase"]:
                pr = phrase_search(chunks, q)
                ids = {c["index"] for c in merged}
                for p in pr:
                    if p["index"] not in ids: merged.insert(0, p)

            if cfg["rerank"]:
                top5 = rerank_openrouter(q, merged[:20], top_k=5)
            else:
                top5 = merged[:5]

            overlap, rank = check_gt(tc["ground_truth"], top5)
            hit = overlap >= 0.7
            results.append({"hit": hit, "overlap": round(overlap, 3), "rank": rank})
            icon = "✓" if hit else "✗"
            log.info(f"  [{i+1}/{len(ground_truth)}] {icon} {overlap:.0%} r{rank} | {q[:50]}...")

        hits = sum(r["hit"] for r in results)
        log.info(f"  → {hits}/{len(results)} ({100*hits/len(results):.0f}%)")

    log.info("\nDone.")

if __name__ == "__main__":
    main()
