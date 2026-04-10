#!/usr/bin/env python3
"""A Thousand Plateaus evaluation harness: exact citation retrieval + general query generation.

Two evaluation modes:
  1. Exact Citation Retrieval — retrieve top-10 chunks, check if ground truth appears
  2. General Query Answer Generation — full RAG pipeline with LLM, verify quotes against PDF

Usage:
    uv run scripts/eval_atp.py --mode both --configs all
    uv run scripts/eval_atp.py --mode exact --configs dense_only,hybrid
    uv run scripts/eval_atp.py --mode general --configs full_pipeline --colbert
"""

import argparse
import json
import logging
import os
import re
import subprocess
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
log = logging.getLogger("eval_atp")

# ── Environment ────────────────────────────────────────────────────────
CHROMA_HOST = os.environ.get("CHROMA_HOST", "http://localhost:8200")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "gutenberg-qwen3-v3")
WINDOW_COLLECTION = CHROMA_COLLECTION + "-windows"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b")
LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "qwen3:8b")

# ── ChromaDB ───────────────────────────────────────────────────────────
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


# ── Source filtering ───────────────────────────────────────────────────
_SOURCE_PATTERNS = [
    (re.compile(r"in\s+A\s+Thousand\s+Plateaus", re.IGNORECASE), "1980 A Thousand Plateaus"),
    (re.compile(r"in\s+Anti-Oedipus", re.IGNORECASE), "1972 Anti-Oedipus"),
    (re.compile(r"in\s+Kafka", re.IGNORECASE), "Kafka"),
    (re.compile(r"in\s+Difference\s+and\s+Repetition", re.IGNORECASE), "Difference and Repetition"),
    (re.compile(r"in\s+Logic\s+of\s+Sense", re.IGNORECASE), "Logic of Sense"),
]


def extract_source_filter(query: str) -> str | None:
    for pattern, source_name in _SOURCE_PATTERNS:
        if pattern.search(query):
            return source_name
    return None


# ── Embedding ──────────────────────────────────────────────────────────
def embed_query(text: str) -> list[float]:
    resp = httpx.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBED_MODEL, "input": [text]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


# ── Retrieval components ───────────────────────────────────────────────
def dense_search(
    query_embedding: list[float],
    top_k: int = 200,
    collection_name: str | None = None,
    where: dict | None = None,
) -> list[dict]:
    col = get_collection(collection_name)
    if col.count() == 0:
        return []
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, col.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    results = col.query(**kwargs)
    chunks = []
    if results["ids"] and results["ids"][0]:
        for id_, doc, meta, dist in zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0],
        ):
            chunks.append({"id": id_, "text": doc, "metadata": meta, "dense_score": 1 - dist})
    return chunks


_bm25_cache = None


def build_bm25(where_source: str | None = None):
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


def bm25_search(query: str, top_k: int = 200, source_filter: str | None = None) -> list[dict]:
    corpus, index = build_bm25()
    scores = index.get_scores(query.lower().split())
    scored = []
    for i, score in enumerate(scores):
        if score <= 0:
            continue
        if source_filter and source_filter not in corpus[i].get("metadata", {}).get("source", ""):
            continue
        scored.append({
            "id": corpus[i]["id"], "text": corpus[i]["text"],
            "metadata": corpus[i]["metadata"], "bm25_score": float(score),
        })
    scored.sort(key=lambda x: x["bm25_score"], reverse=True)
    return scored[:top_k]


def rrf_fusion(
    list_a: list[dict], list_b: list[dict], k: int = 60,
    weight_a: float = 1.0, weight_b: float = 1.0,
) -> list[dict]:
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
    if not chunks:
        return []

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

        q_words_list = query_lower.split()
        for j in range(len(q_words_list) - 1):
            bg = f"{q_words_list[j]} {q_words_list[j+1]}"
            if len(bg) > 5 and bg in chunk_lower:
                score += 5.0

        chunk["passage_score"] = score

    ranked = sorted(chunks, key=lambda c: c.get("passage_score", 0), reverse=True)
    return ranked[:top_k]


def phrase_search(query: str, top_k: int = 10, source_filter: str | None = None) -> list[dict]:
    m = re.search(r'"([^"]{10,})"', query)
    if not m:
        return []
    phrase = re.sub(r"\s+", " ", m.group(1).lower()).strip()
    corpus, _ = build_bm25()
    matches = []
    for chunk in corpus:
        if source_filter and source_filter not in chunk.get("metadata", {}).get("source", ""):
            continue
        chunk_norm = re.sub(r"\s+", " ", chunk["text"].lower())
        if phrase[:30] in chunk_norm:
            matches.append(dict(chunk, phrase_score=100.0))
    return matches[:top_k]


# ── SPLADE query expansion (stub) ─────────────────────────────────────
def splade_expand_query(query: str) -> list[str]:
    """Expand query terms using SPLADE sparse model.

    Falls back to simple synonym expansion if the model is unavailable.
    """
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        import torch

        model_name = "naver/splade-cocondenser-ensembledistil"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.eval()

        tokens = tokenizer(query, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            output = model(**tokens)
        logits = output.logits
        # SPLADE: max-pool over sequence dimension, then ReLU + log1p
        sparse = torch.max(torch.log1p(torch.relu(logits)), dim=1).values
        # Top expansion terms
        top_indices = torch.topk(sparse[0], k=20).indices.tolist()
        expanded = [tokenizer.decode([idx]).strip() for idx in top_indices]
        # Filter out subword tokens and original query words
        query_words = set(query.lower().split())
        expanded = [t for t in expanded if t.isalpha() and len(t) > 2 and t.lower() not in query_words]
        return expanded[:10]
    except Exception:
        log.debug("SPLADE model unavailable, using simple synonym expansion fallback")
        # Minimal D&G-specific synonym expansion
        synonyms = {
            "rhizome": ["multiplicity", "connection", "heterogeneity"],
            "body without organs": ["BwO", "intensity", "desire"],
            "deterritorialization": ["reterritorialization", "decoding", "flow"],
            "assemblage": ["agencement", "machine", "arrangement"],
            "smooth": ["nomad", "open-ended"],
            "striated": ["gridded", "State"],
            "war machine": ["nomad", "exteriority"],
            "faciality": ["face", "landscape", "white wall", "black hole"],
            "becoming": ["becoming-animal", "becoming-woman", "molecular"],
        }
        expanded = []
        query_lower = query.lower()
        for key, syns in synonyms.items():
            if key in query_lower:
                expanded.extend(syns)
        return expanded


# ── ColBERT reranking (stub) ───────────────────────────────────────────
def colbert_rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank chunks using ColBERT late interaction.

    Falls back to passage_score if RAGatouille is unavailable.
    """
    try:
        from ragatouille import RAGPretrainedModel

        model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        texts = [c["text"] for c in chunks]
        results = model.rerank(query=query, documents=texts, k=top_k)
        # Map back to chunk dicts
        reranked = []
        for r in results:
            idx = r["result_index"]
            chunk = dict(chunks[idx])
            chunk["colbert_score"] = r["score"]
            reranked.append(chunk)
        return reranked
    except Exception:
        log.debug("RAGatouille unavailable, falling back to passage_score")
        return passage_score(query, chunks, top_k=top_k)


# ── Multi-query decomposition ─────────────────────────────────────────
def decompose_query(query: str) -> list[str]:
    """Decompose a complex query into 2-4 sub-queries using the LLM."""
    system = (
        "You are a query decomposition assistant. Given a complex question, "
        "break it into 2-4 simpler sub-queries that together cover the full scope. "
        "Return only the sub-queries, one per line, no numbering or bullets."
    )
    try:
        resp = httpx.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                ],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 256},
            },
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json()["message"]["content"].strip()
        sub_queries = [line.strip() for line in text.split("\n") if line.strip()]
        # Sanity: keep between 2-4
        if len(sub_queries) < 2:
            return [query]
        return sub_queries[:4]
    except Exception as e:
        log.warning(f"Multi-query decomposition failed: {e}")
        return [query]


# ── Pipeline configurations ───────────────────────────────────────────
CONFIGS = {
    "dense_only": {
        "description": "Dense vector search only (chunk collection)",
        "use_dense": True, "use_bm25": False, "use_windows": False,
        "use_passage_score": False, "use_phrase": False,
        "use_splade": False, "use_colbert": False, "use_multiquery": False,
    },
    "bm25_only": {
        "description": "BM25 keyword search only",
        "use_dense": False, "use_bm25": True, "use_windows": False,
        "use_passage_score": False, "use_phrase": False,
        "use_splade": False, "use_colbert": False, "use_multiquery": False,
    },
    "hybrid": {
        "description": "Dense + BM25 hybrid (RRF fusion)",
        "use_dense": True, "use_bm25": True, "use_windows": False,
        "use_passage_score": False, "use_phrase": False,
        "use_splade": False, "use_colbert": False, "use_multiquery": False,
    },
    "hybrid_splade": {
        "description": "Hybrid with SPLADE-expanded BM25",
        "use_dense": True, "use_bm25": True, "use_windows": False,
        "use_passage_score": False, "use_phrase": False,
        "use_splade": True, "use_colbert": False, "use_multiquery": False,
    },
    "full_pipeline": {
        "description": "Hybrid + passage scoring + phrase search",
        "use_dense": True, "use_bm25": True, "use_windows": True,
        "use_passage_score": True, "use_phrase": True,
        "use_splade": False, "use_colbert": False, "use_multiquery": False,
    },
    "full_colbert": {
        "description": "Full pipeline with ColBERT reranker",
        "use_dense": True, "use_bm25": True, "use_windows": True,
        "use_passage_score": True, "use_phrase": True,
        "use_splade": False, "use_colbert": True, "use_multiquery": False,
    },
    "full_multiquery": {
        "description": "Full pipeline with multi-query decomposition",
        "use_dense": True, "use_bm25": True, "use_windows": True,
        "use_passage_score": True, "use_phrase": True,
        "use_splade": False, "use_colbert": False, "use_multiquery": True,
    },
    "all_features": {
        "description": "Everything enabled (SPLADE + ColBERT + multi-query)",
        "use_dense": True, "use_bm25": True, "use_windows": True,
        "use_passage_score": True, "use_phrase": True,
        "use_splade": True, "use_colbert": True, "use_multiquery": True,
    },
}


def retrieve(
    query: str, config: dict, top_k: int = 10,
    source_filter: str | None = None,
) -> list[dict]:
    """Run the retrieval pipeline for a single query."""
    retrieve_k = 200
    where = None
    if source_filter:
        where = {"source": {"$contains": source_filter}}

    # Multi-query decomposition
    queries = [query]
    if config.get("use_multiquery"):
        queries = decompose_query(query)
        log.debug(f"Multi-query: {queries}")

    all_dense = []
    all_sparse = []
    all_windows = []

    for q in queries:
        q_emb = embed_query(q)

        if config["use_dense"]:
            all_dense.extend(dense_search(q_emb, top_k=retrieve_k, where=where))

        if config["use_bm25"]:
            bm25_query = q
            if config.get("use_splade"):
                expanded = splade_expand_query(q)
                if expanded:
                    bm25_query = q + " " + " ".join(expanded)
            all_sparse.extend(bm25_search(bm25_query, top_k=retrieve_k, source_filter=source_filter))

        if config.get("use_windows"):
            all_windows.extend(dense_search(q_emb, top_k=retrieve_k, collection_name=WINDOW_COLLECTION, where=where))

    # Deduplicate by id (keep first occurrence)
    def dedup(chunks):
        seen = set()
        result = []
        for c in chunks:
            if c["id"] not in seen:
                seen.add(c["id"])
                result.append(c)
        return result

    dense_results = dedup(all_dense)
    sparse_results = dedup(all_sparse)
    window_results = dedup(all_windows)

    # Combine via RRF
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
        phrase_results = phrase_search(query, source_filter=source_filter)
        if phrase_results:
            existing_ids = {c.get("id") for c in merged}
            for pr in phrase_results:
                if pr.get("id") not in existing_ids:
                    merged.insert(0, pr)

    # ColBERT reranking
    if config.get("use_colbert"):
        return colbert_rerank(query, merged[:retrieve_k], top_k=top_k)

    # Passage scoring
    if config.get("use_passage_score"):
        return passage_score(query, merged[:retrieve_k], top_k=top_k)

    return merged[:top_k]


# ── Exact citation evaluation ──────────────────────────────────────────
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


def run_exact_eval(ground_truth: list[dict], configs_to_run: list[str], enable_source_filter: bool):
    results = {}
    build_bm25()

    for config_name in configs_to_run:
        config = CONFIGS[config_name]
        log.info(f"\n{'='*60}")
        log.info(f"[EXACT] Config: {config_name} -- {config['description']}")
        log.info(f"{'='*60}")

        config_results = []
        t0 = time.time()

        for i, tc in enumerate(ground_truth):
            query = tc["query"]
            gt_text = tc["ground_truth"]

            source_filter = None
            if enable_source_filter:
                source_filter = extract_source_filter(query)

            chunks = retrieve(query, config, top_k=10, source_filter=source_filter)
            o5, r5 = check_gt(gt_text, chunks, top_k=5)
            o10, r10 = check_gt(gt_text, chunks, top_k=10)

            hit5 = o5 >= 0.7
            hit10 = o10 >= 0.7
            config_results.append({
                "query": query, "label": tc.get("label", ""),
                "hit5": hit5, "hit10": hit10,
                "overlap5": round(o5, 3), "overlap10": round(o10, 3),
                "rank5": r5, "rank10": r10,
            })

            icon = "+" if hit5 else "-"
            log.info(f"  [{i+1}/{len(ground_truth)}] {icon} o5={o5:.0%} o10={o10:.0%} r5={r5} r10={r10} | {tc.get('label', query[:40])}")

        elapsed = time.time() - t0
        results[config_name] = {"config": config, "results": config_results, "elapsed": round(elapsed, 1)}

    return results


# ── General query evaluation ───────────────────────────────────────────
def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "unknown")
        heading = chunk["metadata"].get("heading", "")
        page_start = chunk["metadata"].get("page_start", 0)
        page_end = chunk["metadata"].get("page_end", 0)

        label = f"[Source {i}: {source}"
        if page_start and page_end:
            if page_start == page_end:
                label += f", p. {page_start}"
            else:
                label += f", pp. {page_start}-{page_end}"
        if heading:
            label += f" -- {heading}"
        label += "]"
        context_parts.append(f"{label}\n{chunk['text']}")

    context_block = "\n\n---\n\n".join(context_parts)

    system_prompt = f"""You are a scholarly citation assistant. Your purpose is to help researchers find exact passages in their source texts.

## Rules

1. **Quote verbatim.** When citing a passage, reproduce the exact text from the provided context. Do not paraphrase, summarize, or rephrase quotes. Place quoted text in quotation marks.
2. **Cite with page numbers.** After every quote, include the citation in this format: [Source: {{title}}, p. {{page}}]. Use the source name and page numbers provided in the context headers.
3. **Abstain when unsure.** If you cannot find a relevant passage in the provided context with confidence, say: "I could not find a confident match for this query in the provided sources." Never fabricate or guess at quotes.
4. **Multiple sources.** If the answer draws from multiple passages, quote each one separately with its own citation.
5. **Context only.** Only use information from the provided context below. Do not draw on outside knowledge.

## Context

{context_block}"""

    return system_prompt


def llm_generate(system_prompt: str, query: str) -> str:
    resp = httpx.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 1024},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def extract_quotes(text: str) -> list[str]:
    """Extract quoted passages from LLM response."""
    quotes = []
    # Straight double quotes
    for m in re.finditer(r'"([^"]{15,})"', text):
        quotes.append(m.group(1))
    # Curly double quotes
    for m in re.finditer(r'\u201c([^\u201d]{15,})\u201d', text):
        quotes.append(m.group(1))
    # Markdown blockquotes
    blockquote_lines = []
    for line in text.split("\n"):
        if line.strip().startswith("> "):
            blockquote_lines.append(line.strip()[2:])
        else:
            if blockquote_lines:
                bq = " ".join(blockquote_lines)
                if len(bq) > 15:
                    quotes.append(bq)
                blockquote_lines = []
    if blockquote_lines:
        bq = " ".join(blockquote_lines)
        if len(bq) > 15:
            quotes.append(bq)
    return quotes


def extract_cited_pages(text: str) -> list[int]:
    """Extract page numbers from citations in the LLM response."""
    pages = set()
    # [Source: ..., p. 42] or [Source: ..., pp. 42-45]
    for m in re.finditer(r'p{1,2}\.\s*(\d+)(?:\s*-\s*(\d+))?', text):
        p1 = int(m.group(1))
        pages.add(p1)
        if m.group(2):
            p2 = int(m.group(2))
            for p in range(p1, p2 + 1):
                pages.add(p)
    return sorted(pages)


def verify_quote_against_pdf(quote: str, pdf_path: str) -> bool:
    """Verify a quote exists in the PDF using pdftotext."""
    try:
        result = subprocess.run(
            ["pdftotext", pdf_path, "-"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            log.warning(f"pdftotext failed for {pdf_path}")
            return False
        pdf_text = normalize_for_comparison(result.stdout)
        quote_norm = normalize_for_comparison(quote)
        # Check prefix match at various lengths
        for length in [60, 40, 30, 20]:
            if len(quote_norm) >= length and quote_norm[:length] in pdf_text:
                return True
        # Sliding window fallback
        if len(quote_norm) >= 20:
            prefix = quote_norm[:80]
            for start in range(0, len(pdf_text) - len(prefix) + 1, len(prefix) // 4):
                window = pdf_text[start:start + len(prefix)]
                if SequenceMatcher(None, prefix, window).ratio() >= 0.85:
                    return True
        return False
    except Exception as e:
        log.warning(f"PDF verification failed: {e}")
        return False


# Cache for PDF text to avoid re-extracting
_pdf_text_cache: dict[str, str] = {}


def get_pdf_text(source: str) -> str | None:
    """Find and extract text from a PDF by source name."""
    if source in _pdf_text_cache:
        return _pdf_text_cache[source]

    # Search common data directories
    search_dirs = [
        Path(__file__).resolve().parent.parent / "data" / "processed",
        Path(__file__).resolve().parent.parent / "data" / "pdfs",
        Path(__file__).resolve().parent.parent / "data" / "input",
        Path(__file__).resolve().parent.parent / "data",
    ]
    for d in search_dirs:
        if not d.exists():
            continue
        for pdf in d.rglob("*.pdf"):
            if source in pdf.name or pdf.name in source:
                try:
                    result = subprocess.run(
                        ["pdftotext", str(pdf), "-"],
                        capture_output=True, text=True, timeout=60,
                    )
                    if result.returncode == 0:
                        text = normalize_for_comparison(result.stdout)
                        _pdf_text_cache[source] = text
                        return text
                except Exception:
                    pass
    return None


def verify_quote_in_text(quote: str, pdf_text: str) -> bool:
    """Verify a quote exists in pre-extracted PDF text."""
    quote_norm = normalize_for_comparison(quote)
    for length in [60, 40, 30, 20]:
        if len(quote_norm) >= length and quote_norm[:length] in pdf_text:
            return True
    if len(quote_norm) >= 20:
        prefix = quote_norm[:80]
        step = max(1, len(prefix) // 4)
        for start in range(0, max(1, len(pdf_text) - len(prefix) + 1), step):
            window = pdf_text[start:start + len(prefix)]
            if SequenceMatcher(None, prefix, window).ratio() >= 0.85:
                return True
    return False


def run_general_eval(queries: list[dict], configs_to_run: list[str], enable_source_filter: bool):
    results = {}
    build_bm25()

    # Pre-extract PDF text for verification
    sources = {q["source"] for q in queries}
    pdf_texts = {}
    for source in sources:
        text = get_pdf_text(source)
        if text:
            pdf_texts[source] = text
            log.info(f"PDF text loaded for {source} ({len(text)} chars)")
        else:
            log.warning(f"Could not load PDF text for {source} -- quote verification will be skipped")

    for config_name in configs_to_run:
        config = CONFIGS[config_name]
        log.info(f"\n{'='*60}")
        log.info(f"[GENERAL] Config: {config_name} -- {config['description']}")
        log.info(f"{'='*60}")

        config_results = []
        t0 = time.time()

        for i, tc in enumerate(queries):
            query = tc["query"]
            source = tc["source"]
            expected_pages = tc.get("expected_pages", [])
            expected_concepts = tc.get("expected_concepts", [])
            expected_quotes = tc.get("expected_quotes", [])

            source_filter = None
            if enable_source_filter:
                source_filter = extract_source_filter(query)

            # Retrieve + generate
            chunks = retrieve(query, config, top_k=5, source_filter=source_filter)
            prompt = build_rag_prompt(query, chunks)
            answer = llm_generate(prompt, query)

            # Extract quotes from answer
            found_quotes = extract_quotes(answer)
            cited_pages = extract_cited_pages(answer)

            # Verify quotes against retrieved chunks (OCR-deconfused) AND PDF
            verified = 0
            pdf_text = pdf_texts.get(source)
            all_chunk_text_norm = " ".join(normalize_for_comparison(c["text"]) for c in chunks)
            if found_quotes:
                for q in found_quotes:
                    q_norm = normalize_for_comparison(q)
                    # First: check against retrieved chunks (preferred — OCR-cleaned)
                    chunk_verified = False
                    for length in [50, 40, 30, 20]:
                        if len(q_norm) >= length and q_norm[:length] in all_chunk_text_norm:
                            chunk_verified = True
                            break
                    if not chunk_verified and len(q_norm) >= 20:
                        # Sliding window on chunk text
                        prefix = q_norm[:80]
                        step = max(1, len(prefix) // 4)
                        for start in range(0, max(1, len(all_chunk_text_norm) - len(prefix) + 1), step):
                            window = all_chunk_text_norm[start:start + len(prefix)]
                            if SequenceMatcher(None, prefix, window).ratio() >= 0.80:
                                chunk_verified = True
                                break
                    # Fallback: check against raw PDF text
                    if chunk_verified or (pdf_text and verify_quote_in_text(q, pdf_text)):
                        verified += 1

            # Page accuracy: what fraction of cited pages are in expected range
            page_acc = 0.0
            if cited_pages and expected_pages:
                correct = sum(1 for p in cited_pages if p in expected_pages)
                page_acc = correct / len(cited_pages)

            # Concept coverage: what fraction of expected concepts appear in the answer
            concept_hits = 0
            answer_lower = answer.lower()
            for concept in expected_concepts:
                if concept.lower() in answer_lower:
                    concept_hits += 1
            concept_cov = concept_hits / len(expected_concepts) if expected_concepts else 0.0

            # Expected quote coverage: how many expected quotes appear in retrieved chunks
            expected_quote_hits = 0
            all_chunk_text = " ".join(normalize_for_comparison(c["text"]) for c in chunks)
            for eq in expected_quotes:
                eq_norm = normalize_for_comparison(eq)
                if eq_norm[:30] in all_chunk_text:
                    expected_quote_hits += 1
            eq_cov = expected_quote_hits / len(expected_quotes) if expected_quotes else 0.0

            result = {
                "query": query, "label": tc.get("label", ""),
                "quote_count": len(found_quotes),
                "verified_quotes": verified,
                "verification_rate": verified / len(found_quotes) if found_quotes else 0.0,
                "cited_pages": cited_pages,
                "page_accuracy": round(page_acc, 3),
                "concept_coverage": round(concept_cov, 3),
                "expected_quote_coverage": round(eq_cov, 3),
            }
            config_results.append(result)

            vr = result["verification_rate"]
            log.info(
                f"  [{i+1}/{len(queries)}] quotes={len(found_quotes)} verified={verified} "
                f"page_acc={page_acc:.0%} concepts={concept_cov:.0%} | {tc.get('label', query[:40])}"
            )

        elapsed = time.time() - t0
        results[config_name] = {"config": config, "results": config_results, "elapsed": round(elapsed, 1)}

    return results


# ── Report printing ────────────────────────────────────────────────────
def print_exact_report(results: dict, n_queries: int):
    print(f"\n{'='*80}")
    print(f"  ATP EXACT CITATION RETRIEVAL")
    print(f"{'='*80}")

    print(f"\n  {'Config':<20} {'GT@5':>7} {'GT@10':>8} {'Avg Ovlp':>9} {'Time':>6}")
    print(f"  {'_'*52}")

    for name, data in results.items():
        res = data["results"]
        h5 = sum(1 for r in res if r["hit5"])
        h10 = sum(1 for r in res if r["hit10"])
        avg = sum(r["overlap5"] for r in res) / len(res)
        print(f"  {name:<20} {h5:>3}/{n_queries:<3} {h10:>4}/{n_queries:<3} {avg:>8.0%} {data['elapsed']:>5.0f}s")

    # Per-query breakdown
    config_names = list(results.keys())
    print(f"\n  {'#':<3} {'Label':<25}", end="")
    for name in config_names:
        print(f" {name[:10]:>11}", end="")
    print()
    print(f"  {'_'*(29 + 12*len(config_names))}")

    for i in range(n_queries):
        label = list(results.values())[0]["results"][i].get("label", "")[:22]
        if not label:
            label = list(results.values())[0]["results"][i]["query"][:22]
        row = f"  {i+1:<3} {label:<25}"
        for name in config_names:
            r = results[name]["results"][i]
            icon = "+" if r["hit5"] else ("-" if r["hit10"] else "x")
            row += f" {icon} {r['overlap5']:>5.0%} r{r['rank5']:>2}"
        print(row)


def print_general_report(results: dict, n_queries: int):
    print(f"\n{'='*80}")
    print(f"  ATP GENERAL QUERY EVALUATION")
    print(f"{'='*80}")

    # Aggregate per config
    print(f"\n  {'Config':<20} {'Vrfy Rate':>10} {'Page Acc':>9} {'Concepts':>9} {'EQ Cov':>7} {'Time':>6}")
    print(f"  {'_'*63}")

    for name, data in results.items():
        res = data["results"]
        avg_vr = sum(r["verification_rate"] for r in res) / len(res)
        avg_pa = sum(r["page_accuracy"] for r in res) / len(res)
        avg_cc = sum(r["concept_coverage"] for r in res) / len(res)
        avg_eq = sum(r["expected_quote_coverage"] for r in res) / len(res)
        print(f"  {name:<20} {avg_vr:>9.0%} {avg_pa:>8.0%} {avg_cc:>8.0%} {avg_eq:>6.0%} {data['elapsed']:>5.0f}s")

    # Per-query breakdown for first config
    first_config = list(results.keys())[0]
    data = results[first_config]
    print(f"\n  Per-query detail ({first_config}):")
    print(f"  {'#':<3} {'Label':<25} {'Quotes':>7} {'Vrfd':>5} {'PgAcc':>6} {'Cncpt':>6}")
    print(f"  {'_'*55}")

    for i, r in enumerate(data["results"]):
        label = r.get("label", "")[:22] or r["query"][:22]
        print(
            f"  {i+1:<3} {label:<25} {r['quote_count']:>7} {r['verified_quotes']:>5} "
            f"{r['page_accuracy']:>5.0%} {r['concept_coverage']:>5.0%}"
        )


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ATP evaluation harness")
    parser.add_argument("--mode", choices=["exact", "general", "both"], default="both")
    parser.add_argument("--configs", default="all", help="all or comma-separated config names")
    parser.add_argument("--source-filter", action="store_true", help="Enable source filtering from query text")
    parser.add_argument("--multi-query", action="store_true", help="Enable multi-query decomposition")
    parser.add_argument("--colbert", action="store_true", help="Enable ColBERT reranking")
    parser.add_argument("--splade", action="store_true", help="Enable SPLADE query expansion")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON to file")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent

    if args.configs == "all":
        configs_to_run = list(CONFIGS.keys())
    else:
        configs_to_run = [c.strip() for c in args.configs.split(",")]
        for c in configs_to_run:
            if c not in CONFIGS:
                log.error(f"Unknown config: {c}. Available: {', '.join(CONFIGS.keys())}")
                sys.exit(1)

    # Apply CLI overrides to all selected configs
    for name in configs_to_run:
        if args.multi_query:
            CONFIGS[name]["use_multiquery"] = True
        if args.colbert:
            CONFIGS[name]["use_colbert"] = True
        if args.splade:
            CONFIGS[name]["use_splade"] = True

    log.info(f"Chunk collection: {CHROMA_COLLECTION}")
    log.info(f"Window collection: {WINDOW_COLLECTION}")
    log.info(f"LLM model: {LLM_MODEL}")
    log.info(f"Embed model: {EMBED_MODEL}")
    log.info(f"Source filter: {args.source_filter}")

    all_results = {}

    # Exact citation eval
    if args.mode in ("exact", "both"):
        exact_path = base / "data" / "eval" / "atp_exact_citations.json"
        if not exact_path.exists():
            log.error(f"Exact citations file not found: {exact_path}")
            sys.exit(1)
        with open(exact_path) as f:
            exact_gt = json.load(f)
        log.info(f"Loaded {len(exact_gt)} exact citation test cases")

        exact_results = run_exact_eval(exact_gt, configs_to_run, args.source_filter)
        print_exact_report(exact_results, len(exact_gt))
        all_results["exact"] = exact_results

    # General query eval
    if args.mode in ("general", "both"):
        general_path = base / "data" / "eval" / "atp_general_queries.json"
        if not general_path.exists():
            log.error(f"General queries file not found: {general_path}")
            sys.exit(1)
        with open(general_path) as f:
            general_queries = json.load(f)
        log.info(f"Loaded {len(general_queries)} general query test cases")

        general_results = run_general_eval(general_queries, configs_to_run, args.source_filter)
        print_general_report(general_results, len(general_queries))
        all_results["general"] = general_results

    # Save results
    if args.output:
        # Serialize: strip non-JSON-safe config references
        output = {}
        for mode, mode_results in all_results.items():
            output[mode] = {}
            for cname, cdata in mode_results.items():
                output[mode][cname] = {
                    "description": cdata["config"]["description"],
                    "elapsed": cdata["elapsed"],
                    "results": cdata["results"],
                }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        log.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
