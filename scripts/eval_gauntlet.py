#!/usr/bin/env python3
"""5-model LLM gauntlet: compare citation quality across local and API models.

Retrieval is pre-computed once (full_pipeline, top_k=10, source filtering).
Each model gets the same chunks and improved system prompt.

Usage:
    CHROMA_COLLECTION=atp-eval uv run scripts/eval_gauntlet.py
    uv run scripts/eval_gauntlet.py --models qwen3:8b,claude-sonnet-4.6
    uv run scripts/eval_gauntlet.py --dry-run
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
log = logging.getLogger("gauntlet")

# ── Environment ──────────────────────────────────────────────────────
CHROMA_HOST = os.environ.get("CHROMA_HOST", "http://localhost:8200")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "atp-eval")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b")
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Model configs ────────────────────────────────────────────────────
MODEL_CONFIGS = [
    {
        "name": "qwen3:8b",
        "backend": "ollama",
        "model_id": "qwen3:8b",
        "strip_think": True,
        "cost_input": 0, "cost_output": 0,
    },
    {
        "name": "qwen3.5:9b",
        "backend": "ollama",
        "model_id": "qwen3.5:9b",
        "strip_think": True,
        "cost_input": 0, "cost_output": 0,
    },
    {
        "name": "claude-sonnet-4.6",
        "backend": "openrouter",
        "model_id": "anthropic/claude-sonnet-4-6",
        "strip_think": False,
        "cost_input": 3.0, "cost_output": 15.0,  # per M tokens
    },
    {
        "name": "deepseek-r1",
        "backend": "openrouter",
        "model_id": "deepseek/deepseek-r1",
        "strip_think": True,
        "cost_input": 0.5, "cost_output": 2.0,
    },
    {
        "name": "gemma-4-27b",
        "backend": "openrouter",
        "model_id": "google/gemma-4-27b-it",
        "strip_think": False,
        "cost_input": 0.1, "cost_output": 0.2,
    },
    {
        "name": "gpt-4.1",
        "backend": "openrouter",
        "model_id": "openai/gpt-4.1",
        "strip_think": False,
        "cost_input": 2.0, "cost_output": 8.0,
    },
    {
        "name": "gemini-2.5-flash",
        "backend": "openrouter",
        "model_id": "google/gemini-2.5-flash",
        "strip_think": False,
        "cost_input": 0.30, "cost_output": 2.50,
    },
    {
        "name": "qwen3.5-27b",
        "backend": "openrouter",
        "model_id": "qwen/qwen3.5-27b",
        "strip_think": True,
        "cost_input": 0.20, "cost_output": 1.56,
    },
]

# Indices of general queries to SKIP (set empty for full corpus eval)
SKIP_GENERAL = set()

# ── ChromaDB + Retrieval (reused from eval_atp.py) ──────────────────
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

def embed_query(text: str) -> list[float]:
    resp = httpx.post(f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBED_MODEL, "input": [text]}, timeout=30)
    resp.raise_for_status()
    return resp.json()["embeddings"][0]

def dense_search(query_emb, top_k=200, source_filter=None):
    col = get_collection()
    if col.count() == 0:
        return []
    kwargs = {"query_embeddings": [query_emb], "n_results": min(top_k, col.count()),
              "include": ["documents", "metadatas", "distances"]}
    if source_filter:
        kwargs["where"] = {"source": {"$contains": source_filter}}
    try:
        results = col.query(**kwargs)
    except Exception:
        kwargs.pop("where", None)
        results = col.query(**kwargs)
    chunks = []
    if results["ids"] and results["ids"][0]:
        for id_, doc, meta, dist in zip(results["ids"][0], results["documents"][0],
                                         results["metadatas"][0], results["distances"][0]):
            chunks.append({"id": id_, "text": doc, "metadata": meta, "dense_score": 1 - dist})
    return chunks

_bm25_cache = None

def build_bm25():
    global _bm25_cache
    if _bm25_cache is not None:
        return _bm25_cache
    col = get_collection()
    result = col.get(include=["documents", "metadatas"])
    corpus = [{"id": id_, "text": doc, "metadata": meta}
              for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])]
    tokenized = [doc["text"].lower().split() for doc in corpus]
    index = BM25Okapi(tokenized)
    _bm25_cache = (corpus, index)
    log.info(f"BM25 index: {len(corpus)} docs")
    return _bm25_cache

def bm25_search(query, top_k=200, source_filter=None):
    corpus, index = build_bm25()
    scores = index.get_scores(query.lower().split())
    scored = []
    for i, score in enumerate(scores):
        if score > 0:
            if source_filter and not corpus[i]["metadata"].get("source", "").startswith(source_filter):
                continue
            scored.append({"id": corpus[i]["id"], "text": corpus[i]["text"],
                           "metadata": corpus[i]["metadata"], "bm25_score": float(score)})
    scored.sort(key=lambda x: x["bm25_score"], reverse=True)
    return scored[:top_k]

def rrf_fusion(list_a, list_b, k=60, wa=1.0, wb=1.0):
    scores, chunk_map = {}, {}
    for rank, c in enumerate(list_a):
        scores[c["id"]] = scores.get(c["id"], 0) + wa / (k + rank + 1)
        chunk_map[c["id"]] = c
    for rank, c in enumerate(list_b):
        scores[c["id"]] = scores.get(c["id"], 0) + wb / (k + rank + 1)
        if c["id"] not in chunk_map:
            chunk_map[c["id"]] = c
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[cid] for cid, _ in ranked]

def passage_score(query, chunks, top_k=10):
    if not chunks:
        return []
    phrase = None
    m = re.search(r'"([^"]{10,})"', query)
    if m:
        phrase = m.group(1)
    ql = re.sub(r"\s+", " ", query.lower()).strip()
    qw = set(w for w in ql.split() if len(w) > 2)
    for c in chunks:
        cl = re.sub(r"\s+", " ", c["text"].lower()).strip()
        cw = set(w for w in cl.split() if len(w) > 2)
        s = 0.0
        if phrase:
            pl = re.sub(r"\s+", " ", phrase.lower()).strip()
            if pl[:50] in cl: s += 100.0
            elif pl[:30] in cl: s += 80.0
            elif pl[:20] in cl: s += 50.0
        if qw and cw:
            s += len(qw & cw) * 2.0
        wl = ql.split()
        for j in range(len(wl) - 1):
            bg = f"{wl[j]} {wl[j+1]}"
            if len(bg) > 5 and bg in cl:
                s += 5.0
        c["passage_score"] = s
    return sorted(chunks, key=lambda c: c.get("passage_score", 0), reverse=True)[:top_k]

def term_passage_score(term: str, expected_terms: list[str], chunks: list[dict],
                       top_k: int = 50) -> list[dict]:
    """Score and filter chunks for exhaustive term retrieval.

    Keeps only chunks containing the term or expected_terms, scores by
    term frequency and source diversity, sorts chronologically within sources.
    """
    all_terms = {term.lower()} | {t.lower() for t in expected_terms}
    # Filter to chunks containing any target term
    matching = []
    for c in chunks:
        text_lower = c["text"].lower()
        hits = sum(1 for t in all_terms if t in text_lower)
        if hits > 0:
            # Score: term hits + bonus for primary term
            score = hits
            if term.lower() in text_lower:
                score += text_lower.count(term.lower())
            c["term_score"] = score
            matching.append(c)

    # Sort by source (alphabetical ≈ chronological for dated filenames) then page
    matching.sort(key=lambda c: (
        c["metadata"].get("source", ""),
        c["metadata"].get("page_start", 0),
    ))

    # If we have more than top_k, take highest-scored while preserving source diversity
    if len(matching) > top_k:
        # Ensure at least some chunks from each source
        by_source: dict[str, list[dict]] = {}
        for c in matching:
            src = c["metadata"].get("source", "unknown")
            by_source.setdefault(src, []).append(c)

        result = []
        # Round-robin: take top chunks from each source
        per_source = max(3, top_k // max(len(by_source), 1))
        for src, src_chunks in by_source.items():
            src_chunks.sort(key=lambda c: c.get("term_score", 0), reverse=True)
            result.extend(src_chunks[:per_source])

        # Fill remaining slots with highest-scored across all sources
        seen = {c["id"] for c in result}
        remaining = sorted([c for c in matching if c["id"] not in seen],
                           key=lambda c: c.get("term_score", 0), reverse=True)
        result.extend(remaining[:top_k - len(result)])

        # Re-sort chronologically for the LLM prompt
        result.sort(key=lambda c: (
            c["metadata"].get("source", ""),
            c["metadata"].get("page_start", 0),
        ))
        return result[:top_k]

    return matching


def phrase_search(query, top_k=10, source_filter=None):
    m = re.search(r'"([^"]{10,})"', query)
    if not m:
        return []
    phrase = re.sub(r"\s+", " ", m.group(1).lower()).strip()
    corpus, _ = build_bm25()
    matches = []
    for c in corpus:
        if source_filter and not c["metadata"].get("source", "").startswith(source_filter):
            continue
        cn = re.sub(r"\s+", " ", c["text"].lower())
        if phrase[:30] in cn:
            matches.append(dict(c, phrase_score=100.0))
    return matches[:top_k]

SOURCE_PATTERNS = [
    (r"(?i)\bin\s+A\s+Thousand\s+Plateaus\b", "1980 A Thousand Plateaus"),
    (r"(?i)\bin\s+Anti[- ]?Oedipus\b", "1972 Anti-Oedipus"),
    (r"(?i)\bin\s+Difference\s+and\s+Repetition\b", "1968 Difference and repetition"),
    (r"(?i)\bin\s+What\s+Is\s+Philosophy\b", "1991 What Is Philosophy"),
    (r"(?i)\bin\s+Foucault\b", "1988 Foucault"),
    (r"(?i)\bin\s+The\s+Fold\b", "1988 The Fold"),
    (r"(?i)\bin\s+Logic\s+of\s+Sense\b", "1969 The Logic of Sense"),
    (r"(?i)\bin\s+Logic\s+of\s+Sensation\b", "1981 Francis Bacon"),
    (r"(?i)\bin\s+Francis\s+Bacon\b", "1981 Francis Bacon"),
    (r"(?i)\bin\s+Nietzsche\s+and\s+Philosophy\b", "1962 Nietzsche and Philosophy"),
    (r"(?i)\bin\s+Bergsonism\b", "1966 Bergsonism"),
    (r"(?i)\bin\s+Dialogues\b", "1977 Dialogues"),
    (r"(?i)\bin\s+Proust\s+and\s+Signs\b", "1964 Proust and Signs"),
    (r"(?i)\bin\s+Cinema\s+1\b", "1986 Cinema 1"),
    (r"(?i)\bin\s+Cinema\s+2\b", "1989 Cinema 2"),
    (r"(?i)\bin\s+Spinoza\b", "1970 Spinoza"),
    (r"(?i)\bin\s+Movement[- ]Image\b", "1986 Cinema 1"),
    (r"(?i)\bin\s+Time[- ]Image\b", "1989 Cinema 2"),
]

def extract_source_filter(query):
    for pat, prefix in SOURCE_PATTERNS:
        if re.search(pat, query):
            return prefix
    return None

def retrieve_full(query, top_k=10):
    """Full pipeline retrieval: dense + BM25 + phrase search + passage scoring + source filter."""
    sf = extract_source_filter(query)
    q_emb = embed_query(query)
    dense = dense_search(q_emb, 200, sf)
    sparse = bm25_search(query, 200, sf)
    merged = rrf_fusion(dense, sparse, wa=0.6, wb=0.4)

    # Phrase search
    pr = phrase_search(query, source_filter=sf)
    if pr:
        ids = {c["id"] for c in merged}
        for p in pr:
            if p["id"] not in ids:
                merged.insert(0, p)

    return passage_score(query, merged[:200], top_k=top_k)


# ── Improved system prompt ───────────────────────────────────────────

def build_gauntlet_prompt(query: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "unknown")
        ps = chunk["metadata"].get("page_start", 0)
        pe = chunk["metadata"].get("page_end", 0)
        heading = chunk["metadata"].get("heading", "")
        label = f"[Source {i}: {source}"
        if ps and pe:
            label += f", p. {ps}" if ps == pe else f", pp. {ps}-{pe}"
        if heading:
            label += f" -- {heading}"
        label += "]"
        context_parts.append(f"{label}\n{chunk['text']}")

    ctx = "\n\n---\n\n".join(context_parts)

    return f"""You are a scholarly citation assistant. Your ONLY job is to find and reproduce EXACT passages from the provided context.

## CRITICAL RULES

1. **COPY-PASTE ONLY.** Every quote you produce MUST be an exact character-for-character copy from the context below. Do NOT change any words, reorder phrases, or substitute synonyms. If the context says "there was already quite a crowd", you write exactly that.

2. **Citation format.** After every quote, cite like this: [Source: {{title}}, p. {{page}}]

3. **Refuse rather than paraphrase.** If you cannot find a relevant EXACT passage, say: "I could not find an exact match in the provided sources." NEVER paraphrase or summarize and present it as a quote.

4. **Multiple passages.** Quote each relevant passage separately with its own citation.

5. **Context only.** Use ONLY the text provided below. Do not use outside knowledge.

## CORRECT EXAMPLE

"The two of us wrote Anti-Oedipus together. Since each of us was several, there was already quite a crowd." [Source: 1980 A Thousand Plateaus - Deleuze, Gilles.pdf, p. 24]

## WRONG EXAMPLE (paraphrased — DO NOT DO THIS)

Deleuze and Guattari describe how they collaborated on Anti-Oedipus, noting that their combined perspectives created a collective voice.

## Context

{ctx}"""


# ── Unified LLM generation ──────────────────────────────────────────

def llm_generate(model_config: dict, system_prompt: str, query: str,
                  temperature: float | None = None) -> tuple[str, dict]:
    """Generate via Ollama or OpenRouter. Returns (text, usage_stats).

    Args:
        temperature: Override default 0.1. Use 0.0 for deterministic exhaustive mode.
    """
    backend = model_config["backend"]
    model_id = model_config["model_id"]
    temp = temperature if temperature is not None else 0.1
    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            if backend == "ollama":
                resp = httpx.post(
                    f"{OLLAMA_HOST}/api/chat",
                    json={
                        "model": model_id,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": query},
                        ],
                        "stream": False,
                        "options": {"temperature": temp, "num_predict": 4096},
                    },
                    timeout=600,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data.get("message", {}).get("content", "")
                usage["prompt_tokens"] = data.get("prompt_eval_count", 0)
                usage["completion_tokens"] = data.get("eval_count", 0)

            elif backend == "openrouter":
                if not OPENROUTER_KEY:
                    return "(OPENROUTER_KEY not set)", usage

                resp = httpx.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_KEY}",
                        "HTTP-Referer": "https://gutenberg.local",
                    },
                    json={
                        "model": model_id,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": query},
                        ],
                        "temperature": temp,
                        "max_tokens": 4096,
                    },
                    timeout=600,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                u = data.get("usage", {})
                usage["prompt_tokens"] = u.get("prompt_tokens", 0)
                usage["completion_tokens"] = u.get("completion_tokens", 0)

                # Rate limit: 1s between OpenRouter calls
                time.sleep(1)
            else:
                return f"(Unknown backend: {backend})", usage

            break  # success

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                log.warning(f"LLM call failed (attempt {attempt+1}/{max_retries}): {e}, retrying in {wait}s")
                time.sleep(wait)
            else:
                log.error(f"LLM generation failed after {max_retries} attempts ({model_config['name']}): {e}")
                return f"(ERROR: {e})", usage

    # Strip <think> blocks
    if model_config.get("strip_think"):
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    return text.strip(), usage


# ── Metrics ──────────────────────────────────────────────────────────

def check_verbatim(gt_text: str, answer: str) -> dict:
    """Check if ground truth appears verbatim in the LLM answer.

    Two-tier matching:
    1. Exact prefix substring match at 80/50/40/30 chars
    2. Fuzzy sliding window — threshold 0.95 counts as verbatim
       (handles nested-quote normalization artifacts)
    """
    gt_norm = normalize_for_comparison(gt_text)
    ans_norm = normalize_for_comparison(answer)

    for length in [80, 50, 40, 30]:
        if len(gt_norm) >= length and gt_norm[:length] in ans_norm:
            return {"verbatim": True, "match_len": length}

    # Fuzzy sliding window
    best = 0.0
    prefix = gt_norm[:100]
    if len(prefix) > 0 and len(ans_norm) > len(prefix):
        step = max(1, len(prefix) // 4)
        for start in range(0, len(ans_norm) - len(prefix) + 1, step):
            window = ans_norm[start:start + len(prefix)]
            ratio = SequenceMatcher(None, prefix, window).ratio()
            if ratio > best:
                best = ratio

    # Fix 1: 0.95+ fuzzy counts as verbatim (handles nested quotes, minor normalization diffs)
    if best >= 0.95:
        return {"verbatim": True, "match_len": 0, "fuzzy": round(best, 3)}

    return {"verbatim": False, "best_ratio": round(best, 3)}


def check_verbatim_any_passage(query_phrase: str, answer: str, chunks: list[dict]) -> dict:
    """Fix 2: Check if the answer contains ANY verbatim passage from chunks that matches the query phrase.

    Handles cases where multiple chunks contain semantically equivalent passages
    (e.g., a concept stated on p.28 and restated on p.42). If the LLM quotes a valid
    chunk passage containing the query phrase, that counts as correct even if it's not
    the specific GT passage.
    """
    if not query_phrase:
        return {"any_match": False}

    phrase_norm = normalize_for_comparison(query_phrase)
    ans_norm = normalize_for_comparison(answer)

    # Check if any chunk text containing the phrase also appears in the answer
    for chunk in chunks:
        chunk_norm = normalize_for_comparison(chunk["text"])
        # Does this chunk contain the query phrase?
        if phrase_norm[:30] not in chunk_norm:
            continue
        # Find a 60-char window around the phrase in the chunk
        idx = chunk_norm.find(phrase_norm[:30])
        if idx < 0:
            continue
        # Extract a passage window from the chunk
        window_start = max(0, idx - 20)
        window_end = min(len(chunk_norm), idx + len(phrase_norm) + 40)
        passage = chunk_norm[window_start:window_end]
        # Check if this passage appears in the answer
        if len(passage) >= 30 and passage[:30] in ans_norm:
            return {"any_match": True, "source_page": chunk.get("metadata", {}).get("page_start", 0)}

    return {"any_match": False}


def extract_quotes(text: str) -> list[str]:
    quotes = []
    for m in re.finditer(r'"([^"]{15,})"', text):
        quotes.append(m.group(1))
    for m in re.finditer(r'\u201c([^\u201d]{15,})\u201d', text):
        quotes.append(m.group(1))
    return list(dict.fromkeys(quotes))


def extract_cited_pages(text: str) -> list[int]:
    pages = set()
    for m in re.finditer(r'p{1,2}\.\s*(\d+)', text):
        pages.add(int(m.group(1)))
    return sorted(pages)


def verify_quote_in_chunks(quote: str, chunks: list[dict], fuzzy_threshold: float = 0.70) -> bool:
    """Verify quote against OCR-deconfused chunk text.

    Uses 0.70 fuzzy threshold to accommodate OCR artifacts and Surya
    text variants in scanned philosophical books.
    """
    q_norm = normalize_for_comparison(quote)
    all_text = " ".join(normalize_for_comparison(c["text"]) for c in chunks)

    for length in [50, 40, 30, 20]:
        if len(q_norm) >= length and q_norm[:length] in all_text:
            return True

    if len(q_norm) >= 20:
        prefix = q_norm[:80]
        step = max(1, len(prefix) // 4)
        for start in range(0, max(1, len(all_text) - len(prefix) + 1), step):
            window = all_text[start:start + len(prefix)]
            if SequenceMatcher(None, prefix, window).ratio() >= fuzzy_threshold:
                return True
    return False


# ── Exhaustive term retrieval ────────────────────────────────────────

def pre_extract_term_sentences(term: str, expected_terms: list[str],
                                chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Extract sentences containing the term mechanically, bypassing the LLM.

    Returns:
        (pre_extracted, remaining_chunks): pre_extracted is a list of
        {"quote": str, "source": str, "page": int, "chunk_idx": int} dicts.
        remaining_chunks are chunks where only variants (not exact term) appear.
    """
    try:
        from shared.nlp import sentencize, is_available
        use_spacy = is_available()
    except (ImportError, OSError):
        use_spacy = False
        sentencize = None

    term_lower = term.lower()
    variant_set = {t.lower() for t in expected_terms} - {term_lower}
    pre_extracted = []
    remaining = []

    for idx, chunk in enumerate(chunks):
        text = chunk["text"]
        text_lower = text.lower()
        source = chunk["metadata"].get("source", "unknown")
        page = chunk["metadata"].get("page_start", 0)

        if term_lower in text_lower:
            # Exact term match — extract sentences mechanically
            if use_spacy:
                sentences = sentencize(text)
            else:
                sentences = re.split(r'(?<=[.!?])\s+', text)

            for sent in sentences:
                if term_lower in sent.lower() and len(sent.strip()) >= 20:
                    pre_extracted.append({
                        "quote": sent.strip(),
                        "source": source,
                        "page": page,
                        "chunk_idx": idx + 1,
                    })
        elif any(v in text_lower for v in variant_set):
            # Only variant match — send to LLM for interpretation
            remaining.append(chunk)

    return pre_extracted, remaining

def build_exhaustive_prompt(query: str, chunks: list[dict], term: str = "") -> str:
    """System prompt for exhaustive term retrieval — chunk-numbered, selection-based."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "unknown")
        ps = chunk["metadata"].get("page_start", 0)
        pe = chunk["metadata"].get("page_end", 0)
        page_label = ""
        if ps and pe:
            page_label = f", p. {ps}" if ps == pe else f", pp. {ps}-{pe}"
        context_parts.append(f"[CHUNK {i} | {source}{page_label}]\n{chunk['text']}")
    ctx = "\n\n---\n\n".join(context_parts)

    term_display = f"'{term}'" if term else "the requested concept"

    return f"""You are a scholarly citation assistant. Your ONLY job is to scan the numbered chunks below and extract every sentence that mentions {term_display}.

## YOUR TASK

Go through EACH chunk (1 to {len(chunks)}). For each chunk:
- If it contains {term_display} or a direct variant, COPY the exact sentence(s) verbatim.
- If it does NOT contain the term, SKIP it silently.

## OUTPUT FORMAT

For each match, output:
"<exact sentence copied from the chunk>" [Source: <title>, p. <page>] (Chunk <N>)

Group by work. End with: "Total: N citations from M works."

## ABSOLUTE CONSTRAINTS

1. **COPY-PASTE ONLY.** Every quoted sentence must appear VERBATIM in the chunk you cite. Copy character-for-character.
2. **NEVER RECONSTRUCT.** Do NOT generate passages from memory or training data. You are a search tool, not an author.
3. **ONLY USE THE CHUNKS BELOW.** If {term_display} does not appear in a chunk, do not cite that chunk. If zero chunks contain the term, output: "No occurrences of {term_display} found in the provided context."
4. **NO PARAPHRASING.** Do not reword, summarize, or rearrange text. Exact quotes only.

## CORRECT EXAMPLE

Query: Find all occurrences of "rhizome" in A Thousand Plateaus.
[CHUNK 3 | 1980 A Thousand Plateaus, p. 7]
A first type of book is the root-book. ... Let us summarize the principal characteristics of a rhizome: unlike trees or their roots, the rhizome connects any point to any other point ...

Correct output:
"Let us summarize the principal characteristics of a rhizome: unlike trees or their roots, the rhizome connects any point to any other point" [Source: 1980 A Thousand Plateaus, p. 7] (Chunk 3)

## WRONG EXAMPLE

"The rhizome is Deleuze and Guattari's metaphor for non-hierarchical connection" [Source: 1980 A Thousand Plateaus, p. 7] (Chunk 3)
^^^ WRONG: This sentence does NOT appear in Chunk 3. It was generated from memory. NEVER do this.

## Chunks

{ctx}"""


def filter_hallucinated_quotes(quotes: list[str], chunks: list[dict]) -> tuple[list[str], int]:
    """Remove quotes that don't verify against chunks. Returns (verified_quotes, n_filtered)."""
    verified = []
    for q in quotes:
        if verify_quote_in_chunks(q, chunks):
            verified.append(q)
    return verified, len(quotes) - len(verified)


def eval_exhaustive(query_data: dict, answer: str, chunks: list[dict]) -> dict:
    """Evaluate exhaustive term retrieval response."""
    raw_quotes = extract_quotes(answer)
    cited_pages = extract_cited_pages(answer)
    # Filter hallucinated quotes before scoring
    quotes, n_filtered = filter_hallucinated_quotes(raw_quotes, chunks)
    verified = len(quotes)  # all remaining quotes are verified by definition

    # Check source diversity: how many expected sources appear in citations
    answer_lower = answer.lower()
    expected_sources = query_data.get("expected_sources", [])
    sources_cited = 0
    for src in expected_sources:
        # Check if source name fragment appears in the answer
        src_fragment = src.split(" - ")[0].split("Capitalism")[0].strip()
        if src_fragment.lower()[:20] in answer_lower:
            sources_cited += 1

    source_coverage = sources_cited / max(len(expected_sources), 1)

    # Check for expected terms
    expected_terms = query_data.get("expected_terms", [])
    term_hits = sum(1 for t in expected_terms if t.lower() in answer_lower)
    term_coverage = term_hits / max(len(expected_terms), 1)

    # Check if meets minimum citation count
    min_citations = query_data.get("min_citations", 3)
    meets_minimum = len(quotes) >= min_citations

    return {
        "label": query_data.get("label", ""),
        "quotes": len(quotes),
        "verified": verified,
        "vrfy_rate": round(verified / max(len(quotes), 1), 3),
        "hallucinated": n_filtered,
        "raw_quotes": len(raw_quotes),
        "source_coverage": round(source_coverage, 3),
        "term_coverage": round(term_coverage, 3),
        "meets_minimum": meets_minimum,
        "min_required": min_citations,
    }


def consensus_exhaustive(query: str, term: str, chunks: list[dict],
                         model_config: dict, corpus_verify: list[dict],
                         n_runs: int = 3) -> tuple[str, float]:
    """Run exhaustive LLM N times, keep only quotes appearing in 2+ runs AND verified.

    Returns (combined_answer_text, total_cost).
    """
    temps = [0.0] * n_runs  # slight variation doesn't help with greedy; keep deterministic
    if n_runs >= 2:
        temps[-1] = 0.05  # slight perturbation on last run

    all_runs_quotes: list[list[str]] = []
    total_cost = 0.0

    for run_idx, temp in enumerate(temps):
        prompt = build_exhaustive_prompt(query, chunks, term=term)
        answer, usage = llm_generate(model_config, prompt, query, temperature=temp)
        cost = (usage["prompt_tokens"] * model_config["cost_input"] +
                usage["completion_tokens"] * model_config["cost_output"]) / 1_000_000
        total_cost += cost

        quotes = extract_quotes(answer)
        all_runs_quotes.append(quotes)
        log.info(f"      Consensus run {run_idx+1}/{n_runs}: {len(quotes)} quotes extracted")

    # Count occurrences across runs (normalized prefix matching)
    quote_counts: dict[str, str] = {}  # norm_prefix -> original quote
    quote_seen: dict[str, int] = {}    # norm_prefix -> count
    for run_quotes in all_runs_quotes:
        run_seen = set()
        for q in run_quotes:
            norm = normalize_for_comparison(q)[:50]
            if norm not in run_seen:
                run_seen.add(norm)
                quote_seen[norm] = quote_seen.get(norm, 0) + 1
                if norm not in quote_counts:
                    quote_counts[norm] = q

    # Keep quotes in 2+ runs AND verified against corpus
    threshold = max(2, n_runs // 2 + 1)
    consensus_quotes = []
    for norm, count in quote_seen.items():
        if count >= threshold:
            original = quote_counts[norm]
            if verify_quote_in_chunks(original, corpus_verify):
                consensus_quotes.append(original)

    lines = [f'"{q}"' for q in consensus_quotes]
    combined = "\n".join(lines)
    combined += f"\n\nTotal: {len(consensus_quotes)} citations (consensus {threshold}/{n_runs})"

    log.info(f"      Consensus result: {len(consensus_quotes)} verified quotes from "
             f"{sum(len(r) for r in all_runs_quotes)} total across {n_runs} runs")

    return combined, total_cost


def exhaustive_agentic_loop(query_data: dict, model_config: dict,
                            retrieval_cache: dict,
                            max_iterations: int = 3) -> tuple[str, float]:
    """Iterative retrieve-generate-verify loop for exhaustive coverage.

    Each iteration:
    1. Pre-extract exact term sentences from current chunks
    2. Send remaining to LLM
    3. Verify all quotes
    4. Track which sources have been covered
    5. Retrieve new chunks from uncovered sources
    6. Repeat until coverage or max iterations

    Returns: (combined_answer_text, total_cost)
    """
    term = query_data.get("term", "")
    expected_terms = query_data.get("expected_terms", [])
    expected_sources = query_data.get("expected_sources", [])
    query = query_data["query"]

    # Build corpus verification set
    corpus_data = build_bm25()
    term_lower = term.lower()
    corpus_verify = [{"id": c["id"], "text": c["text"], "metadata": c["metadata"]}
                     for c in corpus_data[0] if term_lower in c["text"].lower()]

    all_verified_quotes = []
    all_pre_extracted = []
    covered_sources = set()
    seen_chunk_ids = set()
    total_cost = 0.0

    for iteration in range(max_iterations):
        # Get chunks for this iteration
        if iteration == 0:
            chunks = retrieval_cache.get(query, [])
        else:
            # Retrieve specifically from uncovered sources
            uncovered = [s for s in expected_sources
                         if not any(s.split(" - ")[0][:20].lower() in cs for cs in covered_sources)]
            if not uncovered:
                log.info(f"    Iteration {iteration+1}: all expected sources covered, stopping")
                break

            chunks = []
            for src in uncovered:
                src_prefix = src.split(" - ")[0][:30]
                src_chunks = bm25_search(term, top_k=50, source_filter=src_prefix)
                for c in src_chunks:
                    if c["id"] not in seen_chunk_ids:
                        chunks.append(c)
                for alt_term in expected_terms:
                    if alt_term.lower() != term_lower:
                        alt_chunks = bm25_search(alt_term, top_k=30, source_filter=src_prefix)
                        for c in alt_chunks:
                            if c["id"] not in seen_chunk_ids:
                                chunks.append(c)

            # Dedup and score
            deduped = []
            dedup_ids = set()
            for c in chunks:
                if c["id"] not in dedup_ids:
                    dedup_ids.add(c["id"])
                    deduped.append(c)
            chunks = term_passage_score(term, expected_terms, deduped, top_k=50)

            if not chunks:
                log.info(f"    Iteration {iteration+1}: no new chunks found, stopping")
                break

        # Track seen chunks
        for c in chunks:
            seen_chunk_ids.add(c["id"])

        # Pre-extract exact matches
        pre_extracted, llm_chunks = pre_extract_term_sentences(term, expected_terms, chunks)
        all_pre_extracted.extend(pre_extracted)

        # Build prior-context for the LLM
        prior_quotes = [q for q in all_verified_quotes]
        prior_note = ""
        if prior_quotes and iteration > 0:
            prior_note = (f"\n\nYou have already found {len(prior_quotes)} citations. "
                          f"Find ADDITIONAL occurrences in the NEW chunks below. "
                          f"Do NOT repeat previously found citations.")

        # LLM pass on remaining chunks (or all if no pre-extraction)
        target_chunks = llm_chunks if (pre_extracted or llm_chunks) else chunks
        if target_chunks:
            prompt = build_exhaustive_prompt(query + prior_note, target_chunks, term=term)
            answer, usage = llm_generate(model_config, prompt, query, temperature=0.0)
            cost = (usage["prompt_tokens"] * model_config["cost_input"] +
                    usage["completion_tokens"] * model_config["cost_output"]) / 1_000_000
            total_cost += cost

            # Verify LLM quotes
            llm_quotes = extract_quotes(answer)
            for lq in llm_quotes:
                if verify_quote_in_chunks(lq, corpus_verify):
                    all_verified_quotes.append(lq)

        # Add pre-extracted quotes
        for pe in pre_extracted:
            all_verified_quotes.append(pe["quote"])

        # Update source coverage
        for pe in pre_extracted:
            covered_sources.add(pe["source"].lower()[:20])
        # Also check LLM answer for source mentions
        if target_chunks:
            for src in expected_sources:
                src_frag = src.split(" - ")[0][:20].lower()
                if src_frag in answer.lower():
                    covered_sources.add(src_frag)

        log.info(f"    Iteration {iteration+1}: {len(pre_extracted)} pre-extracted, "
                 f"{len(all_verified_quotes)} total verified, "
                 f"{len(covered_sources)}/{len(expected_sources)} sources covered")

    # Deduplicate quotes
    seen_quotes = set()
    unique_quotes = []
    for q in all_verified_quotes:
        q_norm = normalize_for_comparison(q)[:50]
        if q_norm not in seen_quotes:
            seen_quotes.add(q_norm)
            unique_quotes.append(q)

    # Build combined answer text
    lines = []
    for q in unique_quotes:
        lines.append(f'"{q}"')
    combined = "\n".join(lines)
    combined += f"\n\nTotal: {len(unique_quotes)} citations"

    return combined, total_cost


# ── Evolutionary précis ─────────────────────────────────────────────

def build_precis_prompt(query: str, chunks: list[dict]) -> str:
    """System prompt for evolutionary précis — must trace concept across works."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "unknown")
        ps = chunk["metadata"].get("page_start", 0)
        pe = chunk["metadata"].get("page_end", 0)
        label = f"[Source {i}: {source}"
        if ps and pe:
            label += f", p. {ps}" if ps == pe else f", pp. {ps}-{pe}"
        label += "]"
        context_parts.append(f"{label}\n{chunk['text']}")
    ctx = "\n\n---\n\n".join(context_parts)

    return f"""You are a scholarly citation assistant writing a PRÉCIS tracing a concept's evolution across works.

## CRITICAL RULES

1. **CHRONOLOGICAL STRUCTURE.** Organize by publication date. For each work:
   - State when the concept appears and what it means in that context
   - Quote the key passage verbatim with citation: [Source: {{title}}, p. {{page}}]
   - Note what changed from the previous work's usage
2. **COPY-PASTE ONLY.** Every quoted passage must be exact from the context. No paraphrasing.
3. **SHOW THE DRIFT.** Explicitly articulate how the concept's meaning shifts, expands, narrows, or transforms between works. Use phrases like "In contrast to...", "This marks a shift from...", "Building on..."
4. **MULTIPLE WORKS REQUIRED.** You must cite from at least 2 different source texts. If you can only find the concept in one work, state this explicitly.
5. **Context only.** Use ONLY the provided context.

## Context

{ctx}"""


def eval_precis(query_data: dict, answer: str, chunks: list[dict]) -> dict:
    """Evaluate evolutionary précis response."""
    quotes = extract_quotes(answer)
    cited_pages = extract_cited_pages(answer)
    verified = sum(1 for q in quotes if verify_quote_in_chunks(q, chunks))

    # Count distinct works cited
    answer_text = answer
    works_required = query_data.get("works_required", [])
    works_cited = set()
    for src in works_required:
        src_fragment = src.split(" - ")[0].split("Capitalism")[0].strip()
        if src_fragment.lower()[:20] in answer_text.lower():
            works_cited.add(src)

    min_works = query_data.get("min_works_cited", 2)
    meets_works = len(works_cited) >= min_works

    # Check for evolution markers (drift language)
    drift_markers = ["shift", "contrast", "earlier", "later", "transform", "develop",
                     "evolve", "change", "building on", "unlike", "whereas", "moved from",
                     "marks a", "depart", "extend", "expand", "narrow", "different"]
    drift_hits = sum(1 for m in drift_markers if m.lower() in answer.lower())
    has_drift = drift_hits >= 2

    # Check expected stages
    expected_stages = query_data.get("expected_stages", [])
    stage_hits = 0
    for stage in expected_stages:
        # Check if key terms from each stage appear
        stage_terms = [t.strip().lower() for t in stage.split(",") if len(t.strip()) > 3]
        if stage_terms:
            found = sum(1 for t in stage_terms if t in answer.lower())
            if found >= max(1, len(stage_terms) // 2):
                stage_hits += 1
    stage_coverage = stage_hits / max(len(expected_stages), 1)

    min_quotes = query_data.get("min_quotes", 3)
    meets_quotes = len(quotes) >= min_quotes

    return {
        "label": query_data.get("label", ""),
        "quotes": len(quotes),
        "verified": verified,
        "vrfy_rate": round(verified / max(len(quotes), 1), 3),
        "works_cited": len(works_cited),
        "works_required": len(works_required),
        "meets_works": meets_works,
        "has_drift": has_drift,
        "stage_coverage": round(stage_coverage, 3),
        "meets_quotes": meets_quotes,
    }


# ── Gauntlet runner ──────────────────────────────────────────────────

def run_gauntlet(exact_queries, general_queries, model_configs,
                 exhaustive_queries=None, precis_queries=None,
                 agentic: bool = False, consensus: int = 0):
    if exhaustive_queries is None:
        exhaustive_queries = []
    if precis_queries is None:
        precis_queries = []

    build_bm25()

    # Pre-compute retrieval for all queries
    # Exhaustive and precis queries need more chunks (top_k=20) for broader coverage
    log.info("Pre-computing retrieval for all queries...")
    retrieval_cache = {}
    for q in exact_queries + general_queries:
        chunks = retrieve_full(q["query"], top_k=10)
        retrieval_cache[q["query"]] = chunks
        log.info(f"  Retrieved {len(chunks)} chunks for: {q.get('label', q['query'][:40])}")
    for q in exhaustive_queries:
        # Multi-pass retrieval: retrieve per expected source for better cross-corpus coverage
        all_chunks = []
        seen_ids = set()
        expected_sources = q.get("expected_sources", [])
        expected_terms = q.get("expected_terms", [])
        term = q.get("term", "")

        # Phase 1: BM25 search for the primary term (expanded ceiling)
        term_chunks = bm25_search(term, top_k=500)
        for c in term_chunks:
            if c["id"] not in seen_ids:
                seen_ids.add(c["id"])
                all_chunks.append(c)

        # Phase 1.5: BM25 search for each expected_term variant
        for alt_term in expected_terms:
            if alt_term.lower() != term.lower():
                alt_chunks = bm25_search(alt_term, top_k=100)
                for c in alt_chunks:
                    if c["id"] not in seen_ids:
                        seen_ids.add(c["id"])
                        all_chunks.append(c)

        # Phase 2: Dense search with the full query
        q_emb = embed_query(q["query"])
        dense_chunks = dense_search(q_emb, top_k=50)
        for c in dense_chunks:
            if c["id"] not in seen_ids:
                seen_ids.add(c["id"])
                all_chunks.append(c)

        # Phase 3: Per-source BM25 for expected sources (wider per-source)
        for src in expected_sources:
            src_prefix = src.split(" - ")[0][:30]
            src_chunks = bm25_search(term, top_k=50, source_filter=src_prefix)
            for c in src_chunks:
                if c["id"] not in seen_ids:
                    seen_ids.add(c["id"])
                    all_chunks.append(c)
            # Also search expected_terms per source
            for alt_term in expected_terms:
                if alt_term.lower() != term.lower():
                    alt_src_chunks = bm25_search(alt_term, top_k=30, source_filter=src_prefix)
                    for c in alt_src_chunks:
                        if c["id"] not in seen_ids:
                            seen_ids.add(c["id"])
                            all_chunks.append(c)

        # Store ALL term-matching chunks for verification (before scoring)
        retrieval_cache[q["query"] + "__all"] = list(all_chunks)
        # Term-aware scoring: keep only chunks containing the term/variants, sorted by relevance
        all_chunks = term_passage_score(term, expected_terms, all_chunks, top_k=50)
        retrieval_cache[q["query"]] = all_chunks
        log.info(f"  Retrieved {len(all_chunks)} prompt chunks, "
                 f"{len(retrieval_cache[q['query'] + '__all'])} total candidates "
                 f"for: {q.get('label', q['query'][:40])}")

    for q in precis_queries:
        # Multi-pass: retrieve per required work
        all_chunks = []
        seen_ids = set()
        works = q.get("works_required", [])
        concept = q.get("concept", "")

        for work in works:
            work_prefix = work.split(" - ")[0][:30]
            # BM25 for concept in this specific work
            work_chunks = bm25_search(concept, top_k=15, source_filter=work_prefix)
            for c in work_chunks:
                if c["id"] not in seen_ids:
                    seen_ids.add(c["id"])
                    all_chunks.append(c)
            # Dense for the query in this work
            q_emb = embed_query(f"{concept} {work_prefix}")
            dense_chunks = dense_search(q_emb, top_k=10, source_filter=work_prefix)
            for c in dense_chunks:
                if c["id"] not in seen_ids:
                    seen_ids.add(c["id"])
                    all_chunks.append(c)

        # Also general retrieval
        gen_chunks = retrieve_full(q["query"], top_k=10)
        for c in gen_chunks:
            if c["id"] not in seen_ids:
                seen_ids.add(c["id"])
                all_chunks.append(c)

        all_chunks = passage_score(q["query"], all_chunks, top_k=30)
        retrieval_cache[q["query"]] = all_chunks
        log.info(f"  Retrieved {len(all_chunks)} chunks (per-work) for: {q.get('label', q['query'][:40])}")

    results = {}

    for mc in model_configs:
        name = mc["name"]
        log.info(f"\n{'='*60}\n  MODEL: {name}\n{'='*60}")

        exact_results = []
        general_results = []
        total_cost = 0.0
        t0 = time.time()

        # ── Exact citation queries ────────────────────────────
        for i, q in enumerate(exact_queries):
            chunks = retrieval_cache[q["query"]]
            prompt = build_gauntlet_prompt(q["query"], chunks)
            answer, usage = llm_generate(mc, prompt, q["query"])

            cost = (usage["prompt_tokens"] * mc["cost_input"] +
                    usage["completion_tokens"] * mc["cost_output"]) / 1_000_000
            total_cost += cost

            vb = check_verbatim(q["ground_truth"], answer)

            # Fix 2: if GT check failed, check if LLM quoted ANY valid chunk passage
            # containing the query phrase (handles alternate-passage hits)
            any_pass = {"any_match": False}
            if not vb["verbatim"]:
                phrase_match = re.search(r'"([^"]{10,})"', q["query"])
                if phrase_match:
                    any_pass = check_verbatim_any_passage(phrase_match.group(1), answer, chunks)

            is_pass = vb["verbatim"] or any_pass["any_match"]
            exact_results.append({
                "label": q.get("label", ""),
                "verbatim": vb["verbatim"],
                "any_passage": any_pass["any_match"],
                "pass": is_pass,
                "match_len": vb.get("match_len", 0),
                "best_ratio": vb.get("best_ratio", 0),
                "fuzzy": vb.get("fuzzy", 0),
            })
            icon = "+" if is_pass else "-"
            if vb["verbatim"]:
                detail = f"@{vb['match_len']}" if vb.get("match_len") else f"~{vb.get('fuzzy', 0)}"
            elif any_pass["any_match"]:
                detail = f"alt-p{any_pass.get('source_page', '?')}"
            else:
                detail = f"r={vb.get('best_ratio', 0):.0%}"
            log.info(f"  [E{i+1}] {icon} {detail} | {q.get('label', '')}")

        # ── General queries ───────────────────────────────────
        for i, q in enumerate(general_queries):
            chunks = retrieval_cache[q["query"]]
            prompt = build_gauntlet_prompt(q["query"], chunks)
            answer, usage = llm_generate(mc, prompt, q["query"])

            cost = (usage["prompt_tokens"] * mc["cost_input"] +
                    usage["completion_tokens"] * mc["cost_output"]) / 1_000_000
            total_cost += cost

            quotes = extract_quotes(answer)
            cited_pages = extract_cited_pages(answer)

            verified = sum(1 for q_ in quotes if verify_quote_in_chunks(q_, chunks))
            vrfy_rate = verified / max(len(quotes), 1)

            expected_pages = q.get("expected_pages", [])
            page_hits = sum(1 for p in cited_pages if p in expected_pages)
            page_acc = page_hits / max(len(cited_pages), 1) if cited_pages else 0.0

            expected_concepts = q.get("expected_concepts", [])
            answer_lower = answer.lower()
            concept_hits = sum(1 for c in expected_concepts if c.lower() in answer_lower)
            concept_cov = concept_hits / max(len(expected_concepts), 1)

            general_results.append({
                "label": q.get("label", ""),
                "quotes": len(quotes), "verified": verified,
                "vrfy_rate": round(vrfy_rate, 3),
                "page_acc": round(page_acc, 3),
                "concept_cov": round(concept_cov, 3),
            })

            log.info(f"  [G{i+1}] q={len(quotes)} v={verified} pg={page_acc:.0%} "
                     f"c={concept_cov:.0%} | {q.get('label', '')}")

        # ── Exhaustive term retrieval ─────────────────────────
        exhaustive_results = []
        for i, q in enumerate(exhaustive_queries):
            term = q.get("term", "")
            expected_terms = q.get("expected_terms", [])

            # Build full corpus verification set
            if term:
                corpus_data = build_bm25()
                term_lower = term.lower()
                corpus_verify = [{"id": c["id"], "text": c["text"], "metadata": c["metadata"]}
                                 for c in corpus_data[0] if term_lower in c["text"].lower()]
            else:
                corpus_verify = retrieval_cache[q["query"]]

            if agentic:
                # Agentic multi-iteration loop with coverage tracking
                combined_answer, loop_cost = exhaustive_agentic_loop(
                    q, mc, retrieval_cache, max_iterations=3)
                total_cost += loop_cost
            elif consensus > 0:
                # Multi-sample consensus: run N times, keep verified intersection
                chunks = retrieval_cache[q["query"]]
                combined_answer, cons_cost = consensus_exhaustive(
                    q["query"], term, chunks, mc, corpus_verify, n_runs=consensus)
                total_cost += cons_cost
            else:
                # Single-pass with pre-extraction
                chunks = retrieval_cache[q["query"]]

                # Pre-extract sentences with exact term matches mechanically
                pre_extracted, llm_chunks = pre_extract_term_sentences(
                    term, expected_terms, chunks)

                # Format pre-extracted citations
                pre_lines = []
                for pe in pre_extracted:
                    pre_lines.append(f'"{pe["quote"]}" [Source: {pe["source"]}, p. {pe["page"]}] '
                                     f'(Chunk {pe["chunk_idx"]})')

                # Send remaining chunks to LLM
                llm_answer = ""
                if llm_chunks:
                    prompt = build_exhaustive_prompt(q["query"], llm_chunks, term=term)
                    llm_answer, usage = llm_generate(mc, prompt, q["query"], temperature=0.0)
                    cost = (usage["prompt_tokens"] * mc["cost_input"] +
                            usage["completion_tokens"] * mc["cost_output"]) / 1_000_000
                    total_cost += cost
                elif not pre_extracted:
                    prompt = build_exhaustive_prompt(q["query"], chunks, term=term)
                    llm_answer, usage = llm_generate(mc, prompt, q["query"], temperature=0.0)
                    cost = (usage["prompt_tokens"] * mc["cost_input"] +
                            usage["completion_tokens"] * mc["cost_output"]) / 1_000_000
                    total_cost += cost

                combined_answer = "\n".join(pre_lines)
                if llm_answer:
                    combined_answer += "\n\n" + llm_answer

            result = eval_exhaustive(q, combined_answer, corpus_verify)
            if not agentic and consensus <= 0:
                result["pre_extracted"] = len(pre_extracted)
            exhaustive_results.append(result)

            log.info(f"  [T{i+1}] q={result['quotes']} v={result['verified']} "
                     f"hal={result['hallucinated']} "
                     f"src={result['source_coverage']:.0%} min={'Y' if result['meets_minimum'] else 'N'} "
                     f"| {q.get('label', '')}")

        # ── Evolutionary précis ───────────────────────────────
        precis_results = []
        for i, q in enumerate(precis_queries):
            chunks = retrieval_cache[q["query"]]
            prompt = build_precis_prompt(q["query"], chunks)
            answer, usage = llm_generate(mc, prompt, q["query"])

            cost = (usage["prompt_tokens"] * mc["cost_input"] +
                    usage["completion_tokens"] * mc["cost_output"]) / 1_000_000
            total_cost += cost

            result = eval_precis(q, answer, chunks)
            precis_results.append(result)

            log.info(f"  [P{i+1}] q={result['quotes']} v={result['verified']} "
                     f"wk={result['works_cited']}/{result['works_required']} "
                     f"drift={'Y' if result['has_drift'] else 'N'} "
                     f"stg={result['stage_coverage']:.0%} | {q.get('label', '')}")

        elapsed = time.time() - t0
        results[name] = {
            "exact": exact_results, "general": general_results,
            "exhaustive": exhaustive_results, "precis": precis_results,
            "elapsed": round(elapsed, 1), "cost": round(total_cost, 4),
        }

    return results


# ── Report ───────────────────────────────────────────────────────────

def print_report(results, n_exact, n_general):
    print(f"\n{'='*80}")
    print(f"  5-MODEL LLM GAUNTLET — ATP RAG Citation Pipeline")
    print(f"{'='*80}")

    # Exact citation
    print(f"\n  EXACT CITATION REPRODUCTION ({n_exact} queries)")
    print(f"  {'Model':<22} {'Pass':>6} {'Verbatim':>9} {'AltPass':>8} {'Time':>7} {'Cost':>8}")
    print(f"  {'─'*62}")

    for name, data in results.items():
        ex = data["exact"]
        n_pass = sum(1 for r in ex if r.get("pass", r.get("verbatim")))
        n_verb = sum(1 for r in ex if r["verbatim"])
        n_alt = sum(1 for r in ex if r.get("any_passage", False))

        print(f"  {name:<22} {n_pass:>3}/{n_exact:<2} {n_verb:>5}/{n_exact:<3} {n_alt:>4}/{n_exact:<3} "
              f"{data['elapsed']:>5.0f}s ${data['cost']:>6.2f}")

    # General queries
    print(f"\n  GENERAL QUERY GENERATION ({n_general} queries)")
    print(f"  {'Model':<22} {'Quotes':>7} {'Vrfy%':>7} {'PgAcc':>7} {'Concept':>8} {'Cost':>8}")
    print(f"  {'─'*62}")

    for name, data in results.items():
        gen = data["general"]
        total_q = sum(r["quotes"] for r in gen)
        total_v = sum(r["verified"] for r in gen)
        avg_vrfy = sum(r["vrfy_rate"] for r in gen) / max(len(gen), 1)
        avg_pg = sum(r["page_acc"] for r in gen) / max(len(gen), 1)
        avg_cc = sum(r["concept_cov"] for r in gen) / max(len(gen), 1)

        print(f"  {name:<22} {total_q:>7} {avg_vrfy:>6.0%} {avg_pg:>6.0%} {avg_cc:>7.0%} "
              f"${data['cost']:>6.2f}")

    # Exhaustive term retrieval
    has_exhaustive = any(data.get("exhaustive") for data in results.values())
    if has_exhaustive:
        n_exh = max(len(data.get("exhaustive", [])) for data in results.values())
        print(f"\n  EXHAUSTIVE TERM RETRIEVAL ({n_exh} queries)")
        print(f"  {'Model':<22} {'Quotes':>7} {'Vrfy%':>7} {'Hallu':>6} {'SrcCov':>7} {'MinMet':>7} {'Cost':>8}")
        print(f"  {'─'*70}")

        for name, data in results.items():
            exh = data.get("exhaustive", [])
            if not exh:
                continue
            tq = sum(r["quotes"] for r in exh)
            tv = sum(r["verified"] for r in exh)
            th = sum(r.get("hallucinated", 0) for r in exh)
            avg_src = sum(r["source_coverage"] for r in exh) / len(exh)
            met = sum(1 for r in exh if r["meets_minimum"])
            print(f"  {name:<22} {tq:>7} {tv/max(tq,1):>6.0%} {th:>5} {avg_src:>6.0%} "
                  f"{met:>3}/{len(exh):<3} ${data['cost']:>6.2f}")

    # Evolutionary précis
    has_precis = any(data.get("precis") for data in results.values())
    if has_precis:
        n_pre = max(len(data.get("precis", [])) for data in results.values())
        print(f"\n  EVOLUTIONARY PRECIS ({n_pre} queries)")
        print(f"  {'Model':<22} {'Quotes':>7} {'Vrfy%':>7} {'WksMet':>7} {'Drift':>6} {'Stages':>7} {'Cost':>8}")
        print(f"  {'─'*68}")

        for name, data in results.items():
            pre = data.get("precis", [])
            if not pre:
                continue
            tq = sum(r["quotes"] for r in pre)
            tv = sum(r["verified"] for r in pre)
            wk_met = sum(1 for r in pre if r["meets_works"])
            drift = sum(1 for r in pre if r["has_drift"])
            avg_stg = sum(r["stage_coverage"] for r in pre) / len(pre)
            print(f"  {name:<22} {tq:>7} {tv/max(tq,1):>6.0%} {wk_met:>3}/{len(pre):<3} "
                  f"{drift:>3}/{len(pre):<2} {avg_stg:>6.0%} ${data['cost']:>6.2f}")

    # Per-model exact detail
    for name, data in results.items():
        print(f"\n  {name} — exact detail:")
        for i, r in enumerate(data["exact"]):
            icon = "+" if r.get("pass", r.get("verbatim")) else "-"
            detail = f"@{r['match_len']}" if r.get("verbatim") else f"r={r.get('best_ratio', 0):.0%}"
            print(f"    {i+1:>2}. {icon} {detail:<8} {r['label']}")

    print(f"\n  Total cost: ${sum(d['cost'] for d in results.values()):.2f}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="5-model LLM evaluation gauntlet")
    parser.add_argument("--models", default="all", help="all or comma-separated model names")
    parser.add_argument("--modes", default="all", help="all or comma-separated: exact,general,exhaustive,precis")
    parser.add_argument("--n-exact", type=int, default=10)
    parser.add_argument("--n-general", type=int, default=10)
    parser.add_argument("--exact-gt", default="data/eval/deleuze_exact_citations.json")
    parser.add_argument("--general-gt", default="data/eval/deleuze_general_queries.json")
    parser.add_argument("--exhaustive-gt", default="data/eval/deleuze_term_exhaustive.json")
    parser.add_argument("--precis-gt", default="data/eval/deleuze_precis_evolution.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--agentic", action="store_true",
                        help="Enable agentic multi-iteration loop for exhaustive queries")
    parser.add_argument("--consensus", type=int, default=0, metavar="N",
                        help="Run exhaustive LLM N times, keep quotes in 2+ runs (0=disabled)")
    args = parser.parse_args()

    # Select models
    if args.models == "all":
        configs = MODEL_CONFIGS
    else:
        names = {n.strip() for n in args.models.split(",")}
        configs = [m for m in MODEL_CONFIGS if m["name"] in names]
        if not configs:
            log.error(f"No matching models. Available: {[m['name'] for m in MODEL_CONFIGS]}")
            sys.exit(1)

    # Check prerequisites
    or_models = [m for m in configs if m["backend"] == "openrouter"]
    if or_models and not OPENROUTER_KEY:
        log.error("OPENROUTER_KEY not set. Set it in .env or environment.")
        sys.exit(1)

    # Load queries
    with open(args.exact_gt) as f:
        exact_all = json.load(f)
    with open(args.general_gt) as f:
        general_all = json.load(f)

    modes = set(args.modes.split(",")) if args.modes != "all" else {"exact", "general", "exhaustive", "precis"}

    exact_queries = exact_all[:args.n_exact] if "exact" in modes else []
    general_queries = ([q for i, q in enumerate(general_all) if i not in SKIP_GENERAL][:args.n_general]
                       if "general" in modes else [])

    exhaustive_queries = []
    if "exhaustive" in modes and Path(args.exhaustive_gt).exists():
        with open(args.exhaustive_gt) as f:
            exhaustive_queries = json.load(f)

    precis_queries = []
    if "precis" in modes and Path(args.precis_gt).exists():
        with open(args.precis_gt) as f:
            precis_queries = json.load(f)

    total_queries = len(exact_queries) + len(general_queries) + len(exhaustive_queries) + len(precis_queries)
    log.info(f"Models: {[m['name'] for m in configs]}")
    log.info(f"Queries: {len(exact_queries)} exact + {len(general_queries)} general "
             f"+ {len(exhaustive_queries)} exhaustive + {len(precis_queries)} precis = {total_queries}")
    log.info(f"Collection: {CHROMA_COLLECTION}")

    if args.dry_run:
        print("\nDRY RUN — would run:")
        for m in configs:
            print(f"  {m['name']} ({m['backend']}: {m['model_id']})")
        print(f"\n  {len(exact_queries)} exact + {len(general_queries)} general "
              f"+ {len(exhaustive_queries)} exhaustive + {len(precis_queries)} precis")
        return

    results = run_gauntlet(exact_queries, general_queries, configs,
                           exhaustive_queries=exhaustive_queries,
                           precis_queries=precis_queries,
                           agentic=args.agentic,
                           consensus=args.consensus)
    print_report(results, len(exact_queries), len(general_queries))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
