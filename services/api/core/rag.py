"""RAG pipeline: hybrid search → rerank → prompt assembly."""

import logging
import re

import httpx
import nltk
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

from core.chroma import get_collection
from core.config import settings
from core.reranker import rerank

log = logging.getLogger("gutenberg.rag")

# Per-collection BM25 indexes: {collection_name: (corpus, index)}
_bm25_cache: dict[str, tuple[list[dict], BM25Okapi | None]] = {}

# NLTK tokenizer + stemmer for BM25
_stemmer = PorterStemmer()
_nltk_ready = False


def _ensure_nltk():
    """Download NLTK data on first use."""
    global _nltk_ready
    if not _nltk_ready:
        for resource in ["punkt_tab"]:
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)
        _nltk_ready = True


def _tokenize(text: str) -> list[str]:
    """Tokenize and stem text for BM25. Handles punctuation, hyphens, inflections."""
    _ensure_nltk()
    # Split hyphenated words (e.g., "desiring-machines" → ["desiring", "machines"])
    text = re.sub(r"[-–—]", " ", text.lower())
    tokens = nltk.word_tokenize(text)
    # Filter non-alphanumeric tokens and stem
    return [_stemmer.stem(t) for t in tokens if t.isalnum() and len(t) > 1]


def _clean_query(text: str) -> str:
    """Clean query text the same way chunks are cleaned before embedding.

    Mirrors _clean_for_embedding() in embedder.py to ensure query and chunk
    embeddings are in the same distribution.
    """
    text = text.strip()
    if not text:
        return text
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"-{3,}", "---", text)
    text = re.sub(r"_{3,}", "___", text)
    text = re.sub(r" {3,}", "  ", text)
    return text


def _get_chroma_collection(collection_name: str | None = None):
    return get_collection(collection_name)


def _hyde_expand(query: str) -> str:
    """Generate a hypothetical document passage that answers the query.

    Instead of embedding the short query directly, we embed a hypothetical
    answer paragraph. This is much closer in embedding space to the actual
    passage, dramatically improving dense search for queries with no
    lexical overlap to the target passage.
    """
    try:
        resp = httpx.post(
            f"{settings.ollama_host}/api/generate",
            json={
                "model": settings.ollama_llm_model,
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
        hypothetical = resp.json().get("response", "").strip()
        if hypothetical:
            log.info(f"HyDE expanded query ({len(hypothetical)} chars)")
            return hypothetical
    except Exception:
        log.warning("HyDE expansion failed, using original query")
    return query


def _embed_query(text: str) -> list[float]:
    """Embed a query string via Ollama, with same cleaning as chunks."""
    cleaned = _clean_query(text)
    resp = httpx.post(
        f"{settings.ollama_host}/api/embed",
        json={"model": settings.ollama_embed_model, "input": [cleaned]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def _build_bm25_index(collection_name: str | None = None):
    """Load all documents from ChromaDB and build BM25 index for a collection.

    Uses NLTK tokenization with Porter stemming for better keyword matching.
    """
    col_key = collection_name or settings.chroma_collection

    collection = _get_chroma_collection(collection_name)
    count = collection.count()
    if count == 0:
        _bm25_cache[col_key] = ([], None)
        return

    if count > settings.bm25_max_chunks:
        log.warning(
            f"Corpus {col_key} has {count:,} chunks (limit: {settings.bm25_max_chunks:,}). "
            "Skipping in-memory BM25 index — using ChromaDB text search fallback."
        )
        _bm25_cache[col_key] = ([], None)
        return

    result = collection.get(include=["documents", "metadatas"])
    corpus = [
        {"id": id_, "text": doc, "metadata": meta}
        for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
    ]

    tokenized = [_tokenize(doc["text"]) for doc in corpus]
    index = BM25Okapi(tokenized)
    _bm25_cache[col_key] = (corpus, index)
    log.info(f"BM25 index built for '{col_key}' with {len(corpus)} documents (stemmed)")


def _dense_search(query_embedding: list[float], top_k: int, collection_name: str | None = None) -> list[dict]:
    """Dense vector search via ChromaDB."""
    collection = _get_chroma_collection(collection_name)
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
                "dense_score": 1 - dist,
            })
    return chunks


def _chromadb_text_search(query: str, top_k: int, collection_name: str | None = None) -> list[dict]:
    """Fallback keyword search using ChromaDB's where_document $contains."""
    collection = _get_chroma_collection(collection_name)

    keywords = [w for w in query.lower().split() if len(w) > 3]
    if not keywords:
        keywords = query.lower().split()
    if not keywords:
        return []

    keyword = max(keywords, key=len)
    try:
        result = collection.get(
            where_document={"$contains": keyword},
            include=["documents", "metadatas"],
            limit=top_k * 3,
        )
    except Exception:
        log.warning(f"ChromaDB text search failed for '{keyword}', returning empty")
        return []

    if not result["ids"]:
        return []

    chunks = [
        {
            "id": id_,
            "text": doc,
            "metadata": meta,
            "bm25_score": 1.0,
        }
        for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
    ]
    return chunks[:top_k]


def _bm25_search(query: str, top_k: int, collection_name: str | None = None) -> list[dict]:
    """Sparse keyword search via BM25 with NLTK stemming."""
    col_key = collection_name or settings.chroma_collection

    if col_key not in _bm25_cache:
        _build_bm25_index(collection_name)

    corpus, index = _bm25_cache.get(col_key, ([], None))

    if index is None or not corpus:
        return _chromadb_text_search(query, top_k, collection_name)

    tokenized_query = _tokenize(query)
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


def _reciprocal_rank_fusion(
    dense: list[dict],
    sparse: list[dict],
    k: int = 60,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
) -> list[dict]:
    """Merge dense and sparse results using weighted Reciprocal Rank Fusion."""
    w_d = dense_weight if dense_weight is not None else settings.rrf_dense_weight
    w_s = sparse_weight if sparse_weight is not None else settings.rrf_sparse_weight

    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, chunk in enumerate(dense):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + w_d / (k + rank + 1)
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(sparse):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + w_s / (k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[cid] for cid, _ in ranked]


def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    """Assemble the system prompt with context chunks and citations."""
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
            label += f" — {heading}"
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


def retrieve(query: str, collection: str | None = None) -> tuple[str, list[dict]]:
    """Full retrieval pipeline: embed → hybrid search → rerank → prompt."""
    top_k = settings.retrieval_top_k

    # HyDE: embed a hypothetical answer instead of the raw query for dense search.
    # The hypothetical answer is closer in embedding space to the actual passage.
    # BM25 still uses the original query (keyword matching doesn't benefit from HyDE).
    if settings.hyde_enabled:
        hyde_text = _hyde_expand(query)
        query_embedding = _embed_query(hyde_text)
    else:
        query_embedding = _embed_query(query)

    dense_results = _dense_search(query_embedding, top_k, collection)
    sparse_results = _bm25_search(query, top_k, collection)

    merged = _reciprocal_rank_fusion(dense_results, sparse_results)

    reranked = rerank(query, merged, top_k=settings.reranker_top_k)

    system_prompt = build_rag_prompt(query, reranked)

    return system_prompt, reranked


def refresh_bm25_index(collection: str | None = None):
    """Force rebuild of the BM25 index (call after new documents ingested)."""
    col_key = collection or settings.chroma_collection
    if col_key in _bm25_cache:
        del _bm25_cache[col_key]
    _build_bm25_index(collection)
