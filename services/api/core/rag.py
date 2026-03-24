"""RAG pipeline: hybrid search → rerank → prompt assembly."""

import logging

import chromadb
import httpx
from rank_bm25 import BM25Okapi

from core.config import settings
from core.reranker import rerank

log = logging.getLogger("gutenberg.rag")

# Per-collection BM25 indexes: {collection_name: (corpus, index)}
_bm25_cache: dict[str, tuple[list[dict], BM25Okapi | None]] = {}


def _get_chroma_collection(collection_name: str | None = None):
    host = settings.chroma_host.replace("http://", "").replace("https://", "")
    parts = host.split(":")
    hostname = parts[0]
    port = int(parts[1]) if len(parts) > 1 else 8000
    client = chromadb.HttpClient(host=hostname, port=port)
    name = collection_name or settings.chroma_collection
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def _embed_query(text: str) -> list[float]:
    """Embed a query string via Ollama."""
    resp = httpx.post(
        f"{settings.ollama_host}/api/embed",
        json={"model": settings.ollama_embed_model, "input": [text]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def _build_bm25_index(collection_name: str | None = None):
    """Load all documents from ChromaDB and build BM25 index for a collection.

    If the corpus exceeds bm25_max_chunks, skips building the in-memory
    index. Callers should use _chromadb_text_search() as a fallback.
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

    # Fetch all documents (safe at this scale)
    result = collection.get(include=["documents", "metadatas"])
    corpus = [
        {"id": id_, "text": doc, "metadata": meta}
        for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
    ]

    tokenized = [doc["text"].lower().split() for doc in corpus]
    index = BM25Okapi(tokenized)
    _bm25_cache[col_key] = (corpus, index)
    log.info(f"BM25 index built for '{col_key}' with {len(corpus)} documents")


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
                "dense_score": 1 - dist,  # cosine distance → similarity
            })
    return chunks


def _chromadb_text_search(query: str, top_k: int, collection_name: str | None = None) -> list[dict]:
    """Fallback keyword search using ChromaDB's where_document $contains."""
    collection = _get_chroma_collection(collection_name)

    # Use the most distinctive keyword from the query
    keywords = [w for w in query.lower().split() if len(w) > 3]
    if not keywords:
        keywords = query.lower().split()
    if not keywords:
        return []

    # ChromaDB where_document supports $contains for single-term matching
    # Search with the longest keyword for best selectivity
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
            "bm25_score": 1.0,  # uniform score — ranking left to RRF + reranker
        }
        for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
    ]
    return chunks[:top_k]


def _bm25_search(query: str, top_k: int, collection_name: str | None = None) -> list[dict]:
    """Sparse keyword search via BM25, with ChromaDB fallback for large corpora."""
    col_key = collection_name or settings.chroma_collection

    if col_key not in _bm25_cache:
        _build_bm25_index(collection_name)

    corpus, index = _bm25_cache.get(col_key, ([], None))

    # Fallback to ChromaDB text search if BM25 index wasn't built (corpus too large)
    if index is None or not corpus:
        return _chromadb_text_search(query, top_k, collection_name)

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


def _reciprocal_rank_fusion(dense: list[dict], sparse: list[dict], k: int = 60) -> list[dict]:
    """Merge dense and sparse results using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, chunk in enumerate(dense):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(sparse):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
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

    system_prompt = f"""You are a helpful document assistant. Answer the user's question based on the provided context.
Always cite your sources using [Source N] notation. If the context doesn't contain enough information to answer, say so clearly.

## Context

{context_block}"""

    return system_prompt


def retrieve(query: str, collection: str | None = None) -> tuple[str, list[dict]]:
    """Full retrieval pipeline: embed → hybrid search → rerank → prompt.

    Args:
        query: The user's search query.
        collection: Optional ChromaDB collection name. Defaults to settings.chroma_collection.

    Returns (system_prompt, source_chunks).
    """
    top_k = settings.retrieval_top_k

    # Embed query
    query_embedding = _embed_query(query)

    # Hybrid search
    dense_results = _dense_search(query_embedding, top_k, collection)
    sparse_results = _bm25_search(query, top_k, collection)

    # Fuse
    merged = _reciprocal_rank_fusion(dense_results, sparse_results)

    # Rerank
    reranked = rerank(query, merged, top_k=settings.reranker_top_k)

    # Build prompt
    system_prompt = build_rag_prompt(query, reranked)

    return system_prompt, reranked


def refresh_bm25_index(collection: str | None = None):
    """Force rebuild of the BM25 index (call after new documents ingested)."""
    col_key = collection or settings.chroma_collection
    if col_key in _bm25_cache:
        del _bm25_cache[col_key]
    _build_bm25_index(collection)
