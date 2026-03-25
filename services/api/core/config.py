"""Application settings via pydantic-settings."""

import os

from pydantic_settings import BaseSettings


def _parse_collection_routes() -> dict[str, str]:
    """Parse COLLECTION_ROUTES env var: 'slug:collection,slug2:collection2'."""
    raw = os.environ.get("COLLECTION_ROUTES", "")
    if not raw:
        return {}
    routes = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            slug, collection = pair.split(":", 1)
            routes[slug.strip()] = collection.strip()
    return routes


class Settings(BaseSettings):
    ollama_host: str = "http://ollama:11434"
    ollama_llm_model: str = "llama3.1:8b-instruct-q4_K_M"
    ollama_embed_model: str = "nomic-embed-text"

    chroma_host: str = "http://chromadb:8000"
    chroma_collection: str = "gutenberg"

    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 5
    retrieval_top_k: int = 10

    # BM25 scaling: max chunks to load into memory for BM25.
    # Beyond this threshold, falls back to ChromaDB where_document search.
    bm25_max_chunks: int = 50_000

    # RRF fusion weights: dense (semantic) vs sparse (keyword) search.
    # Higher dense weight favors meaning-based retrieval for philosophical text.
    rrf_dense_weight: float = 0.6
    rrf_sparse_weight: float = 0.4

    # HyDE: generate a hypothetical answer before embedding the query.
    # Dramatically improves dense search for queries with no lexical overlap.
    hyde_enabled: bool = True

    class Config:
        env_file = ".env"


settings = Settings()

# Collection routing: slug → ChromaDB collection name
# Parsed from COLLECTION_ROUTES env var (e.g., "macy:gutenberg-mxbai,dnd:gutenberg")
collection_routes: dict[str, str] = _parse_collection_routes()
