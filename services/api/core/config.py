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

    retrieval_top_k: int = 10
    retrieval_candidate_k: int = 200  # expanded retrieval window before passage scoring

    # BM25 scaling: max chunks to load into memory for BM25.
    # Beyond this threshold, falls back to ChromaDB where_document search.
    bm25_max_chunks: int = 50_000

    # BM25 persistence: path to save/load the pickled BM25 index.
    # Set to empty string to disable persistence (pure in-memory).
    bm25_persist_path: str = "/data/bm25_index.pkl"

    # RRF fusion weights: dense (semantic) vs sparse (keyword) search.
    # Higher dense weight favors meaning-based retrieval for philosophical text.
    rrf_dense_weight: float = 0.6
    rrf_sparse_weight: float = 0.4

    # Query-adaptive RRF: dynamically adjust weights based on query type.
    # Lexical queries (proper nouns, quoted phrases, page refs) get higher
    # sparse weight. Semantic queries (abstract conceptual) get higher dense weight.
    rrf_adaptive: bool = True
    rrf_lexical_dense: float = 0.35  # dense weight when query is lexical
    rrf_lexical_sparse: float = 0.65  # sparse weight when query is lexical
    rrf_semantic_dense: float = 0.70  # dense weight when query is semantic
    rrf_semantic_sparse: float = 0.30  # sparse weight when query is semantic

    # HyDE: generate a hypothetical answer before embedding the query.
    # DISABLED by default — harmful for verbatim citation retrieval in
    # dense philosophical corpora where exact quotes matter more than
    # semantic proximity of paraphrased answers.
    hyde_enabled: bool = False

    # Query-level result cache (LRU, keyed on normalized query + collection).
    query_cache_max_size: int = 100
    query_cache_ttl: int = 300  # seconds

    # Embedding vector cache (LRU, keyed on cleaned text).
    embed_cache_max_size: int = 200

    # Request-level retrieval metrics logging.
    log_retrieval_metrics: bool = True

    # Conversation history: max tokens for history sent to LLM.
    # Prevents silent context truncation by Ollama.
    max_history_tokens: int = 2048

    # SpaCy BM25 enhancements
    bm25_pos_weighting: bool = True  # Weight PROPN 3x, NOUN 2x in BM25 scoring

    # Source filtering: extract book name from query and filter retrieval results
    source_filter_enabled: bool = True

    # Multi-query decomposition: split complex queries into sub-queries
    multi_query_enabled: bool = False
    multi_query_max: int = 4  # max sub-queries to generate

    # ColBERT reranker (via RAGatouille): token-level late interaction reranking
    colbert_reranker_enabled: bool = False
    colbert_model: str = "colbert-ir/colbertv2.0"

    # SPLADE query expansion: neural sparse term expansion for BM25
    splade_enabled: bool = False
    splade_model: str = "naver/splade-v3"

    # Database
    database_path: str = "/data/gutenberg.db"

    # Cloud fallback
    openrouter_api_key: str = ""
    openrouter_model: str = "deepseek/deepseek-r1"
    llm_backend: str = "ollama"  # ollama|openrouter

    runpod_api_key: str = ""
    ocr_backend: str = "auto"  # auto|local|runpod|docling

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Collection routing: slug → ChromaDB collection name
# Parsed from COLLECTION_ROUTES env var (e.g., "macy:gutenberg-mxbai,dnd:gutenberg")
collection_routes: dict[str, str] = _parse_collection_routes()
