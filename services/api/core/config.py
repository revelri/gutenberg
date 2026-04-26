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

    # Post-generation citation page repair: rewrites [Source: …, p. N] tags
    # so the cited page matches the chunk where the quote actually appears.
    enable_citation_repair: bool = True

    # spaCy-driven query expansion: add NER + noun_chunk lemmas to BM25.
    enable_spacy_query_expand: bool = True

    # Per-work retrieval for multi-work queries (précis across titles).
    enable_per_work_recall: bool = True

    # Source filtering: extract book name from query and filter retrieval results
    source_filter_enabled: bool = True

    # Multi-query decomposition: split complex queries into sub-queries
    multi_query_enabled: bool = False
    multi_query_max: int = 4  # max sub-queries to generate

    # Reranker backend: none | gte | colbert
    # - "none"    → passage-score heuristic only (token-overlap fallback)
    # - "gte"     → gte-reranker-modernbert-base cross-encoder (149M, Apache 2.0,
    #               matches 1.2B models on Hit@1 per 2026 benchmarks).
    # - "colbert" → RAGatouille ColBERT (requires ragatouille install; off in
    #               the default Docker image — voyager has no Py 3.13 wheel).
    reranker_backend: str = "gte"
    gte_reranker_model: str = "Alibaba-NLP/gte-reranker-modernbert-base"
    # How many top fusion candidates to feed the cross-encoder. The full
    # retrieval_candidate_k (200) is overkill — on CPU, 200 pairs take ~60s.
    # 30 gives enough recall that top-10 after rerank is stable, and keeps
    # latency under ~10s on CPU.
    reranker_candidate_k: int = 30

    # ColBERT reranker (via RAGatouille): token-level late interaction reranking
    colbert_reranker_enabled: bool = False
    colbert_model: str = "colbert-ir/colbertv2.0"

    # SPLADE query expansion: neural sparse term expansion for BM25
    splade_enabled: bool = False
    splade_model: str = "naver/splade-v3"

    # Filesystem root for all worker-produced artifacts (PDFs, images, indexes).
    # Used to scope filesystem reads in request-handling paths.
    data_root: str = "/data"

    # Database — legacy SQLite path (used when db_backend=sqlite)
    database_path: str = "/data/gutenberg.db"
    # Phase 2 — Postgres / SQLAlchemy backend selector. Values: "sqlite" | "postgres".
    # When set to "postgres", the API uses the SQLAlchemy async engine in
    # core.db with DATABASE_URL; otherwise the legacy aiosqlite layer in
    # core.database is used.
    db_backend: str = "sqlite"
    database_url: str = ""
    db_echo: bool = False
    # Phase 2 — JWT auth. Generate with: python -c 'import secrets; print(secrets.token_urlsafe(64))'
    auth_jwt_secret: str = "CHANGE_ME_IN_PRODUCTION"
    auth_jwt_lifetime_seconds: int = 3600  # access token TTL
    auth_refresh_lifetime_seconds: int = 60 * 60 * 24 * 30  # 30 days

    runpod_api_key: str = ""
    ocr_backend: str = "auto"  # auto|local|runpod|docling

    # P0 — Contextual chunking (Anthropic-style enrichment with prompt cache).
    # Chunks get a 50–100 token doc-aware prefix prepended before embedding and
    # BM25 tokenization. The original chunk text is preserved verbatim for
    # exact-match citation verification.
    feature_contextual_chunking: bool = False
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"
    contextual_prefix_max_tokens: int = 120
    contextual_cache_dir: str = "/data/cache/contextual"
    # Fallback provider when Anthropic is unavailable: "ollama" | "none"
    contextual_fallback: str = "ollama"

    # P1 — SpaCy EntityRuler gazetteer + canonical alias map.
    feature_entity_gazetteer: bool = False
    gazetteer_dir: str = "/app/data/gazetteer"
    # Bounded boost for query/chunk canonical_id overlap during reranking.
    entity_boost_weight: float = 0.15
    entity_boost_cap: int = 3

    # P2 — rapidfuzz verification + anchor validation.
    feature_rapidfuzz_verify: bool = True
    feature_anchor_validation: bool = True
    fuzzy_threshold: float = 0.85

    # P3 — Modal chunks (tables, equations).
    feature_modal_chunks: bool = False
    modal_describe_model: str = ""  # defaults to ollama_llm_model if empty

    # P4 — ColBERTv2 third retriever (parallel to dense + BM25).
    feature_colbert_retrieval: bool = False
    colbert_index_root: str = "/data/colbert"
    colbert_retriever_model: str = "colbert-ir/colbertv2.0"
    colbert_retrieve_k: int = 100
    rrf_colbert_weight: float = 0.4

    # P5 — RAPTOR hierarchical summaries.
    feature_raptor: bool = False
    raptor_max_levels: int = 3
    raptor_cluster_branch: int = 5
    # Summarizer backend: "ollama" (free, serial, slow) or
    # "openrouter" (paid, async-batched, ~30× faster on large corpora).
    raptor_provider: str = "ollama"
    raptor_openrouter_model: str = "google/gemini-2.5-flash-lite"
    raptor_concurrency: int = 40
    openrouter_api_key: str = ""

    # P6 — ALCE citation faithfulness eval (offline script only).
    alce_nli_model: str = "cross-encoder/nli-deberta-v3-base"
    alce_entail_threshold: float = 0.5

    # P7 — Graph-lite entity co-occurrence signal.
    feature_graph_boost: bool = False
    graph_db_path: str = "/data/graph/entities.sqlite"
    graph_boost_weight: float = 0.1

    # P8 — VLM-enhanced answer step.
    feature_vlm_answer: bool = False
    vlm_model: str = "llava:7b"

    # Structured-output hybrid answer (per_work + synthesis JSON schema).
    # Activated automatically when a multi-work query is detected (gazetteer +
    # source patterns) AND OPENROUTER_API_KEY/OPENROUTER_KEY is set.
    feature_structured_answer: bool = False
    feature_structured_answer_single_work: bool = True
    structured_answer_model: str = "google/gemini-2.5-flash"
    structured_answer_timeout: float = 240.0
    # rapidfuzz partial_ratio cutoff (0-100) below which a per_work quote
    # is treated as paraphrase and dropped before render.
    verbatim_min_score: int = 85
    # Per-work targeted retrieval top-up — when ≥2 works are detected in the
    # query, fetch additional dense+BM25 candidates filtered to each work
    # before RRF fusion. Capped to avoid pathological fan-out.
    per_work_fetch_enabled: bool = True
    per_work_fetch_k: int = 15
    per_work_fetch_max_works: int = 4
    vlm_max_images: int = 3

    # P9 — CRAG-lite retrieval gate + query rewrite.
    feature_crag: bool = False
    crag_confident_score: float = 0.6
    crag_ambiguous_score: float = 0.25
    crag_widen_k: int = 40

    # P10 — Contextcite offline audit (sampled, weekly).
    contextcite_sample_size: int = 50

    # P11 — Reindex manifest: API startup refuses index mismatched with flags.
    index_manifest_path: str = "/data/index_manifest.json"
    enforce_index_manifest: bool = False

    # P12 — Structured telemetry for per-query feature-flag A/B.
    telemetry_enabled: bool = True
    telemetry_log_path: str = "/data/telemetry/retrieval.jsonl"
    telemetry_hash_queries: bool = False

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Collection routing: slug → ChromaDB collection name
# Parsed from COLLECTION_ROUTES env var (e.g., "macy:gutenberg-mxbai,dnd:gutenberg")
collection_routes: dict[str, str] = _parse_collection_routes()
