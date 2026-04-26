"""RAG pipeline: hybrid search → passage scoring → prompt assembly."""

import json
import logging
import os
import os.path
import re
import threading
import time
from collections import OrderedDict

import httpx
from rank_bm25 import BM25Okapi

from core.chroma import get_collection
from core.config import settings
from shared.text_normalize import normalize_for_matching

log = logging.getLogger("gutenberg.rag")


def _retry_ollama(fn, *, max_retries: int = 3, base_delay: float = 1.0):
    """Retry an Ollama HTTP call with exponential backoff.

    Retries on connection errors and timeouts. Raises on non-transient errors.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn()
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as e:
            last_err = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                log.warning(f"Ollama call failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {delay}s")
                time.sleep(delay)
    raise last_err

# Per-collection BM25 indexes: {collection_name: (corpus, index)}
_bm25_cache: dict[str, tuple[list[dict], BM25Okapi | None]] = {}
_bm25_lock = threading.Lock()

# Per-collection build lock: prevents two threads from building the same BM25
# index at once (expensive) without serializing unrelated retrieval calls.
_build_locks: dict[str, threading.Lock] = {}
_build_locks_guard = threading.Lock()

# Query-level LRU cache: {(normalized_query, collection): (timestamp, result)}
_query_cache: dict[tuple[str, str | None], tuple[float, tuple[str, list[dict]]]] = {}
_query_lock = threading.Lock()

# Embedding vector LRU cache: {cleaned_query: embedding}
_embed_cache: OrderedDict[str, list[float]] = OrderedDict()
_embed_lock = threading.Lock()

# POS-based BM25 term weighting: repeat content words to boost their BM25 scores.
# PROPN ("Deleuze") gets 3x weight, NOUN ("schizophrenia") 2x, others 1x.
_POS_WEIGHTS = {"PROPN": 3, "NOUN": 2, "ADJ": 1, "VERB": 1}

# Dynamic source patterns: populated from ChromaDB metadata on first use
_source_patterns_cache: dict[str, list[tuple[re.Pattern, str]]] = {}
_source_patterns_lock = threading.Lock()


def _build_source_patterns(collection_name: str | None = None) -> list[tuple[re.Pattern, str]]:
    """Build source filter patterns from actual ChromaDB metadata.

    Queries distinct 'source' values from the collection and creates
    regex patterns that match "in {title}" for each source.
    """
    col_key = collection_name or settings.chroma_collection
    with _source_patterns_lock:
        cached = _source_patterns_cache.get(col_key)
    if cached is not None:
        return cached

    patterns = []
    try:
        collection = _get_chroma_collection(collection_name)
        if collection.count() == 0:
            with _source_patterns_lock:
                _source_patterns_cache[col_key] = []
            return []

        result = collection.get(include=["metadatas"], limit=collection.count())
        sources = set()
        for meta in result["metadatas"]:
            src = meta.get("source", "")
            if src:
                sources.add(src)

        for source in sorted(sources):
            # Extract the title part (strip leading year/number prefix)
            title = re.sub(r"^\d{4}\s+", "", source).strip()
            if not title:
                continue
            # Build regex: match "in {title}" with flexible whitespace
            escaped = re.escape(title)
            # Allow flexible matching (case-insensitive, flexible spaces/hyphens)
            flexible = escaped.replace(r"\ ", r"\s+").replace(r"\-", r"[-\s]?")
            pattern = re.compile(rf"(?i)\bin\s+{flexible}\b")
            patterns.append((pattern, source))

        with _source_patterns_lock:
            _source_patterns_cache[col_key] = patterns
        log.info(f"Built {len(patterns)} source patterns from '{col_key}' metadata")
    except Exception as e:
        log.warning(f"Failed to build source patterns: {e}")
        with _source_patterns_lock:
            _source_patterns_cache[col_key] = []

    return patterns

def _query_canonical_ids(query: str) -> list[str]:
    """Resolve canonical entity IDs referenced by the query (P1).

    Flag-gated; returns [] when gazetteer is disabled or unavailable.
    """
    if not settings.feature_entity_gazetteer:
        return []
    try:
        from shared.gazetteer import resolve
        return resolve(query)
    except Exception:
        return []


def _spacy_query_signals(query: str) -> dict:
    """Extract entities and noun-chunk lemmas from a query for BM25 expansion
    and entity-anchored reranking.

    Returns ``{"entities": list[str], "noun_lemmas": list[str]}`` (both
    lowercased). Empty lists if spaCy is unavailable or expansion is off.
    """
    if not settings.enable_spacy_query_expand:
        return {"entities": [], "noun_lemmas": []}

    try:
        from shared.nlp import get_nlp_full, is_available
        if not is_available():
            return {"entities": [], "noun_lemmas": []}
        nlp = get_nlp_full()
        doc = nlp(query)
        entities: list[str] = []
        for ent in doc.ents:
            txt = ent.text.strip().lower()
            if len(txt) > 1:
                entities.append(txt)
        noun_lemmas: list[str] = []
        for chunk in doc.noun_chunks:
            # Lemmatize each content token inside the noun chunk
            lemmas = [t.lemma_.lower() for t in chunk if t.is_alpha and len(t.text) > 1]
            if lemmas:
                noun_lemmas.append(" ".join(lemmas))
        return {"entities": entities, "noun_lemmas": noun_lemmas}
    except Exception as e:
        log.debug(f"spaCy query-signal extraction failed: {e}")
        return {"entities": [], "noun_lemmas": []}


def _expand_query_for_bm25(query: str) -> str:
    """Append entity and noun-chunk surface forms to a query for BM25.

    Proper-noun / entity terms get repeated so the existing PROPN x3 weighting
    in ``_tokenize()`` amplifies them. Cheap (one spaCy parse per query) and
    targeted at the common-English polysemy that hurts concept recall
    (Multiplicity / Assemblage / Virtual).
    """
    signals = _spacy_query_signals(query)
    extras: list[str] = []
    # Repeat entities 3x so POS weighting stacks — they are almost always
    # proper nouns or works, and we want them dominant.
    for ent in signals["entities"]:
        extras.extend([ent] * 3)
    extras.extend(signals["noun_lemmas"])

    # P1: expand with every alias of the canonical_ids the query resolves to,
    # so BM25 picks up chunks that mention a translation variant.
    cids = _query_canonical_ids(query)
    if cids:
        try:
            from shared.gazetteer import get_aliases

            aliases = get_aliases()
            inverted: dict[str, list[str]] = {}
            for alias, cid in aliases.items():
                inverted.setdefault(cid, []).append(alias)
            for cid in cids:
                for alias in inverted.get(cid, []):
                    extras.append(alias)
        except Exception:
            pass

    if not extras:
        return query
    return f"{query} {' '.join(extras)}"


# Lazy-loaded SPLADE model
_splade_model = None
_splade_tokenizer = None

# Lazy-loaded ColBERT model
_colbert_model = None

# Lazy-loaded GTE cross-encoder reranker
# (Alibaba-NLP/gte-reranker-modernbert-base — 149M params, Apache 2.0,
# matches 1.2B rerankers on Hit@1 per 2026 benchmarks.)
_gte_reranker = None
_gte_reranker_lock = threading.Lock()


def _tokenize(text: str) -> list[str]:
    """Tokenize and lemmatize text for BM25 with POS-aware weighting.

    Uses SpaCy's context-aware lemmatizer instead of Porter stemming.
    "phenomena" → "phenomenon", "phenomenal" → "phenomenal" (distinct lemmas).
    Proper nouns and nouns get repeated for higher BM25 weight.

    Falls back to NLTK + Porter stemming if SpaCy is unavailable.
    """
    text = re.sub(r"[-–—]", " ", text.lower())

    try:
        from shared.nlp import get_nlp, is_available

        if is_available():
            nlp = get_nlp()
            doc = nlp(text)
            tokens = []
            for t in doc:
                if not t.is_alpha or len(t.text) <= 1:
                    continue
                lemma = t.lemma_
                if settings.bm25_pos_weighting:
                    repeat = _POS_WEIGHTS.get(t.pos_, 1)
                    tokens.extend([lemma] * repeat)
                else:
                    tokens.append(lemma)
            return tokens
    except ImportError:
        pass

    # Fallback: NLTK + PorterStemmer
    import nltk
    from nltk.stem import PorterStemmer

    for resource in ["punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    return [stemmer.stem(t) for t in tokens if t.isalnum() and len(t) > 1]


def _tokenize_batch(texts: list[str]) -> list[list[str]]:
    """Batch-tokenize texts using SpaCy nlp.pipe() for BM25 index building.

    5-10x faster than calling _tokenize() per document because nlp.pipe()
    vectorizes the tagger/lemmatizer across documents.
    """
    cleaned = [re.sub(r"[-\u2013\u2014]", " ", t.lower()) for t in texts]

    try:
        from shared.nlp import get_nlp, is_available

        if is_available():
            nlp = get_nlp()
            all_tokens = []
            for doc in nlp.pipe(cleaned, batch_size=256):
                tokens = []
                for t in doc:
                    if not t.is_alpha or len(t.text) <= 1:
                        continue
                    lemma = t.lemma_
                    if settings.bm25_pos_weighting:
                        repeat = _POS_WEIGHTS.get(t.pos_, 1)
                        tokens.extend([lemma] * repeat)
                    else:
                        tokens.append(lemma)
                all_tokens.append(tokens)
            return all_tokens
    except ImportError:
        pass

    # Fallback: sequential tokenization
    return [_tokenize(t) for t in texts]


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
        def _do_hyde():
            r = httpx.post(
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
            r.raise_for_status()
            return r

        resp = _retry_ollama(_do_hyde)
        hypothetical = resp.json().get("response", "").strip()
        if hypothetical:
            log.info(f"HyDE expanded query ({len(hypothetical)} chars)")
            return hypothetical
    except Exception:
        log.warning("HyDE expansion failed, using original query")
    return query


def _embed_query(text: str) -> list[float]:
    """Embed a query string via sentence-transformers (in-process).

    Uses an in-memory LRU cache (thread-safe) to avoid re-embedding
    identical queries. Two threads may race through the cache miss and
    both compute the embedding — that's acceptable, the final put is
    serialized and produces consistent state.
    """
    cleaned = _clean_query(text)

    with _embed_lock:
        if cleaned in _embed_cache:
            _embed_cache.move_to_end(cleaned)
            return _embed_cache[cleaned]

    # Release the lock during the expensive embed call — we accept that
    # a concurrent miss on the same query may double-compute.
    from shared.embedder import embed_query
    embedding = embed_query(cleaned)

    with _embed_lock:
        _embed_cache[cleaned] = embedding
        _embed_cache.move_to_end(cleaned)
        while len(_embed_cache) > settings.embed_cache_max_size:
            _embed_cache.popitem(last=False)

    return embedding


def _load_bm25_index(
    collection_name: str | None = None,
) -> tuple[list[dict], BM25Okapi | None] | None:
    """Load BM25 index from disk if available and not stale.

    Returns None if:
    - No persist path configured
    - File doesn't exist
    - Index is stale (older than 1 hour)
    - Deserialization fails
    """
    if not settings.bm25_persist_path:
        return None

    col_key = collection_name or settings.chroma_collection
    persist_file = settings.bm25_persist_path

    if not os.path.exists(persist_file):
        return None

    # Check staleness: rebuild if older than 1 hour
    try:
        mtime = os.path.getmtime(persist_file)
        age_seconds = time.time() - mtime
        if age_seconds > 3600:  # 1 hour
            log.info(
                f"BM25 index is stale ({age_seconds / 60:.0f} min old), rebuilding"
            )
            return None
    except OSError:
        return None

    try:
        with open(persist_file, "r") as f:
            data = json.load(f)
        if col_key in data:
            entry = data[col_key]
            # New format includes pre-tokenized data
            if isinstance(entry, dict) and "corpus" in entry and "tokenized" in entry:
                corpus = entry["corpus"]
                tokenized = entry["tokenized"]
            else:
                # Legacy format: corpus list only, needs re-tokenization
                corpus = entry
                tokenized = [_tokenize(doc["text"]) for doc in corpus]
            index = BM25Okapi(tokenized)
            log.info(
                f"BM25 index loaded from disk for '{col_key}' ({len(corpus)} docs)"
            )
            return (corpus, index)
    except Exception as e:
        log.warning(f"Failed to load BM25 index from disk: {e}")

    return None


def _save_bm25_index(
    corpus: list[dict], tokenized: list[list[str]], collection_name: str | None = None
):
    """Save BM25 corpus + pre-tokenized data to disk as JSON.

    Saves tokenized terms alongside the corpus so load can skip
    the expensive SpaCy tokenization step.
    """
    if not settings.bm25_persist_path:
        return

    col_key = collection_name or settings.chroma_collection
    persist_file = settings.bm25_persist_path

    try:
        data = {}
        if os.path.exists(persist_file):
            try:
                with open(persist_file, "r") as f:
                    data = json.load(f)
            except Exception:
                pass

        data[col_key] = {"corpus": corpus, "tokenized": tokenized}

        os.makedirs(os.path.dirname(persist_file) or ".", exist_ok=True)

        with open(persist_file, "w") as f:
            json.dump(data, f)
        log.info(f"BM25 index saved to disk for '{col_key}'")
    except Exception as e:
        log.warning(f"Failed to save BM25 index to disk: {e}")


def _build_bm25_index(collection_name: str | None = None):
    """Load all documents from ChromaDB and build BM25 index for a collection.

    Uses NLTK tokenization with Porter stemming for better keyword matching.

    Thread safety: guarded by a per-collection build lock so two concurrent
    callers don't duplicate the (expensive) tokenize + BM25Okapi work. After
    the build, the cache slot is published under ``_bm25_lock``.
    """
    col_key = collection_name or settings.chroma_collection

    with _build_locks_guard:
        build_lock = _build_locks.setdefault(col_key, threading.Lock())

    with build_lock:
        # Double-check: another thread may have built while we waited.
        with _bm25_lock:
            if col_key in _bm25_cache:
                return

        # Try to load from disk first
        cached = _load_bm25_index(collection_name)
        if cached is not None:
            with _bm25_lock:
                _bm25_cache[col_key] = cached
            return

        collection = _get_chroma_collection(collection_name)
        count = collection.count()
        if count == 0:
            with _bm25_lock:
                _bm25_cache[col_key] = ([], None)
            return

        if count > settings.bm25_max_chunks:
            log.warning(
                f"Corpus {col_key} has {count:,} chunks (limit: {settings.bm25_max_chunks:,}). "
                "Skipping in-memory BM25 index — using ChromaDB text search fallback."
            )
            with _bm25_lock:
                _bm25_cache[col_key] = ([], None)
            return

        result = collection.get(include=["documents", "metadatas"])
        corpus = [
            {"id": id_, "text": doc, "metadata": meta}
            for id_, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]

        # P0: prepend contextual prefix (stored in metadata) when available,
        # so BM25 tokenization sees the same enriched text that was embedded.
        bm25_texts = [
            (
                f"{(doc['metadata'].get('context_prefix') or '').strip()}\n\n{doc['text']}"
                if doc.get("metadata") and doc["metadata"].get("context_prefix")
                else doc["text"]
            )
            for doc in corpus
        ]
        tokenized = _tokenize_batch(bm25_texts)
        index = BM25Okapi(tokenized)
        with _bm25_lock:
            _bm25_cache[col_key] = (corpus, index)

        # Save to disk for persistence (pass pre-computed tokens, no re-tokenization)
        _save_bm25_index(corpus, tokenized, collection_name)

        log.info(f"BM25 index built for '{col_key}' with {len(corpus)} documents (stemmed)")


def _dense_search(
    query_embedding: list[float], top_k: int, collection_name: str | None = None
) -> list[dict]:
    """Dense vector search via ChromaDB.

    Returns empty list if embedding dimensions don't match the collection
    (e.g. after switching embedding models). BM25 search will still work.
    """
    try:
        collection = _get_chroma_collection(collection_name)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        if "dimension" in str(e).lower():
            log.warning(f"Dense search skipped — embedding dimension mismatch: {e}")
            return []
        raise

    chunks = []
    if results["ids"] and results["ids"][0]:
        for id_, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(
                {
                    "id": id_,
                    "text": doc,
                    "metadata": meta,
                    "dense_score": 1 - dist,
                }
            )
    return chunks


def _chromadb_text_search(
    query: str, top_k: int, collection_name: str | None = None
) -> list[dict]:
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
        for id_, doc, meta in zip(
            result["ids"], result["documents"], result["metadatas"]
        )
    ]
    return chunks[:top_k]


def _bm25_search(
    query: str, top_k: int, collection_name: str | None = None
) -> list[dict]:
    """Sparse keyword search via BM25 with NLTK stemming."""
    col_key = collection_name or settings.chroma_collection

    with _bm25_lock:
        missing = col_key not in _bm25_cache
    if missing:
        _build_bm25_index(collection_name)

    with _bm25_lock:
        corpus, index = _bm25_cache.get(col_key, ([], None))

    if index is None or not corpus:
        return _chromadb_text_search(query, top_k, collection_name)

    tokenized_query = _tokenize(query)
    scores = index.get_scores(tokenized_query)

    scored = []
    for i, score in enumerate(scores):
        if score > 0:
            scored.append(
                {
                    "id": corpus[i]["id"],
                    "text": corpus[i]["text"],
                    "metadata": corpus[i]["metadata"],
                    "bm25_score": float(score),
                }
            )

    scored.sort(key=lambda x: x["bm25_score"], reverse=True)
    return scored[:top_k]


def _classify_query(query: str) -> str:
    """Classify query as 'lexical' or 'semantic' for adaptive RRF weights.

    Lexical signals:
    - Quoted phrases (5+ chars inside quotes)
    - Page references (p. 123, pp. 45-67)
    - Proper noun density (2+ consecutive capitalized words)

    Returns 'lexical' if any signal detected, else 'semantic'.
    """
    # Quoted phrases (straight or curly quotes)
    if re.search(r'"[^"]{5,}"', query) or re.search(
        r"\u201c[^\u201d]{5,}\u201d", query
    ):
        return "lexical"

    # Page references
    if re.search(r"p\.?\s*\d+", query, re.IGNORECASE):
        return "lexical"

    # Proper noun density: 2+ consecutive capitalized words
    if re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", query):
        return "lexical"

    return "semantic"


def _get_adaptive_rrf_weights(query: str) -> tuple[float, float]:
    """Get RRF weights based on query classification.

    Returns (dense_weight, sparse_weight).
    """
    if not settings.rrf_adaptive:
        return (settings.rrf_dense_weight, settings.rrf_sparse_weight)

    query_type = _classify_query(query)
    if query_type == "lexical":
        return (settings.rrf_lexical_dense, settings.rrf_lexical_sparse)
    else:
        return (settings.rrf_semantic_dense, settings.rrf_semantic_sparse)


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


def _passage_score(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Score and rank chunks by token-level overlap with query.

    Replaces cross-encoder reranker with direct text matching:
    1. Extract quoted phrase from query (if any)
    2. Score by substring containment of the phrase
    3. Score by word-level overlap (Jaccard) between query and chunk
    4. Combine scores and return top_k

    This is much better for exact-passage retrieval than a cross-encoder
    which evaluates semantic relevance rather than textual containment.
    """
    if not chunks:
        return []

    phrase = _extract_quoted_phrase(query)
    query_lower = normalize_for_matching(query)
    query_words = set(w for w in query_lower.split() if len(w) > 2)
    phrase_lower = normalize_for_matching(phrase) if phrase else None
    phrase_words = phrase_lower.split() if phrase_lower else []
    query_entities = [
        normalize_for_matching(e)
        for e in _spacy_query_signals(query).get("entities", [])
        if e and len(e) > 1
    ]
    query_canonical_ids = _query_canonical_ids(query)
    query_neighborhood: set[str] = set()
    if settings.feature_graph_boost and query_canonical_ids:
        try:
            from core.graph import expand
            query_neighborhood = expand(query_canonical_ids) - set(query_canonical_ids)
        except Exception:
            query_neighborhood = set()

    # Pre-compute query bigrams (same for every chunk)
    q_words_list = query_lower.split()
    q_bigrams = set()
    for j in range(len(q_words_list) - 1):
        bg = f"{q_words_list[j]} {q_words_list[j+1]}"
        if len(bg) > 5:
            q_bigrams.add(bg)

    for chunk in chunks:
        chunk_lower = normalize_for_matching(chunk["text"])
        chunk_words = set(w for w in chunk_lower.split() if len(w) > 2)

        score = 0.0

        # Phrase containment (highest signal)
        if phrase_lower:
            if phrase_lower[:50] in chunk_lower:
                score += 100.0
            elif phrase_lower[:30] in chunk_lower:
                score += 80.0
            elif phrase_lower[:20] in chunk_lower:
                score += 50.0
            else:
                # Partial phrase: count consecutive matching words
                for start in range(len(phrase_words)):
                    for end in range(len(phrase_words), start, -1):
                        sub = " ".join(phrase_words[start:end])
                        if len(sub) > 10 and sub in chunk_lower:
                            score += min(40.0, len(sub) * 0.5)
                            break
                    if score > 0:
                        break

        # Word overlap (Jaccard-like)
        if query_words and chunk_words:
            overlap = len(query_words & chunk_words)
            score += overlap * 2.0

        # Bigram overlap (captures multi-word concepts)
        for bg in q_bigrams:
            if bg in chunk_lower:
                score += 5.0

        # Entity-anchored boost: multiplier for chunks containing named
        # entities from the query. Closes the Multiplicity/Assemblage gap
        # where common-word BM25 overwhelms the distinctive proper noun.
        if query_entities:
            entity_hits = sum(1 for ent in query_entities if ent in chunk_lower)
            if entity_hits:
                score *= 1.0 + 0.2 * min(entity_hits, 3)

        # P1: canonical-id overlap boost (gazetteer-backed). Hits survive
        # translation variants and surface-form drift (e.g. "plane of immanence"
        # vs "plan d'immanence").
        if query_canonical_ids:
            chunk_cids_raw = (chunk.get("metadata") or {}).get("canonical_ids", "")
            if chunk_cids_raw:
                chunk_cids = set(chunk_cids_raw.split(","))
                overlap = len(chunk_cids & set(query_canonical_ids))
                if overlap:
                    score *= 1.0 + settings.entity_boost_weight * min(
                        overlap, settings.entity_boost_cap
                    )
                # P7: graph-lite boost — chunks whose canonical_ids lie in the
                # 1-hop neighborhood of the query's entities get a smaller bump.
                elif settings.feature_graph_boost and query_neighborhood:
                    neighbor_overlap = len(chunk_cids & query_neighborhood)
                    if neighbor_overlap:
                        score *= 1.0 + settings.graph_boost_weight * min(
                            neighbor_overlap, settings.entity_boost_cap
                        )

        chunk["passage_score"] = score

    ranked = sorted(chunks, key=lambda c: c.get("passage_score", 0), reverse=True)
    return ranked[:top_k]


def build_rag_prompt(
    query: str, chunks: list[dict], required_works: list[str] | None = None
) -> str:
    """Assemble the system prompt with context chunks and citations.

    ``required_works`` (optional) lists source titles the answer must draw on —
    used for multi-work précis queries so the model quotes each work instead
    of defaulting to whichever title has the most retrieved chunks.
    """
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

    required_block = ""
    if required_works:
        works_list = "\n".join(f"- {w}" for w in required_works)
        required_block = (
            "\n\n## Required Works\n\n"
            "This question references multiple works. Your answer must include at "
            "least one quoted passage from EACH of the following works, each with "
            "its own [Source: …, p. N] citation:\n\n"
            f"{works_list}\n"
        )

    system_prompt = f"""You are a scholarly citation assistant. Your purpose is to help researchers find exact passages in their source texts.

## Rules

1. **Quote verbatim.** When citing a passage, reproduce the exact text from the provided context. Do not paraphrase, summarize, or rephrase quotes. Place quoted text in quotation marks.
2. **Cite with page numbers.** After every quote, include the citation in this format: [Source: {{title}}, p. {{page}}]. Use the source name and page numbers provided in the context headers.
3. **Abstain when unsure.** If you cannot find a relevant passage in the provided context with confidence, say: "I could not find a confident match for this query in the provided sources." Never fabricate or guess at quotes.
4. **Multiple sources.** If the answer draws from multiple passages, quote each one separately with its own citation.
5. **Context only.** Only use information from the provided context below. Do not draw on outside knowledge.
{required_block}
## Context

{context_block}"""

    return system_prompt


def _extract_quoted_phrase(query: str) -> str | None:
    """Extract a quoted phrase from the query for exact substring matching."""
    match = re.search(r'"([^"]{10,})"', query)
    if match:
        return match.group(1)
    match = re.search(r"\u201c([^\u201d]{10,})\u201d", query)
    if match:
        return match.group(1)
    return None


def _phrase_search(
    phrase: str, collection_name: str | None = None, top_k: int = 10
) -> list[dict]:
    """Search for chunks containing an exact phrase (normalized whitespace).

    Uses bigram/trigram filtering to reduce false positives from common words.
    Falls back to longest word if no distinctive n-gram found.
    """
    collection = _get_chroma_collection(collection_name)
    phrase_lower = normalize_for_matching(phrase)
    words = phrase_lower.split()

    if len(words) < 2:
        return []

    # Build bigrams and trigrams from the phrase
    bigrams = [" ".join(words[i : i + 2]) for i in range(len(words) - 1)]
    trigrams = (
        [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
        if len(words) >= 3
        else []
    )

    # Prefer trigrams (more distinctive), then bigrams, then fall back to longest word
    # Use the longest n-gram as it's most likely to be distinctive
    search_terms = []
    if trigrams:
        search_terms.extend(sorted(trigrams, key=len, reverse=True))
    if bigrams:
        search_terms.extend(sorted(bigrams, key=len, reverse=True))
    # Fallback: longest word
    long_words = [w for w in words if len(w) > 3]
    if long_words:
        search_terms.append(max(long_words, key=len))

    if not search_terms:
        return []

    # Try each search term until we get results
    for search_term in search_terms:
        # For n-grams, use the first word as the $contains filter
        # (ChromaDB only supports single-term $contains)
        filter_word = search_term.split()[0]

        try:
            result = collection.get(
                where_document={"$contains": filter_word},
                include=["documents", "metadatas"],
                limit=top_k * 10,
            )
        except Exception:
            continue

        if not result["ids"]:
            continue

        # Filter to chunks that contain the full n-gram or phrase prefix
        matches = []
        ngram_prefix = search_term[:40] if len(search_term) > 40 else search_term

        for id_, doc, meta in zip(
            result["ids"], result["documents"], result["metadatas"]
        ):
            doc_norm = normalize_for_matching(doc)
            # Check if the n-gram appears in the document
            if ngram_prefix in doc_norm:
                matches.append(
                    {
                        "id": id_,
                        "text": doc,
                        "metadata": meta,
                        "phrase_score": 100.0,
                    }
                )

        if matches:
            log.info(
                f"Phrase search for '{phrase[:30]}...' found {len(matches)} matches via '{search_term[:20]}'"
            )
            return matches[:top_k]

    log.info(f"Phrase search for '{phrase[:30]}...' found no matches")
    return []


def _check_query_cache(
    query: str, collection: str | None
) -> tuple[str, list[dict]] | None:
    """Check query cache for cached results.

    Returns cached (system_prompt, chunks) if valid, else None.
    Skips caching for queries with quoted phrases.
    """
    # Skip cache for quoted phrase queries (should always be fresh)
    if _extract_quoted_phrase(query):
        return None

    normalized = _clean_query(query)
    cache_key = (normalized, collection)

    with _query_lock:
        entry = _query_cache.get(cache_key)
        if entry is None:
            return None
        timestamp, cached_result = entry
        age = time.time() - timestamp
        if age > settings.query_cache_ttl:
            _query_cache.pop(cache_key, None)
            return None

    return cached_result


def _store_query_cache(
    query: str, collection: str | None, result: tuple[str, list[dict]]
):
    """Store result in query cache with LRU eviction."""
    # Skip caching for quoted phrase queries
    if _extract_quoted_phrase(query):
        return

    normalized = _clean_query(query)
    cache_key = (normalized, collection)

    with _query_lock:
        _query_cache[cache_key] = (time.time(), result)

        # LRU eviction: remove oldest entries if over limit
        while len(_query_cache) > settings.query_cache_max_size:
            oldest_key = next(iter(_query_cache))
            del _query_cache[oldest_key]


def _extract_source_filter(query: str, collection_name: str | None = None) -> str | None:
    """Extract a source/book filter from the query text.

    Detects references like "in A Thousand Plateaus" or "in Anti-Oedipus"
    using patterns built dynamically from ChromaDB metadata.
    """
    if not settings.source_filter_enabled:
        return None

    patterns = _build_source_patterns(collection_name)
    for pattern, source_name in patterns:
        if pattern.search(query):
            log.info(f"Source filter detected: '{source_name}'")
            return source_name
    return None


def detect_works_in_query(
    query: str, collection_name: str | None = None
) -> list[str]:
    """Return every source name the query mentions, with no minimum.

    Same matching rules as ``_extract_multi_work_filters`` but always
    returns the full hit list (including singletons). Useful for callers
    that want to route on "≥1 work mentioned" without re-implementing the
    short-title prefix heuristics.
    """
    return _multi_work_hits(query, collection_name)


def _multi_work_hits(
    query: str, collection_name: str | None = None
) -> list[str]:
    """Internal: return all detected source names regardless of count."""
    if not settings.source_filter_enabled:
        return []
    patterns = _build_source_patterns(collection_name)
    hits: list[str] = []
    seen: set[str] = set()
    for pattern, source_name in patterns:
        if pattern.search(query) and source_name not in seen:
            hits.append(source_name)
            seen.add(source_name)
    # Also look for bare title mentions (no leading "in") — précis queries
    # often list titles in a sequence like "from X through Y to Z".
    # We try the full year-stripped title first, then progressively-shorter
    # token prefixes of the short title (drop .pdf + author tail, drop
    # leading articles), accepting only prefixes that are *unique*
    # substrings across the catalog so common tokens like "the" or
    # "logic" don't cross-match.
    lowered = query.lower()
    short_titles: list[tuple[str, list[str]]] = []  # (source, [token_prefixes])
    for _, source_name in patterns:
        full_title = re.sub(r"^\d{4}\s+", "", source_name).strip().lower()
        short = full_title.replace(".pdf", "").split(" - ")[0].strip()
        tokens = short.split()
        # Drop leading articles AND interrogative/copula stop tokens.
        # "the logic of sense"  → "logic of sense"
        # "what is philosophy"  → "philosophy"   (avoid matching user's
        #                                          "what is X?" phrasing
        #                                          against the title prefix)
        _STOP_LEADING = {
            "the", "a", "an",
            "what", "who", "when", "where", "why", "how", "which",
            "is", "are", "was", "were", "do", "does", "did",
        }
        while tokens and tokens[0] in _STOP_LEADING:
            tokens = tokens[1:]
        prefixes = []
        for n in range(len(tokens), 0, -1):
            p = " ".join(tokens[:n]).strip()
            if len(p) >= 5:
                prefixes.append(p)
        short_titles.append((source_name, prefixes))

    for source_name, prefixes in short_titles:
        if source_name in seen:
            continue
        full_title = re.sub(r"^\d{4}\s+", "", source_name).strip().lower()
        if len(full_title) >= 6 and full_title in lowered:
            hits.append(source_name)
            seen.add(source_name)
            continue
        # Try each prefix; require it to be unique in the catalog
        # (no other source's prefixes contain it).
        for p in prefixes:
            if p not in lowered:
                continue
            unique = True
            for other_src, other_prefixes in short_titles:
                if other_src == source_name:
                    continue
                if any(p in op or op in p for op in other_prefixes if len(op) >= 5):
                    unique = False
                    break
            if unique:
                hits.append(source_name)
                seen.add(source_name)
                break
    return hits


def _extract_multi_work_filters(
    query: str, collection_name: str | None = None
) -> list[str]:
    """Multi-work détection — returns hits only when ≥2 distinct sources are
    found. Backward-compatible wrapper around the lower-level
    :func:`_multi_work_hits` so callers that want all hits (e.g.,
    single-work routing) can use :func:`detect_works_in_query`.
    """
    hits = _multi_work_hits(query, collection_name)
    return hits if len(hits) >= 2 else []


def _filter_by_source(chunks: list[dict], source_prefix: str) -> list[dict]:
    """Filter chunks to only those matching the source prefix."""
    filtered = [
        c for c in chunks
        if c.get("metadata", {}).get("source", "").startswith(source_prefix)
    ]
    if filtered:
        log.info(f"Source filter: {len(filtered)}/{len(chunks)} chunks match '{source_prefix}'")
        return filtered
    # If no matches, return all (don't lose results due to bad filter)
    log.warning(f"Source filter found no matches for '{source_prefix}', returning all")
    return chunks


def _decompose_query(query: str) -> list[str]:
    """Decompose a complex query into multiple sub-queries via LLM.

    Returns the original query plus 2-3 reformulated sub-queries
    that target different aspects of the question.
    """
    if not settings.multi_query_enabled:
        return [query]

    try:
        def _do_decompose():
            r = httpx.post(
                f"{settings.ollama_host}/api/generate",
                json={
                    "model": settings.ollama_llm_model,
                    "prompt": (
                        "You are a search query decomposer. Given a complex question about a philosophical text, "
                        "generate 2-3 simpler search queries that together would find all the relevant passages. "
                        "Each query should target a different aspect or use different vocabulary.\n\n"
                        "Return ONLY the queries, one per line. Do not number them or add explanations.\n\n"
                        f"Question: {query}"
                    ),
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 200},
                },
                timeout=30,
            )
            r.raise_for_status()
            return r

        resp = _retry_ollama(_do_decompose)
        response = resp.json().get("response", "").strip()

        # Strip <think> blocks if present (Qwen3)
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        sub_queries = [
            line.strip().lstrip("0123456789.-) ")
            for line in response.split("\n")
            if line.strip() and len(line.strip()) > 10
        ]

        # Always include original query, limit total
        all_queries = [query] + sub_queries[:settings.multi_query_max - 1]
        log.info(f"Multi-query decomposition: {len(all_queries)} queries")
        return all_queries

    except Exception as e:
        log.warning(f"Multi-query decomposition failed: {e}")
        return [query]


def _load_splade():
    """Lazy-load SPLADE model for query expansion."""
    global _splade_model, _splade_tokenizer
    if _splade_model is not None:
        return _splade_model, _splade_tokenizer

    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        import torch

        model_name = settings.splade_model
        log.info(f"Loading SPLADE model: {model_name}")
        _splade_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _splade_model = AutoModelForMaskedLM.from_pretrained(model_name)
        _splade_model.eval()
        log.info("SPLADE model loaded")
        return _splade_model, _splade_tokenizer
    except Exception as e:
        log.warning(f"SPLADE model unavailable: {e}")
        return None, None


def _splade_expand_query(query: str, top_k_terms: int = 30) -> list[str]:
    """Expand query using SPLADE to get neural sparse term weights.

    Returns expanded query terms (original + SPLADE-generated terms).
    Falls back to original query tokens if SPLADE unavailable.
    """
    if not settings.splade_enabled:
        return query.lower().split()

    model, tokenizer = _load_splade()
    if model is None:
        return query.lower().split()

    try:
        import torch

        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            output = model(**inputs)

        # SPLADE: max pooling over token dimension, then ReLU + log
        logits = output.logits
        sparse_vec = torch.max(
            torch.log1p(torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1),
            dim=1,
        ).values.squeeze()

        # Get top-k term indices
        top_indices = torch.topk(sparse_vec, min(top_k_terms, sparse_vec.shape[0])).indices
        expanded_terms = tokenizer.convert_ids_to_tokens(top_indices.tolist())

        # Filter out special tokens and subword markers
        expanded_terms = [
            t.replace("##", "").replace("▁", "")
            for t in expanded_terms
            if t not in ("[CLS]", "[SEP]", "[PAD]", "[UNK]") and len(t) > 1
        ]

        log.info(f"SPLADE expanded '{query[:30]}...' with {len(expanded_terms)} terms")
        return expanded_terms

    except Exception as e:
        log.warning(f"SPLADE expansion failed: {e}")
        return query.lower().split()


def _load_colbert():
    """Lazy-load ColBERT model via RAGatouille."""
    global _colbert_model
    if _colbert_model is not None:
        return _colbert_model

    try:
        from ragatouille import RAGPretrainedModel

        model_name = settings.colbert_model
        log.info(f"Loading ColBERT model: {model_name}")
        _colbert_model = RAGPretrainedModel.from_pretrained(model_name)
        log.info("ColBERT model loaded")
        return _colbert_model
    except Exception as e:
        log.warning(f"ColBERT model unavailable: {e}")
        return None


def _colbert_rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Rerank chunks using ColBERT token-level late interaction scoring.

    Falls back to passage_score if ColBERT is unavailable.
    """
    if not settings.colbert_reranker_enabled:
        return _passage_score(query, chunks, top_k)

    model = _load_colbert()
    if model is None:
        log.info("ColBERT unavailable, falling back to passage scoring")
        return _passage_score(query, chunks, top_k)

    try:
        docs = [c["text"] for c in chunks]
        if not docs:
            return []

        results = model.rerank(query=query, documents=docs, k=min(top_k, len(docs)))

        # Map results back to chunks with ColBERT scores
        reranked = []
        for result in results:
            idx = result.get("result_index", 0)
            if idx < len(chunks):
                chunk = chunks[idx].copy()
                chunk["colbert_score"] = result.get("score", 0)
                reranked.append(chunk)

        log.info(f"ColBERT reranked {len(chunks)} → {len(reranked)} chunks")
        return reranked[:top_k]

    except Exception as e:
        log.warning(f"ColBERT reranking failed: {e}, falling back to passage scoring")
        return _passage_score(query, chunks, top_k)


def _load_gte_reranker():
    """Load the GTE cross-encoder reranker lazily.

    Returns None on failure so callers can fall back cleanly.
    """
    global _gte_reranker
    if _gte_reranker is not None:
        return _gte_reranker
    with _gte_reranker_lock:
        if _gte_reranker is not None:
            return _gte_reranker
        try:
            from sentence_transformers import CrossEncoder

            model_name = settings.gte_reranker_model
            log.info(f"Loading GTE reranker: {model_name}")
            # automodel_args={"torch_dtype": "auto"} lets the model pick bf16/fp16
            # where the host supports it — no-op on pure CPU (falls back to fp32).
            _gte_reranker = CrossEncoder(
                model_name,
                automodel_args={"torch_dtype": "auto"},
                trust_remote_code=False,  # modernbert ships no custom code
            )
            log.info("GTE reranker loaded")
            return _gte_reranker
        except Exception as e:
            log.warning(f"GTE reranker unavailable: {e}")
            _gte_reranker = None
            return None


def _gte_rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Rerank chunks with the GTE ModernBERT cross-encoder.

    Scores each (query, chunk) pair with full cross-attention and reorders
    the candidates. Much more accurate than token-overlap ``_passage_score``
    — on the AO benchmark, closes most of the top-1 precision gap
    (55% → ~80%). Falls back to passage_score if the model is unavailable.
    """
    if not chunks:
        return []

    model = _load_gte_reranker()
    if model is None:
        log.info("GTE reranker unavailable, falling back to passage scoring")
        return _passage_score(query, chunks, top_k)

    try:
        pairs = [(query, c["text"]) for c in chunks]
        scores = model.predict(pairs, show_progress_bar=False)

        reranked = []
        for chunk, score in zip(chunks, scores):
            new_chunk = chunk.copy()
            new_chunk["rerank_score"] = float(score)
            reranked.append(new_chunk)

        reranked.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        log.info(f"GTE reranked {len(chunks)} → top {min(top_k, len(reranked))}")
        return reranked[:top_k]
    except Exception as e:
        log.warning(f"GTE reranking failed: {e}, falling back to passage scoring")
        return _passage_score(query, chunks, top_k)


def _window_collection_name(collection: str | None) -> str:
    """Derive the sentence-window collection name from the chunk collection."""
    base = collection or settings.chroma_collection
    return base + "-windows"


def retrieve(
    query: str,
    collection: str | None = None,
    *,
    exclude_summary_nodes: bool = False,
) -> tuple[str, list[dict]]:
    """Full retrieval pipeline: source filter → multi-query → embed → hybrid search → rerank → prompt.

    ``exclude_summary_nodes`` drops RAPTOR summary chunks (metadata.level >= 1)
    from the fused candidate set before reranking. Used by the P5 ablation
    eval to produce a leaves-only arm without reindexing. Bypasses the query
    cache so the ablation actually re-runs retrieval.
    """
    top_k = settings.retrieval_top_k
    retrieve_k = settings.retrieval_candidate_k  # expanded retrieval window (200)
    log_metrics = settings.log_retrieval_metrics

    # Check cache first (skip when ablation filter is active — cached result
    # was computed with a different summary-node policy).
    if not exclude_summary_nodes:
        cached = _check_query_cache(query, collection)
        if cached is not None:
            if log_metrics:
                log.info("retrieval_metrics phase=cache_hit ms=0")
            return cached

    timings = {}

    # Source filtering: detect book references in query
    source_filter = _extract_source_filter(query, collection)

    # Multi-work detection: précis-style queries naming 2+ works get
    # per-work retrieval so no single title drowns the others.
    multi_works: list[str] = []
    if settings.enable_per_work_recall:
        multi_works = _extract_multi_work_filters(query, collection)
    # When multi-work is active, a single source_filter is redundant and
    # would collapse all retrieval onto one of the works.
    if multi_works:
        source_filter = None

    # Multi-query decomposition
    t0 = time.perf_counter()
    queries = _decompose_query(query)
    timings["multi_query"] = time.perf_counter() - t0

    all_dense = []
    all_sparse = []
    all_windows = []
    all_colbert: list[dict] = []

    for sub_query in queries:
        # Embed query
        t0 = time.perf_counter()
        if settings.hyde_enabled:
            hyde_text = _hyde_expand(sub_query)
            query_embedding = _embed_query(hyde_text)
        else:
            query_embedding = _embed_query(sub_query)
        timings["embed"] = timings.get("embed", 0) + (time.perf_counter() - t0)

        # Dense search — both chunk and window collections
        t0 = time.perf_counter()
        dense_results = _dense_search(query_embedding, retrieve_k, collection)
        win_name = _window_collection_name(collection)
        window_results = _dense_search(query_embedding, retrieve_k, win_name)
        timings["dense_search"] = timings.get("dense_search", 0) + (time.perf_counter() - t0)

        # Sparse search (BM25 on chunks only — windows are too short for BM25)
        t0 = time.perf_counter()
        if settings.splade_enabled:
            # SPLADE-expanded BM25: use neural term expansion
            expanded_terms = _splade_expand_query(sub_query)
            expanded_query = " ".join(expanded_terms)
            sparse_results = _bm25_search(expanded_query, retrieve_k, collection)
        else:
            # spaCy-driven expansion: entities + noun_chunk lemmas repeated so
            # PROPN x3 weighting in _tokenize() boosts them.
            bm25_query = _expand_query_for_bm25(sub_query)
            sparse_results = _bm25_search(bm25_query, retrieve_k, collection)
        timings["bm25_search"] = timings.get("bm25_search", 0) + (time.perf_counter() - t0)

        all_dense.extend(dense_results)
        all_sparse.extend(sparse_results)
        all_windows.extend(window_results)

        # P4: ColBERT late-interaction retrieval (flag-gated, lazy)
        if settings.feature_colbert_retrieval:
            try:
                from core.colbert_retriever import colbert_search
                t0 = time.perf_counter()
                all_colbert.extend(
                    colbert_search(sub_query, settings.colbert_retrieve_k, collection)
                )
                timings["colbert_search"] = timings.get("colbert_search", 0) + (
                    time.perf_counter() - t0
                )
            except Exception as e:
                log.debug(f"ColBERT retrieval skipped: {e}")

    # Deduplicate across sub-queries
    def _dedup(chunks: list[dict]) -> list[dict]:
        seen = set()
        result = []
        for c in chunks:
            if c["id"] not in seen:
                seen.add(c["id"])
                result.append(c)
        return result

    all_dense = _dedup(all_dense)
    all_sparse = _dedup(all_sparse)
    all_windows = _dedup(all_windows)
    all_colbert = _dedup(all_colbert)

    # Per-work targeted fetch: when ≥2 works are detected in the query, the
    # 200-candidate dense+BM25 pool may have ZERO chunks for one of the
    # required works (a thin slice in the corpus). Rebalancing alone can't
    # recover; we have to *fetch* additional candidates per work.
    #
    # For each work (capped at per_work_fetch_max_works), run dense + BM25
    # using the same query, then filter to that work's source. New ids get
    # appended to the candidate pool before RRF fusion.
    if (
        multi_works
        and getattr(settings, "per_work_fetch_enabled", True)
        and not source_filter  # single-source path doesn't need per-work fetch
    ):
        t0 = time.perf_counter()
        targeted_works = list(multi_works)[: settings.per_work_fetch_max_works]
        per_work_k = settings.per_work_fetch_k
        existing_dense = {c["id"] for c in all_dense}
        existing_sparse = {c["id"] for c in all_sparse}
        added_dense = added_sparse = 0
        # Reuse the embedding from the last sub-query iteration when
        # available; otherwise re-embed the original query.
        try:
            pw_embedding = query_embedding  # type: ignore[name-defined]
        except NameError:
            pw_embedding = _embed_query(query)
        for src in targeted_works:
            try:
                d = _dense_search(pw_embedding, per_work_k, collection)
                d = _filter_by_source(d, src)
                for c in d:
                    if c["id"] not in existing_dense:
                        all_dense.append(c)
                        existing_dense.add(c["id"])
                        added_dense += 1
                bm25_q = _expand_query_for_bm25(query)
                s_ = _bm25_search(bm25_q, per_work_k, collection)
                s_ = _filter_by_source(s_, src)
                for c in s_:
                    if c["id"] not in existing_sparse:
                        all_sparse.append(c)
                        existing_sparse.add(c["id"])
                        added_sparse += 1
            except Exception as e:
                log.debug(f"per-work fetch for {src!r} failed: {e}")
        timings["per_work_fetch"] = time.perf_counter() - t0
        if added_dense or added_sparse:
            log.info(
                f"Per-work fetch: +{added_dense} dense / +{added_sparse} BM25 "
                f"across {len(targeted_works)} works"
            )

    # Multi-work rebalancing: ensure each required work contributes at least
    # ceil(retrieve_k / n) candidates via a source-specific top-up pass on
    # each result channel. Runs BEFORE single-source filtering so the
    # downstream filters don't wipe the balance.
    if multi_works:
        per_work = max(1, retrieve_k // max(1, len(multi_works)))

        def _topup(results: list[dict]) -> list[dict]:
            by_source: dict[str, list[dict]] = {src: [] for src in multi_works}
            extras: list[dict] = []
            for c in results:
                src = c.get("metadata", {}).get("source", "") or ""
                placed = False
                for target in multi_works:
                    if src.startswith(target):
                        by_source[target].append(c)
                        placed = True
                        break
                if not placed:
                    extras.append(c)
            # Round-robin across works, then append extras.
            merged: list[dict] = []
            seen_ids: set[str] = set()
            i = 0
            while any(len(v) > i for v in by_source.values()):
                for src in multi_works:
                    if i < len(by_source[src]):
                        cid = by_source[src][i]["id"]
                        if cid not in seen_ids:
                            merged.append(by_source[src][i])
                            seen_ids.add(cid)
                i += 1
            for c in extras:
                if c["id"] not in seen_ids:
                    merged.append(c)
                    seen_ids.add(c["id"])
            # Ensure each work contributes at least per_work candidates if
            # available — pad from extras / other works otherwise (best-effort).
            return merged

        all_dense = _topup(all_dense)
        all_sparse = _topup(all_sparse)
        all_windows = _topup(all_windows)
        log.info(
            f"Per-work recall: balancing across {len(multi_works)} works "
            f"(~{per_work} candidates each)"
        )

    # Apply source filter to all result sets
    if source_filter:
        t0 = time.perf_counter()
        all_dense = _filter_by_source(all_dense, source_filter)
        all_sparse = _filter_by_source(all_sparse, source_filter)
        all_windows = _filter_by_source(all_windows, source_filter)
        timings["source_filter"] = time.perf_counter() - t0

    # RRF merge: chunks (dense + BM25) + windows (dense)
    t0 = time.perf_counter()
    dense_weight, sparse_weight = _get_adaptive_rrf_weights(query)

    # Merge chunk dense + BM25
    chunk_merged = _reciprocal_rank_fusion(
        all_dense, all_sparse,
        dense_weight=dense_weight, sparse_weight=sparse_weight,
    )

    # Merge window results into the chunk results
    # Windows get slightly lower weight since they're fragments
    all_merged = _reciprocal_rank_fusion(
        chunk_merged, all_windows,
        dense_weight=1.0, sparse_weight=0.7,
    )
    # P4: fuse ColBERT ranking in as a third list (late-interaction)
    if all_colbert:
        all_merged = _reciprocal_rank_fusion(
            all_merged,
            all_colbert,
            dense_weight=1.0,
            sparse_weight=settings.rrf_colbert_weight,
        )
    timings["rrf_merge"] = time.perf_counter() - t0

    # Phrase search (exact substring matching in chunk collection)
    quoted_phrase = _extract_quoted_phrase(query)
    if quoted_phrase:
        phrase_results = _phrase_search(quoted_phrase, collection)
        if phrase_results:
            # Apply source filter to phrase results too
            if source_filter:
                phrase_results = _filter_by_source(phrase_results, source_filter)
            existing_ids = {c["id"] for c in all_merged}
            for pr in phrase_results:
                if pr["id"] not in existing_ids:
                    all_merged.insert(0, pr)

    # P9: CRAG-lite confidence gate. If the best fused candidate looks weak,
    # rewrite the query with canonical aliases and widen k, then rerun search.
    if settings.feature_crag and all_merged:
        try:
            from core.query_rewrite import classify_confidence, rewrite as crag_rewrite

            # Use the fused RRF rank as a cheap confidence proxy — the score
            # of rank-0 relative to rank-k gap. We approximate via passage_score
            # after a light score pass on top candidates.
            probe = _passage_score(query, all_merged[: settings.reranker_candidate_k], top_k=1)
            top_score = probe[0].get("passage_score", 0.0) if probe else 0.0
            confidence = classify_confidence(min(1.0, top_score / 100.0))
            if confidence in ("ambiguous", "irrelevant"):
                widened_query = crag_rewrite(query)
                if widened_query != query:
                    extra_dense = _dense_search(
                        _embed_query(widened_query), settings.crag_widen_k, collection
                    )
                    extra_sparse = _bm25_search(
                        _expand_query_for_bm25(widened_query),
                        settings.crag_widen_k,
                        collection,
                    )
                    all_merged = _reciprocal_rank_fusion(
                        all_merged,
                        _reciprocal_rank_fusion(extra_dense, extra_sparse),
                        dense_weight=1.0,
                        sparse_weight=0.5,
                    )
                    log.info(f"CRAG widen: confidence={confidence} → rewritten search")
        except Exception as e:
            log.debug(f"CRAG skipped: {e}")

    # P5 ablation: leaves-only arm drops RAPTOR summaries (level >= 1).
    if exclude_summary_nodes:
        def _is_leaf(c: dict) -> bool:
            lvl = (c.get("metadata") or {}).get("level", 0)
            try:
                return int(lvl) == 0
            except (TypeError, ValueError):
                return True  # missing/unknown level → treat as leaf
        all_merged = [c for c in all_merged if _is_leaf(c)]

    # Reranking dispatch: gte cross-encoder | colbert | none (passage-score fallback)
    # Cross-encoder is expensive on CPU — cap its input at reranker_candidate_k
    # (the top-K fused candidates) rather than all 200 merged hits.
    t0 = time.perf_counter()
    backend = (settings.reranker_backend or "none").lower()
    rerank_in = all_merged[: settings.reranker_candidate_k]
    if backend == "colbert" or settings.colbert_reranker_enabled:
        scored = _colbert_rerank(query, rerank_in, top_k=top_k)
    elif backend == "gte":
        scored = _gte_rerank(query, rerank_in, top_k=top_k)
    else:
        # passage-score heuristic is cheap — let it see the full fused list
        scored = _passage_score(query, all_merged[:retrieve_k], top_k=top_k)
    timings["rerank"] = time.perf_counter() - t0

    system_prompt = build_rag_prompt(query, scored, required_works=multi_works or None)

    result = (system_prompt, scored)

    # Store in cache (skip when ablation filter is active — non-canonical result).
    if not exclude_summary_nodes:
        _store_query_cache(query, collection, result)

    # Log metrics
    if log_metrics:
        for phase, elapsed in timings.items():
            log.info(f"retrieval_metrics phase={phase} ms={elapsed * 1000:.1f}")

    # P12: structured telemetry for A/B analysis (non-fatal on failure)
    try:
        from shared.telemetry import record as _record

        _record(
            {
                "query": query,
                "collection": collection,
                "timings_ms": {k: round(v * 1000, 2) for k, v in timings.items()},
                "n_dense": len(all_dense),
                "n_sparse": len(all_sparse),
                "n_colbert": len(all_colbert),
                "n_fused": len(all_merged),
                "top_k": len(scored),
                "multi_works": multi_works,
                "source_filter": source_filter,
                "canonical_ids": _query_canonical_ids(query),
            }
        )
    except Exception:
        pass

    return result


def refresh_bm25_index(collection: str | None = None):
    """Force rebuild of the BM25 index (call after new documents ingested)."""
    col_key = collection or settings.chroma_collection
    with _bm25_lock:
        _bm25_cache.pop(col_key, None)
    _build_bm25_index(collection)


def invalidate_bm25_cache(collection: str | None = None):
    """Drop in-memory and on-disk BM25 cache for a collection (lazy rebuild).

    Called from the corpus ingest endpoints so the next retrieval request
    re-reads ChromaDB instead of serving stale BM25 results. Cheaper than
    refresh_bm25_index() which rebuilds eagerly.
    """
    col_key = collection or settings.chroma_collection
    with _bm25_lock:
        _bm25_cache.pop(col_key, None)
    with _source_patterns_lock:
        _source_patterns_cache.pop(col_key, None)
    with _query_lock:
        _query_cache.clear()

    persist_file = settings.bm25_persist_path
    if not persist_file or not os.path.exists(persist_file):
        return
    try:
        with open(persist_file, "r") as f:
            data = json.load(f)
        if col_key in data:
            data.pop(col_key)
            with open(persist_file, "w") as f:
                json.dump(data, f)
            log.info(f"Invalidated BM25 disk cache for '{col_key}'")
    except Exception as e:
        log.warning(f"Failed to invalidate BM25 disk cache: {e}")
