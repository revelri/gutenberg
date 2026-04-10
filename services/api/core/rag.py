"""RAG pipeline: hybrid search → passage scoring → prompt assembly."""

import json
import logging
import os
import os.path
import re
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

# Query-level LRU cache: {(normalized_query, collection): (timestamp, result)}
_query_cache: dict[tuple[str, str | None], tuple[float, tuple[str, list[dict]]]] = {}

# Embedding vector LRU cache: {cleaned_query: embedding}
_embed_cache: OrderedDict[str, list[float]] = OrderedDict()

# POS-based BM25 term weighting: repeat content words to boost their BM25 scores.
# PROPN ("Deleuze") gets 3x weight, NOUN ("schizophrenia") 2x, others 1x.
_POS_WEIGHTS = {"PROPN": 3, "NOUN": 2, "ADJ": 1, "VERB": 1}

# Dynamic source patterns: populated from ChromaDB metadata on first use
_source_patterns_cache: dict[str, list[tuple[re.Pattern, str]]] = {}


def _build_source_patterns(collection_name: str | None = None) -> list[tuple[re.Pattern, str]]:
    """Build source filter patterns from actual ChromaDB metadata.

    Queries distinct 'source' values from the collection and creates
    regex patterns that match "in {title}" for each source.
    """
    col_key = collection_name or settings.chroma_collection
    if col_key in _source_patterns_cache:
        return _source_patterns_cache[col_key]

    patterns = []
    try:
        collection = _get_chroma_collection(collection_name)
        if collection.count() == 0:
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

        _source_patterns_cache[col_key] = patterns
        log.info(f"Built {len(patterns)} source patterns from '{col_key}' metadata")
    except Exception as e:
        log.warning(f"Failed to build source patterns: {e}")
        _source_patterns_cache[col_key] = []

    return patterns

# Lazy-loaded SPLADE model
_splade_model = None
_splade_tokenizer = None

# Lazy-loaded ColBERT model
_colbert_model = None


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
    """Embed a query string via Ollama, with same cleaning as chunks.

    Uses an in-memory LRU cache to avoid re-embedding identical queries.
    """
    cleaned = _clean_query(text)

    # Check cache
    if cleaned in _embed_cache:
        # Move to end (most recently used)
        _embed_cache.move_to_end(cleaned)
        return _embed_cache[cleaned]

    def _do_embed():
        r = httpx.post(
            f"{settings.ollama_host}/api/embed",
            json={"model": settings.ollama_embed_model, "input": [cleaned]},
            timeout=30,
        )
        r.raise_for_status()
        return r

    resp = _retry_ollama(_do_embed)
    embedding = resp.json()["embeddings"][0]

    # Cache with LRU eviction
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
        # Validate structure: {collection_key: [corpus_list]}
        if col_key in data:
            corpus = data[col_key]
            # Rebuild BM25 index from corpus
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
    corpus: list[dict], index: BM25Okapi, collection_name: str | None = None
):
    """Save BM25 corpus to disk as JSON. Index is rebuilt on load."""
    if not settings.bm25_persist_path:
        return

    col_key = collection_name or settings.chroma_collection
    persist_file = settings.bm25_persist_path

    try:
        # Load existing data or start fresh
        data = {}
        if os.path.exists(persist_file):
            try:
                with open(persist_file, "r") as f:
                    data = json.load(f)
            except Exception:
                pass

        data[col_key] = corpus

        # Ensure directory exists
        os.makedirs(os.path.dirname(persist_file), exist_ok=True)

        with open(persist_file, "w") as f:
            json.dump(data, f)
        log.info(f"BM25 index saved to disk for '{col_key}'")
    except Exception as e:
        log.warning(f"Failed to save BM25 index to disk: {e}")


def _build_bm25_index(collection_name: str | None = None):
    """Load all documents from ChromaDB and build BM25 index for a collection.

    Uses NLTK tokenization with Porter stemming for better keyword matching.
    """
    col_key = collection_name or settings.chroma_collection

    # Try to load from disk first
    cached = _load_bm25_index(collection_name)
    if cached is not None:
        _bm25_cache[col_key] = cached
        return

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
        for id_, doc, meta in zip(
            result["ids"], result["documents"], result["metadatas"]
        )
    ]

    tokenized = [_tokenize(doc["text"]) for doc in corpus]
    index = BM25Okapi(tokenized)
    _bm25_cache[col_key] = (corpus, index)

    # Save to disk for persistence
    _save_bm25_index(corpus, index, collection_name)

    log.info(f"BM25 index built for '{col_key}' with {len(corpus)} documents (stemmed)")


def _dense_search(
    query_embedding: list[float], top_k: int, collection_name: str | None = None
) -> list[dict]:
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

        chunk["passage_score"] = score

    ranked = sorted(chunks, key=lambda c: c.get("passage_score", 0), reverse=True)
    return ranked[:top_k]


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

    if cache_key not in _query_cache:
        return None

    timestamp, cached_result = _query_cache[cache_key]
    age = time.time() - timestamp

    if age > settings.query_cache_ttl:
        # Expired
        del _query_cache[cache_key]
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


def _window_collection_name(collection: str | None) -> str:
    """Derive the sentence-window collection name from the chunk collection."""
    base = collection or settings.chroma_collection
    return base + "-windows"


def retrieve(query: str, collection: str | None = None) -> tuple[str, list[dict]]:
    """Full retrieval pipeline: source filter → multi-query → embed → hybrid search → rerank → prompt."""
    top_k = settings.retrieval_top_k
    retrieve_k = settings.retrieval_candidate_k  # expanded retrieval window (200)
    log_metrics = settings.log_retrieval_metrics

    # Check cache first
    cached = _check_query_cache(query, collection)
    if cached is not None:
        if log_metrics:
            log.info("retrieval_metrics phase=cache_hit ms=0")
        return cached

    timings = {}

    # Source filtering: detect book references in query
    source_filter = _extract_source_filter(query, collection)

    # Multi-query decomposition
    t0 = time.perf_counter()
    queries = _decompose_query(query)
    timings["multi_query"] = time.perf_counter() - t0

    all_dense = []
    all_sparse = []
    all_windows = []

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
            sparse_results = _bm25_search(sub_query, retrieve_k, collection)
        timings["bm25_search"] = timings.get("bm25_search", 0) + (time.perf_counter() - t0)

        all_dense.extend(dense_results)
        all_sparse.extend(sparse_results)
        all_windows.extend(window_results)

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

    # Reranking: ColBERT or passage scoring
    t0 = time.perf_counter()
    if settings.colbert_reranker_enabled:
        scored = _colbert_rerank(query, all_merged[:retrieve_k], top_k=top_k)
    else:
        scored = _passage_score(query, all_merged[:retrieve_k], top_k=top_k)
    timings["rerank"] = time.perf_counter() - t0

    system_prompt = build_rag_prompt(query, scored)

    result = (system_prompt, scored)

    # Store in cache
    _store_query_cache(query, collection, result)

    # Log metrics
    if log_metrics:
        for phase, elapsed in timings.items():
            log.info(f"retrieval_metrics phase={phase} ms={elapsed * 1000:.1f}")

    return result


def refresh_bm25_index(collection: str | None = None):
    """Force rebuild of the BM25 index (call after new documents ingested)."""
    col_key = collection or settings.chroma_collection
    if col_key in _bm25_cache:
        del _bm25_cache[col_key]
    _build_bm25_index(collection)
