"""Tests for the RAG retrieval module (core/rag.py).

Covers: query classification, passage scoring, source filtering,
quoted phrase extraction, reciprocal rank fusion, query cache,
embedding cache, and retry logic. All external dependencies mocked.
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

# Add service paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "api"))
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))


# ---------------------------------------------------------------------------
# _classify_query
# ---------------------------------------------------------------------------

from core.rag import _classify_query


class TestClassifyQuery:
    def test_quoted_phrase_lexical(self):
        """Quoted phrase >= 5 chars triggers lexical."""
        assert _classify_query('find "desiring machines" in text') == "lexical"

    def test_short_quote_semantic(self):
        """Quoted phrase < 5 chars falls through to semantic."""
        assert _classify_query('find "ab" in text') == "semantic"

    def test_curly_quotes_lexical(self):
        """Curly (Unicode) quotes also trigger lexical."""
        assert _classify_query("find \u201cdesiring machines\u201d here") == "lexical"

    def test_page_reference_lexical(self):
        """Page reference like p. 123 triggers lexical."""
        assert _classify_query("what is on p. 42") == "lexical"

    def test_page_reference_no_dot(self):
        """Page reference without dot (p 7) triggers lexical."""
        assert _classify_query("see p 7 of the book") == "lexical"

    def test_proper_noun_density_lexical(self):
        """Two consecutive capitalized words trigger lexical."""
        assert _classify_query("Gilles Deleuze wrote about desire") == "lexical"

    def test_plain_semantic(self):
        """No lexical signals → semantic."""
        assert _classify_query("what is the concept of desire") == "semantic"


# ---------------------------------------------------------------------------
# _extract_quoted_phrase
# ---------------------------------------------------------------------------

from core.rag import _extract_quoted_phrase


class TestExtractQuotedPhrase:
    def test_straight_quotes(self):
        """Extracts phrase from straight double quotes."""
        assert _extract_quoted_phrase('find "the body without organs" here') == "the body without organs"

    def test_curly_quotes(self):
        """Extracts phrase from curly quotes."""
        result = _extract_quoted_phrase("find \u201cthe body without organs\u201d here")
        assert result == "the body without organs"

    def test_short_phrase_returns_none(self):
        """Phrases < 10 chars return None."""
        assert _extract_quoted_phrase('find "short" here') is None

    def test_no_quotes_returns_none(self):
        assert _extract_quoted_phrase("no quotes at all") is None


# ---------------------------------------------------------------------------
# _passage_score
# ---------------------------------------------------------------------------

from core.rag import _passage_score


class TestPassageScore:
    def _chunk(self, text, cid="c1"):
        return {"id": cid, "text": text, "metadata": {"source": "test"}}

    def test_empty_chunks(self):
        assert _passage_score("hello world", [], 5) == []

    def test_exact_phrase_ranked_first(self):
        """Chunk containing the quoted phrase scores highest."""
        chunks = [
            self._chunk("The body without organs is fundamental to Deleuze.", "c1"),
            self._chunk("Desire is productive, not representational.", "c2"),
        ]
        result = _passage_score('"the body without organs"', chunks, 2)
        assert result[0]["id"] == "c1"

    def test_word_overlap_scoring(self):
        """Chunks with more word overlap score higher."""
        chunks = [
            self._chunk("Capitalism and schizophrenia connect in surprising ways.", "c1"),
            self._chunk("The weather is nice today.", "c2"),
        ]
        result = _passage_score("capitalism and schizophrenia", chunks, 2)
        assert result[0]["id"] == "c1"

    def test_top_k_limits_output(self):
        chunks = [self._chunk(f"chunk number {i}", f"c{i}") for i in range(10)]
        result = _passage_score("chunk number", chunks, 3)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _filter_by_source
# ---------------------------------------------------------------------------

from core.rag import _filter_by_source


class TestFilterBySource:
    def _chunk(self, source, cid="c1"):
        return {"id": cid, "text": "text", "metadata": {"source": source}}

    def test_filters_matching_prefix(self):
        chunks = [
            self._chunk("Anti-Oedipus", "c1"),
            self._chunk("A Thousand Plateaus", "c2"),
            self._chunk("Anti-Oedipus ch2", "c3"),
        ]
        result = _filter_by_source(chunks, "Anti-Oedipus")
        assert len(result) == 2
        assert all("Anti-Oedipus" in c["metadata"]["source"] for c in result)

    def test_no_match_returns_all(self):
        """If no chunks match the filter, return everything."""
        chunks = [self._chunk("BookA"), self._chunk("BookB")]
        result = _filter_by_source(chunks, "NonexistentBook")
        assert len(result) == 2

    def test_empty_chunks(self):
        assert _filter_by_source([], "anything") == []


# ---------------------------------------------------------------------------
# _extract_source_filter
# ---------------------------------------------------------------------------

from core.rag import _extract_source_filter


class TestExtractSourceFilter:
    @patch("core.rag._build_source_patterns")
    def test_match_found(self, mock_patterns):
        """Returns source name when pattern matches."""
        import re
        mock_patterns.return_value = [
            (re.compile(r"(?i)\bin\s+Anti[-\s]?Oedipus\b"), "Anti-Oedipus"),
        ]
        result = _extract_source_filter("in Anti-Oedipus")
        assert result == "Anti-Oedipus"

    @patch("core.rag._build_source_patterns")
    def test_no_match(self, mock_patterns):
        import re
        mock_patterns.return_value = [
            (re.compile(r"(?i)\bin\s+Anti[-\s]?Oedipus\b"), "Anti-Oedipus"),
        ]
        assert _extract_source_filter("what is desire") is None

    @patch("core.rag.settings")
    def test_disabled(self, mock_settings):
        mock_settings.source_filter_enabled = False
        assert _extract_source_filter("in Anti-Oedipus") is None


# ---------------------------------------------------------------------------
# _reciprocal_rank_fusion
# ---------------------------------------------------------------------------

from core.rag import _reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def _chunk(self, cid):
        return {"id": cid, "text": f"text {cid}", "metadata": {}}

    def test_both_empty(self):
        assert _reciprocal_rank_fusion([], []) == []

    def test_dense_only(self):
        dense = [self._chunk("a"), self._chunk("b")]
        result = _reciprocal_rank_fusion(dense, [], dense_weight=1.0, sparse_weight=1.0)
        assert len(result) == 2
        assert result[0]["id"] == "a"

    def test_sparse_only(self):
        sparse = [self._chunk("x"), self._chunk("y")]
        result = _reciprocal_rank_fusion([], sparse, dense_weight=1.0, sparse_weight=1.0)
        assert len(result) == 2
        assert result[0]["id"] == "x"

    def test_overlap_merged(self):
        """Shared IDs get combined scores and rank higher."""
        dense = [self._chunk("a"), self._chunk("b")]
        sparse = [self._chunk("b"), self._chunk("c")]
        result = _reciprocal_rank_fusion(dense, sparse, dense_weight=1.0, sparse_weight=1.0)
        ids = [r["id"] for r in result]
        # "b" appears in both lists, should rank first
        assert ids[0] == "b"
        assert set(ids) == {"a", "b", "c"}

    def test_weight_asymmetry(self):
        """Higher sparse weight promotes sparse-first results."""
        dense = [self._chunk("d1")]
        sparse = [self._chunk("s1")]
        result = _reciprocal_rank_fusion(dense, sparse, dense_weight=0.1, sparse_weight=10.0)
        assert result[0]["id"] == "s1"


# ---------------------------------------------------------------------------
# _check_query_cache / _store_query_cache
# ---------------------------------------------------------------------------

from core.rag import _check_query_cache, _store_query_cache, _query_cache


class TestQueryCache:
    def setup_method(self):
        _query_cache.clear()

    def test_cache_hit(self):
        result = ("prompt", [{"text": "chunk"}])
        _store_query_cache("hello world", None, result)
        cached = _check_query_cache("hello world", None)
        assert cached == result

    def test_cache_miss(self):
        assert _check_query_cache("never stored", None) is None

    @patch("core.rag.settings")
    def test_ttl_expiry(self, mock_settings):
        mock_settings.query_cache_ttl = 0  # immediate expiry
        mock_settings.query_cache_max_size = 100
        mock_settings.embed_cache_max_size = 200
        result = ("prompt", [])
        # Manually insert an expired entry
        from core.rag import _clean_query
        normalized = _clean_query("test query")
        _query_cache[(normalized, None)] = (time.time() - 10, result)
        assert _check_query_cache("test query", None) is None

    def test_quoted_query_bypasses_cache(self):
        """Queries with quoted phrases (10+ chars) skip caching entirely."""
        result = ("prompt", [])
        _store_query_cache('"the body without organs"', None, result)
        assert _check_query_cache('"the body without organs"', None) is None

    @patch("core.rag.settings")
    def test_lru_eviction(self, mock_settings):
        mock_settings.query_cache_max_size = 2
        mock_settings.query_cache_ttl = 300
        mock_settings.embed_cache_max_size = 200
        _store_query_cache("q1", None, ("p1", []))
        _store_query_cache("q2", None, ("p2", []))
        _store_query_cache("q3", None, ("p3", []))
        # q1 should have been evicted
        assert _check_query_cache("q1", None) is None
        assert _check_query_cache("q3", None) is not None


# ---------------------------------------------------------------------------
# _embed_query
# ---------------------------------------------------------------------------

from core.rag import _embed_query, _embed_cache


class TestEmbedQuery:
    def setup_method(self):
        _embed_cache.clear()

    @patch("core.rag._retry_ollama")
    def test_cache_miss_calls_ollama(self, mock_retry):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_retry.return_value = mock_resp

        result = _embed_query("test embedding")
        assert result == [0.1, 0.2, 0.3]
        mock_retry.assert_called_once()

    @patch("core.rag._retry_ollama")
    def test_cache_hit_skips_ollama(self, mock_retry):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[1.0, 2.0]]}
        mock_retry.return_value = mock_resp

        _embed_query("same query")
        _embed_query("same query")
        # Only one actual call despite two invocations
        assert mock_retry.call_count == 1

    @patch("core.rag.settings")
    @patch("core.rag._retry_ollama")
    def test_lru_eviction(self, mock_retry, mock_settings):
        mock_settings.embed_cache_max_size = 2
        mock_settings.ollama_host = "http://localhost:11434"
        mock_settings.ollama_embed_model = "test"
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.0]]}
        mock_retry.return_value = mock_resp

        _embed_query("q1")
        _embed_query("q2")
        _embed_query("q3")
        # q1 should have been evicted from cache
        from core.rag import _clean_query
        assert _clean_query("q1") not in _embed_cache


# ---------------------------------------------------------------------------
# _retry_ollama
# ---------------------------------------------------------------------------

from core.rag import _retry_ollama


class TestRetryOllama:
    def test_success_first_try(self):
        fn = MagicMock(return_value="ok")
        assert _retry_ollama(fn, max_retries=3, base_delay=0) == "ok"
        fn.assert_called_once()

    @patch("core.rag.time.sleep")
    def test_success_on_retry(self, mock_sleep):
        fn = MagicMock(side_effect=[httpx.ConnectError("fail"), "ok"])
        assert _retry_ollama(fn, max_retries=3, base_delay=0.001) == "ok"
        assert fn.call_count == 2

    @patch("core.rag.time.sleep")
    def test_max_retries_exceeded(self, mock_sleep):
        fn = MagicMock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(httpx.ConnectError):
            _retry_ollama(fn, max_retries=2, base_delay=0.001)
        assert fn.call_count == 2

    @patch("core.rag.time.sleep")
    def test_read_timeout_retried(self, mock_sleep):
        fn = MagicMock(side_effect=[httpx.ReadTimeout("slow"), "ok"])
        assert _retry_ollama(fn, max_retries=3, base_delay=0.001) == "ok"
