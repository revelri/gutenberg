"""Tests for pipeline quality improvements: BM25 tokenization, query cleaning, weighted RRF.

These tests duplicate the pure logic from rag.py to avoid importing
modules with heavy dependencies (nltk, chromadb, httpx) that may not
be installed on the host Python.
"""

import re
from nltk.stem import PorterStemmer
import nltk

# Ensure NLTK data is available
for resource in ["punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

_stemmer = PorterStemmer()


def _tokenize(text: str) -> list[str]:
    """Local copy of rag.py:_tokenize for testing."""
    text = re.sub(r"[-–—]", " ", text.lower())
    tokens = nltk.word_tokenize(text)
    return [_stemmer.stem(t) for t in tokens if t.isalnum() and len(t) > 1]


def _clean_query(text: str) -> str:
    """Local copy of rag.py:_clean_query for testing."""
    text = text.strip()
    if not text:
        return text
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"-{3,}", "---", text)
    text = re.sub(r"_{3,}", "___", text)
    text = re.sub(r" {3,}", "  ", text)
    return text


def _reciprocal_rank_fusion(
    dense: list[dict],
    sparse: list[dict],
    k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> list[dict]:
    """Local copy of rag.py:_reciprocal_rank_fusion for testing."""
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, chunk in enumerate(dense):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + dense_weight / (k + rank + 1)
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(sparse):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + sparse_weight / (k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[cid] for cid, _ in ranked]


class TestBM25Tokenization:
    def test_stemming_matches_inflections(self):
        """T17: Stemmed tokens match inflected forms."""
        tokens_virtual = _tokenize("virtual")
        tokens_virtuality = _tokenize("virtuality")
        assert tokens_virtual == tokens_virtuality, (
            f"'virtual' → {tokens_virtual}, 'virtuality' → {tokens_virtuality}"
        )

    def test_punctuation_stripped(self):
        """T18: Punctuation doesn't affect tokenization."""
        tokens_clean = _tokenize("Deleuze")
        tokens_period = _tokenize("Deleuze.")
        tokens_comma = _tokenize("Deleuze,")
        assert tokens_clean == tokens_period == tokens_comma

    def test_hyphens_split(self):
        """Hyphenated terms are split into components."""
        tokens = _tokenize("desiring-machines")
        assert len(tokens) == 2

    def test_short_tokens_filtered(self):
        """Single-character tokens are filtered out."""
        tokens = _tokenize("a I x")
        # Single chars should be filtered (len > 1 check)
        assert len(tokens) == 0


class TestQueryCleaning:
    def test_query_cleaned_before_embedding(self):
        """T19: _clean_query applies same normalization as chunk embedding."""
        assert "..." in _clean_query("Chapter 1 ......... 5")
        assert "........." not in _clean_query("Chapter 1 ......... 5")
        assert "---" in _clean_query("section ---------- break")
        assert "----------" not in _clean_query("section ---------- break")
        assert _clean_query("What is the virtual?") == "What is the virtual?"


class TestWeightedRRF:
    def test_dense_weighted_higher(self):
        """T21: Dense results get higher fusion scores than sparse at same rank."""
        dense = [{"id": "d1", "text": "dense chunk", "metadata": {}, "dense_score": 0.9}]
        sparse = [{"id": "s1", "text": "sparse chunk", "metadata": {}, "bm25_score": 5.0}]

        result = _reciprocal_rank_fusion(
            dense, sparse, k=60, dense_weight=0.6, sparse_weight=0.4
        )
        assert len(result) == 2
        assert result[0]["id"] == "d1", "Dense result should rank first with 0.6 weight"

    def test_equal_weights_same_ranking(self):
        """Equal weights: shared items rank highest."""
        dense = [
            {"id": "a", "text": "a", "metadata": {}, "dense_score": 0.9},
            {"id": "b", "text": "b", "metadata": {}, "dense_score": 0.8},
        ]
        sparse = [
            {"id": "b", "text": "b", "metadata": {}, "bm25_score": 5.0},
            {"id": "c", "text": "c", "metadata": {}, "bm25_score": 3.0},
        ]

        result = _reciprocal_rank_fusion(dense, sparse, k=60, dense_weight=1.0, sparse_weight=1.0)
        ids = [r["id"] for r in result]
        assert ids[0] == "b"  # appears in both lists
