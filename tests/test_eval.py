"""Tests for evaluation metric computation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from eval_retrieval import precision_at_k, mrr, ndcg_at_k


class TestMetrics:
    def test_precision_at_k(self):
        """T16a: Precision@K computes correctly."""
        # 3 out of 5 are relevant
        hits = [True, False, True, True, False]
        assert precision_at_k(hits, 5) == 3 / 5
        assert precision_at_k(hits, 3) == 2 / 3
        assert precision_at_k(hits, 1) == 1.0

    def test_precision_empty(self):
        assert precision_at_k([], 5) == 0.0

    def test_mrr(self):
        """T16b: MRR returns 1/rank of first relevant result."""
        assert mrr([False, False, True, True]) == 1 / 3
        assert mrr([True, False, False]) == 1.0
        assert mrr([False, False, False]) == 0.0

    def test_ndcg_at_k(self):
        """T16c: NDCG@K handles relevance scores correctly."""
        # Perfect ranking
        scores = [1.0, 1.0, 0.0, 0.0]
        assert ndcg_at_k(scores, 4) == 1.0

        # All zeros
        assert ndcg_at_k([0.0, 0.0], 2) == 0.0

        # Single item
        assert ndcg_at_k([1.0], 1) == 1.0

    def test_ndcg_empty(self):
        assert ndcg_at_k([], 5) == 0.0
