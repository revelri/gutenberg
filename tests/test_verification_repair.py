"""Tests for citation page repair in services/api/core/verification.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "api"))
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))

from core.verification import repair_citations, repair_citations_with_diff


def _chunk(text: str, source: str, page: int) -> dict:
    return {
        "id": f"{source}:{page}",
        "text": text,
        "metadata": {"source": source, "page_start": page, "page_end": page},
    }


class TestRepairCitations:
    def test_rewrites_wrong_page(self):
        chunks = [
            _chunk("The body without organs is a limit.", "AO", 47),
            _chunk("Machines are everywhere in the flow.", "AO", 12),
        ]
        response = (
            'As D&G put it, "The body without organs is a limit." '
            "[Source: AO, p. 999]."
        )
        out = repair_citations(response, chunks)
        assert "[Source: AO, p. 47]" in out
        assert "p. 999" not in out

    def test_leaves_correct_page_intact(self):
        chunks = [_chunk("The body without organs is a limit.", "AO", 47)]
        response = (
            '"The body without organs is a limit." [Source: AO, p. 47].'
        )
        repaired, diffs = repair_citations_with_diff(response, chunks)
        assert repaired == response
        assert diffs == []

    def test_marks_unverified_when_no_match(self):
        chunks = [_chunk("Totally unrelated passage about rhizomes.", "ATP", 3)]
        response = '"Fabricated quote not in any chunk." [Source: ATP, p. 5].'
        repaired, diffs = repair_citations_with_diff(response, chunks)
        assert "[unverified]" in repaired
        assert len(diffs) == 1

    def test_marks_unverified_when_no_preceding_quote(self):
        chunks = [_chunk("Real passage here.", "AO", 10)]
        response = "Bare citation with no quote. [Source: AO, p. 99]."
        out = repair_citations(response, chunks)
        assert "[unverified]" in out

    def test_multiple_citations(self):
        chunks = [
            _chunk("The body without organs is a limit.", "AO", 47),
            _chunk("Rhizomes have no center and no hierarchy.", "ATP", 21),
        ]
        response = (
            '"The body without organs is a limit." [Source: AO, p. 1]. '
            'Later, "Rhizomes have no center and no hierarchy." [Source: ATP, p. 2].'
        )
        repaired, diffs = repair_citations_with_diff(response, chunks)
        assert "[Source: AO, p. 47]" in repaired
        assert "[Source: ATP, p. 21]" in repaired
        assert len(diffs) == 2

    def test_no_tags_is_noop(self):
        chunks = [_chunk("Some text.", "AO", 1)]
        response = "Plain prose with no citations."
        assert repair_citations(response, chunks) == response

    def test_empty_chunks_returns_original(self):
        response = '"quote" [Source: AO, p. 1].'
        assert repair_citations(response, []) == response
