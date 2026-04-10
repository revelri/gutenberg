"""Tests for the quote verification module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "api"))
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))

from core.verification import extract_quotes, verify_quotes, format_verification_footer


SAMPLE_CHUNKS = [
    {
        "text": "The virtual is not opposed to the real but to the actual. The virtual is fully real in so far as it is virtual.",
        "metadata": {"source": "Difference and Repetition.pdf", "page_start": 208, "page_end": 208},
    },
    {
        "text": "A rhizome ceaselessly establishes connections between semiotic chains, organizations of power, and circumstances relative to the arts, sciences, and social struggles.",
        "metadata": {"source": "A Thousand Plateaus.pdf", "page_start": 7, "page_end": 7},
    },
]


class TestExtractQuotes:
    def test_double_quotes(self):
        """T6: Extract text in double quotation marks."""
        text = 'Deleuze writes: "The virtual is not opposed to the real but to the actual" in Chapter 4.'
        quotes = extract_quotes(text)
        assert len(quotes) == 1
        assert "virtual is not opposed" in quotes[0]

    def test_markdown_blockquotes(self):
        """T7: Extract text from markdown blockquotes."""
        text = "As Deleuze notes:\n\n> The virtual is not opposed to the real\n> but to the actual.\n\nThis is significant."
        quotes = extract_quotes(text)
        assert len(quotes) == 1
        assert "virtual is not opposed" in quotes[0]

    def test_no_quotes(self):
        """T8: No quoted text returns empty list."""
        text = "Deleuze discusses the concept of virtuality in Chapter 4."
        quotes = extract_quotes(text)
        assert quotes == []

    def test_smart_quotes(self):
        """Extract text in curly/smart quotation marks."""
        text = '\u201cThe virtual is not opposed to the real but to the actual.\u201d'
        quotes = extract_quotes(text)
        assert len(quotes) == 1

    def test_deduplication(self):
        """Duplicate quotes are deduplicated."""
        text = '"The virtual is not opposed to the real." And again: "The virtual is not opposed to the real."'
        quotes = extract_quotes(text)
        assert len(quotes) == 1


class TestVerifyQuotes:
    def test_exact_match(self):
        """T9: Exact substring match returns 'verified'."""
        quotes = ["The virtual is not opposed to the real but to the actual"]
        results = verify_quotes(quotes, SAMPLE_CHUNKS)
        assert len(results) == 1
        assert results[0]["status"] == "verified"
        assert results[0]["source"] == "Difference and Repetition.pdf"
        assert results[0]["page"] == 208

    def test_fuzzy_match(self):
        """T10: Near-match returns 'approximate'."""
        # Slightly modified quote (paraphrased)
        quotes = ["The virtual is not opposed to the real, but rather to the actual"]
        results = verify_quotes(quotes, SAMPLE_CHUNKS)
        assert len(results) == 1
        assert results[0]["status"] in ("verified", "approximate")

    def test_no_match(self):
        """T11: Fabricated quote returns 'unverified'."""
        quotes = ["Deleuze argues that capitalism inherently deterritorializes all social bonds"]
        results = verify_quotes(quotes, SAMPLE_CHUNKS)
        assert len(results) == 1
        assert results[0]["status"] == "unverified"

    def test_empty_quotes(self):
        """Empty quotes list returns empty results."""
        results = verify_quotes([], SAMPLE_CHUNKS)
        assert results == []

    def test_empty_chunks(self):
        """Empty chunks list returns empty results."""
        quotes = ["Some quote"]
        results = verify_quotes(quotes, [])
        assert results == []


class TestFormatVerificationFooter:
    def test_mixed_results(self):
        """T12: Footer formats mixed verification results."""
        results = [
            {"quote": "exact match quote", "status": "verified", "source": "test.pdf", "page": 42, "similarity": 1.0},
            {"quote": "fuzzy match quote", "status": "approximate", "source": "test.pdf", "page": 43, "similarity": 0.88},
            {"quote": "no match quote here", "status": "unverified", "source": None, "page": None, "similarity": 0.3},
        ]
        footer = format_verification_footer(results)
        assert "2/3 verified" in footer or "1/3 verified" in footer
        assert "approximate" in footer
        assert "unverified" in footer

    def test_all_verified(self):
        """All verified quotes produce clean footer."""
        results = [
            {"quote": "quote one", "status": "verified", "source": "a.pdf", "page": 1, "similarity": 1.0},
            {"quote": "quote two", "status": "verified", "source": "b.pdf", "page": 5, "similarity": 1.0},
        ]
        footer = format_verification_footer(results)
        assert "2/2 verified" in footer
        # No detail lines for verified quotes
        assert "unverified" not in footer

    def test_empty_results(self):
        """Empty results produce empty footer."""
        assert format_verification_footer([]) == ""
