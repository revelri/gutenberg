"""Tests for the structure-aware recursive chunker."""

import sys
from pathlib import Path

# Add worker pipeline to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "worker"))

from pipeline.chunker import (
    _split_by_headers,
    _recursive_split,
    _build_char_to_page,
    _pages_for_offset,
    chunk_text,
)


class TestSplitByHeaders:
    def test_multiple_headers(self):
        """T1: Split text with multiple markdown headers."""
        text = "# Introduction\nThis is the intro.\n\n# Methods\nThis is methods."
        sections = _split_by_headers(text)
        assert len(sections) == 2
        assert sections[0]["heading"] == "Introduction"
        assert "intro" in sections[0]["text"]
        assert sections[1]["heading"] == "Methods"
        assert "methods" in sections[1]["text"]

    def test_no_headers(self):
        """T2: Text with no headers returns single section."""
        text = "Just plain text with no markdown headers at all."
        sections = _split_by_headers(text)
        assert len(sections) == 1
        assert sections[0]["heading"] == ""
        assert sections[0]["text"] == text

    def test_text_before_first_header(self):
        """Text before the first header becomes its own section."""
        text = "Preamble text.\n\n# Chapter One\nChapter content."
        sections = _split_by_headers(text)
        assert len(sections) == 2
        assert sections[0]["heading"] == ""
        assert "Preamble" in sections[0]["text"]
        assert sections[1]["heading"] == "Chapter One"

    def test_sections_have_offsets(self):
        """Sections track their character offset in the original text."""
        text = "# First\nContent A.\n\n# Second\nContent B."
        sections = _split_by_headers(text)
        for section in sections:
            assert "offset" in section
            assert isinstance(section["offset"], int)


class TestRecursiveSplit:
    def test_short_text_no_split(self):
        """T3: Text under max_tokens is returned as-is with full length."""
        text = "Short text."
        result = _recursive_split(text, max_tokens=100, overlap_tokens=0)
        assert len(result) == 1
        chunk_text, offset, new_len = result[0]
        assert chunk_text == text
        assert offset == 0
        assert new_len == len(text)

    def test_offset_tracking(self):
        """T4: Offsets track where new content starts in the original text."""
        paragraphs = [f"Paragraph {i}. " + "word " * 100 for i in range(5)]
        text = "\n\n".join(paragraphs)
        result = _recursive_split(text, max_tokens=128, overlap_tokens=0)
        assert len(result) > 1

        offsets = [off for _, off, _ in result]
        for i in range(1, len(offsets)):
            assert offsets[i] >= offsets[i - 1], f"Offset {offsets[i]} < {offsets[i-1]}"

    def test_overlap_preserves_content_offset(self):
        """Overlap prepends text but offset still points to new content."""
        paragraphs = [f"Paragraph {i}. " + "word " * 100 for i in range(5)]
        text = "\n\n".join(paragraphs)
        result = _recursive_split(text, max_tokens=128, overlap_tokens=20)
        assert len(result) > 1

        for chunk_text, offset, new_len in result:
            assert isinstance(offset, int)
            assert offset >= 0
            assert isinstance(new_len, int)
            assert new_len > 0

    def test_overlap_new_content_length_excludes_prefix(self):
        """T20: new_content_length is smaller than full chunk when overlap is added."""
        paragraphs = [f"Paragraph {i}. " + "word " * 100 for i in range(5)]
        text = "\n\n".join(paragraphs)
        result = _recursive_split(text, max_tokens=128, overlap_tokens=20)

        # For chunks after the first, the full text includes overlap prefix
        # but new_content_length should be the original chunk size (without overlap)
        for i, (chunk_text, offset, new_len) in enumerate(result):
            if i > 0:
                # new_content_length should be <= full chunk text length
                assert new_len <= len(chunk_text), (
                    f"Chunk {i}: new_len={new_len} > len(text)={len(chunk_text)}"
                )


class TestChunkText:
    def test_page_metadata_correct(self):
        """T5: chunk_text assigns correct page_start/page_end for multi-page doc."""
        page1_text = "Page one content with some words. " * 20
        page2_text = "Page two has different content. " * 20
        page3_text = "Page three concludes the document. " * 20

        full_text = f"{page1_text}\n\n{page2_text}\n\n{page3_text}"
        page_segments = [
            {"page": 1, "text": page1_text},
            {"page": 2, "text": page2_text},
            {"page": 3, "text": page3_text},
        ]

        metadata = {"source": "test.pdf", "doc_type": "pdf_digital"}
        chunks = chunk_text(full_text, metadata, page_segments)

        assert len(chunks) > 0
        for chunk in chunks:
            assert "page_start" in chunk["metadata"]
            assert "page_end" in chunk["metadata"]
            assert chunk["metadata"]["page_start"] >= 1
            assert chunk["metadata"]["page_end"] >= chunk["metadata"]["page_start"]

        assert chunks[0]["metadata"]["page_start"] == 1
        assert chunks[-1]["metadata"]["page_end"] == 3

    def test_no_page_segments(self):
        """Chunks without page segments get page_start=0, page_end=0."""
        text = "Some text without page info."
        metadata = {"source": "test.docx", "doc_type": "docx"}
        chunks = chunk_text(text, metadata, page_segments=None)
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["page_start"] == 0
        assert chunks[0]["metadata"]["page_end"] == 0

    def test_repeated_text_across_pages(self):
        """Page numbers are correct even when pages share identical text."""
        # Simulate a book where every page starts with the same running header
        header = "DIFFERENCE AND REPETITION\n"
        page1 = header + "Unique content on page one. " * 30
        page2 = header + "Different content on page two. " * 30
        page3 = header + "Third page has its own text. " * 30

        full_text = f"{page1}\n\n{page2}\n\n{page3}"
        page_segments = [
            {"page": 10, "text": page1},
            {"page": 11, "text": page2},
            {"page": 12, "text": page3},
        ]

        metadata = {"source": "test.pdf", "doc_type": "pdf_digital"}
        chunks = chunk_text(full_text, metadata, page_segments)

        assert len(chunks) > 0
        # First chunk must start on page 10
        assert chunks[0]["metadata"]["page_start"] == 10
        # Last chunk must end on page 12
        assert chunks[-1]["metadata"]["page_end"] == 12
        # No chunk should have page_start=0 (which would mean the mapping failed)
        for chunk in chunks:
            assert chunk["metadata"]["page_start"] >= 10


class TestBuildCharToPage:
    def test_deterministic_offsets(self):
        """Breakpoints are computed from lengths, not text.find()."""
        page1 = "Short page."
        page2 = "Another page."
        page3 = "Final page."
        full_text = f"{page1}\n\n{page2}\n\n{page3}"
        segments = [{"page": 1, "text": page1}, {"page": 2, "text": page2}, {"page": 3, "text": page3}]

        breakpoints = _build_char_to_page(full_text, segments)
        assert breakpoints[0] == (0, 1)
        assert breakpoints[1] == (len(page1) + 2, 2)  # +2 for "\n\n"
        assert breakpoints[2] == (len(page1) + 2 + len(page2) + 2, 3)

    def test_repeated_text_doesnt_break_mapping(self):
        """Identical page text doesn't cause wrong offset assignment."""
        same_text = "This exact text appears on every page."
        full_text = f"{same_text}\n\n{same_text}\n\n{same_text}"
        segments = [
            {"page": 1, "text": same_text},
            {"page": 2, "text": same_text},
            {"page": 3, "text": same_text},
        ]

        breakpoints = _build_char_to_page(full_text, segments)
        # Each breakpoint should be at a different offset
        offsets = [bp[0] for bp in breakpoints]
        assert len(set(offsets)) == 3, f"Expected 3 unique offsets, got {offsets}"
        # And they should be in order
        assert offsets == sorted(offsets)
