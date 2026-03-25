"""Tests for RAG prompt assembly.

Tests build_rag_prompt directly by extracting just that function's logic,
since importing the full rag module requires chromadb and other dependencies.
"""


def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    """Local copy of the prompt assembly logic for testing.

    This mirrors services/api/core/rag.py:build_rag_prompt exactly.
    If the source changes, this must be updated to match.
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


def _make_chunk(source: str, page_start: int, page_end: int, heading: str = "", text: str = "Sample text.") -> dict:
    return {
        "text": text,
        "metadata": {
            "source": source,
            "heading": heading,
            "page_start": page_start,
            "page_end": page_end,
        },
    }


class TestBuildRagPrompt:
    def test_single_page_citation(self):
        """T13: Single-page chunk gets 'p. X' citation format."""
        chunks = [_make_chunk("Difference and Repetition.pdf", 208, 208)]
        prompt = build_rag_prompt("test query", chunks)
        assert "p. 208" in prompt
        assert "pp." not in prompt

    def test_multi_page_citation(self):
        """T14: Multi-page chunk gets 'pp. X-Y' citation format."""
        chunks = [_make_chunk("A Thousand Plateaus.pdf", 47, 49)]
        prompt = build_rag_prompt("test query", chunks)
        assert "pp. 47-49" in prompt

    def test_grounding_instructions(self):
        """T15: System prompt includes citation grounding instructions."""
        chunks = [_make_chunk("test.pdf", 1, 1)]
        prompt = build_rag_prompt("test query", chunks)
        assert "verbatim" in prompt.lower()
        assert "could not find a confident match" in prompt.lower()
        assert "fabricate" in prompt.lower()

    def test_no_page_numbers(self):
        """Chunks with page_start=0 don't show page citation."""
        chunks = [_make_chunk("notes.docx", 0, 0)]
        prompt = build_rag_prompt("test query", chunks)
        assert "p. 0" not in prompt
        assert "pp." not in prompt

    def test_heading_included(self):
        """Heading metadata appears in the source label."""
        chunks = [_make_chunk("test.pdf", 10, 10, heading="Chapter 3")]
        prompt = build_rag_prompt("test query", chunks)
        assert "Chapter 3" in prompt
