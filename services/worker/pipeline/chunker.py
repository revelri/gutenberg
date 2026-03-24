"""Structure-aware recursive text chunking."""

import logging
import os
import re

import tiktoken

log = logging.getLogger("gutenberg.chunker")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))

# Tokenizer for counting tokens
_enc = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


def _split_by_headers(text: str) -> list[dict]:
    """Split text on markdown headers, preserving heading context."""
    # Match lines starting with # (markdown headers)
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    sections = []
    last_end = 0
    current_heading = ""

    for match in header_pattern.finditer(text):
        # Save text before this header as a section
        if last_end < match.start():
            chunk_text = text[last_end:match.start()].strip()
            if chunk_text:
                sections.append({"heading": current_heading, "text": chunk_text})

        current_heading = match.group(2).strip()
        last_end = match.end()

    # Remaining text after last header
    remaining = text[last_end:].strip()
    if remaining:
        sections.append({"heading": current_heading, "text": remaining})

    # If no headers found, return whole text as one section
    if not sections:
        sections.append({"heading": "", "text": text})

    return sections


def _recursive_split(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Recursively split text into chunks of max_tokens with overlap."""
    if _token_count(text) <= max_tokens:
        return [text] if text.strip() else []

    # Try splitting by progressively finer separators
    separators = ["\n\n", "\n", ". ", " "]

    for sep in separators:
        parts = text.split(sep)
        if len(parts) <= 1:
            continue

        # Merge parts into segments that fit within max_tokens
        segments = []
        current = ""

        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if _token_count(candidate) > max_tokens and current:
                segments.append(current.strip())
                current = part
            else:
                current = candidate

        if current.strip():
            segments.append(current.strip())

        if len(segments) <= 1:
            continue

        # Recursively split any segments still over max_tokens,
        # then apply overlap between final chunks
        final_chunks = []
        for segment in segments:
            if _token_count(segment) > max_tokens:
                final_chunks.extend(_recursive_split(segment, max_tokens, overlap_tokens=0))
            else:
                final_chunks.append(segment)

        # Apply overlap between consecutive chunks
        if overlap_tokens > 0 and len(final_chunks) > 1:
            overlapped = [final_chunks[0]]
            for i in range(1, len(final_chunks)):
                prev_tokens = _enc.encode(final_chunks[i - 1])
                overlap_text = _enc.decode(prev_tokens[-overlap_tokens:]) if len(prev_tokens) > overlap_tokens else ""
                if overlap_text.strip():
                    overlapped.append(f"{overlap_text.strip()} {final_chunks[i]}")
                else:
                    overlapped.append(final_chunks[i])
            return overlapped

        return final_chunks

    # Last resort: hard split by tokens
    tokens = _enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = _enc.decode(tokens[start:end])
        chunks.append(chunk.strip())
        start = end - overlap_tokens if overlap_tokens > 0 else end
        if start >= end:
            break
    return [c for c in chunks if c]


def _build_char_to_page(text: str, page_segments: list[dict]) -> list[tuple[int, int]]:
    """Build a mapping from character offset ranges to page numbers.

    Returns sorted list of (char_offset, page_number) breakpoints.
    """
    if not page_segments:
        return []

    breakpoints = []
    offset = 0
    for seg in page_segments:
        page_text = seg["text"]
        start = text.find(page_text, offset)
        if start == -1:
            # Fallback: approximate position using accumulated offset
            start = offset
        breakpoints.append((start, seg["page"]))
        offset = start + len(page_text)
    return breakpoints


def _pages_for_substring(text: str, substring: str, breakpoints: list[tuple[int, int]]) -> tuple[int, int]:
    """Find page_start and page_end for a substring within the full text."""
    if not breakpoints:
        return (0, 0)

    start_idx = text.find(substring)
    end_idx = start_idx + len(substring) if start_idx >= 0 else 0

    page_start = breakpoints[0][1]
    page_end = breakpoints[0][1]

    for bp_offset, bp_page in breakpoints:
        if bp_offset <= start_idx:
            page_start = bp_page
        if bp_offset <= end_idx:
            page_end = bp_page

    return (page_start, page_end)


def chunk_text(text: str, metadata: dict, page_segments: list[dict] | None = None) -> list[dict]:
    """Split text into chunks with metadata.

    Returns list of dicts with keys: text, metadata (source, heading, chunk_index,
    page_start, page_end).
    """
    breakpoints = _build_char_to_page(text, page_segments or [])
    sections = _split_by_headers(text)
    all_chunks = []
    chunk_index = 0

    for section in sections:
        pieces = _recursive_split(section["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for piece in pieces:
            page_start, page_end = _pages_for_substring(text, piece, breakpoints)
            all_chunks.append({
                "text": piece,
                "metadata": {
                    "source": metadata.get("source", ""),
                    "heading": section["heading"],
                    "chunk_index": chunk_index,
                    "doc_type": metadata.get("doc_type", ""),
                    "page_start": page_start,
                    "page_end": page_end,
                },
            })
            chunk_index += 1

    log.info(f"Chunked into {len(all_chunks)} pieces (avg {_token_count(text) // max(len(all_chunks), 1)} tokens each)")
    return all_chunks
