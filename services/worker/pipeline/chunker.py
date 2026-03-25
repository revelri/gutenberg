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
    """Split text on markdown headers, preserving heading context.

    Returns list of {"heading": str, "text": str, "offset": int} where
    offset is the character position of the section text within the original.
    """
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    sections = []
    last_end = 0
    current_heading = ""

    for match in header_pattern.finditer(text):
        if last_end < match.start():
            chunk_text = text[last_end:match.start()].strip()
            if chunk_text:
                # Find the actual start of stripped text
                strip_offset = last_end + (len(text[last_end:match.start()]) - len(text[last_end:match.start()].lstrip()))
                sections.append({"heading": current_heading, "text": chunk_text, "offset": strip_offset})

        current_heading = match.group(2).strip()
        last_end = match.end()

    remaining = text[last_end:].strip()
    if remaining:
        strip_offset = last_end + (len(text[last_end:]) - len(text[last_end:].lstrip()))
        sections.append({"heading": current_heading, "text": remaining, "offset": strip_offset})

    if not sections:
        sections.append({"heading": "", "text": text, "offset": 0})

    return sections


def _recursive_split(text: str, max_tokens: int, overlap_tokens: int) -> list[tuple[str, int, int]]:
    """Recursively split text into chunks of max_tokens with overlap.

    Returns list of (chunk_text, content_offset, new_content_length) tuples.
    content_offset is the character offset of the chunk's NEW content
    (excluding any overlap prefix) within the input text.
    new_content_length is the length of just the new content (without overlap prefix).
    """
    if _token_count(text) <= max_tokens:
        return [(text, 0, len(text))] if text.strip() else []

    # Try splitting by progressively finer separators
    separators = ["\n\n", "\n", ". ", " "]

    for sep in separators:
        parts = text.split(sep)
        if len(parts) <= 1:
            continue

        # Merge parts into segments that fit within max_tokens,
        # tracking the character offset of each segment.
        segments: list[tuple[str, int]] = []  # (segment_text, offset)
        current = ""
        current_offset = 0

        char_pos = 0
        for i, part in enumerate(parts):
            candidate = f"{current}{sep}{part}" if current else part
            if _token_count(candidate) > max_tokens and current:
                segments.append((current.strip(), current_offset))
                current = part
                current_offset = char_pos
            else:
                if not current:
                    current_offset = char_pos
                current = candidate
            char_pos += len(part) + (len(sep) if i < len(parts) - 1 else 0)

        if current.strip():
            segments.append((current.strip(), current_offset))

        if len(segments) <= 1:
            continue

        # Recursively split any segments still over max_tokens
        final_chunks: list[tuple[str, int, int]] = []
        for seg_text, seg_offset in segments:
            if _token_count(seg_text) > max_tokens:
                sub_chunks = _recursive_split(seg_text, max_tokens, overlap_tokens=0)
                for sub_text, sub_off, sub_len in sub_chunks:
                    final_chunks.append((sub_text, seg_offset + sub_off, sub_len))
            else:
                final_chunks.append((seg_text, seg_offset, len(seg_text)))

        # Apply overlap between consecutive chunks.
        # The overlap prefix is prepended to the chunk text, but the
        # content_offset and new_content_length track only the NEW content.
        if overlap_tokens > 0 and len(final_chunks) > 1:
            overlapped: list[tuple[str, int, int]] = [final_chunks[0]]
            for i in range(1, len(final_chunks)):
                prev_text = final_chunks[i - 1][0]
                chunk_text, chunk_offset, chunk_new_len = final_chunks[i]
                prev_tokens = _enc.encode(prev_text)
                overlap_text = _enc.decode(prev_tokens[-overlap_tokens:]) if len(prev_tokens) > overlap_tokens else ""
                if overlap_text.strip():
                    overlapped.append((f"{overlap_text.strip()} {chunk_text}", chunk_offset, chunk_new_len))
                else:
                    overlapped.append((chunk_text, chunk_offset, chunk_new_len))
            return overlapped

        return final_chunks

    # Last resort: hard split by tokens
    tokens = _enc.encode(text)
    chunks: list[tuple[str, int, int]] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = _enc.decode(tokens[start:end])
        char_offset = len(_enc.decode(tokens[:start]))
        chunk_len = len(chunk.strip())
        chunks.append((chunk.strip(), char_offset, chunk_len))
        start = end - overlap_tokens if overlap_tokens > 0 else end
        if start >= end:
            break
    return [(t, off, cl) for t, off, cl in chunks if t]


def _build_char_to_page(text: str, page_segments: list[dict]) -> list[tuple[int, int]]:
    """Build a mapping from character offset ranges to page numbers.

    Returns sorted list of (char_offset, page_number) breakpoints.

    Uses deterministic offset accumulation based on page text lengths
    and the known "\n\n" separator. Never searches for text in the
    combined string (which fails on repeated content like headers).
    """
    if not page_segments:
        return []

    breakpoints = []
    offset = 0
    for i, seg in enumerate(page_segments):
        breakpoints.append((offset, seg["page"]))
        offset += len(seg["text"])
        if i < len(page_segments) - 1:
            offset += 2  # "\n\n" separator used by extractors to join pages
    return breakpoints


def _pages_for_offset(start_offset: int, length: int, breakpoints: list[tuple[int, int]]) -> tuple[int, int]:
    """Find page_start and page_end for a known character offset range."""
    if not breakpoints:
        return (0, 0)

    end_offset = start_offset + length

    page_start = breakpoints[0][1]
    page_end = breakpoints[0][1]

    for bp_offset, bp_page in breakpoints:
        if bp_offset <= start_offset:
            page_start = bp_page
        if bp_offset <= end_offset:
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
        section_offset = section["offset"]
        pieces = _recursive_split(section["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for piece_text, piece_offset, new_content_len in pieces:
            # piece_offset is relative to the section text;
            # add section_offset to get position in the full document.
            # Use new_content_len (not len(piece_text)) to exclude overlap prefix
            # from the page span calculation.
            abs_offset = section_offset + piece_offset
            page_start, page_end = _pages_for_offset(abs_offset, new_content_len, breakpoints)
            all_chunks.append({
                "text": piece_text,
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
