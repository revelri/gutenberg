"""EPUB text extraction using ebooklib + BeautifulSoup.

Chapters map to page_segments with chapter index as pseudo-page number.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

from shared.text_normalize import clean_for_ingestion

log = logging.getLogger("gutenberg.epub_extractor")


def extract_epub(path: Path) -> tuple[str, dict, list[dict]]:
    """Extract text from EPUB, returning (full_text, metadata, page_segments).

    page_segments uses chapter index as pseudo-page (EPUBs have no physical pages).
    Headings are preserved as markdown for the chunker.
    """
    book = epub.read_epub(str(path), options={"ignore_ncx": True})

    metadata = {
        "source": path.name,
        "doc_type": "epub",
    }

    # Extract book metadata
    titles = book.get_metadata("DC", "title")
    if titles:
        metadata["epub_title"] = titles[0][0]
    creators = book.get_metadata("DC", "creator")
    if creators:
        metadata["epub_author"] = creators[0][0]

    # Extract chapters in spine order
    chapters = []
    chapter_idx = 0

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content()
        soup = BeautifulSoup(content, "html.parser")

        # Extract text preserving heading structure
        parts = []
        for elem in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre"]):
            text = elem.get_text(separator=" ").strip()
            if not text:
                continue

            tag = elem.name
            if tag.startswith("h"):
                level = int(tag[1])
                parts.append(f"{'#' * level} {text}")
            elif tag == "blockquote":
                parts.append(f"> {text}")
            elif tag == "li":
                parts.append(f"- {text}")
            else:
                parts.append(text)

        chapter_text = "\n\n".join(parts)
        if not chapter_text.strip():
            continue

        # Skip front/back matter with heuristic (very short + common keywords)
        if len(chapter_text) < 200:
            lower = chapter_text.lower()
            if any(kw in lower for kw in ["copyright", "isbn", "table of contents",
                                           "all rights reserved", "cover image"]):
                continue

        chapter_idx += 1
        cleaned = clean_for_ingestion(chapter_text)
        chapters.append({
            "page": chapter_idx,  # chapter number as pseudo-page
            "text": cleaned,
        })

    metadata["total_pages"] = len(chapters)  # chapters, not physical pages
    metadata["format_note"] = "EPUB: page numbers are chapter indices"

    full_text = "\n\n".join(ch["text"] for ch in chapters)
    log.info(f"Extracted {len(chapters)} chapters, {len(full_text)} chars from {path.name}")

    return full_text, metadata, chapters
