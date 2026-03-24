"""Classify documents as digital PDF, scanned PDF, or DOCX."""

import logging
from pathlib import Path

import fitz  # PyMuPDF

log = logging.getLogger("gutenberg.detector")

DIGITAL_THRESHOLD = 0.8  # % of pages that must have text
MIN_CHARS_PER_PAGE = 50


def classify_document(path: Path) -> str:
    """Return 'pdf_digital', 'pdf_scanned', or 'docx'."""
    ext = path.suffix.lower()

    if ext == ".docx":
        return "docx"

    if ext != ".pdf":
        raise ValueError(f"Unsupported file type: {ext}")

    doc = fitz.open(str(path))
    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        raise ValueError("PDF has no pages")

    pages_with_text = 0
    for page in doc:
        text = page.get_text("text")
        if len(text.strip()) >= MIN_CHARS_PER_PAGE:
            pages_with_text += 1
    doc.close()

    ratio = pages_with_text / total_pages
    log.info(f"{path.name}: {pages_with_text}/{total_pages} pages have text (ratio={ratio:.2f})")

    if ratio >= DIGITAL_THRESHOLD:
        return "pdf_digital"
    return "pdf_scanned"
