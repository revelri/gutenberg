"""Text extraction for PDF (digital/scanned) and DOCX.

Returns page-segmented text so downstream chunking can track page boundaries.
"""

import logging
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document as DocxDocument

log = logging.getLogger("gutenberg.extractors")


def extract_text(path: Path, doc_type: str) -> tuple[str, dict, list[dict]]:
    """Extract text and metadata from a document.

    Returns (full_text, metadata_dict, page_segments).
    page_segments is a list of {"page": int, "text": str} dicts.
    For formats without page numbers, page_segments is empty.
    """
    metadata = {
        "source": path.name,
        "doc_type": doc_type,
    }

    if doc_type == "pdf_digital":
        return _extract_pdf_digital(path, metadata)
    elif doc_type == "pdf_scanned":
        return _extract_pdf_scanned(path, metadata)
    elif doc_type == "docx":
        return _extract_docx(path, metadata)
    else:
        raise ValueError(f"Unknown doc_type: {doc_type}")


def _extract_pdf_digital(path: Path, metadata: dict) -> tuple[str, dict, list[dict]]:
    """Fast extraction using PyMuPDF for digital PDFs."""
    doc = fitz.open(str(path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    doc.close()

    metadata["total_pages"] = len(pages)
    full_text = "\n\n".join(p["text"] for p in pages)
    return full_text, metadata, pages


def _extract_pdf_scanned(path: Path, metadata: dict) -> tuple[str, dict, list[dict]]:
    """OCR extraction using Docling for scanned PDFs."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(path))
    md_text = result.document.export_to_markdown()

    total = result.document.num_pages if hasattr(result.document, "num_pages") else 0
    metadata["total_pages"] = total
    metadata["ocr"] = True
    # Docling doesn't expose per-page text easily; return single segment
    page_segments = [{"page": 1, "text": md_text}] if md_text.strip() else []
    return md_text, metadata, page_segments


def _extract_docx(path: Path, metadata: dict) -> tuple[str, dict, list[dict]]:
    """Extract text from DOCX using python-docx."""
    doc = DocxDocument(str(path))
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            # Preserve heading structure for chunker
            if para.style and para.style.name.startswith("Heading"):
                level = para.style.name.replace("Heading ", "").strip()
                try:
                    level = int(level)
                except ValueError:
                    level = 1
                paragraphs.append(f"{'#' * level} {text}")
            else:
                paragraphs.append(text)

    # Also extract tables
    for table in doc.tables:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(" | ".join(cells))
        if rows:
            paragraphs.append("\n".join(rows))

    metadata["total_pages"] = 0  # DOCX doesn't have page numbers
    full_text = "\n\n".join(paragraphs)
    return full_text, metadata, []  # no page segments for DOCX
