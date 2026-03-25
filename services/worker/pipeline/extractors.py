"""Text extraction for PDF (digital/scanned) and DOCX.

Returns page-segmented text so downstream chunking can track page boundaries.
"""

import logging
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document as DocxDocument

log = logging.getLogger("gutenberg.extractors")


def extract_text(path: Path, doc_type: str, strategy: str = "default") -> tuple[str, dict, list[dict]]:
    """Extract text and metadata from a document.

    Returns (full_text, metadata_dict, page_segments).
    page_segments is a list of {"page": int, "text": str} dicts.
    For formats without page numbers, page_segments is empty.

    strategy: "default" or "optimized" (for scanned PDFs).
    """
    metadata = {
        "source": path.name,
        "doc_type": doc_type,
    }

    if doc_type == "pdf_digital":
        return _extract_pdf_digital(path, metadata)
    elif doc_type == "pdf_scanned":
        return _extract_pdf_scanned(path, metadata, strategy=strategy)
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


def _extract_pdf_scanned(path: Path, metadata: dict, strategy: str = "default") -> tuple[str, dict, list[dict]]:
    """OCR extraction using Docling for scanned PDFs with per-page text.

    Uses GPU acceleration when available (CUDA) for layout detection and OCR.
    strategy: "default" (full Docling) or "optimized" (skip tables/pictures/formulas, batch processing).
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

    import torch
    device = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
    log.info(f"Docling using device: {device}, strategy: {strategy}")

    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=device),
    )

    if strategy == "optimized":
        # Skip features unnecessary for philosophical books (no tables, figures, formulas)
        pipeline_options.do_table_structure = False
        pipeline_options.do_picture_classification = False
        pipeline_options.do_formula_enrichment = False
        pipeline_options.do_code_enrichment = False
        # Conservative batch sizes for 8GB GPU (layout model needs ~2GB + batch overhead)
        pipeline_options.ocr_batch_size = 8
        pipeline_options.layout_batch_size = 4
        log.info("Docling optimized: tables/pictures/formulas OFF, layout_batch=4, ocr_batch=8")

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    result = converter.convert(str(path))
    doc = result.document

    # Try per-page extraction via Docling's page-aware markdown export
    pages: list[dict] = []
    try:
        num_pages = len(doc.pages) if hasattr(doc, "pages") and doc.pages else 0
        if num_pages > 0:
            for page_no in sorted(doc.pages.keys()):
                page_text = doc.export_to_markdown(page_no=page_no)
                if page_text.strip():
                    pages.append({"page": page_no, "text": page_text})
            log.info(f"Docling per-page extraction: {len(pages)} pages with text out of {num_pages}")
    except (TypeError, AttributeError) as e:
        log.warning(f"Docling per-page extraction failed, falling back to full export: {e}")
        pages = []

    # Fallback: single-page segment from full markdown export
    if not pages:
        md_text = doc.export_to_markdown()
        total = len(doc.pages) if hasattr(doc, "pages") and doc.pages else 0
        metadata["total_pages"] = total
        metadata["ocr"] = True
        page_segments = [{"page": 1, "text": md_text}] if md_text.strip() else []
        return md_text, metadata, page_segments

    metadata["total_pages"] = len(pages)
    metadata["ocr"] = True

    # OCR quality heuristic: check for garbled text
    _check_ocr_quality(pages, metadata)

    full_text = "\n\n".join(p["text"] for p in pages)
    return full_text, metadata, pages


def _check_ocr_quality(pages: list[dict], metadata: dict):
    """Warn if OCR quality appears low (high ratio of non-dictionary words)."""
    import re

    sample_text = " ".join(p["text"] for p in pages[:5])  # sample first 5 pages
    words = re.findall(r"[a-zA-Z]{3,}", sample_text)
    if not words:
        return

    # Simple heuristic: words with >50% non-alpha chars or very short garbled tokens
    garbled = sum(1 for w in words if len(w) <= 2 or not w.isascii())
    ratio = garbled / len(words) if words else 0

    if ratio > 0.20:
        log.warning(
            f"Low OCR quality detected for {metadata.get('source', 'unknown')}: "
            f"{ratio:.0%} potentially garbled words. Citations may be less reliable."
        )


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
