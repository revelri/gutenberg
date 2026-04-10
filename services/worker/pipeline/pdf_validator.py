"""Pre-flight PDF validation — catch corrupt, flipped, encrypted, and empty PDFs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

log = logging.getLogger("gutenberg.pdf_validator")


@dataclass
class ValidationResult:
    valid: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    page_count: int = 0
    is_encrypted: bool = False
    is_landscape_majority: bool = False
    empty_page_ratio: float = 0.0


def validate_pdf(path: Path) -> ValidationResult:
    """Run all pre-flight checks on a PDF. Returns ValidationResult."""
    result = ValidationResult()

    # 1. Corrupt PDF detection
    try:
        doc = fitz.open(str(path))
    except Exception as e:
        result.valid = False
        result.errors.append(f"Corrupt or unreadable PDF: {e}")
        return result

    try:
        result.page_count = len(doc)

        # 2. Zero pages
        if result.page_count == 0:
            result.valid = False
            result.errors.append("PDF has no pages")
            return result

        # 3. Encrypted / password-protected
        if doc.is_encrypted:
            result.is_encrypted = True
            if not doc.authenticate(""):
                result.valid = False
                result.errors.append("PDF is password-protected and cannot be opened")
                return result
            result.warnings.append("PDF was encrypted but opened with empty password")

        # 4. Excessive page count (likely scan artifact or merged files)
        if result.page_count > 2000:
            result.warnings.append(
                f"Unusually high page count ({result.page_count}). "
                "May be a scan artifact or merged file."
            )

        # 5. Landscape/flipped page detection
        # If >50% of pages are wider than tall, likely a spread scan or flipped
        landscape_count = 0
        empty_count = 0
        min_chars = 10

        for page in doc:
            rect = page.rect
            if rect.width > rect.height * 1.2:  # clearly landscape
                landscape_count += 1

            text = page.get_text("text").strip()
            if len(text) < min_chars:
                empty_count += 1

        landscape_ratio = landscape_count / result.page_count
        result.empty_page_ratio = empty_count / result.page_count
        result.is_landscape_majority = landscape_ratio > 0.5

        if result.is_landscape_majority:
            result.warnings.append(
                f"{landscape_count}/{result.page_count} pages are landscape. "
                "May contain two-page spreads that need splitting."
            )

        # 6. Mostly empty pages
        if result.empty_page_ratio > 0.8:
            result.warnings.append(
                f"{empty_count}/{result.page_count} pages have <{min_chars} chars. "
                "PDF may be image-only (needs OCR) or mostly blank."
            )

        # 7. Very small file size relative to page count (possible placeholder/corrupt)
        file_size = path.stat().st_size
        bytes_per_page = file_size / result.page_count if result.page_count > 0 else 0
        if bytes_per_page < 500 and result.page_count > 10:
            result.warnings.append(
                f"Very low bytes/page ({bytes_per_page:.0f}). "
                "PDF may be corrupt or a placeholder."
            )

    finally:
        doc.close()

    if result.errors:
        result.valid = False

    return result


def validate_epub(path: Path) -> ValidationResult:
    """Basic EPUB validation."""
    result = ValidationResult()
    try:
        import ebooklib
        from ebooklib import epub
        book = epub.read_epub(str(path), options={"ignore_ncx": True})
        items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        result.page_count = len(items)  # chapters as "pages"
        if result.page_count == 0:
            result.valid = False
            result.errors.append("EPUB has no content documents")
    except Exception as e:
        result.valid = False
        result.errors.append(f"Invalid or unreadable EPUB: {e}")
    return result


def validate_document(path: Path) -> ValidationResult:
    """Validate any supported document type."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return validate_pdf(path)
    elif ext == ".epub":
        return validate_epub(path)
    elif ext == ".docx":
        # DOCX validation is minimal — python-docx handles most errors gracefully
        result = ValidationResult()
        try:
            from docx import Document
            doc = Document(str(path))
            if len(doc.paragraphs) == 0:
                result.warnings.append("DOCX has no paragraphs")
        except Exception as e:
            result.valid = False
            result.errors.append(f"Invalid DOCX: {e}")
        return result
    else:
        result = ValidationResult()
        result.valid = False
        result.errors.append(f"Unsupported file type: {ext}")
        return result
