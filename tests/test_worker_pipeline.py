import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "worker"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))

import pytest
import json
import sqlite3

from pipeline.pdf_validator import (
    validate_pdf,
    validate_epub,
    validate_document,
    ValidationResult,
)
from pipeline.extractors import extract_text
from pipeline.epub_extractor import extract_epub
from pipeline.chunker import chunk_text
from pipeline.progress import update_job_progress, update_document_status


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pdf_path(tmp_path):
    p = tmp_path / "test.pdf"
    p.write_bytes(b"not a pdf")
    return p


def _make_minimal_epub(path: Path) -> None:
    import ebooklib
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_title("Test Book")
    book.set_language("en")
    book.add_author("Test Author")

    chapter = epub.EpubHtml(title="Chapter 1", file_name="chap1.xhtml")
    chapter.content = b"<html><body><h1>Chapter One</h1><p>This is test content for extraction.</p></body></html>"
    book.add_item(chapter)

    chapter2 = epub.EpubHtml(title="Chapter 2", file_name="chap2.xhtml")
    chapter2.content = b"<html><body><p>More content in chapter two.</p></body></html>"
    book.add_item(chapter2)

    book.toc = [chapter, chapter2]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(path), book)


@pytest.fixture
def epub_path(tmp_path):
    p = tmp_path / "test.epub"
    _make_minimal_epub(p)
    return p


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    # Patch module-level DB_PATH since it's read at import time
    monkeypatch.setattr("pipeline.progress.DB_PATH", str(db_path))
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_job (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'pending',
            completed_files INTEGER DEFAULT 0,
            current_file TEXT,
            current_step TEXT,
            error TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS document (
            id TEXT PRIMARY KEY,
            corpus_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            chunks INTEGER DEFAULT 0,
            sha256 TEXT,
            file_type TEXT,
            error TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# PDF Validation tests
# ---------------------------------------------------------------------------


class TestPDFValidation:
    def test_validate_corrupt_pdf(self, pdf_path):
        result = validate_pdf(pdf_path)
        assert result.valid is False
        assert len(result.errors) > 0
        assert any("Corrupt" in e or "unreadable" in e for e in result.errors)

    def test_validate_result_type(self, pdf_path):
        result = validate_pdf(pdf_path)
        assert isinstance(result, ValidationResult)
        assert hasattr(result, "valid")
        assert hasattr(result, "warnings")
        assert hasattr(result, "errors")
        assert hasattr(result, "page_count")
        assert hasattr(result, "is_encrypted")

    def test_validate_nonexistent_file(self, tmp_path):
        result = validate_pdf(tmp_path / "does_not_exist.pdf")
        assert result.valid is False
        assert len(result.errors) > 0

    def test_validate_valid_pdf(self, tmp_path):
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hello, world!")
        pdf_bytes = doc.tobytes()
        doc.close()

        pdf_path = tmp_path / "valid.pdf"
        pdf_path.write_bytes(pdf_bytes)
        result = validate_pdf(pdf_path)
        assert result.valid is True
        assert result.page_count == 1
        assert result.is_encrypted is False

    @pytest.mark.skip(reason="PyMuPDF version does not support encryption in test env")
    def test_validate_encrypted_pdf(self, tmp_path):
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Secret content")
        try:
            doc.save(
                tmp_path / "encrypted.pdf",
                encryption=fitz.PDF_ENCRYPT_AES_256,
                owner_pw="test123",
            )
            doc.close()
        except Exception:
            pytest.skip("PyMuPDF encryption not supported in this version")

        result = validate_pdf(tmp_path / "encrypted.pdf")
        assert result.is_encrypted is True
        assert result.valid is False
        assert any("password" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# EPUB Validation tests
# ---------------------------------------------------------------------------


class TestEPUBValidation:
    def test_validate_valid_epub(self, epub_path):
        result = validate_epub(epub_path)
        assert result.valid is True
        assert result.page_count >= 1

    def test_validate_corrupt_epub(self, tmp_path):
        bad = tmp_path / "bad.epub"
        bad.write_bytes(b"not an epub")
        result = validate_epub(bad)
        assert result.valid is False
        assert len(result.errors) > 0

    def test_validate_document_epub(self, epub_path):
        result = validate_document(epub_path)
        assert result.valid is True


# ---------------------------------------------------------------------------
# EPUB Extraction tests
# ---------------------------------------------------------------------------


class TestEPUBExtraction:
    def test_epub_extraction(self, epub_path):
        full_text, metadata, chapters = extract_epub(epub_path)
        assert isinstance(full_text, str)
        assert len(full_text) > 0
        assert "test content" in full_text.lower() or "chapter" in full_text.lower()
        assert isinstance(metadata, dict)
        assert metadata.get("doc_type") == "epub"
        assert isinstance(chapters, list)
        assert len(chapters) >= 1

    def test_epub_extraction_metadata(self, epub_path):
        _, metadata, chapters = extract_epub(epub_path)
        assert "source" in metadata
        assert metadata["source"] == "test.epub"
        assert "total_pages" in metadata
        assert metadata["total_pages"] == len(chapters)

    def test_epub_extraction_chapters_have_page_keys(self, epub_path):
        _, _, chapters = extract_epub(epub_path)
        for ch in chapters:
            assert "page" in ch
            assert "text" in ch
            assert isinstance(ch["page"], int)
            assert len(ch["text"]) > 0


# ---------------------------------------------------------------------------
# Progress reporting tests
# ---------------------------------------------------------------------------


class TestProgressReporting:
    def test_update_job_progress(self, test_db, monkeypatch):
        conn = sqlite3.connect(str(test_db))
        conn.execute(
            "INSERT INTO ingestion_job (id, status) VALUES (?, ?)",
            ("job-123", "pending"),
        )
        conn.commit()
        conn.close()

        update_job_progress("job-123", status="processing", current_file="book.pdf")

        conn = sqlite3.connect(str(test_db))
        row = conn.execute(
            "SELECT status, current_file FROM ingestion_job WHERE id = ?", ("job-123",)
        ).fetchone()
        conn.close()
        assert row[0] == "processing"
        assert row[1] == "book.pdf"

    def test_update_job_progress_completed_files(self, test_db, monkeypatch):
        conn = sqlite3.connect(str(test_db))
        conn.execute(
            "INSERT INTO ingestion_job (id, status) VALUES (?, ?)",
            ("job-456", "pending"),
        )
        conn.commit()
        conn.close()

        update_job_progress("job-456", completed_files=5)

        conn = sqlite3.connect(str(test_db))
        row = conn.execute(
            "SELECT completed_files FROM ingestion_job WHERE id = ?", ("job-456",)
        ).fetchone()
        conn.close()
        assert row[0] == 5

    def test_update_job_progress_error(self, test_db, monkeypatch):
        conn = sqlite3.connect(str(test_db))
        conn.execute(
            "INSERT INTO ingestion_job (id, status) VALUES (?, ?)",
            ("job-789", "pending"),
        )
        conn.commit()
        conn.close()

        update_job_progress("job-789", status="failed", error="PDF is corrupt")

        conn = sqlite3.connect(str(test_db))
        row = conn.execute(
            "SELECT status, error FROM ingestion_job WHERE id = ?", ("job-789",)
        ).fetchone()
        conn.close()
        assert row[0] == "failed"
        assert "corrupt" in row[1].lower()

    def test_update_document_status_by_id(self, test_db, monkeypatch):
        conn = sqlite3.connect(str(test_db))
        conn.execute(
            "INSERT INTO document (id, corpus_id, filename, status) VALUES (?, ?, ?, ?)",
            ("doc-1", "corpus-1", "book.pdf", "pending"),
        )
        conn.commit()
        conn.close()

        update_document_status("doc-1", status="completed", chunks=42, file_type="pdf")

        conn = sqlite3.connect(str(test_db))
        row = conn.execute(
            "SELECT status, chunks, file_type FROM document WHERE id = ?", ("doc-1",)
        ).fetchone()
        conn.close()
        assert row[0] == "completed"
        assert row[1] == 42
        assert row[2] == "pdf"

    def test_update_document_status_by_filename_corpus(self, test_db, monkeypatch):
        conn = sqlite3.connect(str(test_db))
        conn.execute(
            "INSERT INTO document (id, corpus_id, filename, status) VALUES (?, ?, ?, ?)",
            ("doc-2", "corpus-1", "book2.epub", "pending"),
        )
        conn.commit()
        conn.close()

        update_document_status(
            filename="book2.epub", corpus_id="corpus-1", status="chunking"
        )

        conn = sqlite3.connect(str(test_db))
        row = conn.execute(
            "SELECT status FROM document WHERE filename = ? AND corpus_id = ?",
            ("book2.epub", "corpus-1"),
        ).fetchone()
        conn.close()
        assert row[0] == "chunking"

    def test_update_document_status_error(self, test_db, monkeypatch):
        conn = sqlite3.connect(str(test_db))
        conn.execute(
            "INSERT INTO document (id, corpus_id, filename, status) VALUES (?, ?, ?, ?)",
            ("doc-3", "corpus-1", "bad.pdf", "pending"),
        )
        conn.commit()
        conn.close()

        update_document_status(
            "doc-3", status="failed", error="Encryption not supported"
        )

        conn = sqlite3.connect(str(test_db))
        row = conn.execute(
            "SELECT status, error FROM document WHERE id = ?", ("doc-3",)
        ).fetchone()
        conn.close()
        assert row[0] == "failed"
        assert "encryption" in row[1].lower()

    def test_update_document_status_noop_without_fields(self, test_db, monkeypatch):
        conn = sqlite3.connect(str(test_db))
        conn.execute(
            "INSERT INTO document (id, corpus_id, filename, status) VALUES (?, ?, ?, ?)",
            ("doc-4", "corpus-1", "noop.pdf", "pending"),
        )
        conn.commit()
        conn.close()

        update_document_status("doc-4")

        conn = sqlite3.connect(str(test_db))
        row = conn.execute(
            "SELECT status FROM document WHERE id = ?", ("doc-4",)
        ).fetchone()
        conn.close()
        assert row[0] == "pending"


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------


class TestChunker:
    def test_chunk_text_basic(self):
        text = (
            "Philosophy is the study of general and fundamental problems concerning matters "
            "such as existence, knowledge, values, reason, mind, and language. It is a rational "
            "and critical inquiry that reflects on its own methods and assumptions."
        )
        metadata = {"source": "test.pdf", "doc_type": "pdf_digital"}
        chunks = chunk_text(text, metadata)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "text" in chunk
            assert len(chunk["text"]) > 0

    def test_chunk_text_metadata(self):
        text = (
            "# Introduction\n\nThis is the introduction to our philosophical inquiry. "
            "We shall examine the nature of reality and existence.\n\n"
            "# Chapter One\n\nReality is that which, when you stop believing in it, "
            "doesn't go away. Philip K. Dick proposed this pragmatic definition."
        )
        metadata = {"source": "book.pdf", "doc_type": "pdf_digital"}
        chunks = chunk_text(text, metadata)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "metadata" in chunk
            m = chunk["metadata"]
            assert "source" in m
            assert m["source"] == "book.pdf"
            assert "chunk_index" in m
            assert isinstance(m["chunk_index"], int)
            assert "page_start" in m
            assert "page_end" in m

    @pytest.mark.skipif(
        not __import__("spacy").util.is_package("en_core_web_sm"),
        reason="en_core_web_sm model not installed",
    )
    def test_chunk_text_with_page_segments(self):
        text = (
            "Page one content here. " * 100 + "\n\n" + "Page two content here. " * 100
        )
        metadata = {"source": "paged.pdf", "doc_type": "pdf_digital"}
        page_segments = [
            {"page": 1, "text": "Page one content here. " * 100},
            {"page": 2, "text": "Page two content here. " * 100},
        ]
        chunks = chunk_text(text, metadata, page_segments=page_segments)
        assert len(chunks) >= 1
        has_page_info = any(c["metadata"]["page_start"] >= 1 for c in chunks)
        assert has_page_info

    def test_chunk_text_heading_preservation(self):
        text = "# Chapter One\n\nSome introductory text.\n\n# Chapter Two\n\nMore text follows."
        metadata = {"source": "headings.epub", "doc_type": "epub"}
        chunks = chunk_text(text, metadata)
        has_heading = any(c["metadata"].get("heading") for c in chunks)
        assert has_heading

    def test_chunk_text_empty_input(self):
        metadata = {"source": "empty.pdf", "doc_type": "pdf_digital"}
        chunks = chunk_text("", metadata)
        assert chunks == []

    def test_chunk_text_custom_max_tokens(self):
        short_text = "A brief sentence."
        metadata = {"source": "short.txt", "doc_type": "docx"}
        chunks = chunk_text(short_text, metadata, max_tokens=50)
        assert len(chunks) >= 1
