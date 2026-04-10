"""Tests for PDF serving infrastructure (B.1, B.2, B.3)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "api"))

import asyncio

import pytest
import fitz

from core.database import set_db_path, init_db


def _make_test_pdf(path: Path, filename: str = "test.pdf") -> Path:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Chapter One: The Beginning")
    page.insert_text((72, 100), "This is the first page of our test document.")
    page.insert_text((72, 128), "It contains some sample text for search testing.")

    page2 = doc.new_page(width=612, height=792)
    page2.insert_text((72, 72), "Chapter Two: The Middle")
    page2.insert_text((72, 100), "On the second page, more content appears.")
    page2.insert_text((72, 128), "Philosophy is the love of wisdom.")

    page3 = doc.new_page(width=612, height=792)
    page3.insert_text((72, 72), "Chapter Three: The End")
    page3.insert_text((72, 100), "The final page concludes our document.")
    page3.insert_text((72, 128), "Every ending is a new beginning.")

    out = path / filename
    doc.save(str(out))
    doc.close()
    return out


@pytest.fixture
def client(tmp_path):
    set_db_path(str(tmp_path / "test.db"))
    asyncio.run(init_db())

    processed = tmp_path / "data" / "processed"
    inbox = tmp_path / "data" / "inbox"
    processed.mkdir(parents=True, exist_ok=True)
    inbox.mkdir(parents=True, exist_ok=True)

    _make_test_pdf(processed, "test.pdf")
    _make_test_pdf(inbox, "inbox-doc.pdf")

    from routers import pdf as pdf_mod
    from core import pdf_search as pdf_search_mod

    pdf_mod.DATA_DIR = str(tmp_path / "data")
    pdf_search_mod.DATA_DIR = str(tmp_path / "data")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(pdf_mod.router)
    return TestClient(app)


class TestServePDF:
    def test_pdf_endpoint_returns_pdf(self, client):
        r = client.get("/api/pdf/test.pdf")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/pdf"
        assert len(r.content) > 4
        assert r.content[:4] == b"%PDF"

    def test_pdf_endpoint_404(self, client):
        r = client.get("/api/pdf/nonexistent.pdf")
        assert r.status_code == 404

    def test_pdf_endpoint_rejects_path_traversal(self, client):
        r = client.get("/api/pdf/..%2F..%2Fetc%2Fpasswd")
        assert r.status_code in (400, 404)

    def test_pdf_endpoint_rejects_slash(self, client):
        r = client.get("/api/pdf/sub/dir/file.pdf")
        assert r.status_code in (400, 404)

    def test_pdf_endpoint_rejects_null_byte(self, client):
        r = client.get("/api/pdf/test%00.pdf")
        assert r.status_code in (400, 404)

    def test_pdf_endpoint_serves_from_inbox(self, client):
        r = client.get("/api/pdf/inbox-doc.pdf")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/pdf"

    def test_pdf_endpoint_has_accept_ranges(self, client):
        r = client.get("/api/pdf/test.pdf")
        assert r.status_code == 200
        assert r.headers.get("accept-ranges") == "bytes"


class TestPageImage:
    def test_page_image_returns_jpeg(self, client):
        r = client.get("/api/pdf/test.pdf/page/1/image")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        assert r.content[:2] == b"\xff\xd8"

    def test_page_image_404_on_bad_file(self, client):
        r = client.get("/api/pdf/nonexistent.pdf/page/1/image")
        assert r.status_code == 404

    def test_page_image_second_page(self, client):
        r = client.get("/api/pdf/test.pdf/page/2/image")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"

    def test_page_image_out_of_range(self, client):
        r = client.get("/api/pdf/test.pdf/page/999/image")
        assert r.status_code == 404

    def test_page_image_with_highlight(self, client):
        r = client.get("/api/pdf/test.pdf/page/1/image?highlight=Chapter+One")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"

    def test_page_image_custom_dpi(self, client):
        r = client.get("/api/pdf/test.pdf/page/1/image?dpi=150")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"


class TestFindQuotePage:
    def test_find_page_for_quote(self, client):
        from core.pdf_search import find_quote_page

        page = find_quote_page("test.pdf", "Philosophy is the love of wisdom")
        assert page == 2

    def test_find_page_returns_none_for_missing(self, client):
        from core.pdf_search import find_quote_page

        page = find_quote_page("test.pdf", "zzz_nonexistent_text_zzz")
        assert page is None

    def test_find_page_first_page(self, client):
        from core.pdf_search import find_quote_page

        page = find_quote_page("test.pdf", "The Beginning")
        assert page == 1

    def test_find_page_inbox_file(self, client):
        from core.pdf_search import find_quote_page

        page = find_quote_page("inbox-doc.pdf", "The Beginning")
        assert page == 1

    def test_find_page_nonexistent_file(self, client):
        from core.pdf_search import find_quote_page

        page = find_quote_page("nope.pdf", "anything")
        assert page is None
