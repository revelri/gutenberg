import asyncio
import importlib
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

API_PATH = str(Path(__file__).resolve().parent.parent / "services" / "api")
SERVICES_PATH = str(Path(__file__).resolve().parent.parent / "services")
sys.path.insert(0, API_PATH)
sys.path.insert(0, SERVICES_PATH)

FAKE_COLLECTION = MagicMock()
FAKE_COLLECTION.count.return_value = 0


def _mock_chroma_client(*a, **kw):
    client = MagicMock()
    client.get_or_create_collection.return_value = FAKE_COLLECTION
    client.get_collection.return_value = FAKE_COLLECTION
    client.delete_collection.return_value = None
    return client


@pytest.fixture
def client(tmp_path):
    import os

    os.environ["DATABASE_PATH"] = str(tmp_path / "test.db")

    from core.database import set_db_path, init_db
    from core import config

    config.settings = config.Settings()
    set_db_path(str(tmp_path / "test.db"))
    asyncio.run(init_db())

    async def _noop(*a, **kw):
        pass

    spec = importlib.util.spec_from_file_location("api_main", f"{API_PATH}/main.py")
    api_main = importlib.util.module_from_spec(spec)
    sys.modules["shared"] = MagicMock()
    sys.modules["shared.chroma"] = MagicMock()
    sys.modules["shared.nlp"] = MagicMock()
    sys.modules["shared.embeddings"] = MagicMock()
    sys.modules["pipeline"] = MagicMock()
    sys.modules["pipeline.progress"] = MagicMock()
    spec.loader.exec_module(api_main)
    api_main.auto_populate_exemplar = _noop
    sys.modules["routers.corpus"]._get_chroma_client = _mock_chroma_client
    sys.modules["routers.corpus"].chromadb = MagicMock()

    with TestClient(api_main.app) as c:
        yield c


def _create_corpus(client, name="Test Corpus", tags="test"):
    resp = client.post("/api/corpus", data={"name": name, "tags": tags})
    return resp


def _create_conversation(client, corpus_id, mode="general"):
    resp = client.post(f"/api/corpus/{corpus_id}/conversations", json={"mode": mode})
    return resp


class TestCreateCorpus:
    def test_create_corpus(self, client):
        resp = _create_corpus(client)
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["collection_name"].startswith("gutenberg-")

    def test_create_corpus_with_tags(self, client):
        resp = _create_corpus(client, name="Philosophy", tags="deleuze,guattari")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Philosophy"


class TestListCorpora:
    def test_list_corpora(self, client):
        create_resp = _create_corpus(client, name="Listable Corpus")
        corpus_id = create_resp.json()["id"]

        list_resp = client.get("/api/corpus")
        assert list_resp.status_code == 200
        corpora = list_resp.json()
        ids = [c["id"] for c in corpora]
        assert corpus_id in ids


class TestGetCorpusDetail:
    def test_get_corpus_detail(self, client):
        create_resp = _create_corpus(client)
        corpus_id = create_resp.json()["id"]

        detail_resp = client.get(f"/api/corpus/{corpus_id}")
        assert detail_resp.status_code == 200
        data = detail_resp.json()
        assert "documents" in data
        assert isinstance(data["documents"], list)


class TestDeleteCorpus:
    def test_delete_corpus(self, client):
        create_resp = _create_corpus(client)
        corpus_id = create_resp.json()["id"]

        conv_resp = _create_conversation(client, corpus_id)
        conv_id = conv_resp.json()["id"]

        del_resp = client.delete(f"/api/corpus/{corpus_id}")
        assert del_resp.status_code == 200

        conv_check = client.get(f"/api/conversations/{conv_id}")
        assert conv_check.status_code == 404


class TestCreateConversation:
    def test_create_conversation(self, client):
        create_resp = _create_corpus(client)
        corpus_id = create_resp.json()["id"]

        resp = _create_conversation(client, corpus_id, mode="exhaustive")
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["mode"] == "exhaustive"


class TestListConversations:
    def test_list_conversations(self, client):
        create_resp = _create_corpus(client)
        corpus_id = create_resp.json()["id"]

        conv_resp = _create_conversation(client, corpus_id)
        conv_id = conv_resp.json()["id"]

        list_resp = client.get(f"/api/corpus/{corpus_id}/conversations")
        assert list_resp.status_code == 200
        convs = list_resp.json()
        ids = [c["id"] for c in convs]
        assert conv_id in ids


class TestGetConversationMessages:
    def test_get_conversation_messages(self, client):
        create_resp = _create_corpus(client)
        corpus_id = create_resp.json()["id"]

        conv_resp = _create_conversation(client, corpus_id)
        conv_id = conv_resp.json()["id"]

        msg_resp = client.get(f"/api/conversations/{conv_id}")
        assert msg_resp.status_code == 200
        data = msg_resp.json()
        assert "messages" in data
        assert isinstance(data["messages"], list)


class TestDeleteConversation:
    def test_delete_conversation(self, client):
        create_resp = _create_corpus(client)
        corpus_id = create_resp.json()["id"]

        conv_resp = _create_conversation(client, corpus_id)
        conv_id = conv_resp.json()["id"]

        del_resp = client.delete(f"/api/conversations/{conv_id}")
        assert del_resp.status_code == 200

        check_resp = client.get(f"/api/conversations/{conv_id}")
        assert check_resp.status_code == 404


class TestUploadRejectsBadTypes:
    @pytest.mark.skip(reason="Requires writable /data/inbox (integration test)")
    def test_upload_rejects_bad_types(self, client):
        create_resp = _create_corpus(client)
        corpus_id = create_resp.json()["id"]

        resp = client.post(
            f"/api/corpus/{corpus_id}/ingest",
            files=[("files", ("bad.txt", b"hello world", "text/plain"))],
        )
        assert resp.status_code == 400


class TestUploadAcceptsPdf:
    @pytest.mark.skip(reason="Requires writable /data/inbox (integration test)")
    def test_upload_accepts_pdf(self, client):
        create_resp = _create_corpus(client)
        corpus_id = create_resp.json()["id"]

        minimal_pdf = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF"
        resp = client.post(
            f"/api/corpus/{corpus_id}/ingest",
            files=[("files", ("test.pdf", minimal_pdf, "application/pdf"))],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data


class TestHealthEndpoint:
    def test_health_endpoint(self, client, monkeypatch):
        import httpx as httpx_mod

        mock_get = MagicMock()
        mock_get.status_code = 200
        monkeypatch.setattr(httpx_mod, "get", MagicMock(return_value=mock_get))

        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
