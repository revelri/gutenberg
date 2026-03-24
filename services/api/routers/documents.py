"""Document management endpoints."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter

from core.config import settings
from core.rag import refresh_bm25_index

log = logging.getLogger("gutenberg.documents")
router = APIRouter(prefix="/api/documents")

STATE_FILE = Path("/data/state/documents.jsonl")


@router.get("")
async def list_documents():
    """List all ingested documents."""
    documents = []
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            for line in f:
                try:
                    documents.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return {"documents": documents, "total": len(documents)}


@router.delete("/{filename}")
async def delete_document(filename: str):
    """Delete a document's chunks from ChromaDB."""
    import chromadb

    host = settings.chroma_host.replace("http://", "").replace("https://", "")
    parts = host.split(":")
    hostname = parts[0]
    port = int(parts[1]) if len(parts) > 1 else 8000
    client = chromadb.HttpClient(host=hostname, port=port)

    collection = client.get_or_create_collection(name=settings.chroma_collection)

    # Find and delete chunks by source filename
    results = collection.get(
        where={"source": filename},
        include=["metadatas"],
    )

    if results["ids"]:
        collection.delete(ids=results["ids"])
        refresh_bm25_index()
        return {"deleted": len(results["ids"]), "filename": filename}

    return {"deleted": 0, "filename": filename, "message": "not found"}


@router.post("/refresh-index")
async def refresh_index():
    """Force rebuild of the BM25 index."""
    refresh_bm25_index()
    return {"status": "ok"}
