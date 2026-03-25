"""Document management endpoints."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter

from core.chroma import get_collection
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
    collection = get_collection()

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
