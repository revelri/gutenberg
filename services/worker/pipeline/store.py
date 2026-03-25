"""ChromaDB storage and dedup tracking."""

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path

import chromadb

log = logging.getLogger("gutenberg.store")

CHROMA_HOST = os.environ.get("CHROMA_HOST", "http://chromadb:8000")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "gutenberg")


def _get_collection(collection_name: str | None = None):
    """Get or create a ChromaDB collection."""
    host = CHROMA_HOST.replace("http://", "").replace("https://", "")
    parts = host.split(":")
    hostname = parts[0]
    port = int(parts[1]) if len(parts) > 1 else 8000

    client = chromadb.HttpClient(host=hostname, port=port)
    return client.get_or_create_collection(
        name=collection_name or COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def is_duplicate(path: Path, state_file: Path) -> bool:
    """Check if file SHA-256 already exists in state log."""
    sha = _file_sha256(path)
    if not state_file.exists():
        return False
    with open(state_file) as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("sha256") == sha:
                    return True
            except json.JSONDecodeError:
                continue
    return False


def store_chunks(chunks: list[dict], embeddings: list[list[float]], collection_name: str | None = None):
    """Store chunks and embeddings in ChromaDB."""
    collection = _get_collection(collection_name)

    ids = [str(uuid.uuid4()) for _ in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # ChromaDB has a batch limit, insert in batches of 100
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = i + batch_size
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )


def record_document(path: Path, chunk_count: int, state_file: Path):
    """Append document record to state file."""
    import datetime

    record = {
        "filename": path.name,
        "sha256": _file_sha256(path),
        "chunks": chunk_count,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(state_file, "a") as f:
        f.write(json.dumps(record) + "\n")
