"""ChromaDB storage and dedup tracking."""

import fcntl
import hashlib
import json
import logging
import os
import uuid
from pathlib import Path

from shared.chroma import get_collection as _shared_get_collection

log = logging.getLogger("gutenberg.store")

CHROMA_HOST = os.environ.get("CHROMA_HOST", "http://chromadb:8000")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "gutenberg")


def _get_collection(collection_name: str | None = None):
    """Get or create a ChromaDB collection using the shared client."""
    return _shared_get_collection(CHROMA_HOST, collection_name or COLLECTION_NAME)


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
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("sha256") == sha:
                        return True
                except json.JSONDecodeError:
                    continue
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return False


def write_pending_marker(filename: str, state_dir: Path) -> Path:
    """Write a pending marker before starting ingestion.

    The marker file contains chunk IDs so partial ingestion can be cleaned up.
    Returns the marker file path.
    """
    marker = state_dir / f".pending-{filename}"
    marker.write_text(json.dumps({"filename": filename, "chunk_ids": [], "status": "pending"}))
    return marker


def update_pending_marker(marker: Path, chunk_ids: list[str]):
    """Update pending marker with stored chunk IDs."""
    data = json.loads(marker.read_text())
    data["chunk_ids"].extend(chunk_ids)
    marker.write_text(json.dumps(data))


def remove_pending_marker(marker: Path):
    """Remove pending marker after successful ingestion."""
    if marker.exists():
        marker.unlink()


def cleanup_partial_ingestion(state_dir: Path, collection_name: str | None = None):
    """On startup, find pending markers and clean up partial ChromaDB entries."""
    if not state_dir.exists():
        return
    for marker in state_dir.glob(".pending-*"):
        try:
            data = json.loads(marker.read_text())
            chunk_ids = data.get("chunk_ids", [])
            filename = data.get("filename", "unknown")
            if chunk_ids:
                collection = _get_collection(collection_name)
                collection.delete(ids=chunk_ids)
                log.info(f"Cleaned up {len(chunk_ids)} partial chunks for '{filename}'")
            marker.unlink()
            log.info(f"Removed pending marker for '{filename}'")
        except Exception as e:
            log.warning(f"Failed to clean up pending marker {marker}: {e}")


def store_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    collection_name: str | None = None,
    pending_marker: Path | None = None,
):
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
        # Track stored IDs in pending marker for crash recovery
        if pending_marker and pending_marker.exists():
            update_pending_marker(pending_marker, ids[i:end])


def record_document(path: Path, chunk_count: int, state_file: Path):
    """Append document record to state file with exclusive lock."""
    import datetime

    record = {
        "filename": path.name,
        "sha256": _file_sha256(path),
        "chunks": chunk_count,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(state_file, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(record) + "\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
