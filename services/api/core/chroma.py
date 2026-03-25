"""Shared ChromaDB connection helper."""

import chromadb

from core.config import settings


def get_chroma_client() -> chromadb.HttpClient:
    """Create a ChromaDB HTTP client from settings."""
    host = settings.chroma_host.replace("http://", "").replace("https://", "")
    parts = host.split(":")
    hostname = parts[0]
    port = int(parts[1]) if len(parts) > 1 else 8000
    return chromadb.HttpClient(host=hostname, port=port)


def get_collection(name: str | None = None):
    """Get or create a ChromaDB collection with cosine similarity."""
    client = get_chroma_client()
    collection_name = name or settings.chroma_collection
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
