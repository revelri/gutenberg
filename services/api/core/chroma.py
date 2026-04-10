"""ChromaDB connection helper — delegates to shared module."""

from core.config import settings
from shared.chroma import get_collection as _shared_get_collection


def get_chroma_client():
    """Create a ChromaDB HTTP client from settings.

    Prefer get_collection() directly — this exists for backward compat.
    """
    import chromadb

    host = settings.chroma_host.replace("http://", "").replace("https://", "")
    parts = host.split(":")
    hostname = parts[0]
    port = int(parts[1]) if len(parts) > 1 else 8000
    return chromadb.HttpClient(host=hostname, port=port)


def get_collection(name: str | None = None):
    """Get or create a ChromaDB collection with cosine similarity."""
    collection_name = name or settings.chroma_collection
    return _shared_get_collection(settings.chroma_host, collection_name)
