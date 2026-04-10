"""Shared ChromaDB connection helper — singleton HTTP client.

Both the API and worker containers import from here to avoid
duplicating host-parsing and client-creation logic.
"""

import chromadb

from typing import Optional, Union

_client: Optional[chromadb.HttpClient] = None
_collection_cache: dict[str, chromadb.Collection] = {}


def _get_chroma_client(chroma_host: str) -> chromadb.HttpClient:
    """Get or create a singleton ChromaDB HTTP client."""
    global _client
    if _client is None:
        host = chroma_host.replace("http://", "").replace("https://", "")
        parts = host.split(":")
        hostname = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 8000
        _client = chromadb.HttpClient(host=hostname, port=port)
    return _client


def get_collection(
    chroma_host: str,
    collection_name: str,
) -> chromadb.Collection:
    """Get or create a ChromaDB collection with cosine similarity.

    Results are cached per (host, name) pair.
    """
    cache_key = f"{chroma_host}:{collection_name}"
    if cache_key not in _collection_cache:
        client = _get_chroma_client(chroma_host)
        _collection_cache[cache_key] = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection_cache[cache_key]
