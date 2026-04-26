"""Gutenborg MCP server — structured citation retrieval for LLM clients.

Exposes the retrieval pipeline (BM25 + dense + SpaCy + ChromaDB) as MCP tools.
Any MCP-compatible client (Claude Desktop, Cursor, etc.) can connect and search
the corpus without needing Ollama or any LLM in the loop.

Usage:
    # stdio transport (for Claude Desktop, Cursor, etc.)
    python services/mcp_server/server.py

    # Or via mcp CLI
    mcp run services/mcp_server/server.py

Configure in Claude Desktop's claude_desktop_config.json:
    {
      "mcpServers": {
        "gutenborg": {
          "command": "python",
          "args": ["services/mcp_server/server.py"],
          "cwd": "/path/to/gutenborg"
        }
      }
    }
"""

import os
import re
import sys
from pathlib import Path

# Add service directories to sys.path so we can import core + shared modules
_services_dir = str(Path(__file__).parent.parent)
_api_dir = str(Path(__file__).parent.parent / "api")
_project_root = str(Path(__file__).parent.parent.parent)
for _p in (_api_dir, _services_dir, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("gutenborg")


@mcp.tool()
def search_corpus(
    query: str,
    collection: str | None = None,
    top_k: int = 10,
    source_filter: str | None = None,
) -> dict:
    """Search the corpus for passages matching a query.

    Returns ranked chunks with source citations, page numbers, and relevance scores.
    Uses hybrid search (BM25 keyword + dense semantic) with adaptive fusion.

    Args:
        query: The search query. May include quoted phrases for exact matching.
        collection: Corpus collection name. Defaults to the primary collection.
        top_k: Number of results to return (default 10).
        source_filter: Restrict results to a specific document/book name.
    """
    from core.rag import retrieve

    _, chunks = retrieve(query, collection)

    results = []
    for i, chunk in enumerate(chunks[:top_k]):
        meta = chunk.get("metadata", {})
        entry = {
            "text": chunk.get("document", chunk.get("text", "")),
            "source": meta.get("source", ""),
            "page_start": meta.get("page_start", 0),
            "page_end": meta.get("page_end", 0),
            "heading": meta.get("heading", ""),
            "rank": i + 1,
        }
        if source_filter and source_filter.lower() not in entry["source"].lower():
            continue
        results.append(entry)

    return {"results": results, "total_retrieved": len(chunks)}


@mcp.tool()
def verify_quote(
    quote: str,
    collection: str | None = None,
) -> dict:
    """Verify whether a quoted passage exists in the corpus and identify its source.

    Uses exact, fuzzy, and lemma-normalized matching to check if the quote
    appears in any indexed document.

    Args:
        quote: The quoted text to verify.
        collection: Corpus collection name. Defaults to the primary collection.
    """
    from core.rag import retrieve
    from core.verification import verify_quotes

    _, chunks = retrieve(quote, collection)
    if not chunks:
        return {"status": "unverified", "source": None, "page": None, "similarity": 0.0}

    results = verify_quotes([quote], chunks)
    if results:
        v = results[0]
        return {
            "status": v.get("status", "unverified"),
            "source": v.get("source"),
            "page": v.get("page"),
            "similarity": v.get("similarity", 0.0),
        }
    return {"status": "unverified", "source": None, "page": None, "similarity": 0.0}


@mcp.tool()
def get_page_text(source: str, page: int) -> dict:
    """Get the raw text of a specific page from a source document.

    Args:
        source: Filename of the source document (e.g. "Deleuze - Difference and Repetition.pdf").
        page: Page number (1-indexed).
    """
    import fitz

    safe = source
    if "/" in safe or "\\" in safe or ".." in safe or "\0" in safe:
        return {"error": "Invalid filename"}

    data_dir = os.environ.get("DATA_DIR", "/data")
    base = Path(data_dir)

    pdf_path = None
    for subdir in ("processed", "inbox"):
        p = base / subdir / safe
        if p.is_file():
            pdf_path = p
            break

    if not pdf_path:
        # Try local data directory
        local_data = Path(_project_root) / "data"
        for subdir in ("processed", "inbox"):
            p = local_data / subdir / safe
            if p.is_file():
                pdf_path = p
                break

    if not pdf_path:
        return {"error": f"Source document not found: {source}"}

    doc = fitz.open(str(pdf_path))
    if page < 1 or page > len(doc):
        doc.close()
        return {"error": f"Page {page} out of range (document has {len(doc)} pages)"}

    text = doc[page - 1].get_text()
    doc.close()

    return {"text": text, "source": source, "page": page}


@mcp.tool()
def list_documents(collection: str | None = None) -> dict:
    """List all documents in a corpus with metadata.

    Args:
        collection: Corpus collection name. Defaults to the primary collection.
    """
    from core.chroma import get_collection
    from core.config import settings

    col = get_collection(collection)
    if col.count() == 0:
        return {"documents": []}

    result = col.get(include=["metadatas"], limit=col.count())
    sources = {}
    for meta in result["metadatas"]:
        src = meta.get("source", "")
        if src not in sources:
            sources[src] = {"filename": src, "chunk_count": 0}
        sources[src]["chunk_count"] += 1

    return {"documents": sorted(sources.values(), key=lambda d: d["filename"])}


@mcp.tool()
def list_corpora() -> dict:
    """List all corpus projects with document and chunk counts."""
    import sqlite3

    db_path = os.environ.get("DATABASE_PATH", "/data/gutenberg.db")
    # Try local path if container path doesn't exist
    if not Path(db_path).exists():
        local_db = Path(_project_root) / "data" / "gutenberg.db"  # corrected
        if local_db.exists():
            db_path = str(local_db)

    if not Path(db_path).exists():
        return {"corpora": [], "error": "Database not found"}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, name, collection_name, status FROM corpus ORDER BY name"
    ).fetchall()

    corpora = []
    for row in rows:
        doc_count = conn.execute(
            "SELECT COUNT(*) FROM document WHERE corpus_id = ?", (row["id"],)
        ).fetchone()[0]
        corpora.append({
            "id": row["id"],
            "name": row["name"],
            "collection_name": row["collection_name"],
            "status": row["status"],
            "document_count": doc_count,
        })
    conn.close()

    return {"corpora": corpora}


@mcp.tool()
def get_chunk_context(chunk_id: str, window: int = 1) -> dict:
    """Get a chunk with its surrounding chunks for expanded context.

    Retrieves the specified chunk plus adjacent chunks (by chunk_index)
    from the same source document.

    Args:
        chunk_id: The ChromaDB chunk ID.
        window: Number of adjacent chunks to include on each side (default 1).
    """
    from core.chroma import get_collection
    from core.config import settings

    col = get_collection()

    # Get the target chunk
    result = col.get(ids=[chunk_id], include=["documents", "metadatas"])
    if not result["ids"]:
        return {"error": f"Chunk not found: {chunk_id}"}

    meta = result["metadatas"][0]
    source = meta.get("source", "")
    chunk_index = meta.get("chunk_index", 0)

    # Get surrounding chunks from the same source
    target_indices = list(range(max(0, chunk_index - window), chunk_index + window + 1))

    # Query all chunks from same source
    all_chunks = col.get(
        where={"source": source},
        include=["documents", "metadatas"],
    )

    # Filter to our target window
    chunks = []
    for i, (doc, m) in enumerate(zip(all_chunks["documents"], all_chunks["metadatas"])):
        if m.get("chunk_index", -1) in target_indices:
            chunks.append({
                "text": doc,
                "source": m.get("source", ""),
                "page_start": m.get("page_start", 0),
                "page_end": m.get("page_end", 0),
                "heading": m.get("heading", ""),
                "chunk_index": m.get("chunk_index", 0),
            })

    chunks.sort(key=lambda c: c["chunk_index"])
    return {"chunks": chunks, "target_chunk_index": chunk_index}


if __name__ == "__main__":
    mcp.run()
