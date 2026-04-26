"""Corpus management — create, list, delete corpus projects."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
import shutil
import uuid
from pathlib import Path

import chromadb
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from sse_starlette.sse import EventSourceResponse

from core.config import settings
from core.database import get_db

log = logging.getLogger("gutenberg.corpus")
router = APIRouter(prefix="/api/corpus", tags=["corpus"])


def _sanitize_name(name: str) -> str:
    name = html.unescape(name)
    name = name.strip()[:100]
    for char in "/\\:<>":
        name = name.replace(char, "")
    return name


def _sanitize_tags(tags: str) -> str:
    tags = html.unescape(tags)
    parts = [t.strip()[:50] for t in tags.split(",") if t.strip()]
    return ",".join(parts[:10])


def _sanitize_filename(filename: str) -> str:
    filename = Path(filename).name
    filename = filename.replace("..", "").replace("\x00", "")
    filename = re.sub(r"[\x00-\x1f]", "", filename)
    return filename


def _slug(name: str) -> str:
    """Generate a ChromaDB-safe collection name from corpus name."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]
    return f"gutenberg-{slug}"


def _get_chroma_client():
    host = settings.chroma_host.replace("http://", "").replace("https://", "")
    parts = host.split(":")
    return chromadb.HttpClient(
        host=parts[0], port=int(parts[1]) if len(parts) > 1 else 8000
    )


def _invalidate_retrieval_cache(collection_name: str) -> None:
    """Clear BM25/query caches so next retrieval re-reads ChromaDB.

    Called after ingest/reingest so newly ingested chunks aren't hidden
    behind the stale in-memory BM25 index. Cheap: no eager rebuild.
    """
    try:
        from core.rag import invalidate_bm25_cache

        invalidate_bm25_cache(collection_name)
    except Exception as e:
        log.warning(f"Could not invalidate retrieval cache for {collection_name}: {e}")


@router.post("")
async def create_corpus(name: str = Form(...), tags: str = Form("")):
    """Create a new corpus project with an empty ChromaDB collection."""
    name = _sanitize_name(name)
    tags = _sanitize_tags(tags)
    corpus_id = str(uuid.uuid4())
    collection_name = _slug(name)

    # Create ChromaDB collection
    client = _get_chroma_client()
    client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO corpus (id, name, tags, collection_name, status) VALUES (?, ?, ?, ?, ?)",
            (corpus_id, name, tags, collection_name, "empty"),
        )
        await db.commit()
    finally:
        await db.close()

    return {
        "id": corpus_id,
        "name": name,
        "collection_name": collection_name,
        "status": "empty",
    }


@router.get("")
async def list_corpora():
    """List all corpus projects with document and chunk counts.

    Chunk count is aggregated from document.chunks (populated by the worker
    on ingest) rather than round-tripping to ChromaDB per corpus — avoids
    O(N) HTTP calls when listing many corpora.
    """
    db = await get_db()
    try:
        cursor = await db.execute("""
            SELECT c.id, c.name, c.tags, c.collection_name, c.status, c.created_at,
                   COUNT(d.id) as document_count,
                   COALESCE(SUM(CASE WHEN d.status = 'done' THEN d.chunks ELSE 0 END), 0) as chunk_count
            FROM corpus c
            LEFT JOIN document d ON d.corpus_id = c.id AND d.status = 'done'
            GROUP BY c.id
            ORDER BY c.created_at DESC
        """)
        rows = await cursor.fetchall()
    finally:
        await db.close()

    return [
        {
            "id": row["id"],
            "name": row["name"],
            "tags": row["tags"],
            "collection_name": row["collection_name"],
            "status": row["status"],
            "document_count": row["document_count"],
            "chunk_count": row["chunk_count"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


@router.get("/{corpus_id}")
async def get_corpus(corpus_id: str):
    """Get corpus detail with documents."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM corpus WHERE id = ?", (corpus_id,))
        corpus = await cursor.fetchone()
        if not corpus:
            raise HTTPException(status_code=404, detail="Corpus not found")

        cursor = await db.execute(
            "SELECT * FROM document WHERE corpus_id = ? ORDER BY created_at",
            (corpus_id,),
        )
        docs = await cursor.fetchall()
    finally:
        await db.close()

    chunk_count = sum(d["chunks"] for d in docs if d["status"] == "done")

    return {
        "id": corpus["id"],
        "name": corpus["name"],
        "tags": corpus["tags"],
        "collection_name": corpus["collection_name"],
        "status": corpus["status"],
        "chunk_count": chunk_count,
        "created_at": corpus["created_at"],
        "documents": [dict(d) for d in docs],
    }


@router.delete("/{corpus_id}")
async def delete_corpus(corpus_id: str):
    """Delete corpus, its ChromaDB collection, and all related data."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT collection_name FROM corpus WHERE id = ?", (corpus_id,)
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Corpus not found")

        # Delete ChromaDB collection
        try:
            client = _get_chroma_client()
            client.delete_collection(row["collection_name"])
        except Exception as e:
            log.warning(f"Could not delete ChromaDB collection: {e}")

        # Cascade deletes conversations, messages, documents, jobs
        await db.execute("DELETE FROM corpus WHERE id = ?", (corpus_id,))
        await db.commit()
    finally:
        await db.close()

    # Clean up inbox files
    inbox = Path(f"/data/inbox/{corpus_id}")
    if inbox.exists():
        shutil.rmtree(inbox, ignore_errors=True)

    return {"deleted": True}


@router.delete("/{corpus_id}/documents/{doc_id}")
async def delete_document(corpus_id: str, doc_id: str):
    """Remove a document from corpus: delete chunks from ChromaDB + DB record."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT d.filename, c.collection_name FROM document d "
            "JOIN corpus c ON d.corpus_id = c.id "
            "WHERE d.id = ? AND d.corpus_id = ?",
            (doc_id, corpus_id),
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        filename = row["filename"]
        collection_name = row["collection_name"]

        # Delete chunks from ChromaDB by source metadata
        try:
            client = _get_chroma_client()
            col = client.get_collection(collection_name)
            # ChromaDB where filter on metadata
            results = col.get(where={"source": filename}, include=[])
            if results["ids"]:
                col.delete(ids=results["ids"])
                log.info(f"Deleted {len(results['ids'])} chunks for {filename}")
        except Exception as e:
            log.warning(f"Could not delete chunks from ChromaDB: {e}")

        # Delete DB record
        await db.execute("DELETE FROM document WHERE id = ?", (doc_id,))
        await db.commit()
    finally:
        await db.close()

    return {"deleted": True, "filename": filename}


@router.post("/{corpus_id}/documents/{doc_id}/reingest")
async def reingest_document(corpus_id: str, doc_id: str):
    """Re-process a document: delete existing chunks and create a new ingestion job."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT d.filename, c.collection_name FROM document d "
            "JOIN corpus c ON d.corpus_id = c.id "
            "WHERE d.id = ? AND d.corpus_id = ?",
            (doc_id, corpus_id),
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        filename = row["filename"]
        collection_name = row["collection_name"]

        # Delete existing chunks from ChromaDB
        try:
            client = _get_chroma_client()
            col = client.get_collection(collection_name)
            results = col.get(where={"source": filename}, include=[])
            if results["ids"]:
                col.delete(ids=results["ids"])
        except Exception as e:
            log.warning(f"Could not delete existing chunks: {e}")

        # Reset document status
        await db.execute(
            "UPDATE document SET status = 'pending', chunks = 0, error = NULL WHERE id = ?",
            (doc_id,),
        )

        # Ensure file exists in inbox for worker to pick up
        processed_file = Path(f"/data/processed/{filename}")
        if not processed_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Original file not found: {filename}",
            )
        inbox_dir = Path(f"/data/inbox/{corpus_id}")
        inbox_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(processed_file), str(inbox_dir / filename))

        # Create ingestion job
        job_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO ingestion_job (id, corpus_id, total_files, status) VALUES (?, ?, 1, 'pending')",
            (job_id, corpus_id),
        )
        await db.execute(
            "UPDATE corpus SET status = 'ingesting' WHERE id = ?", (corpus_id,)
        )
        await db.commit()
    finally:
        await db.close()

    _invalidate_retrieval_cache(collection_name)

    return {"job_id": job_id, "filename": filename}


@router.post("/{corpus_id}/ingest")
async def upload_documents(corpus_id: str, files: list[UploadFile] = File(...)):
    """Upload documents and create an ingestion job."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, collection_name FROM corpus WHERE id = ?", (corpus_id,)
        )
        corpus = await cursor.fetchone()
        if not corpus:
            raise HTTPException(status_code=404, detail="Corpus not found")

        # Save files to inbox
        inbox = Path(f"/data/inbox/{corpus_id}")
        inbox.mkdir(parents=True, exist_ok=True)

        saved_files = []
        MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
        for f in files:
            f.filename = _sanitize_filename(f.filename)
            ext = Path(f.filename or "").suffix.lower()
            if ext not in {".pdf", ".epub", ".docx"}:
                continue
            content = await f.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413, detail=f"{f.filename} exceeds 500MB limit"
                )
            dest = inbox / f.filename
            dest.write_bytes(content)
            saved_files.append(f.filename)

            # Create document record
            doc_id = str(uuid.uuid4())
            from core.database import _parse_filename_metadata

            author, title, year = _parse_filename_metadata(f.filename)
            await db.execute(
                """INSERT INTO document (id, corpus_id, filename, status, author, title, year)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (doc_id, corpus_id, f.filename, "pending", author, title, year),
            )

        if not saved_files:
            raise HTTPException(status_code=400, detail="No valid files uploaded")

        # Create ingestion job
        job_id = str(uuid.uuid4())
        await db.execute(
            """INSERT INTO ingestion_job (id, corpus_id, total_files, status)
               VALUES (?, ?, ?, ?)""",
            (job_id, corpus_id, len(saved_files), "pending"),
        )

        # Update corpus status
        await db.execute(
            "UPDATE corpus SET status = 'ingesting' WHERE id = ?", (corpus_id,)
        )
        await db.commit()
    finally:
        await db.close()

    _invalidate_retrieval_cache(corpus["collection_name"])

    return {"job_id": job_id, "files": saved_files, "total": len(saved_files)}


@router.get("/{corpus_id}/ingest/status")
async def ingest_status(corpus_id: str):
    """SSE stream of ingestion progress."""

    async def event_generator():
        while True:
            db = await get_db()
            try:
                cursor = await db.execute(
                    """SELECT * FROM ingestion_job WHERE corpus_id = ?
                       ORDER BY created_at DESC LIMIT 1""",
                    (corpus_id,),
                )
                job = await cursor.fetchone()
            finally:
                await db.close()

            if not job:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": "No ingestion job found"}),
                }
                return

            data = {
                "job_id": job["id"],
                "total_files": job["total_files"],
                "completed_files": job["completed_files"],
                "current_file": job["current_file"],
                "current_step": job["current_step"],
                "status": job["status"],
                "error": job["error"],
            }
            yield {"event": "progress", "data": json.dumps(data)}

            if job["status"] in ("done", "failed"):
                return

            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())
