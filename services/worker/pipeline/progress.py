"""Ingestion progress reporting via SQLite.

The worker updates progress rows; the API's SSE endpoint reads them.
Uses synchronous sqlite3 (worker is sync, API is async via aiosqlite).
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime

log = logging.getLogger("gutenberg.progress")

DB_PATH = os.environ.get("DATABASE_PATH", "/data/gutenberg.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def update_job_progress(
    job_id: str,
    *,
    status: str | None = None,
    completed_files: int | None = None,
    current_file: str | None = None,
    current_step: str | None = None,
    error: str | None = None,
) -> None:
    """Update an ingestion job's progress in SQLite."""
    conn = _get_conn()
    try:
        sets = ["updated_at = datetime('now')"]
        params: list = []
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if completed_files is not None:
            sets.append("completed_files = ?")
            params.append(completed_files)
        if current_file is not None:
            sets.append("current_file = ?")
            params.append(current_file)
        if current_step is not None:
            sets.append("current_step = ?")
            params.append(current_step)
        if error is not None:
            sets.append("error = ?")
            params.append(error)

        params.append(job_id)
        conn.execute(f"UPDATE ingestion_job SET {', '.join(sets)} WHERE id = ?", params)
        conn.commit()
    finally:
        conn.close()


def update_document_status(
    doc_id: str | None = None,
    *,
    filename: str | None = None,
    corpus_id: str | None = None,
    status: str | None = None,
    chunks: int | None = None,
    sha256: str | None = None,
    file_type: str | None = None,
    error: str | None = None,
) -> None:
    """Update a document record in SQLite."""
    conn = _get_conn()
    try:
        if doc_id:
            where = "id = ?"
            where_param = doc_id
        elif filename and corpus_id:
            where = "filename = ? AND corpus_id = ?"
            where_param = None  # handled below
        else:
            return

        sets = []
        params: list = []
        if status:
            sets.append("status = ?")
            params.append(status)
        if chunks is not None:
            sets.append("chunks = ?")
            params.append(chunks)
        if sha256:
            sets.append("sha256 = ?")
            params.append(sha256)
        if file_type:
            sets.append("file_type = ?")
            params.append(file_type)
        if error:
            sets.append("error = ?")
            params.append(error)

        if not sets:
            return

        if doc_id:
            params.append(doc_id)
        else:
            params.extend([filename, corpus_id])

        conn.execute(f"UPDATE document SET {', '.join(sets)} WHERE {where}", params)
        conn.commit()
    finally:
        conn.close()


def update_corpus_status(corpus_id: str, status: str) -> None:
    """Update corpus status in SQLite."""
    conn = _get_conn()
    try:
        conn.execute("UPDATE corpus SET status = ? WHERE id = ?", (status, corpus_id))
        conn.commit()
    finally:
        conn.close()


def get_corpus_collection(corpus_id: str) -> str | None:
    """Get the ChromaDB collection name for a corpus."""
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "SELECT collection_name FROM corpus WHERE id = ?", (corpus_id,)
        )
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        conn.close()
