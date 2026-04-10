"""SQLite database — async via aiosqlite, WAL mode for concurrent access."""

from __future__ import annotations

import logging
import json
import sqlite3
from pathlib import Path

import aiosqlite

log = logging.getLogger("gutenberg.db")

_DB_PATH: str = "/data/gutenberg.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS corpus (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    tags TEXT DEFAULT '',
    collection_name TEXT UNIQUE NOT NULL,
    status TEXT DEFAULT 'empty',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS document (
    id TEXT PRIMARY KEY,
    corpus_id TEXT NOT NULL REFERENCES corpus(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    sha256 TEXT,
    file_type TEXT,
    chunks INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',
    author TEXT DEFAULT '',
    title TEXT DEFAULT '',
    year INTEGER DEFAULT 0,
    publisher TEXT DEFAULT '',
    error TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS conversation (
    id TEXT PRIMARY KEY,
    corpus_id TEXT NOT NULL REFERENCES corpus(id) ON DELETE CASCADE,
    title TEXT,
    mode TEXT DEFAULT 'general',
    citation_style TEXT DEFAULT 'chicago',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS message (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ingestion_job (
    id TEXT PRIMARY KEY,
    corpus_id TEXT NOT NULL REFERENCES corpus(id) ON DELETE CASCADE,
    total_files INTEGER DEFAULT 0,
    completed_files INTEGER DEFAULT 0,
    current_file TEXT,
    current_step TEXT,
    status TEXT DEFAULT 'pending',
    error TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_document_corpus ON document(corpus_id);
CREATE INDEX IF NOT EXISTS idx_conversation_corpus ON conversation(corpus_id);
CREATE INDEX IF NOT EXISTS idx_message_conversation ON message(conversation_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_job_corpus ON ingestion_job(corpus_id);
"""


def set_db_path(path: str) -> None:
    global _DB_PATH
    _DB_PATH = path


async def get_db() -> aiosqlite.Connection:
    """Open a connection with WAL mode and foreign keys enabled."""
    db = await aiosqlite.connect(_DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db() -> None:
    """Create tables if they don't exist, run migrations."""
    Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    db = await get_db()
    try:
        await db.executescript(SCHEMA)
        await db.commit()
        log.info(f"Database initialized at {_DB_PATH}")

        # Migrate from documents.jsonl if it exists and corpus table is empty
        cursor = await db.execute("SELECT COUNT(*) FROM corpus")
        row = await cursor.fetchone()
        if row[0] == 0:
            await _migrate_from_jsonl(db)
    finally:
        await db.close()


async def _migrate_from_jsonl(db: aiosqlite.Connection) -> None:
    """Import existing documents.jsonl into the database on first run."""
    jsonl_path = Path("/data/state/documents.jsonl")
    if not jsonl_path.exists():
        return

    log.info("Migrating from documents.jsonl...")
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return

    # Group by collection (infer from existing ChromaDB collections)
    import uuid
    corpus_id = str(uuid.uuid4())

    await db.execute(
        "INSERT INTO corpus (id, name, tags, collection_name, status) VALUES (?, ?, ?, ?, ?)",
        (corpus_id, "Imported Corpus", "migrated", "gutenberg", "ready"),
    )

    for rec in records:
        doc_id = str(uuid.uuid4())
        filename = rec.get("filename", "unknown")
        # Parse metadata from filename pattern: "YYYY Title - Author.pdf"
        author, title, year = _parse_filename_metadata(filename)

        await db.execute(
            """INSERT INTO document (id, corpus_id, filename, sha256, chunks, status, author, title, year)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, corpus_id, filename, rec.get("sha256", ""),
             rec.get("chunks", 0), "done", author, title, year),
        )

    await db.commit()
    log.info(f"Migrated {len(records)} documents into corpus '{corpus_id}'")


def _parse_filename_metadata(filename: str) -> tuple[str, str, int]:
    """Parse 'YYYY Title - Author, First.pdf' → (author, title, year)."""
    import re
    name = filename.rsplit(".", 1)[0]  # strip extension

    # Try pattern: "YYYY Title - Author, First"
    m = re.match(r"^(\d{4})\s+(.+?)\s+-\s+(.+)$", name)
    if m:
        year = int(m.group(1))
        title = m.group(2).strip()
        author = m.group(3).strip()
        return author, title, year

    return "", name, 0


async def auto_populate_exemplar(chroma_host: str) -> None:
    """Detect existing deleuze-surya ChromaDB collection and create corpus entry."""
    import uuid
    try:
        import chromadb
        host = chroma_host.replace("http://", "").replace("https://", "")
        parts = host.split(":")
        client = chromadb.HttpClient(
            host=parts[0], port=int(parts[1]) if len(parts) > 1 else 8000
        )

        try:
            col = client.get_collection("deleuze-surya")
        except Exception:
            return  # collection doesn't exist

        chunk_count = col.count()
        if chunk_count == 0:
            return

        db = await get_db()
        try:
            # Check if already populated
            cursor = await db.execute(
                "SELECT id FROM corpus WHERE collection_name = ?", ("deleuze-surya",)
            )
            if await cursor.fetchone():
                return  # already exists

            corpus_id = str(uuid.uuid4())
            await db.execute(
                """INSERT INTO corpus (id, name, tags, collection_name, status)
                   VALUES (?, ?, ?, ?, ?)""",
                (corpus_id, "Deleuze Corpus (Surya OCR)",
                 "philosophy,deleuze,exemplar", "deleuze-surya", "ready"),
            )

            # Get unique sources from ChromaDB metadata
            result = col.get(include=["metadatas"])
            sources = {}
            for meta in result["metadatas"]:
                src = meta.get("source", "")
                if src and src not in sources:
                    sources[src] = 0
                if src:
                    sources[src] += 1

            for source, count in sources.items():
                doc_id = str(uuid.uuid4())
                author, title, year = _parse_filename_metadata(source)
                await db.execute(
                    """INSERT INTO document
                       (id, corpus_id, filename, chunks, status, author, title, year)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (doc_id, corpus_id, source, count, "done", author, title, year),
                )

            await db.commit()
            log.info(f"Auto-populated exemplar corpus: {len(sources)} books, {chunk_count} chunks")
        finally:
            await db.close()

    except Exception as e:
        log.warning(f"Could not auto-populate exemplar corpus: {e}")
