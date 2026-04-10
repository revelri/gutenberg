"""Shared Pydantic models — contract between frontend, API, and worker."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────

class CorpusStatus(str, Enum):
    empty = "empty"
    ingesting = "ingesting"
    ready = "ready"
    error = "error"


class QueryMode(str, Enum):
    exact = "exact"
    general = "general"
    exhaustive = "exhaustive"
    precis = "precis"


class CitationStyle(str, Enum):
    mla = "mla"
    apa = "apa"
    chicago = "chicago"
    harvard = "harvard"
    asa = "asa"
    sage = "sage"


class DocType(str, Enum):
    pdf_digital = "pdf_digital"
    pdf_scanned = "pdf_scanned"
    epub = "epub"
    docx = "docx"


class DocStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    failed = "failed"


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


# ── Corpus ───────────────────────────────────────────────────────────

class CorpusCreate(BaseModel):
    name: str
    tags: str = ""


class CorpusResponse(BaseModel):
    id: str
    name: str
    tags: str
    collection_name: str
    status: CorpusStatus
    document_count: int = 0
    chunk_count: int = 0
    created_at: str


# ── Document ─────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: str
    corpus_id: str
    filename: str
    file_type: str | None = None
    chunks: int = 0
    status: DocStatus
    author: str = ""
    title: str = ""
    year: int = 0
    error: str | None = None
    created_at: str


# ── Conversation ─────────────────────────────────────────────────────

class ConversationCreate(BaseModel):
    mode: QueryMode = QueryMode.general
    citation_style: CitationStyle = CitationStyle.chicago
    title: str | None = None


class ConversationResponse(BaseModel):
    id: str
    corpus_id: str
    title: str | None
    mode: QueryMode
    citation_style: CitationStyle
    created_at: str
    updated_at: str


class ConversationDetail(ConversationResponse):
    messages: list[MessageResponse] = []


# ── Message ──────────────────────────────────────────────────────────

class MessageCreate(BaseModel):
    content: str
    term: str | None = None  # for exhaustive mode: the term to search for


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    metadata_json: str = "{}"
    created_at: str


# ── Ingestion ────────────────────────────────────────────────────────

class IngestionStatus(BaseModel):
    job_id: str
    corpus_id: str
    total_files: int
    completed_files: int
    current_file: str | None
    current_step: str | None  # validating|ocr|chunking|embedding|storing
    status: JobStatus
    error: str | None = None
