"""Structured retrieval endpoint — raw JSON, no LLM."""

import asyncio
import re
from pathlib import Path

import fitz
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.rag import retrieve
from core.verification import verify_quotes

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    collection: str | None = None
    top_k: int | None = None
    source_filter: str | None = None


class ChunkResult(BaseModel):
    text: str
    source: str
    page_start: int
    page_end: int
    heading: str = ""
    score: float = 0.0
    rank: int = 0
    chunk_id: str = ""


class SearchResponse(BaseModel):
    results: list[ChunkResult]
    total_candidates: int


class VerifyRequest(BaseModel):
    quote: str
    collection: str | None = None


class VerifyResponse(BaseModel):
    status: str
    source: str | None = None
    page: int | None = None
    similarity: float = 0.0


@router.post("/api/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """Hybrid retrieval: BM25 + dense search, returns ranked chunks."""
    system_prompt, chunks = await asyncio.to_thread(
        retrieve, req.query, req.collection
    )

    results = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        results.append(ChunkResult(
            text=chunk.get("document", chunk.get("text", "")),
            source=meta.get("source", ""),
            page_start=meta.get("page_start", 0),
            page_end=meta.get("page_end", 0),
            heading=meta.get("heading", ""),
            score=chunk.get("score", 0.0),
            rank=i + 1,
            chunk_id=chunk.get("id", ""),
        ))

    if req.source_filter:
        results = [r for r in results if req.source_filter.lower() in r.source.lower()]

    if req.top_k:
        results = results[: req.top_k]

    return SearchResponse(results=results, total_candidates=len(chunks))


@router.post("/api/verify", response_model=VerifyResponse)
async def verify(req: VerifyRequest):
    """Verify whether a quoted passage exists in the corpus."""
    _, chunks = await asyncio.to_thread(retrieve, req.quote, req.collection)

    if not chunks:
        return VerifyResponse(status="unverified")

    verification = verify_quotes([req.quote], chunks)
    if verification:
        v = verification[0]
        return VerifyResponse(
            status=v.get("status", "unverified"),
            source=v.get("source"),
            page=v.get("page"),
            similarity=v.get("similarity", 0.0),
        )
    return VerifyResponse(status="unverified")


DATA_DIR = "/data"


def _sanitize_filename(filename: str) -> str:
    if (
        "/" in filename
        or "\\" in filename
        or ".." in filename
        or "\0" in filename
        or re.search(r"%2[fF]|%5[cC]", filename)
    ):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return filename


@router.get("/api/pages/{source}/{page}")
async def get_page_text(source: str, page: int):
    """Get the raw text of a specific page from a source document."""
    safe = _sanitize_filename(source)
    base = Path(DATA_DIR)

    pdf_path = None
    for subdir in ("processed", "inbox"):
        p = base / subdir / safe
        if p.is_file():
            pdf_path = p
            break

    if not pdf_path:
        raise HTTPException(status_code=404, detail="Source document not found")

    doc = fitz.open(str(pdf_path))
    if page < 1 or page > len(doc):
        doc.close()
        raise HTTPException(status_code=404, detail="Page out of range")

    text = doc[page - 1].get_text()
    doc.close()

    return {"text": text, "source": source, "page": page}
