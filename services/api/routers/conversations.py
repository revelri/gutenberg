"""Conversation management — CRUD + mode-aware chat with RAG."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from core.config import settings
from core.database import get_db


# ── Request models ──────────────────────────────────────────────────

class CreateConversationRequest(BaseModel):
    mode: str = Field(default="general", pattern="^(exact|general|exhaustive|precis)$")
    citation_style: str = Field(default="chicago", pattern="^(mla|apa|chicago|harvard|asa|sage)$")
    title: str | None = None


class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)
    term: str = ""


class UpdateConversationRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
from core.modes import (
    build_exact_prompt,
    build_general_prompt,
    build_exhaustive_prompt,
    build_precis_prompt,
)

log = logging.getLogger("gutenberg.conversations")
router = APIRouter(tags=["conversations"])


# ── CRUD ─────────────────────────────────────────────────────────────


@router.get("/api/corpus/{corpus_id}/conversations")
async def list_conversations(corpus_id: str):
    db = await get_db()
    try:
        cursor = await db.execute(
            """SELECT id, corpus_id, title, mode, citation_style, created_at, updated_at
               FROM conversation WHERE corpus_id = ? ORDER BY updated_at DESC""",
            (corpus_id,),
        )
        rows = await cursor.fetchall()
    finally:
        await db.close()
    return [dict(r) for r in rows]


@router.post("/api/corpus/{corpus_id}/conversations")
async def create_conversation(corpus_id: str, body: CreateConversationRequest):
    db = await get_db()
    try:
        cursor = await db.execute("SELECT id FROM corpus WHERE id = ?", (corpus_id,))
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Corpus not found")

        conv_id = str(uuid.uuid4())

        await db.execute(
            """INSERT INTO conversation (id, corpus_id, title, mode, citation_style)
               VALUES (?, ?, ?, ?, ?)""",
            (conv_id, corpus_id, body.title, body.mode, body.citation_style),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM conversation WHERE id = ?", (conv_id,))
        row = await cursor.fetchone()
    finally:
        await db.close()

    return dict(row)


@router.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM conversation WHERE id = ?", (conv_id,))
        conv = await cursor.fetchone()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        cursor = await db.execute(
            "SELECT * FROM message WHERE conversation_id = ? ORDER BY created_at",
            (conv_id,),
        )
        messages = await cursor.fetchall()
    finally:
        await db.close()

    return {**dict(conv), "messages": [dict(m) for m in messages]}


@router.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    db = await get_db()
    try:
        await db.execute("DELETE FROM conversation WHERE id = ?", (conv_id,))
        await db.commit()
    finally:
        await db.close()
    return {"deleted": True}


@router.patch("/api/conversations/{conv_id}")
async def update_conversation(conv_id: str, body: UpdateConversationRequest):
    """Update conversation title."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id FROM conversation WHERE id = ?", (conv_id,)
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Conversation not found")

        await db.execute(
            "UPDATE conversation SET title = ?, updated_at = datetime('now') WHERE id = ?",
            (body.title, conv_id),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM conversation WHERE id = ?", (conv_id,))
        row = await cursor.fetchone()
    finally:
        await db.close()

    return dict(row)


# ── Chat (mode-aware RAG) ───────────────────────────────────────────


@router.post("/api/conversations/{conv_id}/messages")
async def send_message(conv_id: str, body: SendMessageRequest):
    """Send a message → RAG retrieve → LLM stream → persist."""
    content = body.content.strip()

    db = await get_db()
    try:
        cursor = await db.execute(
            """SELECT c.*, corpus.collection_name
               FROM conversation c JOIN corpus ON c.corpus_id = corpus.id
               WHERE c.id = ?""",
            (conv_id,),
        )
        conv = await cursor.fetchone()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Save user message
        user_msg_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO message (id, conversation_id, role, content) VALUES (?, ?, ?, ?)",
            (user_msg_id, conv_id, "user", content),
        )

        # Set conversation title from first message
        if not conv["title"]:
            title = content[:80] + ("..." if len(content) > 80 else "")
            await db.execute(
                "UPDATE conversation SET title = ? WHERE id = ?", (title, conv_id)
            )

        await db.commit()
    finally:
        await db.close()

    mode = conv["mode"]
    collection_name = conv["collection_name"]
    term = body.term

    # Retrieve chunks via the existing RAG pipeline. Offload to a thread
    # because retrieve() is synchronous and performs blocking network I/O
    # (ChromaDB, Ollama, sentence-transformers).
    retrieval_error: str | None = None
    try:
        from core.rag import retrieve

        system_prompt_unused, chunks = await asyncio.to_thread(
            retrieve, content, collection=collection_name
        )
    except Exception as e:
        log.exception("Retrieval failed")
        chunks = []
        retrieval_error = str(e) or e.__class__.__name__

    # Build mode-specific prompt
    citation_style = conv["citation_style"]
    if mode == "exact":
        system_prompt = build_exact_prompt(
            content, chunks, citation_style=citation_style
        )
    elif mode == "exhaustive":
        system_prompt = _build_exhaustive_with_preextract(
            content, chunks, term, citation_style
        )
    elif mode == "precis":
        system_prompt = build_precis_prompt(
            content, chunks, citation_style=citation_style
        )
    else:
        system_prompt = build_general_prompt(
            content, chunks, citation_style=citation_style
        )

    # Stream LLM response
    async def stream_and_persist():
        if retrieval_error:
            yield {
                "event": "warning",
                "data": json.dumps(
                    {
                        "message": (
                            "Source retrieval failed — answering without corpus context. "
                            f"({retrieval_error})"
                        )
                    }
                ),
            }
        elif not chunks:
            yield {
                "event": "warning",
                "data": json.dumps(
                    {
                        "message": "No relevant passages found in the corpus for this query."
                    }
                ),
            }
        accumulated = []

        async for chunk_text in _stream_llm(system_prompt, content, mode):
            accumulated.append(chunk_text)
            yield {
                "event": "message",
                "data": json.dumps({"content": chunk_text}),
            }

        # Persist assistant response
        full_response = "".join(accumulated)
        assistant_msg_id = str(uuid.uuid4())
        db = await get_db()
        try:
            await db.execute(
                "INSERT INTO message (id, conversation_id, role, content) VALUES (?, ?, ?, ?)",
                (assistant_msg_id, conv_id, "assistant", full_response),
            )
            await db.execute(
                "UPDATE conversation SET updated_at = datetime('now') WHERE id = ?",
                (conv_id,),
            )
            await db.commit()
        finally:
            await db.close()

        # Run verification on completed response
        try:
            from core.verification import extract_quotes, verify_quotes

            quotes = extract_quotes(full_response)
            citations = verify_quotes(quotes, chunks)
            yield {
                "event": "verification",
                "data": json.dumps({"citations": citations}),
            }
        except Exception as e:
            log.warning(f"Verification failed: {e}")

        yield {"event": "done", "data": json.dumps({"message_id": assistant_msg_id})}

    return EventSourceResponse(stream_and_persist())


def _build_exhaustive_with_preextract(
    query: str, chunks: list[dict], term: str, citation_style: str = "chicago"
) -> str:
    """For exhaustive mode: pre-extract exact term sentences, build prompt for remainder."""
    if not term:
        # Try to extract term from query
        m = re.search(
            r"['\u2018\u2019\u201c\u201d\"']([^'\"]+)['\u2018\u2019\u201c\u201d\"']",
            query,
        )
        if m:
            term = m.group(1)

    if term:
        try:
            from shared.nlp import sentencize, is_available

            use_spacy = is_available()
        except (ImportError, OSError):
            use_spacy = False

        # Pre-extract sentences with exact term
        pre_lines = []
        remaining = []
        term_lower = term.lower()

        for idx, chunk in enumerate(chunks):
            text = chunk.get("text", chunk.get("document", ""))
            if term_lower in text.lower():
                if use_spacy:
                    sentences = sentencize(text)
                else:
                    sentences = re.split(r"(?<=[.!?])\s+", text)
                for sent in sentences:
                    if term_lower in sent.lower() and len(sent.strip()) >= 20:
                        source = chunk.get("metadata", {}).get("source", "unknown")
                        page = chunk.get("metadata", {}).get("page_start", 0)
                        pre_lines.append(
                            f'"{sent.strip()}" [Source: {source}, p. {page}] (Chunk {idx + 1})'
                        )
            else:
                remaining.append(chunk)

        # Build prompt for remaining chunks only
        if pre_lines and remaining:
            prompt = build_exhaustive_prompt(
                query, remaining, term=term, citation_style=citation_style
            )
            pre_block = "\n".join(pre_lines)
            return f"""## Pre-extracted citations (verified mechanically)

{pre_block}

## Additional chunks to scan

{prompt}"""
        elif pre_lines:
            pre_block = "\n".join(pre_lines)
            return f"""## Pre-extracted citations (verified mechanically)

{pre_block}

Total: {len(pre_lines)} citations pre-extracted. Scan complete."""

    return build_exhaustive_prompt(
        query, chunks, term=term, citation_style=citation_style
    )


async def _stream_llm(system_prompt: str, query: str, mode: str):
    """Stream LLM response from Ollama or OpenRouter."""
    temp = 0.0 if mode == "exhaustive" else 0.1
    max_tokens = 4096 if mode in ("exhaustive", "precis") else 2048

    if settings.llm_backend == "openrouter" and settings.openrouter_api_key:
        async for chunk in _stream_openrouter(system_prompt, query, temp, max_tokens):
            yield chunk
    else:
        async for chunk in _stream_ollama(system_prompt, query, temp, max_tokens):
            yield chunk


async def _stream_ollama(system_prompt: str, query: str, temp: float, max_tokens: int):
    """Stream from local Ollama."""
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            async with client.stream(
                "POST",
                f"{settings.ollama_host}/api/chat",
                json={
                    "model": settings.ollama_llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ],
                    "stream": True,
                    "options": {"temperature": temp, "num_predict": max_tokens},
                },
            ) as resp:
                if resp.status_code != 200:
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": "LLM service unavailable"}),
                    }
                    return
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            if "💭" not in content and "⏺" not in content:
                                yield content
                    except json.JSONDecodeError:
                        continue
    except httpx.ConnectError:
        yield {
            "event": "error",
            "data": json.dumps({"error": "LLM service unavailable"}),
        }
        return


async def _stream_openrouter(
    system_prompt: str, query: str, temp: float, max_tokens: int
):
    """Stream from OpenRouter API."""
    async with httpx.AsyncClient(timeout=600) as client:
        async with client.stream(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "HTTP-Referer": "https://gutenberg.local",
            },
            json={
                "model": settings.openrouter_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                "temperature": temp,
                "max_tokens": max_tokens,
                "stream": True,
            },
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    return
                try:
                    data = json.loads(payload)
                    content = (
                        data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    )
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue
