"""OpenAI-compatible /v1/chat/completions with SSE streaming."""

import asyncio
import json
import logging
import os
import re
import time
import uuid

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from core.config import settings, collection_routes
from core.rag import retrieve, _extract_multi_work_filters, detect_works_in_query
from core.verification import (
    extract_quotes,
    verify_quotes,
    verify_against_source,
    format_verification_footer,
    repair_citations,
)

log = logging.getLogger("gutenberg.chat")
router = APIRouter()


class Message(BaseModel):
    role: str
    content: str


def _estimate_tokens(text: str) -> int:
    """Estimate token count using ~4 chars per token heuristic."""
    return len(text) // 4


class ChatRequest(BaseModel):
    model: str = "gutenberg-rag"
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = True


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions with RAG."""
    # Extract the user's latest message
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    # Parse collection from model name (e.g., "gutenberg-rag/macy" → slug "macy")
    target_collection = None
    if "/" in request.model:
        _, slug = request.model.split("/", 1)
        target_collection = collection_routes.get(slug)
        if target_collection:
            log.info(f"Routing to collection '{target_collection}' (slug: {slug})")

    # RAG retrieval — offload to thread so blocking I/O doesn't stall event loop
    try:
        system_prompt, sources = await asyncio.to_thread(
            retrieve, user_message, collection=target_collection
        )
    except Exception:
        log.exception("RAG retrieval failed, falling back to direct")
        system_prompt = (
            "You are a helpful assistant. Note: source retrieval failed, "
            "so you are answering without corpus grounding; acknowledge this briefly."
        )
        sources = []

    # Structured-answer route: detect multi-work intent at runtime via the same
    # gazetteer/source-pattern logic retrieve() uses internally. When the user
    # asks for a précis across ≥1 work AND we have an OpenRouter key, route
    # the answer through the JSON-schema path so citation breadth is structural
    # AND the single-work path bypasses qwen3:8b's empty-content failure mode.
    or_key = (
        settings.openrouter_api_key
        or os.environ.get("OPENROUTER_API_KEY", "")
        or os.environ.get("OPENROUTER_KEY", "")
    )
    multi_works: list[str] = []
    if settings.feature_structured_answer and or_key and sources:
        try:
            multi_works = await asyncio.to_thread(
                _extract_multi_work_filters, user_message, target_collection
            )
        except Exception as e:
            log.debug(f"multi-work detection failed: {e}")
            multi_works = []
        # Single-work fallback: the multi-work helper requires ≥2 hits.
        # When that fails, ``detect_works_in_query`` returns ALL detected
        # works without minimum, so a 1-work mention also routes through
        # the structured path. Avoids qwen3:8b's empty-content failure
        # mode by giving the user a consistent OpenRouter answerer.
        if not multi_works and settings.feature_structured_answer_single_work:
            try:
                all_hits = await asyncio.to_thread(
                    detect_works_in_query, user_message, target_collection
                )
                if all_hits:
                    multi_works = all_hits
            except Exception:
                pass

    if multi_works:
        log.info(
            f"structured-answer route engaged for {len(multi_works)} works "
            f"({settings.structured_answer_model})"
        )
        rendered, parsed, validation = await _structured_answer_via_openrouter(
            user_message, sources, multi_works, or_key
        )
        if request.stream:
            # Progressive: synthesis → separator → per_work entries → coverage
            # gets each emitted as its own delta. Falls back to one-shot when
            # parsed JSON wasn't produced (call failure or downstream error).
            if parsed:
                return EventSourceResponse(
                    _stream_structured(parsed, validation, request),
                    media_type="text/event-stream",
                )
            return EventSourceResponse(
                _stream_oneshot(rendered, request, sources),
                media_type="text/event-stream",
            )
        else:
            return _wrap_oneshot_response(rendered, request)

    # Build messages for Ollama
    ollama_messages = [{"role": "system", "content": system_prompt}]

    # Include conversation history with token budget
    max_tokens = settings.max_history_tokens
    history = []
    total_tokens = 0
    for msg in reversed(request.messages[:-1]):
        msg_tokens = _estimate_tokens(msg.content)
        if total_tokens + msg_tokens > max_tokens and history:
            break
        history.insert(0, msg)
        total_tokens += msg_tokens

    for msg in history:
        ollama_messages.append({"role": msg.role, "content": msg.content})

    # Always include the latest user message
    latest = {"role": request.messages[-1].role, "content": request.messages[-1].content}

    # P8: VLM-enhanced answer — attach images from retrieved chunks when the
    # feature flag is on and any source carries an image_path. Ollama's /api/chat
    # accepts a base64 "images" list on the user message; non-VLM models ignore it.
    if settings.feature_vlm_answer and sources:
        images_b64 = _gather_source_images(sources, limit=settings.vlm_max_images)
        if images_b64:
            latest["images"] = images_b64

    ollama_messages.append(latest)

    if request.stream:
        return EventSourceResponse(
            _stream_response(ollama_messages, request, sources),
            media_type="text/event-stream",
        )
    else:
        return await _non_stream_response(ollama_messages, request, sources)


async def _stream_response(
    messages: list[dict], request: ChatRequest, sources: list[dict]
):
    """Stream response from Ollama as SSE events, with verification footer."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    accumulated_text = []

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST",
            f"{settings.ollama_host}/api/chat",
            json={
                "model": (
                    settings.vlm_model
                    if settings.feature_vlm_answer
                    and any("images" in m for m in messages)
                    else settings.ollama_llm_model
                ),
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                },
            },
        ) as resp:
            in_think = False
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = data.get("message", {}).get("content", "")
                done = data.get("done", False)

                # Filter out <think>...</think> tokens from streaming
                if "<think>" in content:
                    in_think = True
                    content = content.split("<think>")[0]
                if in_think:
                    if "</think>" in content:
                        in_think = False
                        content = content.split("</think>")[-1]
                    else:
                        content = ""

                if content:
                    accumulated_text.append(content)

                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content} if content else {},
                            "finish_reason": None,
                        }
                    ],
                }
                yield {"data": json.dumps(chunk)}

                if done:
                    # Run verification on the accumulated response
                    full_response = "".join(accumulated_text)
                    _, footer = _run_verification(full_response, sources)

                    if footer:
                        footer_chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": footer},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield {"data": json.dumps(footer_chunk)}

                    # Final stop chunk
                    stop_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield {"data": json.dumps(stop_chunk)}
                    yield {"data": "[DONE]"}
                    return


async def _non_stream_response(
    messages: list[dict], request: ChatRequest, sources: list[dict]
) -> dict:
    """Non-streaming response from Ollama."""
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{settings.ollama_host}/api/chat",
            json={
                "model": (
                    settings.vlm_model
                    if settings.feature_vlm_answer
                    and any("images" in m for m in messages)
                    else settings.ollama_llm_model
                ),
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()

    content = data.get("message", {}).get("content", "")
    # Strip qwen3-style <think>...</think> blocks from response
    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)

    # Repair inline citations in-place (non-streaming) + append footer
    content, footer = _run_verification(content, sources)
    if footer:
        content += footer

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0)
            + data.get("eval_count", 0),
        },
    }


def _run_verification(response_text: str, sources: list[dict]) -> tuple[str, str]:
    """Run quote verification and optional citation repair.

    Returns ``(repaired_text, footer)``. ``repaired_text`` is the response with
    inline [Source: …, p. N] tags rewritten to match the chunks their preceding
    quotes actually came from (no-op if :attr:`settings.enable_citation_repair`
    is False). ``footer`` is the verification summary plus — when repair ran —
    a "Citation corrections" block listing the rewrites.
    """
    if not sources:
        return response_text, ""

    repaired = response_text
    corrections: list[dict] = []
    try:
        if settings.enable_citation_repair:
            from core.verification import repair_citations_with_diff
            repaired, corrections = repair_citations_with_diff(response_text, sources)
    except Exception:
        log.exception("Citation repair failed")
        repaired = response_text
        corrections = []

    try:
        quotes = extract_quotes(repaired)
        if not quotes:
            return repaired, _format_corrections_footer(corrections)
        results = verify_quotes(quotes, sources)
        results = verify_against_source(results)
        footer = format_verification_footer(results)
        footer += _format_corrections_footer(corrections)
        return repaired, footer
    except Exception:
        log.exception("Quote verification failed")
        return repaired, _format_corrections_footer(corrections)


async def _structured_answer_via_openrouter(
    query: str,
    chunks: list[dict],
    required_works: list[str],
    api_key: str,
) -> tuple[str, dict | None, dict]:
    """Run the JSON-schema structured-answer path off-thread.

    Returns ``(rendered_markdown_with_coverage, parsed_json, validation)``.
    ``parsed_json`` is ``None`` only on call failure — callers can fall
    back to the one-shot stream in that case.
    """
    from core.structured_answer import answer_structured

    def _do_call() -> tuple[str, dict, dict]:
        return answer_structured(
            query,
            chunks,
            required_works,
            model=settings.structured_answer_model,
            api_key=api_key,
            timeout=settings.structured_answer_timeout,
        )

    try:
        rendered, parsed, validation = await asyncio.to_thread(_do_call)
    except Exception:
        log.exception("structured answer failed; falling back to plain text")
        return "", None, {}

    # Coverage report: surface gaps the schema couldn't prevent. We do NOT
    # run the existing _run_verification → repair_citations pipeline here:
    # the structured renderer already guarantees verbatim quotes (via
    # _enforce_verbatim) and correct [Source: work, p. page] tags (via
    # per_work metadata). Re-running citation repair only confuses adjacent
    # tags and rewrites short titles into ugly long filenames.
    coverage_lines: list[str] = []
    if validation.get("missing_in_per_work"):
        coverage_lines.append(
            f"- ⚠ {len(validation['missing_in_per_work'])} required work(s) "
            "had no per_work entry: " +
            ", ".join(validation["missing_in_per_work"][:3])
        )
    if validation.get("unverified_quotes"):
        coverage_lines.append(
            f"- ⚠ {len(validation['unverified_quotes'])} quote(s) flagged "
            "non-verbatim and removed before render"
        )
    coverage_block = (
        ("\n\n---\n**Coverage report**\n" + "\n".join(coverage_lines))
        if coverage_lines else ""
    )

    return rendered + coverage_block, parsed, validation


def _build_coverage_block(validation: dict) -> str:
    """Same coverage-report block the non-stream path appends, exposed for
    reuse by the streaming path."""
    if not validation:
        return ""
    coverage_lines: list[str] = []
    if validation.get("missing_in_per_work"):
        coverage_lines.append(
            f"- ⚠ {len(validation['missing_in_per_work'])} required work(s) "
            "had no per_work entry: " +
            ", ".join(validation["missing_in_per_work"][:3])
        )
    if validation.get("unverified_quotes"):
        coverage_lines.append(
            f"- ⚠ {len(validation['unverified_quotes'])} quote(s) flagged "
            "non-verbatim and removed before render"
        )
    if not coverage_lines:
        return ""
    return "\n\n---\n**Coverage report**\n" + "\n".join(coverage_lines)


def _sse_chunk(chat_id: str, model: str, content: str, *, role: str | None = None):
    """Build one OpenAI-compatible SSE event with a content delta."""
    delta: dict = {"content": content}
    if role:
        delta["role"] = role
    return {"data": json.dumps({
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    })}


async def _stream_structured(
    parsed: dict, validation: dict, request: ChatRequest
):
    """Progressive SSE stream for the structured-answer route.

    Emits one delta per logical block so clients render
    synthesis-then-evidence-then-coverage incrementally rather than as
    a single jumbo blob. Same OpenAI envelope; multiple ``delta.content``
    chunks share one chat_id, ending with stop + ``[DONE]``.

    Honest gap signalling: works listed in ``validation.works_without_chunks``
    that the model didn't include get an explicit "no chunk for this work"
    line so users can tell corpus-side gaps from model-side refusal.
    """
    from core.structured_answer import (
        render_evidence_line, _name_match, _short_title, _stem,
    )

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    synthesis = (parsed.get("synthesis") or "").strip()
    per_work = parsed.get("per_work") or []
    works_without_chunks = set(validation.get("works_without_chunks") or [])
    first = True

    if synthesis:
        yield _sse_chunk(chat_id, request.model, synthesis + "\n\n", role="assistant")
        first = False

    if per_work or works_without_chunks:
        sep = "### Evidence\n\n"
        if first:
            yield _sse_chunk(chat_id, request.model, sep, role="assistant")
            first = False
        else:
            yield _sse_chunk(chat_id, request.model, sep)

        rendered_works: set[str] = set()
        n_emitted = 0
        for entry in per_work:
            entry_work = (entry.get("work") or "").strip()
            no_chunks = any(
                _name_match(entry_work, w) for w in works_without_chunks
            )
            line = render_evidence_line(entry, no_corpus_chunks=no_chunks)
            if not line:
                continue
            yield _sse_chunk(chat_id, request.model, line + "\n")
            rendered_works.add(entry_work)
            n_emitted += 1
        # Emit a synthetic gap entry for any zero-chunk work the model
        # didn't already list.
        for w in works_without_chunks:
            already = any(_name_match(rw, w) for rw in rendered_works)
            if already:
                continue
            short = _short_title(_stem(w)) or w
            line = render_evidence_line(
                {"work": short.title(), "gloss": ""}, no_corpus_chunks=True
            )
            if line:
                yield _sse_chunk(chat_id, request.model, line + "\n")
                n_emitted += 1
        if n_emitted == 0 and first:
            yield _sse_chunk(chat_id, request.model, "", role="assistant")
            first = False

    coverage = _build_coverage_block(validation)
    if coverage:
        if first:
            yield _sse_chunk(chat_id, request.model, coverage, role="assistant")
            first = False
        else:
            yield _sse_chunk(chat_id, request.model, coverage)

    yield {"data": json.dumps({
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    })}
    yield {"data": "[DONE]"}


async def _stream_oneshot(
    content: str, request: ChatRequest, sources: list[dict]
):
    """Emit a complete pre-rendered answer as a single OpenAI-compatible
    streaming chunk + stop chunk + [DONE]. Used by the structured route."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    yield {"data": json.dumps({
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": content},
            "finish_reason": None,
        }],
    })}
    yield {"data": json.dumps({
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    })}
    yield {"data": "[DONE]"}


def _wrap_oneshot_response(content: str, request: ChatRequest) -> dict:
    """Non-streaming OpenAI-compatible envelope around a pre-rendered answer."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _gather_source_images(sources: list[dict], limit: int) -> list[str]:
    """Return base64-encoded images from retrieved chunks carrying image paths.

    image_path comes from chunk metadata written by the worker, but we still
    scope reads to settings.data_root so a poisoned/legacy chunk cannot escape.
    """
    import base64
    from pathlib import Path

    try:
        data_root = Path(settings.data_root).resolve()
    except Exception:
        return []

    out: list[str] = []
    for src in sources:
        if len(out) >= limit:
            break
        meta = src.get("metadata") or {}
        path = meta.get("image_path")
        if not path:
            continue
        try:
            p = Path(path).resolve()
            if not p.is_relative_to(data_root):
                log.warning(f"image_path outside data_root rejected: {path}")
                continue
            if p.is_file():
                out.append(base64.b64encode(p.read_bytes()).decode("ascii"))
        except Exception as e:
            log.debug(f"image attach skipped ({path}): {e}")
    return out


def _format_corrections_footer(corrections: list[dict]) -> str:
    if not corrections:
        return ""
    lines = [f"\n\n**Citation corrections:** {len(corrections)} rewritten"]
    for c in corrections[:10]:
        lines.append(f"- {c['original']} → {c['corrected']}  (\u201c{c['quote']}\u201d)")
    if len(corrections) > 10:
        lines.append(f"- \u2026 {len(corrections) - 10} more")
    return "\n".join(lines)
