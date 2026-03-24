"""OpenAI-compatible /v1/chat/completions with SSE streaming."""

import json
import logging
import re
import time
import uuid

import httpx
from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from core.config import settings, collection_routes
from core.rag import retrieve

log = logging.getLogger("gutenberg.chat")
router = APIRouter()


class Message(BaseModel):
    role: str
    content: str


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
        return {"error": "No user message found"}

    # Parse collection from model name (e.g., "gutenberg-rag/macy" → slug "macy")
    target_collection = None
    if "/" in request.model:
        _, slug = request.model.split("/", 1)
        target_collection = collection_routes.get(slug)
        if target_collection:
            log.info(f"Routing to collection '{target_collection}' (slug: {slug})")

    # RAG retrieval
    try:
        system_prompt, sources = retrieve(user_message, collection=target_collection)
    except Exception:
        log.exception("RAG retrieval failed, falling back to direct")
        system_prompt = "You are a helpful assistant."
        sources = []

    # Build messages for Ollama
    ollama_messages = [{"role": "system", "content": system_prompt}]

    # Include conversation history (last few turns for context)
    for msg in request.messages[-6:]:
        ollama_messages.append({"role": msg.role, "content": msg.content})

    if request.stream:
        return EventSourceResponse(
            _stream_response(ollama_messages, request),
            media_type="text/event-stream",
        )
    else:
        return await _non_stream_response(ollama_messages, request)


async def _stream_response(messages: list[dict], request: ChatRequest):
    """Stream response from Ollama as SSE events."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{settings.ollama_host}/api/chat",
            json={
                "model": settings.ollama_llm_model,
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

                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content} if content else {},
                            "finish_reason": "stop" if done else None,
                        }
                    ],
                }
                yield {"data": json.dumps(chunk)}

                if done:
                    yield {"data": "[DONE]"}
                    return


async def _non_stream_response(messages: list[dict], request: ChatRequest) -> dict:
    """Non-streaming response from Ollama."""
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{settings.ollama_host}/api/chat",
            json={
                "model": settings.ollama_llm_model,
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
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        },
    }
