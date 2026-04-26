"""Modal processors — convert non-text content to retrievable chunks.

Each processor receives a raw modal payload (table markdown, LaTeX string,
image path + surrounding context) and returns a chunk dict compatible with
the rest of the pipeline. The produced chunk text is ``summary ∥ raw_payload``
so exact-match verification can still hit the original markup.

Flag-gated via ``settings.feature_modal_chunks``. Uses Ollama by default
(cheap, local). If an ``modal_describe_model`` is configured it wins.
"""

from __future__ import annotations

import logging
from typing import Iterable

import httpx

try:
    from core.config import settings
except Exception:
    from services.api.core.config import settings  # type: ignore

log = logging.getLogger("gutenberg.modal")


def _describe(prompt: str, max_tokens: int = 160) -> str:
    model = settings.modal_describe_model or settings.ollama_llm_model
    try:
        r = httpx.post(
            f"{settings.ollama_host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": max_tokens},
            },
            timeout=60,
        )
        r.raise_for_status()
        return (r.json().get("response") or "").strip()
    except Exception as e:
        log.warning(f"modal describe failed: {e}")
        return ""


def describe_table(markdown: str, surrounding_text: str = "") -> str:
    prompt = (
        "Describe the table below in 2 sentences. Include what it compares, "
        "its axes, and any notable values. Preserve proper names.\n\n"
        f"Context: {surrounding_text[:1200]}\n\n"
        f"Table:\n{markdown[:2400]}"
    )
    return _describe(prompt)


def describe_equation(latex: str, surrounding_text: str = "") -> str:
    prompt = (
        "Describe the equation below in one sentence, naming its domain "
        "(mathematics, physics, logic) and the concepts it relates. Do not "
        "rewrite the equation.\n\n"
        f"Context: {surrounding_text[:1200]}\n\n"
        f"LaTeX: {latex[:1200]}"
    )
    return _describe(prompt)


def make_modal_chunks(elements: Iterable[dict], source_metadata: dict) -> list[dict]:
    """Convert a list of modal elements into chunk dicts.

    Each element shape:
        {"kind": "table"|"equation"|"image",
         "content": str,                # markdown / LaTeX / caption
         "page": int | None,
         "surrounding_text": str}
    """
    if not settings.feature_modal_chunks:
        return []

    out: list[dict] = []
    base_source = source_metadata.get("source", "")
    for el in elements:
        kind = el.get("kind")
        content = el.get("content") or ""
        if not content.strip():
            continue

        if kind == "table":
            summary = describe_table(content, el.get("surrounding_text", ""))
        elif kind == "equation":
            summary = describe_equation(content, el.get("surrounding_text", ""))
        else:
            summary = el.get("caption") or ""

        # ``summary ∥ content`` keeps the original markup searchable so exact
        # verification can still hit the raw payload.
        text = f"{summary}\n\n{content}".strip() if summary else content.strip()
        if not text:
            continue

        page = el.get("page") or 0
        out.append(
            {
                "text": text,
                "contextual_text": text,
                "metadata": {
                    "source": base_source,
                    "heading": el.get("heading", ""),
                    "chunk_index": -1,  # assigned downstream if needed
                    "doc_type": source_metadata.get("doc_type", ""),
                    "page_start": page,
                    "page_end": page,
                    "modality": kind,
                },
            }
        )
    if out:
        log.info(f"Modal chunks produced: {len(out)}")
    return out
