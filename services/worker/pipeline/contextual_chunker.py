"""Contextual chunking: prepend LLM-generated context to each chunk.

Based on Anthropic's Contextual Retrieval technique (Sep 2024):
each chunk gets a 50-100 token context prefix summarizing its position
in the document. This improves retrieval accuracy by 35-49%.

VRAM constraint: runs AFTER extraction completes. Docling and LLM
cannot coexist on an 8GB GPU. The caller must ensure GPU is free.
"""

import logging
import os

import httpx

log = logging.getLogger("gutenberg.contextual_chunker")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
CONTEXT_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "qwen3:8b")
BATCH_SIZE = 5  # chunks per LLM call


def add_context_to_chunks(
    chunks: list[dict],
    full_text: str,
    doc_title: str,
    ollama_host: str | None = None,
    model: str | None = None,
) -> list[dict]:
    """Prepend LLM-generated context summary to each chunk.

    For each chunk, generates a 1-2 sentence context like:
    "<context>This passage from Chapter 3 of Difference and Repetition
    discusses the distinction between virtual and actual.</context>"

    Then prepends it to the chunk text before embedding.
    """
    host = ollama_host or OLLAMA_HOST
    mdl = model or CONTEXT_MODEL

    log.info(f"Adding context to {len(chunks)} chunks using {mdl}")

    # Process in batches to reduce LLM calls
    enriched = []
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        contexts = _generate_contexts_batch(batch, doc_title, host, mdl)

        for chunk, context in zip(batch, contexts):
            if context:
                enriched_text = f"<context>{context}</context>\n\n{chunk['text']}"
            else:
                enriched_text = chunk["text"]

            enriched.append({
                "text": enriched_text,
                "metadata": chunk["metadata"],
            })

        log.info(f"  Contextualized {min(batch_start + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

    return enriched


def _generate_contexts_batch(
    chunks: list[dict], doc_title: str, host: str, model: str
) -> list[str]:
    """Generate context summaries for a batch of chunks in one LLM call."""
    # Build a prompt that asks for all contexts at once
    chunk_descriptions = []
    for i, chunk in enumerate(chunks):
        heading = chunk["metadata"].get("heading", "")
        page = chunk["metadata"].get("page_start", "?")
        preview = chunk["text"][:200].replace("\n", " ")
        chunk_descriptions.append(
            f"CHUNK {i+1} (p. {page}, section: {heading or 'none'}):\n{preview}..."
        )

    chunks_block = "\n\n".join(chunk_descriptions)

    prompt = f"""Document: "{doc_title}"

For each chunk below, write a single sentence (max 30 words) describing what this passage is about and where it fits in the document. Format: one line per chunk, numbered to match.

{chunks_block}

Respond with ONLY the numbered context lines, nothing else:"""

    try:
        resp = httpx.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 256,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        response_text = resp.json().get("response", "")

        # Parse numbered lines
        contexts = _parse_numbered_contexts(response_text, len(chunks))
        return contexts

    except Exception:
        log.exception("Context generation failed, returning empty contexts")
        return [""] * len(chunks)


def _parse_numbered_contexts(text: str, expected_count: int) -> list[str]:
    """Parse numbered context lines from LLM response."""
    import re

    lines = text.strip().split("\n")
    contexts = []
    for line in lines:
        # Match patterns like "1. ...", "1: ...", "CHUNK 1: ..."
        match = re.match(r"(?:CHUNK\s+)?(\d+)[.):]\s*(.*)", line.strip())
        if match:
            contexts.append(match.group(2).strip())

    # Pad or trim to expected count
    while len(contexts) < expected_count:
        contexts.append("")
    return contexts[:expected_count]
