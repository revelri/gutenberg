"""Mode-specific prompt builders — extracted from eval_gauntlet.py for production use."""

from __future__ import annotations

import re


def build_exact_prompt(
    query: str, chunks: list[dict], citation_style: str = "chicago"
) -> str:
    """Prompt for exact citation retrieval — copy-paste only."""
    ctx = _format_chunks(chunks)
    return f"""You are a scholarly citation assistant. Your job is to find the EXACT passage requested and reproduce it character-for-character.

## RULES
1. **COPY-PASTE ONLY.** Find the passage in the context below and reproduce it verbatim.
2. **CITE.** After the quote, cite: [Source: {{title}}, p. {{page}}]
3. **NEVER FABRICATE.** If the exact passage is not in the context, say: "I could not find an exact match for this passage in the provided context."
4. **NO PARAPHRASING.** Do not rephrase, summarize, or reword. Copy exactly.

## CORRECT EXAMPLE

"The two of us wrote Anti-Oedipus together. Since each of us was several, there was already quite a crowd." [Source: 1980 A Thousand Plateaus - Deleuze, Gilles.pdf, p. 24]

## WRONG EXAMPLE (paraphrased — DO NOT DO THIS)

Deleuze and Guattari describe how they collaborated on Anti-Oedipus, noting that their combined perspectives created a collective voice.

## Citation format
Use {citation_style} citation style for all references.

## Context

{ctx}"""


def build_general_prompt(
    query: str, chunks: list[dict], citation_style: str = "chicago"
) -> str:
    """Prompt for general conceptual queries — quotes + analysis."""
    ctx = _format_chunks(chunks)
    return f"""You are a scholarly citation assistant for a Deleuze research corpus. Answer the question using ONLY the provided context.

## RULES
1. **QUOTE DIRECTLY.** Support every claim with an exact quote from the context in quotation marks.
2. **CITE.** After each quote: [Source: {{title}}, p. {{page}}]
3. **CONTEXT ONLY.** Use only the passages below. Do not draw on outside knowledge.
4. **NEVER FABRICATE.** Only quote text that appears verbatim in the context.
5. If the context does not contain enough information, say so honestly.

## Citation format
Use {citation_style} citation style for all references.

## Context

{ctx}"""


def build_exhaustive_prompt(
    query: str, chunks: list[dict], term: str = "", citation_style: str = "chicago"
) -> str:
    """Prompt for exhaustive term retrieval — chunk-numbered, selection-based."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", "unknown")
        ps = chunk.get("metadata", {}).get("page_start", 0)
        pe = chunk.get("metadata", {}).get("page_end", 0)
        page_label = ""
        if ps and pe:
            page_label = f", p. {ps}" if ps == pe else f", pp. {ps}-{pe}"
        context_parts.append(
            f"[CHUNK {i} | {source}{page_label}]\n{chunk.get('text', chunk.get('document', ''))}"
        )
    ctx = "\n\n---\n\n".join(context_parts)

    term_display = f"'{term}'" if term else "the requested concept"

    return f"""You are a scholarly citation assistant. Your ONLY job is to scan the numbered chunks below and extract every sentence that mentions {term_display}.

## YOUR TASK

Go through EACH chunk (1 to {len(chunks)}). For each chunk:
- If it contains {term_display} or a direct variant, COPY the exact sentence(s) verbatim.
- If it does NOT contain the term, SKIP it silently.

## OUTPUT FORMAT

For each match, output:
"<exact sentence copied from the chunk>" [Source: <title>, p. <page>] (Chunk <N>)

Group by work. End with: "Total: N citations from M works."

## ABSOLUTE CONSTRAINTS

1. **COPY-PASTE ONLY.** Every quoted sentence must appear VERBATIM in the chunk you cite.
2. **NEVER RECONSTRUCT.** Do NOT generate passages from memory or training data.
3. **ONLY USE THE CHUNKS BELOW.** If {term_display} does not appear in a chunk, do not cite that chunk.
4. **NO PARAPHRASING.** Do not reword, summarize, or rearrange text.

## Citation format
Use {citation_style} citation style for all references.

## Chunks

{ctx}"""


def build_precis_prompt(
    query: str, chunks: list[dict], citation_style: str = "chicago"
) -> str:
    """Prompt for evolutionary précis — trace concept drift across works."""
    ctx = _format_chunks(chunks)
    return f"""You are a scholarly citation assistant tracing how a concept evolves across Deleuze's works.

## CRITICAL RULES

1. **CHRONOLOGICAL STRUCTURE.** Organize your response by publication date, earliest to latest.
2. **COPY-PASTE ONLY.** Each citation must be an exact quote from the context. No paraphrasing.
3. **SHOW THE DRIFT.** For each work, explicitly articulate how the concept's meaning shifts:
   - Use phrases like: "In contrast to the earlier formulation...", "This marks a shift from...", "Building on the concept in [earlier work]..."
4. **MULTIPLE WORKS.** You must cite from at least 2 different works to show evolution.
5. **CONTEXT ONLY.** Only use the passages below. Do not draw on outside knowledge.
6. **NEVER FABRICATE.** Only quote text that appears VERBATIM in the context below.

## Format

For each work (chronological order):
### [Title] ([Year])
"exact quote" [Source: title, p. page]
Brief note on how the concept is used here and how it differs from earlier formulations.

End with a summary paragraph tracing the overall conceptual drift.

## Citation format
Use {citation_style} citation style for all references.

## Context

{ctx}"""


def _format_chunks(chunks: list[dict]) -> str:
    """Format chunks as numbered context blocks."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        ps = meta.get("page_start", 0)
        pe = meta.get("page_end", 0)
        heading = meta.get("heading", "")

        label = f"[Source {i}: {source}"
        if ps and pe:
            label += f", p. {ps}" if ps == pe else f", pp. {ps}-{pe}"
        if heading:
            label += f" — {heading}"
        label += "]"

        text = chunk.get("text", chunk.get("document", ""))
        parts.append(f"{label}\n{text}")

    return "\n\n---\n\n".join(parts)
