"""Hybrid answer composer — guaranteed citation breadth via structured output.

The LLM emits a JSON object with two parts:

  * ``per_work``  — one entry per required work, each with a verbatim quote,
                    page number, and short gloss. Schema-required to be
                    non-empty when ``required_works`` is supplied; the model
                    cannot return without populating an entry per work.
  * ``synthesis`` — a free 2-4 sentence comparative paragraph that draws on
                    the per_work entries. No citation tags here — citations
                    live in the deterministically-rendered evidence section.

We render the JSON to markdown ourselves, so citation breadth is structural
rather than a property of the prose. A quick spaCy/regex validator surfaces
mismatches between ``per_work`` content and what the synthesis actually
discusses, but the citation tags themselves are guaranteed correct by
construction.

OpenRouter / OpenAI-compatible endpoints with ``response_format={"type":
"json_schema", ...}`` are required. Tested against
``google/gemini-2.5-flash`` and ``google/gemini-2.5-flash-lite``; falls
back to permissive parsing when ``strict`` JSON-schema validation isn't
supported by the model.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

try:
    from core.rag import build_rag_prompt
except Exception:  # pragma: no cover — for offline tooling
    build_rag_prompt = None  # type: ignore

log = logging.getLogger("gutenberg.structured_answer")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ── Schema ──────────────────────────────────────────────────────────────

def _schema(required_works: list[str] | None) -> dict:
    """JSON schema for the structured response.

    ``per_work.minItems`` is set to ``len(required_works)`` so the model
    cannot return a response missing a work entry. ``additionalProperties``
    is locked off — keeps the response shape clean for rendering.
    """
    min_items = len(required_works) if required_works else 1
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["synthesis", "per_work"],
        "properties": {
            "synthesis": {
                "type": "string",
                "description": (
                    "2-4 sentence comparative paragraph drawing on the "
                    "per_work entries. NO inline [Source: …] tags here — "
                    "citations live in the evidence section. Name each work "
                    "by short title at least once."
                ),
            },
            "per_work": {
                "type": "array",
                "minItems": min_items,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["work", "quote", "page", "gloss"],
                    "properties": {
                        "work": {
                            "type": "string",
                            "description": (
                                "SHORT canonical title only — drop the year "
                                "prefix and the ' - Author' tail. "
                                "Example: for source label "
                                "'1972 Anti-Oedipus Capitalism and Schizophr "
                                "- Deleuze, Gilles.pdf', emit "
                                "'Anti-Oedipus'. For "
                                "'1980 A Thousand Plateaus - Deleuze, Gilles"
                                ".pdf', emit 'A Thousand Plateaus'."
                            ),
                        },
                        "quote": {
                            "type": "string",
                            "description": (
                                "VERBATIM quote from the cited chunk. Do "
                                "not paraphrase. 8-40 words. Strip leading "
                                "and trailing quotation marks."
                            ),
                        },
                        "page": {
                            "type": "string",
                            "description": (
                                "Page number or range as it appears in the "
                                "Source label (e.g., '47' or '47-49')."
                            ),
                        },
                        "gloss": {
                            "type": "string",
                            "description": (
                                "One short sentence explaining how this "
                                "passage answers the question."
                            ),
                        },
                    },
                },
            },
        },
    }


# ── Prompt ──────────────────────────────────────────────────────────────

def _build_system_prompt(
    query: str, chunks: list[dict], required_works: list[str] | None
) -> str:
    """Assemble system prompt with context + structured-output instructions."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata") or {}
        source = meta.get("source", "unknown")
        page_start = meta.get("page_start", 0)
        page_end = meta.get("page_end", 0)
        label = f"[Source {i}: {source}"
        if page_start and page_end:
            if page_start == page_end:
                label += f", p. {page_start}"
            else:
                label += f", pp. {page_start}-{page_end}"
        label += "]"
        context_parts.append(f"{label}\n{chunk['text']}")
    context_block = "\n\n---\n\n".join(context_parts)

    required_block = ""
    if required_works:
        works_list = "\n".join(f"- {w}" for w in required_works)
        required_block = (
            "\n\n## Required Works\n\n"
            "This question references multiple works. The per_work array MUST "
            "contain one entry for EACH of the following — no exceptions, no "
            "merging:\n\n"
            f"{works_list}\n"
        )

    return f"""You are a scholarly citation assistant. Return a single JSON object matching the response_format schema.

## Rules

1. **per_work** — one entry per required work. Each entry's ``quote`` MUST be verbatim from the corresponding source's chunk in the context below. Do not paraphrase. Pick the single most relevant passage per work for the question.
2. **synthesis** — a brief comparative paragraph (2-4 sentences). Name each work by short title at least once. Do NOT include any [Source: …] tags or page numbers — citations live in the evidence section.
3. **Verbatim only.** If you cannot find a verbatim passage from a required work in the context, set its ``quote`` to "" and ``gloss`` to a brief explanation of why no passage was found. Do not fabricate.
4. **Context only.** Use only the provided context. No outside knowledge.
{required_block}
## Context

{context_block}"""


# ── OpenRouter call ─────────────────────────────────────────────────────

def _call_openrouter(
    system_prompt: str,
    user_query: str,
    schema: dict,
    *,
    model: str,
    api_key: str,
    timeout: float = 180.0,
) -> dict:
    """Returns the parsed JSON object. Raises on transport / parse failure."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "temperature": 0.1,
        "max_tokens": 1500,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "precis_response",
                "strict": True,
                "schema": schema,
            },
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/revelri/gutenborg",
        "X-Title": "gutenborg-structured-answer",
    }
    r = httpx.post(OPENROUTER_URL, headers=headers, json=body, timeout=timeout)
    if r.status_code == 400 and "json_schema" in r.text.lower():
        # Model rejected strict json_schema — retry with looser json_object.
        log.warning("model rejected json_schema; retrying with json_object")
        body["response_format"] = {"type": "json_object"}
        r = httpx.post(OPENROUTER_URL, headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    return _parse_loose_json(content)


def _parse_loose_json(text: str) -> dict:
    """Best-effort JSON parse — tolerates code-fence wrappers and trailing text."""
    if not text:
        raise ValueError("empty response from model")
    text = text.strip()
    # Strip markdown fences if any.
    fence = re.match(r"^```(?:json)?\s*(.+?)\s*```\s*$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Find the first balanced JSON object substring.
        start = text.find("{")
        if start < 0:
            raise
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
        raise


# ── Rendering ───────────────────────────────────────────────────────────

def render_evidence_line(entry: dict, *, no_corpus_chunks: bool = False) -> str:
    """Format a single per_work entry as a single markdown bullet line.

    Public so the SSE streamer in chat.py can emit each evidence card as
    its own delta without duplicating format logic. Returns "" if the
    entry has no usable ``work`` field (caller should skip it).

    When ``no_corpus_chunks=True``, render an explicit "passage not in
    retrieved context for this work" message — distinct from the model's
    "no verbatim passage found" so users can tell corpus-side absence
    from model-side refusal to quote.
    """
    work = (entry.get("work") or "").strip()
    if not work:
        return ""
    if no_corpus_chunks:
        gloss = (entry.get("gloss") or "").strip()
        line = (
            f"- **{work}:** _no chunk for this work was retrieved — "
            "the corpus may not contain a passage answering this query._"
        )
        if gloss:
            line += f" — {gloss}"
        return line
    quote = (entry.get("quote") or "").strip().strip('"').strip("“”")
    page = str(entry.get("page") or "").strip()
    gloss = (entry.get("gloss") or "").strip()
    if quote and page:
        line = f'- **{work}:** "{quote}" [Source: {work}, p. {page}]'
    elif quote:
        line = f'- **{work}:** "{quote}"'
    else:
        line = f"- **{work}:** _no verbatim passage found in retrieved context._"
    if gloss:
        line += f" — {gloss}"
    return line


def chunks_per_work(
    chunks: list[dict], required_works: list[str]
) -> dict[str, int]:
    """Count retrieved chunks per required work via short-title fuzzy match.

    Returns ``{work: chunk_count}``; works with 0 chunks signal a
    corpus-side gap that the answerer can't resolve no matter what.
    """
    counts: dict[str, int] = {w: 0 for w in (required_works or [])}
    for c in chunks or []:
        src = (c.get("metadata") or {}).get("source", "") or ""
        if not src:
            continue
        for w in counts:
            if _name_match(src, w):
                counts[w] += 1
                break
    return counts


def _render_markdown(
    parsed: dict,
    required_works: list[str] | None,
    works_without_chunks: set[str] | None = None,
) -> str:
    """Deterministically render parsed JSON to user-facing markdown."""
    synthesis = (parsed.get("synthesis") or "").strip()
    per_work = parsed.get("per_work") or []
    works_without_chunks = works_without_chunks or set()

    lines: list[str] = []
    if synthesis:
        lines.append(synthesis)
        lines.append("")

    if per_work or works_without_chunks:
        lines.append("### Evidence")
        lines.append("")
        rendered_works: set[str] = set()
        for entry in per_work:
            entry_work = (entry.get("work") or "").strip()
            no_chunks = any(
                _name_match(entry_work, w) for w in works_without_chunks
            )
            line = render_evidence_line(entry, no_corpus_chunks=no_chunks)
            if line:
                lines.append(line)
                rendered_works.add(entry_work)
        # Synthesise an entry for any work the model omitted that has no
        # corpus chunks — guarantees the gap is visible to the user.
        for w in works_without_chunks:
            already = any(_name_match(rw, w) for rw in rendered_works)
            if already:
                continue
            short = _short_title(_stem(w)) or w
            line = render_evidence_line(
                {"work": short.title(), "gloss": ""}, no_corpus_chunks=True
            )
            if line:
                lines.append(line)
    return "\n".join(lines).rstrip()


# ── Validation ──────────────────────────────────────────────────────────

def _stem(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"^\d{4}\s+", "", s)
    s = re.sub(r"\.pdf$", "", s)
    return s.strip()


def _short_title(stem_str: str) -> str:
    """Best-effort short title from a long source stem.

    "anti-oedipus capitalism and schizophr - deleuze, gilles"
        → "anti-oedipus"
    "a thousand plateaus - deleuze, gilles"
        → "a thousand plateaus"
    Picks the segment before the publisher/author tail (separator " - ").
    Caps at 4 tokens or 30 chars to stay readable.
    """
    head = stem_str.split(" - ")[0].strip()
    tokens = head.split()
    short = " ".join(tokens[:4]).strip()
    return short[:30].rstrip(",.;:")


def _name_match(candidate: str, required: str) -> bool:
    a, b = _short_title(_stem(candidate)), _short_title(_stem(required))
    if not a or not b:
        return False
    return a in b or b in a


def validate_coverage(
    parsed: dict,
    required_works: list[str] | None,
    chunks: list[dict] | None = None,
) -> dict:
    """Surface gaps between parsed JSON and required_works.

    Returns a dict with:
      * per_work_coverage  — share of required_works that have a per_work entry
      * synthesis_coverage — share named in the synthesis paragraph (deterministic
                             stem match, no NLP)
      * unverified_quotes  — entries whose quote isn't found verbatim in any
                             provided chunk (rapidfuzz partial_ratio < 0.85)
      * missing_in_per_work / missing_in_synthesis — string lists
    """
    required_works = required_works or []
    per_work = parsed.get("per_work") or []
    synthesis = (parsed.get("synthesis") or "").lower()

    per_work_works = [(e.get("work") or "") for e in per_work]
    in_per_work = [
        w for w in required_works
        if any(_name_match(pw, w) for pw in per_work_works)
    ]
    in_syn = [
        w for w in required_works
        if _short_title(_stem(w)) and _short_title(_stem(w)) in synthesis
    ]

    unverified: list[dict] = []
    if chunks:
        try:
            from rapidfuzz import fuzz
            chunk_text = "\n".join((c.get("text") or "") for c in chunks).lower()
            for e in per_work:
                q = (e.get("quote") or "").strip().lower()
                if len(q) < 12:
                    continue
                score = fuzz.partial_ratio(q, chunk_text)
                if score < 85:
                    unverified.append(
                        {"work": e.get("work"), "quote": q[:80], "score": int(score)}
                    )
        except Exception as ex:
            log.debug(f"rapidfuzz unavailable: {ex}")

    return {
        "required_works_n": len(required_works),
        "per_work_coverage": (
            round(len(in_per_work) / len(required_works), 4)
            if required_works else 1.0
        ),
        "synthesis_coverage": (
            round(len(in_syn) / len(required_works), 4)
            if required_works else 1.0
        ),
        "missing_in_per_work": [
            w for w in required_works if w not in in_per_work
        ],
        "missing_in_synthesis": [
            w for w in required_works if w not in in_syn
        ],
        "unverified_quotes": unverified,
    }


def _enforce_verbatim(
    parsed: dict, chunks: list[dict], min_score: int = 85
) -> int:
    """Replace any per_work quote that isn't found verbatim in the chunks
    with an empty string. Returns count of dropped quotes.

    The renderer already shows ``_no verbatim passage found in retrieved
    context._`` when ``quote`` is empty, so this enforces ALCE precision
    mechanically rather than trusting the LLM to refuse paraphrase.
    """
    if not chunks:
        return 0
    try:
        from rapidfuzz import fuzz
    except ImportError:
        log.warning("rapidfuzz unavailable — skipping verbatim enforcement")
        return 0

    chunk_text = "\n".join((c.get("text") or "") for c in chunks).lower()
    dropped = 0
    for entry in parsed.get("per_work", []):
        q = (entry.get("quote") or "").strip().lower()
        if len(q) < 12:
            continue
        if fuzz.partial_ratio(q, chunk_text) < min_score:
            entry["quote"] = ""
            entry["page"] = ""
            dropped += 1
    return dropped


# ── Public API ──────────────────────────────────────────────────────────

def _settings_default_min_score() -> int:
    """Read the verbatim threshold from settings if available; fall back to 85.

    Lets callers (chat router) pick up runtime config without explicit
    plumbing while keeping unit-test callers free to override.
    """
    try:
        from core.config import settings  # type: ignore
        return int(getattr(settings, "verbatim_min_score", 85))
    except Exception:
        return 85


def answer_structured(
    query: str,
    chunks: list[dict],
    required_works: list[str] | None,
    *,
    model: str,
    api_key: str,
    timeout: float = 180.0,
    enforce_verbatim: bool = True,
    verbatim_min_score: int | None = None,
) -> tuple[str, dict, dict]:
    """Build prompt → call OR with json_schema → enforce verbatim →
    render markdown + validate.

    Returns ``(rendered_markdown, parsed_json, validation_report)``.

    With ``enforce_verbatim=True`` (default), quotes whose rapidfuzz
    ``partial_ratio`` against the concatenated chunk text falls below
    ``verbatim_min_score`` are blanked before rendering. The renderer
    surfaces these as ``_no verbatim passage found in retrieved context_``
    rather than emitting a possibly-paraphrased citation tag.
    """
    if verbatim_min_score is None:
        verbatim_min_score = _settings_default_min_score()

    # Pre-flight: which required works have ZERO retrieved chunks?
    # These are corpus-side gaps the LLM cannot resolve regardless of
    # prompt. We exclude them from the schema's required per_work entries
    # (so the model isn't pushed to fabricate) and surface them with an
    # explicit "no chunk for this work" line in the rendered output.
    counts = chunks_per_work(chunks, required_works or [])
    works_without_chunks: set[str] = {w for w, n in counts.items() if n == 0}
    works_with_chunks = [
        w for w in (required_works or []) if w not in works_without_chunks
    ]
    if works_without_chunks:
        log.info(
            f"structured_answer: {len(works_without_chunks)}/{len(required_works or [])} "
            f"required work(s) had no retrieved chunks — surfacing as corpus gap"
        )

    schema = _schema(works_with_chunks if required_works else None)
    system_prompt = _build_system_prompt(
        query, chunks, works_with_chunks if required_works else None
    )
    parsed = _call_openrouter(
        system_prompt, query, schema,
        model=model, api_key=api_key, timeout=timeout,
    )
    if enforce_verbatim:
        dropped = _enforce_verbatim(parsed, chunks, min_score=verbatim_min_score)
        if dropped:
            log.info(
                f"structured_answer: dropped {dropped} non-verbatim quote(s) "
                f"(threshold={verbatim_min_score})"
            )
    rendered = _render_markdown(parsed, required_works, works_without_chunks)
    validation = validate_coverage(parsed, required_works, chunks)
    validation["works_without_chunks"] = sorted(works_without_chunks)
    return rendered, parsed, validation
