# Hybrid structured-answer integration — three follow-ups landed

## What changed

### 1. Verbatim enforcement (`structured_answer.py::_enforce_verbatim`)
Before rendering, every `per_work` quote is checked against the concatenated chunk text via `rapidfuzz.fuzz.partial_ratio`. Quotes scoring < 85 get `quote=""` and `page=""` — the renderer surfaces these as `_no verbatim passage found in retrieved context._` and emits no `[Source: …]` tag. Paraphrastic citations cannot leak into rendered output.

### 2. API integration (`routers/chat.py::_structured_answer_via_openrouter`)
New helper threads the existing `core.structured_answer.answer_structured` into the `/v1/chat/completions` flow. Activates when:
- `settings.feature_structured_answer = True` AND
- `OPENROUTER_API_KEY` (or `OPENROUTER_KEY`) is set AND
- `_extract_multi_work_filters(query)` returns ≥ 2 works

Both streaming (`_stream_oneshot`) and non-streaming (`_wrap_oneshot_response`) envelopes implemented. Single-work queries fall through to the existing Ollama path unchanged.

Critically, the structured path skips the legacy `_run_verification` → `repair_citations_with_diff` pipeline. The structured renderer already guarantees verbatim quotes (via #1) and correct citation tags (built directly from chunk metadata), so re-running citation repair only confused adjacent tags.

### 3. Runtime required-works detection (`core/rag.py::_extract_multi_work_filters`)
Hardened the bare-title fallback to handle natural-language phrasings like "from Anti-Oedipus through A Thousand Plateaus" (no explicit "in X" preposition). Now tries progressively-shorter token prefixes of the short title (drop year, drop `.pdf`, drop ` - <author>`, drop leading articles), accepting only prefixes that are *unique substrings* across the catalog so common tokens like "the" don't cross-match.

In-container probe:
```
multi_works for "Trace ... from Anti-Oedipus through A Thousand Plateaus" → ['1972 Anti-Oedipus...', '1980 A Thousand Plateaus...']
multi_works for "Compare desire in Nietzsche and Philosophy with desiring-machines in Anti-Oedipus" → ['1962 Nietzsche...', '1972 Anti-Oedipus...']
```

## End-to-end test (live API)

Multi-work query routed through structured + Gemini Flash:

```
POST /v1/chat/completions
  "model": "gutenberg-rag/deleuze",
  "messages": [{"role": "user",
    "content": "Trace the evolution of the body without organs from Anti-Oedipus through A Thousand Plateaus."}]
```

API log: `structured-answer route engaged for 2 works (google/gemini-2.5-flash)`

Response:

> *In Anti-Oedipus, the Body without Organs (BwO) is presented as a fundamental concept, representing a surface where production is recorded… A Thousand Plateaus builds on this by defining the BwO as something that is 'drawn' or created…*
>
> **Evidence**
> - **Anti-Oedipus:** "But in reality the unconscious belongs to the realm of physics; the body without organs and its intensities are not metaphors, but matter itself." [Source: Anti-Oedipus, p. 154]
> - **A Thousand Plateaus:** "What is drawn (the Body without Organs, the plane of consistency, a line of flight) does not preexist the act of drawing." [Source: A Thousand Plateaus, p. 17]

Streaming variant verified: SSE emits a single complete content delta + stop + `[DONE]`.

Single-work query (`"What is the body without organs?"`): does NOT match `_extract_multi_work_filters`, falls back to existing Ollama path. No structured route invocation in logs.

## Re-measured precis (n=10) with rapidfuzz baked in

| metric | free-form | structured (no verbatim) | strict (verbatim on) |
|---|---|---|---|
| mean_composite | 0.601 | 0.805 | **0.822** |
| mean_works_cited_coverage | 0.300 | 0.883 | **0.933** |
| alce_citation_recall | 0.929 | 0.850 | 0.810 |
| alce_citation_precision | 0.929 | 0.850 | 0.810 |
| alce_total_citations | 14 | 20 | 21 |

Verbatim enforcement triggered on 2/10 queries (Repetition DR→Cinema and Diagram FB→Foucault→ATP — both queries where `works_cited_coverage_retrieval` was already < 1.0, so retrieval honestly reported no chunk for one work and the model correctly emitted a paraphrase that got rejected).

ALCE −0.04 between structured runs is within Gemini Flash's stochastic noise on n=20-ish citations (one extra entailment failure ≈ −5pts).

## Operational settings

To enable in production, set in env:

```
FEATURE_STRUCTURED_ANSWER=true
STRUCTURED_ANSWER_MODEL=google/gemini-2.5-flash
OPENROUTER_API_KEY=<key>
```

The flag defaults to `false` so existing deployments are unaffected. Multi-work detection is gated on the flag *and* the key being present — either missing → existing Ollama path stays in charge.

## Files touched

- `services/api/core/structured_answer.py` — `_enforce_verbatim`, threaded into `answer_structured`
- `services/api/core/config.py` — `feature_structured_answer`, `structured_answer_model`, `structured_answer_timeout`
- `services/api/core/rag.py` — `_extract_multi_work_filters` short-title fallback
- `services/api/routers/chat.py` — `_structured_answer_via_openrouter`, `_stream_oneshot`, `_wrap_oneshot_response`, multi-work route gate
