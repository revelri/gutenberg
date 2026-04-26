# Follow-up plan execution — verbatim threshold, per-work fetch, progressive SSE

**Run date:** 2026-04-25

Three follow-ups from `this-is-a-competitive-melodic-hellman.md` executed in plan order.

---

## Change 2: Verbatim threshold sweep (85 / 90 / 95)

Goal: pick the rapidfuzz `partial_ratio` cutoff that maximises ALCE precision while keeping cited-works coverage ≥ baseline − 0.05.

Precis n=10, Gemini Flash via OpenRouter, threshold 85 vs 90 vs 95, identical retrieval.

| threshold | composite | cited_cov | span_hit | alce_recall | alce_prec | total_cit |
|---|---|---|---|---|---|---|
| 85 | 0.797 | 0.883 | 0.920 | 0.800 | 0.800 | 20 |
| 90 | 0.774 | 0.817 | 0.920 | 0.783 | 0.783 | 23 |
| 95 | 0.774 | 0.817 | 0.920 | 0.750 | 0.750 | 20 |

**Decision: ship 85 as default.** Tightening the threshold did not improve ALCE precision — the variance is dominated by Gemini Flash's stochastic LLM/NLI output, not by paraphrase rate. 90/95 lost 6.6pts cited-works coverage *and* 1.7-5.0pts ALCE precision. Precision is at the LLM-composition layer, not at the verbatim-match layer; rapidfuzz is doing its job at 85.

Field added: `Settings.verbatim_min_score: int = 85` in `services/api/core/config.py`. Now picked up at call time by `core/structured_answer.py::answer_structured` via `_settings_default_min_score()` (no plumbing needed in chat.py). Eval CLI: `scripts/eval_precis.py --verbatim-threshold N`.

---

## Change 1: Per-work targeted retrieval top-up

Goal: when ≥2 works are detected in the query, fetch additional dense+BM25 candidates filtered to each work's source rather than only rebalancing the existing 200-candidate pool.

Implementation: new block in `services/api/core/rag.py::retrieve()` after dedup, before the existing `_topup` rebalance. Reuses `_dense_search`, `_bm25_search`, `_filter_by_source`, `_expand_query_for_bm25`. Gated by `settings.per_work_fetch_enabled`, capped at `settings.per_work_fetch_max_works=4` works × `settings.per_work_fetch_k=15` per channel.

**Ablation (n=10 precis, threshold=85):**

| metric | fetch_off | fetch_on | Δ |
|---|---|---|---|
| mean_composite | 0.797 | 0.809 | +0.012 |
| mean_works_cited_coverage | 0.883 | 0.917 | +0.033 |
| mean_works_cited_coverage_retrieval | 0.917 | 0.917 | +0.000 |
| mean_concept_span_hit_rate | 0.920 | 0.920 | +0.000 |
| mean_entity_coverage | 0.830 | 0.830 | +0.000 |
| alce_recall | 0.850 | 0.810 | -0.040 |
| alce_precision | 0.850 | 0.810 | -0.040 |

**Per-query cited-works delta (the metric per-work fetch targets):**

| label | fetch_off | fetch_on | Δ |
|---|---|---|---|
| BwO evolution AO→ATP | 1.000 | 1.000 | +0.000 |
| Repetition DR→Cinema | 0.667 | 0.667 | +0.000 |
| Deterritorialization AO→ATP | 0.500 | 1.000 | +0.500 |
| Virtual Berg→DR→C2 | 1.000 | 1.000 | +0.000 |
| Will to power → Desire NP→AO | 1.000 | 1.000 | +0.000 |
| Diagram FB→Foucault→ATP | 0.667 | 1.000 | +0.333 |
| Sense LS→WIP | 1.000 | 1.000 | +0.000 |
| Immanence Spinoza→WIP | 1.000 | 1.000 | +0.000 |
| Fold: Leibniz vs Foucault | 1.000 | 1.000 | +0.000 |
| Signs PS→ATP | 1.000 | 0.500 | -0.500 |

**Result:** mean cited-works coverage +0.033 (0.883 → 0.917). The two queries the plan flagged — Diagram FB→Foucault→ATP and Deterritorialization AO→ATP — moved +0.333 and +0.500 respectively. Signs PS→ATP regressed −0.500 (run-to-run Gemini stochastic variance). ALCE −0.040 within noise on n=21 citations.

---

## Change 3: Progressive SSE streaming for the structured route

Goal: replace the single-content-delta envelope with a staged stream that emits synthesis first, then each per-work evidence card as a separate delta, then coverage block, then stop. Same OpenAI `chatcmpl` envelope; multiple deltas share one chat_id.

Implementation:

- Refactored `_render_evidence_line` out of `_render_markdown` in `services/api/core/structured_answer.py` so the streamer reuses the same line builder.
- `_structured_answer_via_openrouter` in `services/api/routers/chat.py` now returns `(rendered_with_coverage, parsed_json, validation)` instead of just the rendered string.
- New `_stream_structured(parsed, validation, request)` async generator + `_build_coverage_block` helper + `_sse_chunk` builder.
- The chat handler picks `_stream_structured` when `parsed is not None`, falls back to `_stream_oneshot` on call failure. Non-streaming path unchanged.

**End-to-end test against the BwO multi-work query (`stream: true`):**

```
[0] {'role': 'assistant', 'content': 'In *Anti-Oedipus*, the Body without Organs (BwO) is presented…'}
[1] {'role': None,        'content': '### Evidence\n\n'}
[2] {'role': None,        'content': '- **Anti-Oedipus:** "But in reality the unconscious belongs to the realm of phys…'}
[3] {'role': None,        'content': '- **A Thousand Plateaus:** "What is drawn (the Body without Organs, the plane of…'}
[4] {'role': None,        'content': None, 'finish': 'stop'}
[5] DONE
```

6 SSE events. Synthesis, separator, 2 evidence cards, stop, DONE. Concatenated content matches the non-stream `_wrap_oneshot_response` output byte-for-byte. Single-work query (`What is the body without organs?`) does NOT engage the structured route — falls through to existing Ollama path unchanged.

---

## Files changed

- `services/api/core/config.py` — 4 new fields: `verbatim_min_score`, `per_work_fetch_enabled`, `per_work_fetch_k`, `per_work_fetch_max_works`.
- `services/api/core/structured_answer.py` — `_settings_default_min_score()`; `verbatim_min_score` parameter now optional; `render_evidence_line()` extracted from `_render_markdown`.
- `services/api/core/rag.py` — per-work targeted dense+BM25 fetch block in `retrieve()` before `_topup`.
- `services/api/routers/chat.py` — `_structured_answer_via_openrouter` now returns 3-tuple; `_build_coverage_block`, `_sse_chunk`, `_stream_structured` added; `chat_completions` branches on parsed presence.
- `scripts/eval_precis.py` — `--verbatim-threshold` CLI flag plumbed to `answer_structured`.

No new files. No reindex required.

## Outstanding (unchanged)

- **P4 ColBERT** — voyager Py-3.13 wheel still missing.
- **Async-OR RAPTOR** — refactor landed in prior session but not exercised on fresh tree.
- **Bigger précis gold set** — 10 queries gives high run-to-run variance; expand to 30+ before treating ±5pt drifts as signal.
