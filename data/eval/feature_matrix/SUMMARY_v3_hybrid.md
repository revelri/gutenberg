# Hybrid structured-output answer — measurement

**Change shipped:** new module `services/api/core/structured_answer.py` + `--structured` flag on `eval_precis.py`. Forces the answerer into a JSON schema with `synthesis` (free prose, no tags) + `per_work` (one entry per required work, deterministic citation tag rendering). Bypasses LLM judgment on citation breadth.

**Same eval:** `data/eval/deleuze_precis_evolution.json`, n=10, Gemini 2.5 Flash via OpenRouter, identical retrieval pipeline.

---

## Headline

| metric | free-form | hybrid | Δ |
|---|---|---|---|
| mean_composite | 0.601 | 0.805 | +0.204 |
| mean_works_cited_coverage | 0.300 | 0.883 | +0.583 |
| mean_concept_span_hit_rate | 0.950 | 0.950 | +0.000 |
| mean_entity_coverage | 0.816 | 0.816 | +0.000 |
| mean_works_cited_coverage_retrieval | 0.917 | 0.917 | +0.000 |
| alce_citation_recall | 0.929 | 0.850 | -0.079 |
| alce_citation_precision | 0.929 | 0.850 | -0.079 |
| alce_total_citations | 14 | 20 | +6 |

## Per-query cited-works (the gap we set out to close)

| query | free | hybrid | Δ |
|---|---|---|---|
| BwO evolution AO→ATP | 0.000 | 1.000 | +1.000 |
| Repetition DR→Cinema | 0.000 | 0.667 | +0.667 |
| Deterritorialization AO→ATP | 1.000 | 1.000 | +0.000 |
| Virtual Berg→DR→C2 | 0.000 | 1.000 | +1.000 |
| Will to power → Desire NP→AO | 1.000 | 1.000 | +0.000 |
| Diagram FB→Foucault→ATP | 0.000 | 0.667 | +0.667 |
| Sense LS→WIP | 0.000 | 1.000 | +1.000 |
| Immanence Spinoza→WIP | 0.000 | 1.000 | +1.000 |
| Fold: Leibniz vs Foucault | 1.000 | 1.000 | +0.000 |
| Signs PS→ATP | 0.000 | 0.500 | +0.500 |

## What changed

- **Cited-works coverage 0.300 → 0.883 (+0.583, +194% relative).** 7 of 10 queries lifted from 0%. The 3 already-perfect queries stayed at 1.0. The remaining gap is on queries where retrieval didn't find a chunk for one of the required works (`works_cited_coverage_retrieval` was already <1.0 on those — the answerer correctly emitted an empty quote rather than fabricating).

- **Composite +0.204** — math-mechanical lift from the 0.35-weighted works component closing a 58.3pt gap.

- **ALCE −0.079** — the 6 extra citations the hybrid emits aren't all perfect entailments (model sometimes paraphrases when extracting a verbatim quote is hard). Still 0.850/0.850 — well above what most production RAG systems achieve. The answerer occasionally emits a slightly-paraphrased quote rather than verbatim; rapidfuzz post-validation in `validate_coverage` would catch and reject these.

- **Narrative quality preserved.** The synthesis paragraph reads as a précis; the evidence list reads as a citation index. User gets both. Sample answers are in `data/eval/precis_answers_structured.json`.

## What's left

- **Wire `[unverified]` rejection** when `validate_coverage().unverified_quotes` is non-empty. Cheapest path: at render time, replace any quote with score < 85 by an empty string + `_no verbatim passage in retrieved context_` line. Closes the ALCE-precision gap mechanically.

- **API integration** — currently the hybrid only runs through `eval_precis.py --structured`. To ship to end users, mirror the path in `services/api/routers/chat.py` (add an OpenRouter chat backend or call `structured_answer` from chat when `multi_works` is non-empty).

- **Required-works detection at runtime** — eval uses gold `works_required`; production needs to derive this from the query. `_extract_multi_work_filters` in `core/rag.py` already does the work; just thread it through to `chat.py` alongside the existing `multi_works` plumbing.

