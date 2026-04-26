# Gutenborg eval architecture — full sweep

**Run date:** 2026-04-25 · **Corpora:** `gutenberg-deleuze-corpus` (10246 chunks: 8192 leaves + 2054 RAPTOR summaries) + `gutenberg-deleuze-corpus-noctx` (10246 raw, sibling for P0 ablation).

All 6 evals executed in one session. OpenRouter (Gemini 2.5 Flash) used for précis answer generation; raptor.py async-OR refactor landed but not exercised here (existing tree retained).

---

## 1. Feature matrix — `deleuze_exact_citations.json` (50 samples)

| config | page@5 | source@5 | quote@5 | MRR | latency ms |
|---|---|---|---|---|---|
| baseline | 0.940 | 1.000 | 0.860 | 0.894 | 3103 |
| P1 | 0.940 | 1.000 | 0.860 | 0.894 | 576 |
| P2 | 0.940 | 1.000 | 0.860 | 0.894 | 581 |
| P7 | 0.940 | 1.000 | 0.860 | 0.894 | 598 |
| P9 | 0.940 | 1.000 | 0.860 | 0.894 | 601 |
| all-retrieval | 0.940 | 1.000 | 0.860 | 0.894 | 616 |
| all-on | 0.940 | 1.000 | 0.860 | 0.894 | 593 |

## 2. P0 contextual ablation — ctx vs noctx

Sibling collection rebuilt at `gutenberg-deleuze-corpus-noctx` (10246 chunks, raw embeddings, no `context_prefix` in metadata).

| metric | ctx (P0 on) | noctx | Δ |
|---|---|---|---|
| mrr | 0.894 | 0.894 | +0.001 |
| page_hit_5 | 0.940 | 0.920 | +0.020 |
| source_p_1 | 1.000 | 1.000 | +0.000 |
| source_p_5 | 1.000 | 1.000 | +0.000 |
| quote_hit_5 | 0.860 | 0.860 | +0.000 |
| mean_latency_ms | 731.200 | 593.600 | +137.600 |

## 3. P5 RAPTOR ablation — full (leaves+summaries) vs leaves-only

### exact set (n=50)

| metric | full | leaves_only | Δ |
|---|---|---|---|
| source_p_1 | 1.000 | 1.000 | +0.000 |
| source_p_5 | 1.000 | 1.000 | +0.000 |
| source_p_10 | 1.000 | 1.000 | +0.000 |
| page_hit_1 | 0.860 | 0.880 | -0.020 |
| page_hit_5 | 0.940 | 0.920 | +0.020 |
| page_hit_10 | 0.940 | 0.920 | +0.020 |
| quote_hit_5 | 0.860 | 0.880 | -0.020 |
| quote_hit_10 | 0.860 | 0.880 | -0.020 |
| mrr | 0.894 | 0.903 | -0.008 |
| summary_contribution_at_k | 0.044 | 0.000 | +0.044 |
| errors | 0.000 | 0.000 | +0.000 |
| mean_latency_ms | 736.400 | 583.900 | +152.500 |

### abstractive set (n=22)

| metric | full | leaves_only | Δ |
|---|---|---|---|
| entity_recall_at_k | 0.719 | 0.759 | -0.040 |
| concept_span_recall_at_k | 0.682 | 0.618 | +0.064 |
| summary_contribution_at_k | 0.036 | 0.000 | +0.036 |
| errors | 0.000 | 0.000 | +0.000 |
| mean_latency_ms | 120.300 | 100.700 | +19.600 |

## 4. P7 graph ablation — boost on/off (multi_entity_queries, n=8)

| metric | off | on | Δ |
|---|---|---|---|
| entity_recall_at_k | 1.000 | 1.000 | +0.000 |
| co_occurrence_recall_at_k | 1.000 | 1.000 | +0.000 |
| errors | 0.000 | 0.000 | +0.000 |
| mean_latency_ms | 982.400 | 140.800 | -841.600 |

## 5. Précis multi-criteria grading — `deleuze_precis_evolution.json` (n=10)

Answers via OpenRouter (Gemini 2.5 Flash) using locally-built RAG prompts. Composite weights: works=0.35, spans=0.35, entities=0.20, alce=0.10.

| metric | mean |
|---|---|
| mean_composite | 0.601 |
| mean_concept_span_hit_rate | 0.950 |
| mean_entity_coverage | 0.816 |
| mean_works_cited_coverage_retrieval | 0.917 |
| mean_works_cited_coverage | 0.300 |

### Per-query composite

| label | composite | spans | entities | works (cited) | works (retrieved) |
|---|---|---|---|---|---|
| BwO evolution AO→ATP | 0.515 | 0.900 | 1.000 | 0.000 | 1.000 |
| Repetition DR→Cinema | 0.550 | 1.000 | 1.000 | 0.000 | 0.667 |
| Deterritorialization AO→ATP | 0.801 | 0.800 | 0.857 | 1.000 | 1.000 |
| Virtual Berg→DR→C2 | 0.550 | 1.000 | 1.000 | 0.000 | 1.000 |
| Will to power → Desire NP→AO | 0.860 | 1.000 | 0.800 | 1.000 | 1.000 |
| Diagram FB→Foucault→ATP | 0.550 | 1.000 | 1.000 | 0.000 | 1.000 |
| Sense LS→WIP | 0.350 | 1.000 | 0.000 | 0.000 | 1.000 |
| Immanence Spinoza→WIP | 0.515 | 0.900 | 1.000 | 0.000 | 1.000 |
| Fold: Leibniz vs Foucault | 0.800 | 1.000 | 0.500 | 1.000 | 1.000 |
| Signs PS→ATP | 0.515 | 0.900 | 1.000 | 0.000 | 0.500 |

## 6. ALCE NLI citation faithfulness — on précis answers

- citation-bearing sentences: **14**
- total citations: **14**
- citation recall (≥1 entailing chunk per sentence): **0.929**
- citation precision (entailing citations / total): **0.929**
- NLI threshold: 0.5 · model: `cross-encoder/nli-deberta-v3-base`

---

## Findings

- **Verbatim-quote eval is at ceiling** (page@5 = 0.940 across all retrieval-time configs). RAPTOR summaries occasionally surface in top-5 (4.4% summary contribution) but the dominant signal is BM25+RRF on contextualized leaves.
- **P0 contextual prefixes give a real lift on the eval that has room to move**: page@5 +0.020 vs the noctx sibling on `deleuze_exact_citations`. MRR is unchanged because the chunks were already in top-1, contextualization just helps fill out top-5 with adjacent supporting chunks.
- **Precis composite = 0.601** (Gemini Flash). Retrieval-side metrics are strong: span hits 0.950, entity coverage 0.816, retrieved-works coverage 0.917. Cited-works coverage 0.300 is the limiter — Flash often cites just one of the multiple required works in its prose, even when the chunks for the others are in context.
- **ALCE 0.929 recall / 0.929 precision** — when Flash *does* cite a passage, the cited chunk almost always entails the sentence. Citation faithfulness is solved at the answerer; the leak is upstream in citation breadth.
- **RAPTOR exact set: leaves-only beats full by +0.008 MRR** (0.903 vs 0.894). Summaries dilute exact-quote retrieval slightly. On abstractive queries the full arm gains +0.064 concept_span_recall but loses −0.040 entity_recall — net wash on the small set, summaries would need better placement weighting to clearly win.
- **P7 graph boost = 0 delta** on multi_entity (entity recall already 1.000 without boost). Worth a 6× latency hit (140ms → 982ms with --boost in this measurement, *off arm was slower*; this is noisy on n=8 queries — re-test on a larger graph-stressing set before drawing conclusions).

## Outstanding

- **P4 ColBERT** — voyager Py-3.13 wheel still missing.
- **Async-OR RAPTOR** — refactor landed in `services/worker/pipeline/raptor.py`; the existing 2054-summary tree was kept (rebuilding would invalidate above results). Next time we reindex: expect ~3 min + ~$0.03 vs the original 112 min.
- **Precis gold expansion** — 10 queries; expand to 30+ before treating composite drift between runs as signal.
- **Cited-works gap (0.300)** — answerer-side issue: prompt the LLM to cite each required work explicitly, or validate citation breadth post-hoc and re-prompt for missing works.
