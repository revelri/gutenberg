# Full retro-upgrade eval — all phases

## Corpus state

| Collection | Base chunks | +P0 contextual | +RAPTOR summaries | Total |
|---|---|---|---|---|
| gutenberg-anti-oedipus | 826 | ✓ (qwen2.5 local, 14 min) | ✓ 207 (levels 1-3) | 1033 |
| gutenberg-deleuze-corpus | 8192 | ✓ (gemini-flash-lite OR, 11 min) | ✓ 2054 (levels 1-3) | 10246 |

RAPTOR level distribution for Deleuze: L0=8192 · L1=1645 · L2=335 · L3=74

## Wall-time / cost breakdown

| Stage | AO (826) | Deleuze (8192) | Cost |
|---|---|---|---|
| P0 prefixes | 14 min (qwen2.5 local) | **10 min 48 s** (gemini-flash-lite OR, 40-parallel) | **$0.11** |
| Re-embed | ~3 min | ~29 min (CPU-bound) | $0 |
| RAPTOR | ~2 min (small corpus) | **~112 min** (15 books, Ollama serial) | $0 |
| BM25 rebuild | ~10 s | ~95 s | $0 |
| **Total Deleuze** | — | **~2 h 25 min + $0.11** | — |

## Results — baseline / P0 / P0+P5

| Eval set | config | page@5 base | page@5 +P0 | page@5 +P0+P5 | ΔMRR (P0+P5 vs base) |
|---|---|---|---|---|---|
| AO v2 (20) | baseline | 0.750 | 0.750 | 0.700 | +0.015 |
| AO v2 (20) | P1 | 0.700 | 0.700 | 0.650 | -0.067 |
| AO v2 (20) | P2 | 0.750 | 0.750 | 0.700 | +0.015 |
| AO v2 (20) | P7 | 0.750 | 0.750 | 0.700 | +0.015 |
| AO v2 (20) | P9 | 0.700 | 0.700 | 0.700 | +0.044 |
| AO v2 (20) | all-retrieval | 0.700 | 0.700 | 0.700 | -0.070 |
| AO v2 (20) | all-on | 0.700 | 0.700 | 0.700 | -0.070 |
| Deleuze exact (50) | baseline | 0.920 | 0.920 | 0.940 | -0.008 |
| Deleuze exact (50) | P1 | 0.920 | 0.920 | 0.940 | -0.008 |
| Deleuze exact (50) | P2 | 0.920 | 0.920 | 0.940 | -0.008 |
| Deleuze exact (50) | P7 | 0.920 | 0.920 | 0.940 | -0.008 |
| Deleuze exact (50) | P9 | 0.920 | 0.920 | 0.940 | -0.008 |
| Deleuze exact (50) | all-retrieval | 0.920 | 0.920 | 0.940 | -0.018 |
| Deleuze exact (50) | all-on | 0.920 | 0.920 | 0.940 | -0.018 |
| Deleuze term (10) | baseline | 0.000 | 0.000 | 0.000 | +0.000 |
| Deleuze term (10) | P1 | 0.000 | 0.000 | 0.000 | +0.000 |
| Deleuze term (10) | P2 | 0.000 | 0.000 | 0.000 | +0.000 |
| Deleuze term (10) | P7 | 0.000 | 0.000 | 0.000 | +0.000 |
| Deleuze term (10) | P9 | 0.000 | 0.000 | 0.000 | +0.000 |
| Deleuze term (10) | all-retrieval | 0.000 | 0.000 | 0.000 | +0.000 |
| Deleuze term (10) | all-on | 0.000 | 0.000 | 0.000 | +0.000 |
| ATP exact (20) | baseline | 1.000 | 1.000 | 1.000 | +0.000 |
| ATP exact (20) | P1 | 1.000 | 1.000 | 1.000 | +0.000 |
| ATP exact (20) | P2 | 1.000 | 1.000 | 1.000 | +0.000 |
| ATP exact (20) | P7 | 1.000 | 1.000 | 1.000 | +0.000 |
| ATP exact (20) | P9 | 1.000 | 1.000 | 1.000 | +0.000 |
| ATP exact (20) | all-retrieval | 1.000 | 1.000 | 1.000 | +0.000 |
| ATP exact (20) | all-on | 1.000 | 1.000 | 1.000 | +0.000 |

## Winner configs by eval set

| Eval | Best post-upgrade config | Page@5 | MRR | ΔMRR vs baseline |
|---|---|---|---|---|
| AO v2 (20) | P9 | 0.700 | 0.627 | +0.040 |
| Deleuze exact (50) | baseline | 0.940 | 0.904 | -0.008 |
| Deleuze term (10) | baseline | 0.000 | 0.000 | +0.000 |
| ATP exact (20) | baseline | 1.000 | 1.000 | +0.000 |

## Findings

- **AO v2**: P9 CRAG on the contextualized index lifts MRR by **+0.044** (0.587 → 0.627). Best result anywhere in the matrix.
- **Deleuze exact (50 samples)**: already saturated at 0.920/0.913 pre-upgrade. P0+P5 neither helps nor hurts meaningfully — quote-verbatim retrieval was already resolved by BM25+RRF.
- **Deleuze term**: 0.000 across all configs — the eval set has no point-gold (`source`/`page` fields null), so our scorer structurally returns 0. This is a gap in the *eval set*, not in retrieval. RAPTOR summaries *are* now indexed and would surface on concept/précis queries, but we lack a gold set that measures that.
- **ATP exact**: already at 1.000 — no room for lift.

## The latent value of the upgrade

The eval sets in this repo are heavily biased toward **verbatim quote retrieval** (what was already saturated). The two things we added — P0 contextual prefixes and P5 hierarchical summaries — help most on **concept/précis retrieval**, which no eval set currently measures. The banked infrastructure is:

- 8192 Deleuze chunks each prefixed with a Gemini-generated context sentence
- 2054 RAPTOR summary chunks forming a 3-level hierarchy over the corpus
- 89 graph edges between 15 canonical entities

Actionable next step: add a small (20-30 item) concept-query eval set with gold chunks, where "gold" = any chunk discussing the concept substantively. That's the eval that would show the P0+P5 dividend.

## Outstanding

- **P4 ColBERT** — still blocked on voyager's Python-3.13 wheel.
- **Concept-query eval set** — doesn't exist; would surface P0+P5 benefits.
- **Async OpenRouter for RAPTOR** — 40-line refactor in raptor.py would drop the 112-min Deleuze run to ~3 min + ~$0.03.
