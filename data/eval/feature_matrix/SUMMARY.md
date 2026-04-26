# Full retro-upgrade eval — OpenRouter-accelerated run

## Upgrades applied

| Collection | P0 contextual | P1 tags | P5 RAPTOR | P7 graph | Wall time | Cost |
|---|---|---|---|---|---|---|
| gutenberg-anti-oedipus (826) | ✓ qwen2.5 (local) | ✓ | ✓ 207 summaries | ✓ 35 edges | ~14 min | $0 |
| gutenberg-deleuze-corpus (8192) | ✓ gemini-2.5-flash-lite (OpenRouter, 40-parallel) | ✓ | ⧗ pending | ✓ 89 edges | ~11 min prefixes + ~29 min re-embed | ~$0.11 |

## Speedup vs local Ollama

| Pass | Ollama (serial) | OpenRouter (40-parallel) | Speedup |
|---|---|---|---|
| P0 contextual prefixes, Deleuze 8192 chunks | ~170 min extrapolated | 10 min 48 s measured | 15.7× |
| Per-prefix cost | $0 | ~$0.0000135 | — |

## Results — pre vs post retro-upgrade

| Eval set | N | config | page@5 pre | page@5 post | Δ | MRR pre | MRR post | Δ MRR |
|---|---|---|---|---|---|---|---|---|
| AO v2 | 20 | baseline | 0.750 | 0.700 | -0.050 | 0.587 | 0.603 | +0.015 |
| AO v2 | 20 | P1 | 0.700 | 0.650 | -0.050 | 0.583 | 0.516 | -0.067 |
| AO v2 | 20 | P2 | 0.750 | 0.700 | -0.050 | 0.587 | 0.603 | +0.015 |
| AO v2 | 20 | P7 | 0.750 | 0.700 | -0.050 | 0.587 | 0.603 | +0.015 |
| AO v2 | 20 | P9 | 0.700 | 0.700 | +0.000 | 0.583 | 0.627 | +0.044 |
| AO v2 | 20 | all-retrieval | 0.700 | 0.700 | +0.000 | 0.583 | 0.513 | -0.070 |
| AO v2 | 20 | all-on | 0.700 | 0.700 | +0.000 | 0.583 | 0.513 | -0.070 |
| Deleuze exact | 50 | baseline | 0.920 | 0.920 | +0.000 | 0.913 | 0.909 | -0.003 |
| Deleuze exact | 50 | P1 | 0.920 | 0.920 | +0.000 | 0.913 | 0.912 | -0.000 |
| Deleuze exact | 50 | P2 | 0.920 | 0.920 | +0.000 | 0.913 | 0.909 | -0.003 |
| Deleuze exact | 50 | P7 | 0.920 | 0.920 | +0.000 | 0.913 | 0.909 | -0.003 |
| Deleuze exact | 50 | P9 | 0.920 | 0.920 | +0.000 | 0.913 | 0.909 | -0.003 |
| Deleuze exact | 50 | all-retrieval | 0.920 | 0.920 | +0.000 | 0.913 | 0.902 | -0.010 |
| Deleuze exact | 50 | all-on | 0.920 | 0.920 | +0.000 | 0.913 | 0.902 | -0.010 |
| Deleuze term | 10 | baseline | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Deleuze term | 10 | P1 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Deleuze term | 10 | P2 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Deleuze term | 10 | P7 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Deleuze term | 10 | P9 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Deleuze term | 10 | all-retrieval | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Deleuze term | 10 | all-on | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| ATP exact | 20 | baseline | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| ATP exact | 20 | P1 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| ATP exact | 20 | P2 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| ATP exact | 20 | P7 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| ATP exact | 20 | P9 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| ATP exact | 20 | all-retrieval | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| ATP exact | 20 | all-on | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |

## Outstanding

- **P5 RAPTOR on Deleuze** — pending. Ollama path would take ~30 min serial; wiring async OpenRouter into raptor.py requires a ~40-line refactor (per-level parallel cluster summarize). Deferred.
- **P4 ColBERT** — blocked on voyager Python-3.13 wheel.
- **P0 on Deleuze added to image** — retro script updates Chroma in place; next re-ingest of any Deleuze PDF would lose prefixes unless the worker pipeline runs with FEATURE_CONTEXTUAL_CHUNKING=true and ANTHROPIC or OpenRouter keys in env.
