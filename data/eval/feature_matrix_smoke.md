# Feature-matrix eval — `deleuze_exact_citations.json`

## Feature taxonomy

| P | Class | Name | Flag | Reindex required |
|---|---|---|---|---|
| P0 | INDEX | Contextual chunking | `feature_contextual_chunking` | yes |
| P1 | RETRIEVAL | SpaCy gazetteer + alias map | `feature_entity_gazetteer` | no |
| P2 | RETRIEVAL | rapidfuzz + anchor validation | `feature_rapidfuzz_verify` | no |
| P3 | INDEX | Modal chunks (tables, equations) | `feature_modal_chunks` | yes |
| P4 | INDEX | ColBERTv2 late-interaction | `feature_colbert_retrieval` | yes |
| P5 | INDEX | RAPTOR summary tree | `feature_raptor` | yes |
| P6 | EVAL_ONLY | ALCE NLI citation eval | `_(script)_` | no |
| P7 | RETRIEVAL | Graph-lite entity neighborhood | `feature_graph_boost` | no |
| P8 | ANSWER | VLM-enhanced answer | `feature_vlm_answer` | no |
| P9 | RETRIEVAL | CRAG-lite gate + rewrite | `feature_crag` | no |
| P10 | EVAL_ONLY | Contextcite offline audit | `_(script)_` | no |
| P11 | EVAL_ONLY | Reindex automation + manifest | `_(script)_` | no |
| P12 | EVAL_ONLY | Structured telemetry | `telemetry_enabled` | no |


## Configurations to evaluate

- **baseline** — baseline (no flags)
- **P1** — P1 only — SpaCy gazetteer + alias map
- **P2** — P2 only — rapidfuzz + anchor validation
- **P7** — P7 only — Graph-lite entity neighborhood
- **P9** — P9 only — CRAG-lite gate + rewrite
- **all-retrieval** — all retrieval-time flags on
- **all-on** — all flags on


## Results — `deleuze_exact_citations.json` (3 samples)

| config | page_hit_1 | page_hit_5 | page_hit_10 | source_p_5 | quote_hit_5 | mrr | mean_latency_ms | Δ page_hit_5 |
|---|---|---|---|---|---|---|---|---|
| baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 34866.300 | +0.000 |
| P1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 305.900 | +0.000 |
| P2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 314.200 | +0.000 |
| P7 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 314.800 | +0.000 |
| P9 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 331.700 | +0.000 |
| all-retrieval | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 321.700 | +0.000 |
| all-on | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 315.900 | +0.000 |


Δ columns are signed lift vs baseline on the same eval set.
