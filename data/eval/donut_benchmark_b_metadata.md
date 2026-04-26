# Benchmark B — Metadata Extraction

Gold: `data/eval/donut_metadata_gold.json` — 20 pages, 6 fields each. Ollama available: True.

## Normalized-match accuracy by field

| field | regex | spacy | ollama | donut |
|---|---|---|---|---|
| title | 25% | 5% | 90% | 20% |
| author | 75% | 40% | 80% | 0% |
| translator | 60% | 60% | 75% | 40% |
| publisher | 80% | 55% | 80% | 50% |
| year | 65% | 70% | 75% | 25% |
| isbn | 80% | 80% | 80% | 15% |
| **overall** | **64%** | **52%** | **80%** | **25%** |

## Latency (seconds per page)

| extractor | p50 | p95 | peak CUDA (MB) |
|---|---|---|---|
| regex | 0.00 | 0.00 | 0 |
| spacy | 0.03 | 0.06 | 0 |
| ollama | 1.71 | 2.20 | 0 |
| donut | 3.94 | 4.10 | 3669 |

## Decision rule

Donut wins a field if its normalized-match accuracy exceeds the best non-Donut extractor by ≥10 percentage points **and** its latency p95 is within 2× of that extractor.

- `title`: donut 20% vs best-other (ollama) 90% — Δ = -70pp ✗
- `author`: donut 0% vs best-other (ollama) 80% — Δ = -80pp ✗
- `translator`: donut 40% vs best-other (ollama) 75% — Δ = -35pp ✗
- `publisher`: donut 50% vs best-other (regex) 80% — Δ = -30pp ✗
- `year`: donut 25% vs best-other (ollama) 75% — Δ = -50pp ✗
- `isbn`: donut 15% vs best-other (regex) 80% — Δ = -65pp ✗

**Donut wins 0 / 6 fields.** Integration threshold: ≥2. Recommendation: DO NOT INTEGRATE.

