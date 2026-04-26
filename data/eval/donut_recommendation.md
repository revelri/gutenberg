# Donut DocVQA vs. gutenborg pipeline — final recommendation

**Model evaluated:** `p786/donut-base-finetuned-docvqa` (Swin image encoder + BART text decoder, fine-tuned on DocVQA).

**Original questions**

1. Does Donut outperform any element of the current pipeline?
2. Should it be integrated? Where? How? Why?

**Short answers: No. No.** Full reasoning below.

---

## 1. Per-stage verdict

Donut's input is `(page_image, question) → short answer string`. That shape only even plausibly competes with pipeline stages that (a) operate on a page image and (b) produce a short factoid. For every other stage the question is a category error — Donut does not work on already-extracted text, and cannot be made to.

| pipeline stage | current tech | Donut verdict | evidence |
|---|---|---|---|
| PDF detection (digital vs scanned) | PyMuPDF text-density heuristic (`detector.py`) | **n/a** | Donut has no classifier head; DocVQA fine-tune does not expose page-type labels. |
| OCR / body-text extraction | Surya → Docling → PyMuPDF → OCRmyPDF (`extractors.py:53-236`) | **underperforms, stipulated** | Body-text transcription is out-of-distribution for DocVQA; BART decoder caps generation far below a book page. Benchmark skipped by construction — user confirmed. |
| OCR quality gate | spaCy POS tagging, >15% "X" tokens flagged (`extractors.py:197-236`) | **n/a** | Operates on extracted strings, no image signal. |
| Chunking | Header/sentence-aware, tiktoken-sized (`chunker.py`) | **n/a** | Text-only; no image. |
| Embedding | `gte-large-en-v1.5`, 1024-dim (`embedder.py`) | **n/a** | Donut does not produce retrieval embeddings; using its Swin CLS token would require separate training. |
| Retrieval (dense + BM25 + RRF) | Chroma + BM25Okapi + RRF (`rag.py`) | **n/a** | Retrieval operates on chunks/embeddings; images are not in the index. |
| Generation | Ollama `llama3.1:8b-instruct-q4_K_M` | **n/a** | Donut cannot take a retrieved multi-chunk context + generate an answer with strict citation format. Different job. |
| Quote verification | substring / fuzzy / spaCy-lemma, page cross-check (`verification.py`) | **n/a** | Existing verifier operates on extracted text and is near-ceiling on `deleuze_exact_citations.json`. Skipped by construction. |
| **Bibliographic metadata extraction** (**Benchmark B**) | currently implicit — filename heuristics + PDF metadata | **underperforms — 25% vs 80% (Ollama), 64% (regex)** | See §2. |

## 2. Benchmark B results — the one stage where Donut was testable

Head-to-head on 20 title / copyright / combined pages across 11 Deleuze monographs, 6 fields each. Full data in `donut_benchmark_b_metadata.json`, table in `donut_benchmark_b_metadata.md`.

| field | regex | spacy NER | Ollama llama3.1 | **Donut** |
|---|---|---|---|---|
| title | 25% | 5% | **90%** | 20% |
| author | 75% | 40% | **80%** | 0% |
| translator | 60% | 60% | **75%** | 40% |
| publisher | 80% | 55% | **80%** | 50% |
| year | 65% | 70% | **75%** | 25% |
| isbn | **80%** | 80% | 80% | 15% |
| **overall** | **64%** | **52%** | **80%** | **25%** |

Latency p50 / p95 per page: regex 0.00 / 0.00 s · spaCy 0.03 / 0.06 s · Ollama 1.71 / 2.20 s · **Donut 3.94 / 4.10 s**.
Peak CUDA: Donut 3.7 GB (book-page resolution blows up Swin's activation memory).

**Donut wins 0 / 6 fields under the decision rule (≥10 pp delta + latency within 2× of best other).** The delta is negative on every field.

### Failure modes (from per-page audit)

- **Field cross-contamination** — the model puts the answer to one question in another field's slot: `"columbia university press"` returned for *author*, `"contents"` returned for *title*, `"zone books"` returned for *author*. Trained on business docs, it answers "what is this page about" instead of the requested field.
- **Text hallucination on typeset prose** — `"focusult"` for Foucault, `"nietzsche and philosophiy"`, `"anti- pedialsm and schizophrenia"`. Donut's decoder guesses when the page is not the DocVQA distribution (mostly forms/receipts).
- **Date hallucination on null fields** — on pages where `year` is genuinely absent, Donut fabricates a year (e.g. `"1978"` when gold is `null`). The probability of hallucination rose sharply on 2-up scanned spreads.
- **2-up spread handling** — 7/15 corpus PDFs are scanned as 2-page book-spread images. On these, Donut returns arbitrary text from either half.

The one bright spot: **translators** come through 40% of the time when phrased as "Translated by X" on a title page (matching a DocVQA-like key–value pattern).

## 3. Should it be integrated? No.

### Stage-by-stage

- **Body-text OCR** — No. Surya + Docling + OCRmyPDF is already the right tool class. Donut's DocVQA fine-tune is not an OCR model for philosophy books and cannot be turned into one by prompting.
- **Metadata extraction** — No. Ollama llama3.1 (already loaded for RAG) at 80% trounces Donut's 25%. Regex alone beats Donut on 5 of 6 fields with zero latency and zero GPU. The correct metadata-extraction design is a prompt on the Docling-extracted page text — not a second 3.7 GB vision model competing with Ollama and the reranker for GPU memory on an 8 GB card.
- **Retrieval / verification / generation** — Wrong modality. Not replaceable.

### Why the user's framing ("instead of an LLM and/or instead of spaCy") does not hold

- **Instead of spaCy:** spaCy operates on strings to produce POS/lemma/sentence boundaries consumed by the chunker and BM25. Donut takes images and produces strings. There is no overlap at all.
- **Instead of the RAG LLM:** the generation step takes a system prompt + retrieved chunks + user query → an answer with strict citation format. Donut takes one page image + one short question → one short answer. It cannot accept retrieved-chunk context, cannot follow a multi-paragraph system prompt, and has no mechanism for citation-format adherence.

### Concrete operational cost of integrating anyway

On this 8 GB RTX 3070 the first run of the benchmark (before restructuring) confirmed that Ollama (5.3 GB resident) and Donut (0.9 GB params + 2.8 GB activations at 200 dpi) **cannot coexist on GPU**. Every Donut call OOM'd. Shipping Donut into the pipeline would force Ollama-unload / Donut-load / Donut-unload / Ollama-reload per document — turning a 2 s Ollama call into a multi-second GPU churn, for worse accuracy.

## 4. If an integration is pursued despite this

(Documented for completeness; I do not recommend executing this.)

The only plausible non-absurd use of Donut on this corpus would be **scanned-PDF title-page metadata, gated by OCR-quality flag**:

```python
# services/worker/pipeline/extractors.py — proposed hook (~30 lines)
# AFTER extract_* returns, BEFORE chunking:
if metadata.get("ocr_quality") == "low" and segments and segments[0]["page"] == 1:
    from donut_metadata import donut_answer  # sidecar
    img = render_page(pdf_path, 1, dpi=200)
    # Only fields where Donut was non-terrible in Benchmark B:
    translator = donut_answer(img, "Who is the translator?")
    metadata.setdefault("translator", translator)
```

Even this is likely net-negative: Ollama on the same Docling text achieves 75% on translator at 2 s with no new GPU resident model. The only scenario where Donut would win is if the Docling text were so corrupt that an LLM call on it fails — in which case the correct fix is better OCR (Surya, or re-running Docling with tables/formulas enabled), not a DocVQA model.

## 5. Recommendation

**Do not integrate `p786/donut-base-finetuned-docvqa`.**

If the underlying motivation is "can we do bibliographic metadata extraction automatically?", the right follow-up is a ~50-line addition to `extractors.py` that runs the existing Ollama model over the Docling-extracted first-page text with a structured-extraction prompt, yielding ~80% accuracy on all six fields at the GPU cost of a single extra 2 s LLM call per document. That is the result Benchmark B's Ollama column already demonstrates.

## Artifacts

- `data/eval/donut_metadata_gold.json` — hand-labeled ground truth, 20 pages × 6 fields.
- `scripts/eval_donut_metadata.py` — sidecar evaluator (reproducible: `uv run python scripts/eval_donut_metadata.py`).
- `data/eval/donut_benchmark_b_metadata.json` — raw predictions + aggregates.
- `data/eval/donut_benchmark_b_metadata.md` — comparison table.
- `/home/revelri/.claude/plans/review-the-testing-and-lovely-gray.md` — the plan this executed.
