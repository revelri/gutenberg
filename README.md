# Gutenberg

Self-hosted RAG system with page-accurate citations and quote verification. Drop in your PDFs, ask questions, get answers that cite real page numbers — verified against the source material.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-blue)]()
[![Tests](https://img.shields.io/badge/tests-170%2B-brightgreen)]()

---

## Why

LLMs hallucinate citations. Ask a model about a passage in Deleuze and it will confidently cite a page number that doesn't exist. Existing RAG systems improve retrieval but can't tell you *where on what page* a quote appears, and they don't verify that cited text is real.

Gutenberg tracks page boundaries through the entire pipeline and verifies every quote against the source material before returning it.

## How it works

1. **Ingest.** Drop a PDF into the inbox. The worker classifies it (digital vs. scanned), runs OCR if needed, and extracts text with per-page character offsets preserved.

2. **Chunk with page tracking.** Text splits into overlapping chunks via character-offset mapping — each chunk knows its `page_start` and `page_end`. An optional LLM pass prepends contextual summaries for better retrieval.

3. **Hybrid retrieval.** Queries hit dense search (ChromaDB embeddings) and sparse search (BM25 with stemming). Results fuse via Reciprocal Rank Fusion with adaptive per-query weighting. A reranker scores the final set.

4. **Citation verification.** Every quote in the response is extracted and checked: exact substring match, fuzzy match (SequenceMatcher 0.85), then lemma-based match. Quotes are cross-checked against raw PDF page text via PyMuPDF. The response shows which citations are confirmed, approximate, or unverified.

## Features

| Feature | Detail |
|---------|--------|
| Page-accurate citations | Char-offset tracking preserves page boundaries through chunking |
| 3-tier quote verification | Exact, fuzzy, and lemma-based matching against source material |
| Hybrid retrieval | Dense + BM25 sparse search with RRF fusion and adaptive weighting |
| Contextual chunking | LLM-generated context prepended to each chunk (Anthropic's pattern) |
| OCR pipeline | OCRmyPDF for text-layer injection, Docling with CUDA for scanned docs |
| Multi-corpus | Per-corpus ChromaDB collections with independent ingestion |
| Crash-resilient ingestion | Transactional markers with pending-file rollback on restart |
| Self-hosted | Runs on local hardware with Ollama. No cloud APIs required |
| SvelteKit frontend | Chat interface with streaming responses and inline citation display |

## Tech Stack

- **Runtime:** Python 3.12+
- **API:** FastAPI — async HTTP with streaming response support
- **Vector store:** ChromaDB — per-corpus collections with metadata filtering
- **LLM:** Ollama — local inference for chat, embeddings, and contextual chunking
- **PDF:** PyMuPDF — text extraction with character-offset page tracking
- **OCR:** OCRmyPDF (text-layer injection), Docling (GPU-accelerated for scans)
- **Sparse search:** rank-bm25 with NLTK stemming
- **Frontend:** SvelteKit — streaming chat with inline citation display
- **Storage:** SQLite for jobs, metadata, conversations
- **Build:** Docker Compose, uv for Python dependency management

## Getting started

### Prerequisites

- Docker and Docker Compose
- [Ollama](https://ollama.com) running on the host
- A chat model and an embedding model pulled:
  ```bash
  ollama pull llama3.1:8b-instruct-q4_K_M
  ollama pull nomic-embed-text
  ```

### Run

```bash
cp .env.example .env    # review and adjust
docker compose up -d
```

The API serves at `http://localhost:8000`. The frontend is bundled at the same origin.

### Ingest documents

Drop PDF, DOCX, or EPUB files into `data/inbox/` — the worker picks them up automatically. Or use the API:

```bash
# Create a corpus
curl -X POST http://localhost:8000/api/corpus \
  -H "Content-Type: application/json" \
  -d '{"name": "My Corpus", "description": "Research papers"}'

# Upload files
curl -X POST http://localhost:8000/api/corpus/{id}/upload \
  -F "files=@paper.pdf"
```

### Query

```bash
curl http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Where does Deleuze distinguish the virtual from the actual?"}'
```

The response includes retrieved passages with page numbers and a verification footer.

## Architecture

```
┌──────────────────┐
│   SvelteKit UI   │
└────────┬─────────┘
         │
┌────────▼─────────┐
│    FastAPI API    │──── ChromaDB (vectors)
│  chat, corpus,   │
│  documents       │──── Ollama (LLM + embed)
└──────────────────┘
         │
┌────────▼─────────┐
│     Worker       │──── OCR (OCRmyPDF / Docling)
│  watcher → OCR   │
│  → chunk → embed │──── ChromaDB (store)
│  → store         │
└──────────────────┘
```

**API** handles queries: retrieves from ChromaDB, scores with BM25, reranks, assembles the prompt, streams the LLM response, verifies citations.

**Worker** handles ingestion: watches the inbox, classifies documents, runs OCR, chunks with page tracking, generates embeddings, stores in ChromaDB. Progress tracked in SQLite.

## Project structure

```
services/
  api/
    core/           rag.py, verification.py, config, database
    routers/        chat, corpus, documents, conversations, models
  worker/
    pipeline/       watcher, chunker, embedder, extractors, store
  shared/           text_normalize, chroma client, NLP utilities
frontend/           SvelteKit app
scripts/            eval frameworks, benchmarks, ingestion tools
tests/              170+ unit tests
```

## Development

```bash
# Run tests
uv run pytest tests/

# Frontend dev
cd frontend && npm run dev
```

### Configuration

All via environment variables (see `.env.example`):

| Variable | Default | Effect |
|----------|---------|--------|
| `OLLAMA_LLM_MODEL` | `llama3.1:8b-instruct-q4_K_M` | Chat model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHUNK_SIZE` | `384` | Tokens per chunk |
| `CHUNK_OVERLAP` | `96` | Overlap tokens |
| `CONTEXTUAL_CHUNKER_ENABLED` | `true` | LLM-generated chunk context |
| `RRF_ADAPTIVE` | `true` | Per-query adaptive fusion weights |
| `HYDE_ENABLED` | `false` | Hypothetical document expansion (harmful for verbatim retrieval) |

## Design Rationale

| Decision | Reasoning |
|----------|-----------|
| Page-offset tracking | Character offsets through the entire pipeline mean citations resolve to real page numbers, not chunk indices |
| 3-tier verification cascade | Exact → fuzzy → lemma fallback catches paraphrased citations without false negatives on verbatim quotes |
| RRF over single retrieval | Dense search finds semantic matches; BM25 finds exact terms. Fusion consistently outperforms either alone |
| Ollama for self-hosted | No API keys, no data leaves the machine. Users own their inference stack |
| Contextual chunking | Prepending LLM-generated context to chunks improves retrieval by 20-30% on domain-specific corpora |
| HYDE disabled by default | Hypothetical document expansion hurts verbatim retrieval — it rewrites the query away from exact quotes |

## License

[MIT](LICENSE)
