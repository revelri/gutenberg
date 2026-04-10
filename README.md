# Gutenberg

Self-hosted RAG system that retrieves passages from your document corpus and returns page-accurate citations. Built for researchers who need to trust that quoted text actually appears in the source material.

## The Problem

LLMs hallucinate citations. Ask a model about a passage in Deleuze and it will confidently cite a page number that doesn't exist. Existing RAG systems improve retrieval but still can't tell you *where on what page* a quote appears, and they have no mechanism to verify that the text they're citing is real.

Gutenberg solves both problems: it tracks page boundaries through the entire ingestion pipeline and verifies every quote against the source material before returning it to the user.

## How It Works

1. **Drop a PDF** into the inbox (or upload via API). The worker classifies it (digital vs. scanned), runs OCR if needed, and extracts text with per-page boundaries preserved.

2. **Chunking with page tracking.** Text is split into overlapping chunks using character-offset mapping so each chunk knows its `page_start` and `page_end`. An optional LLM pass prepends contextual summaries to each chunk for better retrieval.

3. **Hybrid retrieval.** Queries hit both dense search (ChromaDB embeddings) and sparse search (BM25 with stemming). Results are fused via Reciprocal Rank Fusion with adaptive per-query weighting. A reranker scores the final candidate set.

4. **Citation verification.** Every quote in the LLM's response is extracted and checked against the retrieved chunks (exact substring match, fuzzy match, then lemma-based match). Quotes are then cross-checked against the raw PDF page text. The response includes a verification footer showing which citations are confirmed, approximate, or unverified.

## Features

- **Page-accurate citations** -- char-offset tracking preserves page boundaries through chunking
- **3-tier quote verification** -- exact, fuzzy (SequenceMatcher), and lemma-based matching
- **Source PDF cross-check** -- verified quotes are checked against the actual page via PyMuPDF
- **Hybrid retrieval** -- dense + BM25 sparse search with RRF fusion and adaptive weighting
- **Contextual chunking** -- LLM-generated context prepended to each chunk for retrieval quality
- **Multi-corpus** -- per-corpus ChromaDB collections with independent ingestion jobs
- **OCR pipeline** -- OCRmyPDF for text-layer injection, Docling with CUDA for scanned documents
- **Crash-resilient ingestion** -- transactional markers with pending-file rollback on restart
- **Fully self-hosted** -- runs on local hardware with Ollama, no cloud APIs required
- **SvelteKit frontend** -- chat interface with streaming responses and citation display

## Stack

| Component | Role |
|-----------|------|
| **FastAPI** | API server -- chat, corpus management, document upload |
| **ChromaDB** | Vector store for dense retrieval |
| **Ollama** | Local LLM (chat + embeddings + contextual chunking) |
| **PyMuPDF** | PDF text extraction and page rendering |
| **OCRmyPDF** | Text-layer injection for scanned PDFs |
| **Docling** | Heavy-duty OCR with GPU acceleration (CUDA) |
| **rank-bm25** | Sparse retrieval with NLTK stemming |
| **SvelteKit** | Frontend chat interface |
| **SQLite** | Job tracking, corpus metadata, conversation history |

## Getting Started

### Prerequisites

- Docker and Docker Compose
- [Ollama](https://ollama.com) running on the host
- A chat model and an embedding model pulled in Ollama:
  ```
  ollama pull llama3.1:8b-instruct-q4_K_M
  ollama pull nomic-embed-text
  ```

### Run

```bash
cp .env.example .env    # review and adjust
docker compose up -d
```

The API is at `http://localhost:8000`. The frontend is at `http://localhost:5173` (dev) or served by the API in production.

### Ingest Documents

Drop PDF, DOCX, or EPUB files into `data/inbox/` -- the worker picks them up automatically. Or use the API:

```bash
curl -X POST http://localhost:8000/api/corpus \
  -H "Content-Type: application/json" \
  -d '{"name": "My Corpus", "description": "Research papers"}'

# Upload files to the corpus
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
                    +------------------+
                    |   SvelteKit UI   |
                    +--------+---------+
                             |
                    +--------v---------+
                    |    FastAPI API    |
                    |  (chat, corpus,   |
                    |   documents)      |
                    +---+----------+---+
                        |          |
              +---------v--+  +----v------+
              |  ChromaDB   |  |  Ollama   |
              | (vectors)   |  | (LLM +    |
              +-------------+  |  embed)   |
                               +-----------+
                    +------------------+
                    |     Worker       |
                    | (watcher, OCR,   |
                    |  chunk, embed,   |
                    |  store)          |
                    +------------------+
```

The **API** handles queries: retrieves from ChromaDB, scores with BM25, reranks, assembles the prompt, streams the LLM response, and verifies citations.

The **Worker** handles ingestion: watches the inbox, classifies documents, runs OCR, chunks with page tracking, generates embeddings, and stores in ChromaDB. Progress is tracked in SQLite for the API's SSE endpoint.

## Development

### Project Structure

```
services/
  api/
    core/           # rag.py, verification.py, config, database
    routers/        # chat, corpus, documents, conversations, models
  worker/
    pipeline/       # watcher, chunker, embedder, extractors, store
  shared/           # text_normalize, chroma client, NLP utilities
frontend/           # SvelteKit app
scripts/            # eval frameworks, benchmarks, ingestion tools
tests/              # 170+ unit tests
```

### Running Tests

```bash
uv run pytest tests/
```

### Key Configuration

All configuration is via environment variables (see `.env.example`). Notable settings:

- `OLLAMA_LLM_MODEL` / `OLLAMA_EMBED_MODEL` -- which models Ollama serves
- `CHUNK_SIZE` / `CHUNK_OVERLAP` -- chunking parameters (tokens)
- `CONTEXTUAL_CHUNKER_ENABLED` -- LLM-generated chunk context (default: on)
- `RRF_ADAPTIVE` -- per-query adaptive fusion weights (default: on)
- `HYDE_ENABLED` -- hypothetical document expansion (default: off, harmful for verbatim retrieval)

## License

MIT
