# Gutenborg — TODOs

## Re-ingest corpus with updated chunker
**Priority:** High (blocks P4 dogfooding)
**What:** P0 (page metadata) and the citation accuracy pipeline are now shipped. All existing ChromaDB data was chunked with the old pipeline (before char offset tracking fix). Need to re-ingest to get accurate `page_start`/`page_end` metadata.
**Context:** Run `python scripts/reingest.py` to clear ChromaDB and re-process all documents in `/data/processed/`. The Deleuze & Guattari corpus (founder's test data) should be the primary corpus after re-ingestion. The reingest script already exists.
**Approach:** `python scripts/reingest.py` — clears collection + state file, re-runs full pipeline on all processed documents.
**Depends on:** Nothing — ready to run now.

## Evaluate embedding models for dense philosophical text
**Priority:** Medium (after baseline accuracy established)
**What:** Compare retrieval quality of nomic-embed-text (current) against academic-focused models (SPECTER2, SciBERT, mxbai-embed-large, or instruction-tuned models) on the Deleuze & Guattari corpus.
**Context:** Design doc Open Question #2. nomic-embed-text is general-purpose and may underperform on queries like "where does Deleuze distinguish his use of deterritorialization from Anti-Oedipus." The P4 eval framework makes A/B comparison straightforward: run the same 20 gold-standard queries with different embedding models and compare retrieval relevance scores.
**Approach:** Use the eval framework to test each model. Requires re-embedding the corpus for each model (compute cost). Start with mxbai-embed-large (Ollama-native) as the easiest swap.
**Depends on:** P4 eval framework complete + gold-standard test set collected.

## Migrate JSONL files to database for multi-tenancy
**Priority:** P3 (not needed until multi-tenancy)
**What:** `data/bibliography.jsonl`, `data/bookmarks.jsonl`, and `data/state/documents.jsonl` are flat files. They work for single-user but don't support concurrent access, querying by field, or user isolation.
**Context:** Deliberate simplicity for the PoC (CEO review decision). When multi-tenancy arrives, migrate to SQLite (for self-hosted) or Postgres (for cloud SaaS). The bookmark CRUD endpoints (`/api/bookmarks`) and corpus stats endpoint (`/api/corpus/stats`) will need to swap their file I/O for DB queries.
**Approach:** SQLite is the path of least resistance — single file, no server, Python stdlib. Add a `data/gutenborg.db` with tables for `documents`, `bibliography`, and `bookmarks`.
**Depends on:** Multi-tenancy decision.

## ~~BM25 memory scaling guard~~ (DONE)
**Status:** Implemented in `rag.py:56-62`. BM25 index has a `bm25_max_chunks` threshold (default 50,000). When exceeded, falls back to ChromaDB `where_document` text search. The fallback search quality should be evaluated if corpus exceeds ~100 books at institutional scale.
