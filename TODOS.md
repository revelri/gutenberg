# Gutenborg — TODOs

## Re-ingest corpus after P0 page metadata ships
**Priority:** High (blocks P4 dogfooding)
**What:** After the page metadata pipeline (P0) lands, all existing ChromaDB data lacks `page_start`/`page_end` metadata. Need to clear the ChromaDB collection and `documents.jsonl` state file, then re-ingest the test corpus with the new pipeline.
**Context:** The Deleuze & Guattari corpus (founder's test data) will need fresh ingestion. The existing D&D sourcebooks can optionally be re-ingested or discarded depending on whether they're still useful for testing.
**Approach:** Either (a) wipe ChromaDB collection + state file and re-drop PDFs into inbox, or (b) build a `scripts/reingest.py` that reads `documents.jsonl`, finds the original files in `/data/processed/`, and re-runs the pipeline.
**Depends on:** P0 complete.

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

## BM25 memory scaling guard
**Priority:** Low (fine at current scale, needed for institutional deployments)
**What:** The BM25 index in `rag.py:42-62` loads ALL chunks from ChromaDB into memory. At 30 books (~50-100MB) this is fine. At 500+ books it will OOM the API container.
**Context:** We explicitly chose to leave this as-is for the single-user PoC (eng review Issue 6B). When multi-tenancy or larger corpora arrive, options include: (a) ChromaDB's native `where_document` text search, (b) paginated BM25 loading, (c) external search index (Elasticsearch/Meilisearch). Option (a) is the simplest and avoids a new dependency.
**Depends on:** Nothing — can be done anytime. Becomes urgent if corpus exceeds ~100 books.
