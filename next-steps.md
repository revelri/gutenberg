# Gutenberg Beta 1 — Next Steps

Exhaustive implementation guide, TDD-first, phased by dependency order.
Every task includes: what to test, what to build, and what to verify.

---

## Priority 0: Blockers (must-fix before anything else runs)

### 0.1 Missing Python dependencies

**Test first:**
```bash
cd services/api && python -c "import aiosqlite; print('OK')"
cd services/worker && python -c "import ebooklib; from bs4 import BeautifulSoup; print('OK')"
```
Both should fail. Fix:

**API** (`services/api/requirements.txt`) — add:
```
aiosqlite>=0.20
```

**Worker** (`services/worker/requirements.txt`) — add:
```
ebooklib>=0.18
beautifulsoup4>=4.12
```

**Root** (`pyproject.toml`) — add:
```
ebooklib>=0.18
beautifulsoup4>=4.12
```

Re-run test commands. Both must print `OK`.

### 0.2 Static files mount for frontend

The API must serve the built SvelteKit frontend. Without this, users can't load the app from the API URL.

**File:** `services/api/main.py`

Add at the VERY END (after all router includes, before nothing):
```python
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")
```

**Test:** Build frontend (`cd frontend && npm run build`), copy `frontend/build/` to `services/api/static/`, start the API, and `curl http://localhost:8000/` should return HTML containing "Gutenberg".

### 0.3 Frontend vite proxy for development

During development, the SvelteKit dev server needs to proxy API calls to FastAPI.

**File:** `frontend/vite.config.ts`

```typescript
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
    plugins: [sveltekit()],
    server: {
        proxy: {
            '/api': 'http://localhost:8000',
            '/v1': 'http://localhost:8000',
        }
    }
});
```

**Test:** `npm run dev`, open browser, corpus list should load from the API.

---

## Phase A: Make the existing UI functional

Everything here fixes things that are built but not wired up.

### A.1 Corpus modal — state bindings and file upload

**Tests to write first** (`tests/test_frontend_api.py` — integration):
```python
def test_create_corpus():
    """POST /api/corpus with name and tags → returns corpus with collection_name."""
    resp = client.post("/api/corpus", data={"name": "Test Corpus", "tags": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["collection_name"].startswith("gutenberg-")

def test_upload_rejects_invalid_types():
    """POST /api/corpus/{id}/ingest rejects .txt files."""
    # create corpus first, then upload a .txt file
    resp = client.post(f"/api/corpus/{cid}/ingest", files=[("files", ("bad.txt", b"hello", "text/plain"))])
    assert resp.status_code == 400

def test_upload_accepts_pdf():
    """POST /api/corpus/{id}/ingest accepts .pdf files."""
    pdf_bytes = b"%PDF-1.4 ..."  # minimal valid PDF
    resp = client.post(f"/api/corpus/{cid}/ingest", files=[("files", ("test.pdf", pdf_bytes, "application/pdf"))])
    assert resp.status_code == 200
    assert resp.json()["total"] == 1
```

**Implementation** (`frontend/src/routes/+page.svelte`):

The corpus modal needs:
- `let corpusName = $state('')` and `let corpusTags = $state('')` bound to inputs
- `let selectedFiles: File[] = $state([])` for the file list
- `ondrop` / `ondragover` / `ondragleave` handlers on the drop zone
- `<input type="file" multiple accept=".pdf,.epub,.docx">` hidden, triggered by Browse button
- File list display (name, size, remove button per file)
- Build button onclick: `createCorpus(name, tags)` → `uploadDocuments(id, files)` → `streamIngestionStatus(id, ...)` → close modal on done
- Progress bar bound to `IngestionStatus.completed_files / total_files`
- Validation: name required, at least 1 file
- Error display if upload or ingestion fails

### A.2 Mode selector sync with conversation

When user selects a conversation, the mode dropdown must update to match `conversation.mode`. Currently `selectConversation()` sets `selectedMode = conv.mode` which is correct, but mode changes in the dropdown don't update the active conversation's mode.

**Fix:** Disable mode dropdown when a conversation is active (mode is set at creation time, not changeable). Or: create a new conversation when mode changes.

### A.3 Chat error feedback

When RAG retrieval returns 0 chunks, the LLM gets empty context and hallucinates freely.

**File:** `services/api/routers/conversations.py`

After the retrieve call, if chunks is empty:
```python
if not chunks:
    yield {"event": "warning", "data": json.dumps({
        "message": "No relevant passages found in the corpus for this query."
    })}
```

Frontend should display this warning above the response.

### A.4 Citation style passthrough

**File:** `services/api/core/modes.py`

Each prompt builder should accept `citation_style` and append a formatting instruction:
```
## Citation format
Use {style_name} citation style for all references.
```

**File:** `services/api/routers/conversations.py`

Pass `conv["citation_style"]` to the prompt builder.

---

## Phase B: "Go to Passage" PDF viewer

### B.1 PDF serving endpoint

**Tests first** (`tests/test_pdf_endpoint.py`):
```python
def test_pdf_endpoint_returns_pdf():
    """GET /api/pdf/{filename} returns application/pdf with correct headers."""
    resp = client.get("/api/pdf/test.pdf")
    assert resp.headers["content-type"] == "application/pdf"
    assert "accept-ranges" in resp.headers

def test_pdf_endpoint_supports_range():
    """GET /api/pdf/{filename} with Range header returns 206."""
    resp = client.get("/api/pdf/test.pdf", headers={"Range": "bytes=0-1023"})
    assert resp.status_code == 206

def test_pdf_endpoint_404():
    """GET /api/pdf/nonexistent.pdf returns 404."""
    resp = client.get("/api/pdf/nonexistent.pdf")
    assert resp.status_code == 404

def test_pdf_endpoint_rejects_path_traversal():
    """GET /api/pdf/../../../etc/passwd returns 400."""
    resp = client.get("/api/pdf/..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code in (400, 404)
```

**Implementation** (`services/api/routers/pdf.py`):
```python
@router.get("/api/pdf/{filename}")
async def serve_pdf(filename: str, request: Request):
    # Sanitize filename (reject path traversal)
    # Look in /data/processed/ first, then /data/inbox/
    # Return FileResponse with media_type="application/pdf"
    # Support Range header for streaming
```

### B.2 Page image fallback endpoint

**Tests first:**
```python
def test_page_image_returns_jpeg():
    """GET /api/pdf/{filename}/page/1/image returns JPEG."""
    resp = client.get("/api/pdf/test.pdf/page/1/image")
    assert resp.headers["content-type"] == "image/jpeg"

def test_page_image_with_highlight():
    """GET /api/pdf/{filename}/page/1/image?highlight=TEXT adds highlight annotation."""
    resp = client.get("/api/pdf/test.pdf/page/1/image?highlight=rhizome")
    assert resp.status_code == 200
    # Image should be different from non-highlighted version
```

**Implementation** (`services/api/routers/pdf.py`):
```python
@router.get("/api/pdf/{filename}/page/{page_num}/image")
async def page_image(filename: str, page_num: int, highlight: str = "", dpi: int = 200):
    # Open PDF with fitz
    # Get page (0-indexed internally, 1-indexed in URL)
    # If highlight: page.search_for(highlight) → page.add_highlight_annot(rects)
    # Render: page.get_pixmap(dpi=dpi)
    # Return as JPEG StreamingResponse
```

### B.3 On-demand page search for Surya chunks (page_start=0)

**Tests first:**
```python
def test_find_page_for_quote():
    """Given a quote and PDF, find the page number it appears on."""
    page = find_quote_page("test.pdf", "rhizome connects any point")
    assert page > 0

def test_find_page_returns_none_for_missing():
    """If quote not in PDF, return None."""
    page = find_quote_page("test.pdf", "this text does not exist anywhere")
    assert page is None
```

**Implementation** (`services/api/core/pdf_search.py`):
```python
def find_quote_page(filename: str, quote: str, search_dirs=None) -> int | None:
    """Search PDF pages for a quote, return 1-indexed page number."""
    # Open PDF with fitz
    # Normalize quote (lowercase, collapse whitespace)
    # For each page: search_for(quote[:60])
    # If found, return page_num + 1
    # If not found with full prefix, try sliding window with shorter prefixes
    # Return None if not found
```

### B.4 Preserve OCR'd PDFs during ingestion

**File:** `services/worker/pipeline/watcher.py`

Currently OCRmyPDF output goes to `_ocr_tmp/` and is discarded. Change `process_file()` and `process_corpus_job()` to copy the OCR'd PDF to `/data/processed/` with a `-ocr` suffix:
```python
ocr_dest = self.processed / f"{fname.rsplit('.', 1)[0]}-ocr.pdf"
shutil.copy2(str(extract_path), str(ocr_dest))
```

Update the PDF serving endpoint to prefer `-ocr.pdf` over the original (it has the text layer for search).

### B.5 Bundle PDF.js viewer

Download the PDF.js prebuilt release and place it in `services/api/static/pdfjs/`:
```
static/pdfjs/
  web/
    viewer.html
    viewer.js
    viewer.css
  build/
    pdf.js
    pdf.worker.js
```

Access via: `/pdfjs/web/viewer.html?file=/api/pdf/FILENAME.pdf#page=7&search=TEXT`

### B.6 PdfViewer.svelte component

**Implementation** (`frontend/src/components/PdfViewer.svelte`):
```svelte
<script lang="ts">
  let { source, page, quote, onclose } = $props<{
    source: string; page: number; quote: string; onclose: () => void;
  }>();

  const searchText = quote.slice(0, 60);
  const viewerUrl = `/pdfjs/web/viewer.html?file=/api/pdf/${encodeURIComponent(source)}#page=${page}&search=${encodeURIComponent(searchText)}`;
</script>

<div class="pdf-modal-overlay" role="dialog">
  <div class="pdf-modal">
    <div class="pdf-header">
      <span>{source}, p. {page}</span>
      <button onclick={onclose}>Close</button>
    </div>
    <iframe src={viewerUrl} title="PDF viewer"></iframe>
  </div>
</div>
```

### B.7 Wire citation badges to viewer

Modify `formatResponse()` in `+page.svelte` to make citation badges clickable:
```typescript
.replace(
  /\[Source:\s*([^,\]]+),\s*p\.\s*(\d+)\]/g,
  '<span class="citation-badge" data-source="$1" data-page="$2">[Source: $1, p. $2]</span>' +
  '<button class="btn-ghost view-btn" data-source="$1" data-page="$2">View</button>'
)
```

Add click handler that opens `PdfViewer` with the right source/page/quote.

### B.8 Structured verification data in chat responses

**File:** `services/api/routers/conversations.py`

After streaming completes, run verification on the full response and store in message metadata:
```python
from core.verification import extract_quotes, verify_quotes
citations = verify_quotes(full_response, chunks)
metadata = {"citations": citations}
await db.execute(
    "UPDATE message SET metadata_json = ? WHERE id = ?",
    (json.dumps(metadata), assistant_msg_id)
)
```

Emit a final SSE event with verification results:
```python
yield {"event": "verification", "data": json.dumps({"citations": citations})}
```

---

## Phase C: Worker robustness

### C.1 Per-file timeout in corpus ingestion

Wrap each file's processing in `process_corpus_job()` with a timeout:
```python
import signal

class TimeoutError(Exception): pass

def _timeout_handler(signum, frame):
    raise TimeoutError("File processing timed out")

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(300)  # 5 minutes per file
try:
    # ... process file ...
finally:
    signal.alarm(0)  # cancel
```

### C.2 Atomic job claiming (prevent double-processing)

In `_check_corpus_jobs()`, atomically set status to "running" before processing:
```python
conn.execute(
    "UPDATE ingestion_job SET status = 'running' WHERE id = ? AND status = 'pending'",
    (job["id"],)
)
if conn.total_changes == 0:
    return  # another worker got it
conn.commit()
```

### C.3 Resume after crash

Add a `last_completed_file` column to `ingestion_job`. On startup, check for jobs with status "running" — these crashed mid-flight. Resume from last_completed_file + 1.

### C.4 File size limits

**File:** `services/api/routers/corpus.py`

Add validation in `upload_documents()`:
```python
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
for f in files:
    content = await f.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, f"{f.filename} exceeds 500MB limit")
```

---

## Phase D: Tests

### D.1 API router tests (`tests/test_api_routers.py`)

Use FastAPI's `TestClient` with a temporary SQLite database:

```python
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client(tmp_path):
    from services.api.core.database import set_db_path, init_db
    import asyncio
    set_db_path(str(tmp_path / "test.db"))
    asyncio.run(init_db())
    from services.api.main import app
    return TestClient(app)
```

**Tests needed:**
- `test_create_corpus` — POST name+tags → 200 with id
- `test_list_corpora` — GET after create → includes new corpus
- `test_get_corpus_detail` — GET /{id} → documents list
- `test_delete_corpus` — DELETE /{id} → cascades
- `test_create_conversation` — POST with mode → 200
- `test_list_conversations` — GET by corpus → includes new conv
- `test_get_conversation_messages` — GET /{id} → messages array
- `test_delete_conversation` — DELETE /{id} → 200
- `test_upload_rejects_bad_types` — POST .txt → 400
- `test_upload_accepts_pdf` — POST .pdf → 200 with job_id
- `test_health_endpoint` — GET /api/health → status field

### D.2 Worker pipeline tests (`tests/test_worker_pipeline.py`)

```python
def test_validate_corrupt_pdf(tmp_path):
    (tmp_path / "corrupt.pdf").write_bytes(b"not a pdf")
    result = validate_pdf(tmp_path / "corrupt.pdf")
    assert not result.valid

def test_validate_encrypted_pdf(tmp_path):
    # Create encrypted PDF with fitz
    result = validate_pdf(encrypted_path)
    assert result.is_encrypted

def test_epub_extraction(tmp_path):
    # Create minimal EPUB with ebooklib
    text, meta, pages = extract_epub(epub_path)
    assert len(text) > 0
    assert meta["doc_type"] == "epub"
    assert len(pages) > 0

def test_progress_updates(tmp_path):
    # Set DB_PATH to tmp, create tables, update progress, read back
    update_job_progress(job_id, status="running", current_step="chunking")
    # Verify via direct sqlite3 read
```

### D.3 Citation formatter edge cases (`tests/test_citation_edge_cases.py`)

```python
def test_empty_author():
    c = Citation(quote="x", author="", title="Test", year=2000, page=1)
    result = format_inline(c, Style.APA)
    assert "2000" in result  # should not crash

def test_three_authors():
    c = Citation(quote="x", author="Deleuze, Gilles and Felix Guattari and Claire Parnet",
                 title="Test", year=2000, page=1)
    result = format_inline(c, Style.APA)
    assert "et al." in result

def test_page_zero():
    c = Citation(quote="x", author="Deleuze, Gilles", title="Test", year=2000, page=0)
    result = format_inline(c, Style.MLA)
    assert "(Deleuze 0)" == result
```

### D.4 Database tests (`tests/test_database.py`)

```python
async def test_init_creates_tables(tmp_path):
    set_db_path(str(tmp_path / "test.db"))
    await init_db()
    db = await get_db()
    cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in await cursor.fetchall()}
    assert "corpus" in tables
    assert "conversation" in tables

async def test_auto_populate_exemplar(tmp_path):
    # Requires running ChromaDB with deleuze-surya collection
    # Skip if not available
```

---

## Phase E: Polish and UX

### E.1 Loading states

Add a `.spinner` CSS class and use it for:
- Corpus list loading on first open
- Conversation list loading when switching corpus
- Message streaming indicator (pulsing dot)
- Ingestion progress (animated progress bar)

### E.2 Empty states with guidance

- No corpora: "Welcome to Gutenberg. Create your first corpus to get started."
- No conversations: "Start a new conversation to query this corpus."
- No messages: Show mode-specific example queries as clickable chips
- Exhaustive mode: "Tip: put the concept in single quotes for best results"

### E.3 Keyboard shortcuts

- `Ctrl+N` — new conversation
- `Ctrl+Shift+N` — new corpus
- `Escape` — close modal / cancel streaming
- `Ctrl+Enter` — send message (alternative to Enter)

### E.4 Conversation search

Search bar in the left pane to filter conversations by title text.

### E.5 Export citations

"Export" button in the center pane that copies all citations to clipboard in the selected citation style. Supports:
- Plain text (for pasting into Word)
- BibTeX (for LaTeX users)
- Markdown (for note-taking apps)

### E.6 Corpus statistics dashboard

When a corpus is selected but no conversation is active, show:
- Total documents, chunks, estimated tokens
- Document list with status badges (done/failed/processing)
- Source breakdown chart (chunks per book)
- Last ingestion date

### E.7 Responsive layout

Below 1200px: collapse left pane to icon strip.
Below 900px: hide left pane entirely, add hamburger menu.
Right pane should stack below center pane on narrow screens.

### E.8 Accessibility

- Tab navigation through all interactive elements
- ARIA labels on buttons, modal, panels
- High contrast mode (increase border/text contrast)
- Screen reader: announce new messages, streaming status

---

## Phase F: Additional checks and hardening

### F.1 Input sanitization

- Corpus name: strip HTML, limit to 100 chars, no path characters (`/\:`)
- Tags: strip HTML, split on comma, trim whitespace, limit to 10 tags
- Chat input: limit to 10,000 chars
- File upload: validate MIME type matches extension, not just extension alone

### F.2 Rate limiting

- `/api/corpus/{id}/ingest`: max 1 concurrent ingestion per corpus
- `/api/conversations/{id}/messages`: max 1 concurrent stream per conversation
- File upload: max 10 files per request, max 500MB per file

### F.3 Graceful degradation

- Ollama down: return clear error "LLM service unavailable" (not generic 500)
- ChromaDB down: return clear error "Vector database unavailable"
- Worker down: ingestion jobs stay "pending" — show "Ingestion queued, waiting for worker"
- OpenRouter key invalid: fall back to Ollama, log warning

### F.4 Cleanup and garbage collection

- Delete orphaned files in `/data/inbox/` older than 24 hours
- Delete ingestion jobs with status "failed" after 7 days
- BM25 index rebuild when corpus changes (new documents ingested)
- ChromaDB collection compaction after bulk ingestion

### F.5 Logging and observability

- Structured JSON logging (for aggregation)
- Request ID tracking across API → worker
- Ingestion timing per step (validate: Xms, extract: Xms, chunk: Xms, embed: Xms, store: Xms)
- Chat latency tracking: retrieval time + LLM time + verification time

### F.6 Security

- PDF serving endpoint must reject path traversal (`../`, encoded variants)
- File upload filenames must be sanitized (strip `..`, control chars, null bytes)
- SQLite parameterized queries only (already done — verify no string formatting)
- CORS: restrict to known origins in production (currently `*`)
- No secrets in error messages (OpenRouter key, DB path, etc.)

---

## Phase G: Docker and deployment

### G.1 Multi-stage API Dockerfile

```dockerfile
# Stage 1: Build SvelteKit
FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python API + static frontend
FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
COPY services/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && python -m spacy download en_core_web_sm
COPY services/api/ .
COPY services/shared/ ./shared/
COPY --from=frontend /app/frontend/build ./static
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### G.2 Simplified docker-compose.yml

Three services only: `chromadb`, `api`, `worker`.
Move LibreChat/MongoDB/Meilisearch to `docker-compose.librechat.yml`.

### G.3 Health checks in compose

```yaml
api:
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
    interval: 30s
    timeout: 5s
    retries: 3
```

### G.4 Volume mounts

```yaml
volumes:
  - ./data:/data                    # PDFs, SQLite, BM25 index
  - ./frontend/build:/app/static    # (dev override, not needed with multi-stage build)
```

### G.5 Environment file

Update `.env.example` with ALL new variables:
```
# Database
DATABASE_PATH=/data/gutenberg.db

# Cloud fallback
OPENROUTER_API_KEY=
OPENROUTER_MODEL=deepseek/deepseek-r1
LLM_BACKEND=ollama
RUNPOD_API_KEY=
OCR_BACKEND=auto

# Worker
INBOX_DIR=/data/inbox
PROCESSING_DIR=/data/processing
PROCESSED_DIR=/data/processed
FAILED_DIR=/data/failed
```

---

## Phase H: Eval and regression

### H.1 Exhaustive retrieval regression test

After any pipeline change, run:
```bash
CHROMA_COLLECTION=deleuze-surya uv run scripts/eval_gauntlet.py \
  --modes exhaustive --models deepseek-r1 \
  --output data/eval/exhaustive_regression_$(date +%Y%m%d).json
```

Expect: 100% verification, ~1200 quotes, ~200 hallucinations filtered, <$0.04 cost.

### H.2 Citation style visual regression

For each style, generate a citation for a known Deleuze passage and screenshot the rendered output. Compare against golden screenshots stored in `data/eval/citation_screenshots/`.

### H.3 End-to-end smoke test

Script that:
1. Starts docker compose
2. Waits for health check
3. Creates a corpus
4. Uploads a small test PDF (< 10 pages)
5. Waits for ingestion to complete
6. Creates a conversation (general mode)
7. Sends a query
8. Verifies response contains citations
9. Tears down

### H.4 Performance baselines

Measure and record:
- Retrieval latency (BM25 + dense + RRF + rerank): target < 500ms
- Embedding latency (single query): target < 200ms
- Streaming first-token latency: target < 2s (local), < 5s (OpenRouter)
- Ingestion throughput: target > 50 pages/min (digital PDF)
- Frontend build size: target < 500KB gzipped

---

## Laundry list: everything else

### Features that should exist before shipping to real users

- [ ] Conversation rename (click title to edit)
- [ ] Conversation reorder (pin favorites)
- [ ] Bulk delete conversations
- [ ] Document deletion from corpus (remove from ChromaDB + re-index BM25)
- [ ] Re-ingest single document (update without full corpus rebuild)
- [ ] Copy citation to clipboard (single click on citation badge)
- [ ] Copy full response to clipboard
- [ ] Print / export conversation as PDF
- [ ] Bookmark specific citations across conversations
- [ ] Search across all conversations in a corpus
- [ ] Corpus sharing (export corpus config + ChromaDB dump for colleagues)
- [ ] Auto-detect term for exhaustive mode from query (currently requires explicit `term` param or regex)
- [ ] Citation count badge on conversation list items
- [ ] Streaming progress indicator (word count, elapsed time)
- [ ] "Stop generating" button during streaming
- [ ] Retry failed messages
- [ ] Dark mode persistence (save to localStorage, currently resets on reload)
- [ ] Browser tab title updates with active conversation
- [ ] Favicon badge for streaming status
- [ ] PDF viewer: next/prev page navigation
- [ ] PDF viewer: zoom controls
- [ ] PDF viewer: download highlighted page as image
- [ ] Notification when ingestion completes (if user navigated away)
- [ ] Batch mode: run all 10 exhaustive queries and show comparison table
- [ ] A/B test mode: compare two models' responses side by side
- [ ] Token usage tracking per conversation (input/output tokens, cost estimate)
- [ ] Admin panel: view all corpora, users, ingestion history
- [ ] Changelog / release notes accessible from the app
- [ ] "About" dialog with version, model info, corpus stats
- [ ] Keyboard-navigable conversation list (arrow keys)
- [ ] Drag-to-resize panes
- [ ] Collapsible left/right panes
- [ ] Mobile-responsive layout (at minimum: single-pane mode)
- [ ] PWA manifest (installable on desktop)
- [ ] Offline indicator (warn if API unreachable)
- [ ] Session persistence (resume conversation on page reload)
- [ ] Undo delete (30-second grace period)
