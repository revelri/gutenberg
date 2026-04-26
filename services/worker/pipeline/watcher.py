"""File system watcher with debounce and stable-size check.

Supports two modes:
1. File-watch: monitors /data/inbox/ for new files (original behavior)
2. Corpus job: processes files for a specific corpus project (triggered by API)
"""

import hashlib
import logging
import os
import shutil
import signal
import time
from pathlib import Path


class FileTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise FileTimeoutError("File processing timed out (5 min)")


from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from pipeline.detector import classify_document
from pipeline.extractors import extract_text, extract_modal_elements
from pipeline.pdf_validator import validate_document
from pipeline.chunker import chunk_text
from pipeline.embedder import embed_chunks
from pipeline.ocrmypdf_preprocess import (
    preprocess as ocrmypdf_preprocess,
    ocrmypdf_available,
)
from pipeline.store import (
    store_chunks, is_duplicate, record_document,
    write_pending_marker, remove_pending_marker, cleanup_partial_ingestion,
)
from pipeline.progress import (
    update_job_progress,
    update_document_status,
    update_corpus_status,
    get_corpus_collection,
)

log = logging.getLogger("gutenberg.watcher")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".epub"}
DEBOUNCE_SECONDS = 2


class InboxHandler(FileSystemEventHandler):
    """Handles new files appearing in the inbox directory."""

    def __init__(self):
        self._pending: dict[str, float] = {}

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            self._pending[str(path)] = time.time()

    def on_modified(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            self._pending[str(path)] = time.time()

    def get_stable_files(self) -> list[Path]:
        """Return files that haven't changed for DEBOUNCE_SECONDS."""
        now = time.time()
        stable = []
        still_pending = {}
        for fpath, last_modified in self._pending.items():
            if now - last_modified >= DEBOUNCE_SECONDS:
                p = Path(fpath)
                if p.exists():
                    stable.append(p)
            else:
                still_pending[fpath] = last_modified
        self._pending = still_pending
        return stable


class InboxWatcher:
    """Watches inbox directory and processes stable files."""

    def __init__(self, inbox_dir: str):
        self.inbox = Path(inbox_dir)
        self.processing = Path(os.environ.get("PROCESSING_DIR", "/data/processing"))
        self.processed = Path(os.environ.get("PROCESSED_DIR", "/data/processed"))
        self.failed = Path(os.environ.get("FAILED_DIR", "/data/failed"))
        self.state_file = Path(
            os.environ.get("STATE_FILE", "/data/state/documents.jsonl")
        )

        for d in [self.inbox, self.processing, self.processed, self.failed]:
            d.mkdir(parents=True, exist_ok=True)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Clean up any partial ingestion from previous crashes
        cleanup_partial_ingestion(self.state_file.parent)

    def process_file(self, source: Path):
        """Full ingestion pipeline for a single file."""
        fname = source.name
        work_path = self.processing / fname

        try:
            # Move to processing (atomic on same filesystem)
            shutil.move(str(source), str(work_path))
            log.info(f"Processing: {fname}")

            # Validate
            validation = validate_document(work_path)
            if not validation.valid:
                raise ValueError(f"Validation failed: {'; '.join(validation.errors)}")
            for w in validation.warnings:
                log.warning(f"Validation warning for {fname}: {w}")

            # Dedup check
            if is_duplicate(work_path, self.state_file):
                log.info(f"Skipping duplicate: {fname}")
                shutil.move(str(work_path), str(self.processed / fname))
                return

            # Classify
            doc_type = classify_document(work_path)
            log.info(f"Classified {fname} as {doc_type}")

            # For scanned PDFs: OCRmyPDF adds a text layer, then PyMuPDF extracts fast.
            # Falls back to Docling if OCRmyPDF is not available.
            extract_path = work_path
            if doc_type == "pdf_scanned" and ocrmypdf_available():
                try:
                    ocr_dir = self.processing / "_ocr_tmp"
                    ocr_dir.mkdir(exist_ok=True)
                    extract_path = ocrmypdf_preprocess(work_path, ocr_dir)
                    doc_type = classify_document(extract_path)
                    log.info(f"OCRmyPDF preprocessed → reclassified as {doc_type}")
                    ocr_dest = self.processed / f"{fname.rsplit('.', 1)[0]}-ocr.pdf"
                    shutil.copy2(str(extract_path), str(ocr_dest))
                except Exception:
                    log.warning(f"OCRmyPDF failed for {fname}, falling back to Docling")
                    extract_path = work_path

            text, metadata, page_segments = extract_text(extract_path, doc_type)
            metadata["source"] = fname  # always use original filename
            if not text.strip():
                raise ValueError(f"No text extracted from {fname}")

            log.info(f"Extracted {len(text)} chars from {fname}")

            # Chunk
            chunks = chunk_text(text, metadata, page_segments)
            log.info(f"Created {len(chunks)} chunks from {fname}")

            # P3: modal chunks (tables, equations) — flag-gated, appended
            try:
                from pipeline.modal import make_modal_chunks
                modal_elements = extract_modal_elements(extract_path, doc_type)
                if modal_elements:
                    extra = make_modal_chunks(modal_elements, metadata)
                    chunks.extend(extra)
                    log.info(f"Added {len(extra)} modal chunks")
            except Exception as e:
                log.debug(f"modal chunks skipped: {e}")

            # Embed
            # Embed contextualized text (P0); falls back to raw text if absent.
            embeddings = embed_chunks([c.get("contextual_text") or c["text"] for c in chunks])
            log.info(f"Generated {len(embeddings)} embeddings")

            # P5: RAPTOR summary tree — appended as additional chunks
            try:
                from pipeline.raptor import build_tree
                summaries = build_tree(chunks, embeddings)
                if summaries:
                    summary_embeds = embed_chunks(
                        [s.get("contextual_text") or s["text"] for s in summaries]
                    )
                    chunks.extend(summaries)
                    embeddings.extend(summary_embeds)
                    log.info(f"RAPTOR appended {len(summaries)} summary chunks")
            except Exception as e:
                log.debug(f"RAPTOR skipped: {e}")

            # Store with pending marker for crash recovery
            state_dir = self.state_file.parent
            state_dir.mkdir(parents=True, exist_ok=True)
            marker = write_pending_marker(fname, state_dir)
            store_chunks(chunks, embeddings, pending_marker=marker)
            log.info(f"Stored {len(chunks)} chunks in ChromaDB")

            # Record success, remove marker, and move
            record_document(work_path, len(chunks), self.state_file)
            remove_pending_marker(marker)
            shutil.move(str(work_path), str(self.processed / fname))
            log.info(f"Done: {fname}")

        except Exception:
            log.exception(f"Failed to process {fname}")
            if work_path.exists():
                shutil.move(str(work_path), str(self.failed / fname))

    def process_corpus_job(self, job_id: str, corpus_id: str, resume_from: int = 0):
        """Process all files for a corpus ingestion job.

        Files are expected at /data/inbox/{corpus_id}/.
        Progress is reported via SQLite for the API's SSE endpoint.
        """
        inbox_dir = self.inbox / corpus_id
        if not inbox_dir.exists():
            update_job_progress(
                job_id, status="failed", error="Inbox directory not found"
            )
            return

        collection_name = get_corpus_collection(corpus_id)
        if not collection_name:
            update_job_progress(
                job_id, status="failed", error="Corpus not found in database"
            )
            return

        files = [
            f for f in inbox_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not files:
            update_job_progress(
                job_id, status="failed", error="No supported files found"
            )
            return

        if resume_from > 0:
            files = sorted(files)[resume_from:]
            completed = resume_from
        else:
            completed = 0

        update_job_progress(job_id, status="running", completed_files=completed)

        for fpath in sorted(files):
            fname = fpath.name
            update_job_progress(job_id, current_file=fname, current_step="validating")

            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(300)  # 5 minutes per file
            try:
                # Validate
                validation = validate_document(fpath)
                if not validation.valid:
                    error_msg = "; ".join(validation.errors)
                    update_document_status(
                        filename=fname,
                        corpus_id=corpus_id,
                        status="failed",
                        error=error_msg,
                    )
                    log.warning(f"Validation failed for {fname}: {error_msg}")
                    completed += 1
                    update_job_progress(job_id, completed_files=completed)
                    continue

                for w in validation.warnings:
                    log.warning(f"Validation warning for {fname}: {w}")

                # Classify
                update_job_progress(job_id, current_step="classifying")
                doc_type = classify_document(fpath)
                update_document_status(
                    filename=fname,
                    corpus_id=corpus_id,
                    file_type=doc_type,
                    status="processing",
                )

                # OCR preprocessing for scanned PDFs
                extract_path = fpath
                if doc_type == "pdf_scanned" and ocrmypdf_available():
                    update_job_progress(job_id, current_step="ocr")
                    try:
                        ocr_dir = self.processing / "_ocr_tmp"
                        ocr_dir.mkdir(exist_ok=True)
                        extract_path = ocrmypdf_preprocess(fpath, ocr_dir)
                        doc_type = classify_document(extract_path)
                        ocr_dest = self.processed / f"{fname.rsplit('.', 1)[0]}-ocr.pdf"
                        shutil.copy2(str(extract_path), str(ocr_dest))
                        log.info(f"Saved OCR'd PDF: {ocr_dest.name}")
                    except Exception:
                        log.warning(f"OCRmyPDF failed for {fname}, using original")
                        extract_path = fpath

                # Extract text
                update_job_progress(job_id, current_step="extracting")
                text, metadata, page_segments = extract_text(extract_path, doc_type)
                metadata["source"] = fname

                if not text.strip():
                    raise ValueError(f"No text extracted from {fname}")

                # Compute SHA-256
                sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
                update_document_status(
                    filename=fname,
                    corpus_id=corpus_id,
                    sha256=sha,
                )

                # Chunk
                update_job_progress(job_id, current_step="chunking")
                chunks = chunk_text(text, metadata, page_segments)

                # P3: modal chunks (tables, equations)
                try:
                    from pipeline.modal import make_modal_chunks
                    modal_elements = extract_modal_elements(extract_path, doc_type)
                    if modal_elements:
                        extra = make_modal_chunks(modal_elements, metadata)
                        chunks.extend(extra)
                except Exception as e:
                    log.debug(f"modal chunks skipped: {e}")

                # Embed
                update_job_progress(job_id, current_step="embedding")
                # Embed contextualized text (P0); falls back to raw text if absent.
                embeddings = embed_chunks([c.get("contextual_text") or c["text"] for c in chunks])

                # Store in corpus-specific collection
                update_job_progress(job_id, current_step="storing")
                store_chunks(chunks, embeddings, collection_name=collection_name)

                # Update document record
                update_document_status(
                    filename=fname,
                    corpus_id=corpus_id,
                    status="done",
                    chunks=len(chunks),
                )

                log.info(f"Corpus {corpus_id}: ingested {fname} → {len(chunks)} chunks")

            except Exception as e:
                log.exception(f"Failed to process {fname} for corpus {corpus_id}")
                update_document_status(
                    filename=fname,
                    corpus_id=corpus_id,
                    status="failed",
                    error=str(e),
                )
            finally:
                signal.alarm(0)

            completed += 1
            update_job_progress(job_id, completed_files=completed)

        # Finalize
        update_job_progress(job_id, status="done", current_step=None, current_file=None)
        update_corpus_status(corpus_id, "ready")

        # Clean up inbox
        shutil.rmtree(str(inbox_dir), ignore_errors=True)
        log.info(
            f"Corpus {corpus_id}: ingestion complete ({completed}/{len(files)} files)"
        )

    def run(self):
        """Start watching and processing loop.

        Monitors two sources:
        1. /data/inbox/ for file-drop ingestion (original behavior)
        2. SQLite ingestion_jobs for corpus-triggered jobs (from API)
        """
        handler = InboxHandler()
        observer = Observer()
        observer.schedule(handler, str(self.inbox), recursive=False)
        observer.start()

        # Process any files already in inbox at startup
        for f in self.inbox.iterdir():
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                handler._pending[str(f)] = 0

        try:
            while True:
                # Check for file-drop files
                stable = handler.get_stable_files()
                for fpath in stable:
                    self.process_file(fpath)

                # Check for pending corpus jobs
                self._check_corpus_jobs()

                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    def _check_corpus_jobs(self):
        """Poll SQLite for pending corpus ingestion jobs."""
        import sqlite3

        db_path = os.environ.get("DATABASE_PATH", "/data/gutenberg.db")
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "UPDATE ingestion_job SET status = 'running', updated_at = datetime('now') WHERE id = (SELECT id FROM ingestion_job WHERE status = 'pending' ORDER BY created_at LIMIT 1) AND status = 'pending'"
            )
            if cursor.rowcount == 0:
                conn.close()
            else:
                conn.commit()
                cursor = conn.execute(
                    "SELECT id, corpus_id FROM ingestion_job WHERE status = 'running' ORDER BY updated_at DESC LIMIT 1"
                )
                job = cursor.fetchone()
                conn.close()

                if job:
                    log.info(f"Claimed job {job['id']} for corpus {job['corpus_id']}")
                    self.process_corpus_job(job["id"], job["corpus_id"])
                    return
        except Exception as e:
            log.debug(f"Could not check corpus jobs: {e}")
            return

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT id, corpus_id, completed_files FROM ingestion_job WHERE status = 'running' ORDER BY created_at DESC LIMIT 1"
            )
            crashed_job = cursor.fetchone()
            conn.close()

            if crashed_job:
                log.info(
                    f"Found crashed job {crashed_job['id']}, resuming from file {crashed_job['completed_files']}"
                )
                self.process_corpus_job(
                    crashed_job["id"],
                    crashed_job["corpus_id"],
                    resume_from=crashed_job["completed_files"],
                )
                return
        except Exception as e:
            log.debug(f"Could not check crashed jobs: {e}")
