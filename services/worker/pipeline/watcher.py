"""File system watcher with debounce and stable-size check."""

import logging
import os
import shutil
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from pipeline.detector import classify_document
from pipeline.extractors import extract_text
from pipeline.chunker import chunk_text
from pipeline.embedder import embed_chunks
from pipeline.store import store_chunks, is_duplicate, record_document

log = logging.getLogger("gutenberg.watcher")

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
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
        self.state_file = Path(os.environ.get("STATE_FILE", "/data/state/documents.jsonl"))

        for d in [self.inbox, self.processing, self.processed, self.failed]:
            d.mkdir(parents=True, exist_ok=True)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def process_file(self, source: Path):
        """Full ingestion pipeline for a single file."""
        fname = source.name
        work_path = self.processing / fname

        try:
            # Move to processing (atomic on same filesystem)
            shutil.move(str(source), str(work_path))
            log.info(f"Processing: {fname}")

            # Dedup check
            if is_duplicate(work_path, self.state_file):
                log.info(f"Skipping duplicate: {fname}")
                shutil.move(str(work_path), str(self.processed / fname))
                return

            # Classify and extract
            doc_type = classify_document(work_path)
            log.info(f"Classified {fname} as {doc_type}")

            text, metadata, page_segments = extract_text(work_path, doc_type)
            if not text.strip():
                raise ValueError(f"No text extracted from {fname}")

            log.info(f"Extracted {len(text)} chars from {fname}")

            # Chunk
            chunks = chunk_text(text, metadata, page_segments)
            log.info(f"Created {len(chunks)} chunks from {fname}")

            # Embed
            embeddings = embed_chunks([c["text"] for c in chunks])
            log.info(f"Generated {len(embeddings)} embeddings")

            # Store
            store_chunks(chunks, embeddings)
            log.info(f"Stored {len(chunks)} chunks in ChromaDB")

            # Record success and move
            record_document(work_path, len(chunks), self.state_file)
            shutil.move(str(work_path), str(self.processed / fname))
            log.info(f"Done: {fname}")

        except Exception:
            log.exception(f"Failed to process {fname}")
            if work_path.exists():
                shutil.move(str(work_path), str(self.failed / fname))

    def run(self):
        """Start watching and processing loop."""
        handler = InboxHandler()
        observer = Observer()
        observer.schedule(handler, str(self.inbox), recursive=False)
        observer.start()

        # Also process any files already in inbox at startup
        for f in self.inbox.iterdir():
            if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                handler._pending[str(f)] = 0  # immediately stable

        try:
            while True:
                stable = handler.get_stable_files()
                for fpath in stable:
                    self.process_file(fpath)
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
