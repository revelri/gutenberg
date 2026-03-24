#!/usr/bin/env python3
"""Re-ingest all documents from /data/processed/ through the pipeline.

Clears the ChromaDB collection and state file, then re-runs extraction,
chunking, embedding, and storage for every file in the processed directory.

Usage (inside worker container):
    python /app/scripts/reingest.py

Usage (via docker exec):
    docker exec gutenberg-worker python /app/scripts/reingest.py [--dry-run]
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add worker to path so pipeline modules are importable
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "worker"))

from pipeline.detector import classify_document
from pipeline.extractors import extract_text
from pipeline.chunker import chunk_text
from pipeline.embedder import embed_chunks
from pipeline.store import store_chunks, record_document, _get_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("reingest")

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def clear_chromadb():
    """Delete all documents from the ChromaDB collection."""
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        log.info("ChromaDB collection already empty")
        return

    # ChromaDB requires fetching IDs to delete
    batch_size = 1000
    while True:
        result = collection.get(limit=batch_size)
        ids = result["ids"]
        if not ids:
            break
        collection.delete(ids=ids)
        log.info(f"Deleted {len(ids)} chunks from ChromaDB")

    log.info(f"Cleared {count} total chunks from ChromaDB")


def clear_state_file(state_file: Path):
    """Truncate the state file."""
    if state_file.exists():
        state_file.write_text("")
        log.info(f"Cleared {state_file}")


def get_processable_files(processed_dir: Path) -> list[Path]:
    """Find all supported files in the processed directory."""
    files = []
    for f in sorted(processed_dir.iterdir()):
        if f.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(f)
    return files


def reingest_file(path: Path, state_file: Path) -> tuple[int, float]:
    """Run the full pipeline on a single file. Returns (chunk_count, elapsed)."""
    t0 = time.time()

    doc_type = classify_document(path)
    log.info(f"  Classified as {doc_type}")

    text, metadata, page_segments = extract_text(path, doc_type)
    if not text.strip():
        raise ValueError(f"No text extracted from {path.name}")
    log.info(f"  Extracted {len(text):,} chars, {len(page_segments)} page segments")

    chunks = chunk_text(text, metadata, page_segments)
    log.info(f"  Chunked into {len(chunks)} pieces")

    embeddings = embed_chunks([c["text"] for c in chunks])
    log.info(f"  Generated {len(embeddings)} embeddings")

    store_chunks(chunks, embeddings)
    record_document(path, len(chunks), state_file)

    elapsed = time.time() - t0
    return len(chunks), elapsed


def main():
    parser = argparse.ArgumentParser(description="Re-ingest all processed documents")
    parser.add_argument("--dry-run", action="store_true", help="List files without processing")
    parser.add_argument("--processed-dir", default=os.environ.get("PROCESSED_DIR", "/data/processed"))
    parser.add_argument("--state-file", default=os.environ.get("STATE_FILE", "/data/state/documents.jsonl"))
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    state_file = Path(args.state_file)

    if not processed_dir.exists():
        log.error(f"Processed directory not found: {processed_dir}")
        sys.exit(1)

    files = get_processable_files(processed_dir)
    log.info(f"Found {len(files)} files in {processed_dir}")

    if args.dry_run:
        for f in files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name} ({size_mb:.1f} MB)")
        print(f"\n{len(files)} files would be re-ingested.")
        return

    # Clear existing data
    log.info("Clearing ChromaDB collection...")
    clear_chromadb()
    clear_state_file(state_file)

    # Re-ingest each file
    total_chunks = 0
    total_time = 0
    failed = []

    for i, f in enumerate(files, 1):
        log.info(f"[{i}/{len(files)}] Processing {f.name}")
        try:
            chunks, elapsed = reingest_file(f, state_file)
            total_chunks += chunks
            total_time += elapsed
            log.info(f"  Done: {chunks} chunks in {elapsed:.1f}s")
        except Exception:
            log.exception(f"  Failed: {f.name}")
            failed.append(f.name)

    # Summary
    log.info(f"\nRe-ingestion complete:")
    log.info(f"  Files: {len(files) - len(failed)}/{len(files)} succeeded")
    log.info(f"  Chunks: {total_chunks:,}")
    log.info(f"  Time: {total_time:.1f}s")
    if failed:
        log.warning(f"  Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
