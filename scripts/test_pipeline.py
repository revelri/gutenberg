#!/usr/bin/env python3
"""Test the OCR/extraction pipeline locally without Docker services."""

import sys
import time
from pathlib import Path

# Add worker to path
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "worker"))

from pipeline.detector import classify_document
from pipeline.extractors import extract_text
from pipeline.chunker import chunk_text

PDF_FILES = [
    Path.home() / "Downloads" / "Xanathar's Guide to Everything Deluxe.pdf",
    Path.home() / "Downloads" / "Xanathar's Lost Notes to Everything Else v1.1.pdf",
    Path.home() / "Downloads" / "Tasha\u2019s Cauldron of Everything (HQ, Both Covers).pdf",
    Path.home() / "Downloads" / "Wizards RPG Team - Xanathar's Guide To Everything.pdf",
    Path.home() / "Downloads" / "DnD5e - Player's Handbook.pdf",
]


def test_file(path: Path):
    print(f"\n{'='*70}")
    print(f"FILE: {path.name}")
    print(f"SIZE: {path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"{'='*70}")

    # Classify
    t0 = time.time()
    doc_type = classify_document(path)
    t_classify = time.time() - t0
    print(f"TYPE: {doc_type} ({t_classify:.2f}s)")

    # Extract (skip Docling/scanned for now — test digital path)
    if doc_type == "pdf_scanned":
        print("SCANNED — skipping extraction (needs Docling)")
        return doc_type, 0, 0, t_classify

    t0 = time.time()
    text, metadata, page_segments = extract_text(path, doc_type)
    t_extract = time.time() - t0
    print(f"EXTRACTED: {len(text):,} chars, {metadata.get('total_pages', '?')} pages, {len(page_segments)} page segments ({t_extract:.2f}s)")
    print(f"FIRST 300 CHARS:\n{text[:300]}\n...")

    # Chunk
    t0 = time.time()
    chunks = chunk_text(text, metadata, page_segments)
    t_chunk = time.time() - t0
    print(f"CHUNKS: {len(chunks)} ({t_chunk:.2f}s)")

    if chunks:
        print(f"SAMPLE CHUNK [0]: {chunks[0]['text'][:200]}...")
        print(f"SAMPLE META [0]: {chunks[0]['metadata']}")

    return doc_type, len(text), len(chunks), t_classify + t_extract + t_chunk


def main():
    results = []
    for path in PDF_FILES:
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        try:
            result = test_file(path)
            results.append((path.name, *result))
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((path.name, "error", 0, 0, 0))

    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'File':<55} {'Type':<14} {'Chars':>10} {'Chunks':>8} {'Time':>7}")
    print("-" * 100)
    for name, dtype, chars, chunks, elapsed in results:
        short = name[:52] + "..." if len(name) > 55 else name
        print(f"{short:<55} {dtype:<14} {chars:>10,} {chunks:>8} {elapsed:>6.2f}s")


if __name__ == "__main__":
    main()
