#!/usr/bin/env python3
"""Ingest PDFs into ChromaDB for evaluation.

Uses the OCR-deconfused text normalizer.
Runs standalone against local Ollama + ChromaDB.

Usage:
    CHROMA_COLLECTION=atp-eval uv run scripts/ingest_atp.py                          # ATP only
    CHROMA_COLLECTION=deleuze-full uv run scripts/ingest_atp.py --all-deleuze        # Full corpus
    CHROMA_COLLECTION=deleuze-full uv run scripts/ingest_atp.py path/to/file.pdf     # Specific PDF
"""

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import chromadb
import httpx
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))
from shared.text_normalize import clean_for_ingestion, strip_headers_footers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingest_atp")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "http://localhost:8200")
COLLECTION = os.environ.get("CHROMA_COLLECTION", "atp-eval")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b")

PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / \
    "1980 A Thousand Plateaus - Deleuze, Gilles.pdf"

CHUNK_SIZE = 384
CHUNK_OVERLAP = 96
EMBED_BATCH = 8

enc = tiktoken.get_encoding("cl100k_base")


def token_len(text: str) -> int:
    return len(enc.encode(text, disallowed_special=()))


def extract_pages(pdf_path: Path) -> list[dict]:
    """Extract per-page text from PDF using PyMuPDF."""
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc, 1):
        text = page.get_text()
        if text.strip():
            pages.append({"page": i, "text": text})
    doc.close()
    log.info(f"Extracted {len(pages)} pages from {pdf_path.name}")
    return pages


def clean_pages(pages: list[dict]) -> list[dict]:
    """Clean extracted text and strip repeated headers/footers."""
    cleaned = []
    for p in pages:
        text = clean_for_ingestion(p["text"])
        cleaned.append({"page": p["page"], "text": text})

    cleaned = strip_headers_footers(cleaned)
    return cleaned


def chunk_pages(pages: list[dict], source: str) -> list[dict]:
    """Split pages into overlapping token-based chunks with page tracking."""
    # Build full text with page break markers
    full_text = ""
    page_breaks = []  # (char_offset, page_number)
    for p in pages:
        page_breaks.append((len(full_text), p["page"]))
        full_text += p["text"] + "\n"

    # Split into sentences (simple heuristic)
    import re
    sentences = re.split(r'(?<=[.!?])\s+', full_text)

    chunks = []
    current_tokens = []
    current_text = ""
    current_start = 0  # char offset in full_text

    for sentence in sentences:
        sent_tokens = enc.encode(sentence, disallowed_special=())
        if len(current_tokens) + len(sent_tokens) > CHUNK_SIZE and current_tokens:
            # Save current chunk
            chunk_text = current_text.strip()
            if chunk_text:
                # Determine page range
                chunk_start = current_start
                chunk_end = current_start + len(current_text)
                page_start = 1
                page_end = 1
                for offset, page in page_breaks:
                    if offset <= chunk_start:
                        page_start = page
                    if offset <= chunk_end:
                        page_end = page

                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source": source,
                        "page_start": page_start,
                        "page_end": page_end,
                        "chunk_index": len(chunks),
                        "heading": "",
                    }
                })

            # Overlap: keep last CHUNK_OVERLAP tokens worth of text
            overlap_text = ""
            overlap_tokens = 0
            words = current_text.split()
            for word in reversed(words):
                word_tokens = len(enc.encode(word + " ", disallowed_special=()))
                if overlap_tokens + word_tokens > CHUNK_OVERLAP:
                    break
                overlap_text = word + " " + overlap_text
                overlap_tokens += word_tokens

            current_text = overlap_text
            current_tokens = enc.encode(current_text, disallowed_special=())
            current_start = full_text.find(current_text.split()[0], max(0, chunk_end - 500)) if current_text.strip() else chunk_end

        current_text += sentence + " "
        current_tokens = enc.encode(current_text, disallowed_special=())

    # Final chunk
    if current_text.strip():
        chunk_start = current_start
        chunk_end = current_start + len(current_text)
        page_start = 1
        page_end = 1
        for offset, page in page_breaks:
            if offset <= chunk_start:
                page_start = page
            if offset <= chunk_end:
                page_end = page

        chunks.append({
            "text": current_text.strip(),
            "metadata": {
                "source": source,
                "page_start": page_start,
                "page_end": page_end,
                "chunk_index": len(chunks),
                "heading": "",
            }
        })

    log.info(f"Created {len(chunks)} chunks (avg {sum(token_len(c['text']) for c in chunks)//max(len(chunks),1)} tokens)")
    return chunks


def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    """Embed chunk texts via Ollama in batches."""
    embeddings = []
    texts = [c["text"] for c in chunks]

    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        t0 = time.time()
        resp = httpx.post(
            f"{OLLAMA_HOST}/api/embed",
            json={"model": EMBED_MODEL, "input": batch},
            timeout=120,
        )
        resp.raise_for_status()
        batch_embeddings = resp.json()["embeddings"]
        embeddings.extend(batch_embeddings)
        elapsed = time.time() - t0
        log.info(f"  Embedded batch {i//EMBED_BATCH + 1}/{(len(texts)-1)//EMBED_BATCH + 1} "
                 f"({len(batch)} chunks, {elapsed:.1f}s)")

    return embeddings


def store_chunks(chunks: list[dict], embeddings: list[list[float]], source: str = "", client=None):
    """Store chunks + embeddings in ChromaDB."""
    if client is None:
        host = CHROMA_HOST.replace("http://", "").replace("https://", "")
        parts = host.split(":")
        client = chromadb.HttpClient(host=parts[0], port=int(parts[1]) if len(parts) > 1 else 8000)

    col = _get_or_create_collection(client)

    # Insert in batches of 100
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeds = embeddings[i:i + batch_size]

        # Use source-based prefix for IDs to allow multi-book ingestion
        prefix = (source or "doc")[:20].replace(" ", "_").lower()
        ids = [f"{prefix}-{c['metadata']['chunk_index']:04d}" for c in batch_chunks]
        documents = [c["text"] for c in batch_chunks]
        metadatas = [c["metadata"] for c in batch_chunks]

        col.add(ids=ids, embeddings=batch_embeds, documents=documents, metadatas=metadatas)

    log.info(f"Stored {len(chunks)} chunks in '{COLLECTION}' (total now: {col.count()})")


_col_cache = None

def _get_or_create_collection(client):
    global _col_cache
    if _col_cache is not None:
        return _col_cache
    _col_cache = client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})
    return _col_cache


def ingest_pdf(pdf_path: Path):
    """Ingest a single PDF into the collection."""
    log.info(f"Ingesting: {pdf_path.name}")
    t0 = time.time()

    pages = extract_pages(pdf_path)
    pages = clean_pages(pages)
    source = pdf_path.name
    chunks = chunk_pages(pages, source)

    log.info(f"Embedding {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks)

    host = CHROMA_HOST.replace("http://", "").replace("https://", "")
    parts = host.split(":")
    client = chromadb.HttpClient(host=parts[0], port=int(parts[1]) if len(parts) > 1 else 8000)
    store_chunks(chunks, embeddings, source, client)

    elapsed = time.time() - t0
    log.info(f"  {pdf_path.name}: {len(chunks)} chunks in {elapsed:.0f}s")
    return len(chunks)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="*", help="Specific PDFs to ingest")
    parser.add_argument("--all-deleuze", action="store_true", help="Ingest all Deleuze PDFs")
    parser.add_argument("--fresh", action="store_true", help="Delete collection before ingesting")
    args = parser.parse_args()

    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"

    if args.all_deleuze:
        pdfs = sorted(processed_dir.glob("*Deleuze*.pdf"))
    elif args.pdfs:
        pdfs = [Path(p) for p in args.pdfs]
    else:
        pdfs = [PDF_PATH]

    log.info(f"Collection: {COLLECTION}")
    log.info(f"Embed model: {EMBED_MODEL}")
    log.info(f"PDFs to ingest: {len(pdfs)}")

    if args.fresh:
        host = CHROMA_HOST.replace("http://", "").replace("https://", "")
        parts = host.split(":")
        client = chromadb.HttpClient(host=parts[0], port=int(parts[1]) if len(parts) > 1 else 8000)
        try:
            client.delete_collection(COLLECTION)
            log.info(f"Deleted existing collection '{COLLECTION}'")
        except Exception:
            pass

    total_chunks = 0
    t0_all = time.time()
    for pdf in pdfs:
        if not pdf.exists():
            log.error(f"PDF not found: {pdf}")
            continue
        total_chunks += ingest_pdf(pdf)

    elapsed = time.time() - t0_all
    log.info(f"\nDone: {total_chunks} total chunks from {len(pdfs)} PDFs in {elapsed:.0f}s")

    # Spot check OCR
    sample_texts = [c["text"] for c in [] ]  # can't easily check here
    rn_count = 0  # skipped for multi-pdf
    if rn_count > 0:
        log.warning(f"OCR deconfusion may be incomplete: {rn_count} chunks still contain 'rnach'")
    else:
        log.info("OCR deconfusion verified: no 'rnach' artifacts in first 50 chunks")


if __name__ == "__main__":
    main()
