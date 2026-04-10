#!/usr/bin/env python3
"""Ingest Surya markdown OCR output into ChromaDB.

Reads .md files from data/surya_corpus/, chunks them, embeds, and stores.

Usage:
    CHROMA_COLLECTION=deleuze-surya uv run scripts/ingest_surya.py
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import chromadb
import httpx
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))
from shared.text_normalize import clean_for_ingestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingest_surya")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "http://localhost:8200")
COLLECTION = os.environ.get("CHROMA_COLLECTION", "deleuze-surya")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b")
SURYA_DIR = Path(__file__).resolve().parent.parent / "data" / "surya_corpus"

CHUNK_SIZE = 384
CHUNK_OVERLAP = 96
EMBED_BATCH = 8

enc = tiktoken.get_encoding("cl100k_base")


def token_len(text: str) -> int:
    return len(enc.encode(text, disallowed_special=()))


def load_markdown(md_path: Path) -> tuple[str, str]:
    """Load markdown file and derive source name."""
    text = md_path.read_text(encoding="utf-8", errors="replace")
    # Source name: parent directory name + .pdf
    source = md_path.parent.name + ".pdf"
    return text, source


def chunk_markdown(text: str, source: str) -> list[dict]:
    """Split markdown into token-based chunks with page estimation."""
    # Clean the text
    text = clean_for_ingestion(text)

    # Remove image references
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Remove excessive blank lines from image removal
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Split into paragraphs
    paragraphs = re.split(r'\n\n+', text)

    chunks = []
    current_text = ""
    current_tokens = 0
    current_heading = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Track headings
        heading_match = re.match(r'^(#{1,4})\s+(.+)', para)
        if heading_match:
            current_heading = heading_match.group(2)[:80]

        para_tokens = token_len(para)

        if current_tokens + para_tokens > CHUNK_SIZE and current_text:
            # Save current chunk
            chunks.append({
                "text": current_text.strip(),
                "metadata": {
                    "source": source,
                    "heading": current_heading,
                    "chunk_index": len(chunks),
                    "page_start": 0,  # Page tracking not available from markdown
                    "page_end": 0,
                }
            })

            # Overlap: keep last portion
            words = current_text.split()
            overlap_words = []
            overlap_tokens = 0
            for w in reversed(words):
                wt = token_len(w + " ")
                if overlap_tokens + wt > CHUNK_OVERLAP:
                    break
                overlap_words.insert(0, w)
                overlap_tokens += wt

            current_text = " ".join(overlap_words) + "\n\n" + para if overlap_words else para
            current_tokens = token_len(current_text)
        else:
            current_text += ("\n\n" if current_text else "") + para
            current_tokens += para_tokens

    # Final chunk
    if current_text.strip():
        chunks.append({
            "text": current_text.strip(),
            "metadata": {
                "source": source,
                "heading": current_heading,
                "chunk_index": len(chunks),
                "page_start": 0,
                "page_end": 0,
            }
        })

    return chunks


def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    embeddings = []
    texts = [c["text"] for c in chunks]
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        resp = httpx.post(f"{OLLAMA_HOST}/api/embed",
                          json={"model": EMBED_MODEL, "input": batch}, timeout=120)
        resp.raise_for_status()
        embeddings.extend(resp.json()["embeddings"])
        if (i // EMBED_BATCH + 1) % 20 == 0:
            log.info(f"  Embedded {i + len(batch)}/{len(texts)} chunks")
    return embeddings


def store_chunks(chunks, embeddings, source, client):
    col = client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})
    prefix = source[:25].replace(" ", "_").lower()
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        bc = chunks[i:i + batch_size]
        be = embeddings[i:i + batch_size]
        ids = [f"{prefix}-{c['metadata']['chunk_index']:04d}" for c in bc]
        col.add(ids=ids, embeddings=be, documents=[c["text"] for c in bc],
                metadatas=[c["metadata"] for c in bc])
    log.info(f"  Stored {len(chunks)} chunks (total: {col.count()})")


def main():
    host = CHROMA_HOST.replace("http://", "").replace("https://", "")
    parts = host.split(":")
    client = chromadb.HttpClient(host=parts[0], port=int(parts[1]) if len(parts) > 1 else 8000)

    # Delete existing collection
    try:
        client.delete_collection(COLLECTION)
        log.info(f"Deleted existing collection '{COLLECTION}'")
    except Exception:
        pass

    md_files = sorted(SURYA_DIR.glob("*/*.md"))
    log.info(f"Found {len(md_files)} markdown files in {SURYA_DIR}")
    log.info(f"Collection: {COLLECTION}, Embed model: {EMBED_MODEL}")

    total_chunks = 0
    t0 = time.time()

    for md_path in md_files:
        text, source = load_markdown(md_path)
        words = len(text.split())
        log.info(f"\n{source}: {words} words")

        chunks = chunk_markdown(text, source)
        log.info(f"  {len(chunks)} chunks (avg {sum(token_len(c['text']) for c in chunks) // max(len(chunks), 1)} tokens)")

        embeddings = embed_chunks(chunks)
        store_chunks(chunks, embeddings, source, client)
        total_chunks += len(chunks)

    elapsed = time.time() - t0
    log.info(f"\nDone: {total_chunks} chunks from {len(md_files)} books in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
