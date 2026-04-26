#!/usr/bin/env python3
"""Re-index the Deleuze corpus with sentence-transformers embeddings.

Creates a new ChromaDB collection with embeddings from the in-process
sentence-transformers model, replacing the old Ollama-based vectors.

Usage:
    CHROMA_HOST=http://localhost:8001 uv run python scripts/reindex_st.py
    CHROMA_HOST=http://localhost:8001 uv run python scripts/reindex_st.py --dry-run
    CHROMA_HOST=http://localhost:8001 uv run python scripts/reindex_st.py --pattern "*Deleuze*"
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add service paths
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "worker"))
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))

from pipeline.detector import classify_document
from pipeline.extractors import extract_text
from pipeline.chunker import chunk_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("reindex")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".epub"}


def get_files(processed_dir: Path, pattern: str) -> list[Path]:
    files = sorted(processed_dir.glob(pattern))
    return [f for f in files if f.suffix.lower() in SUPPORTED_EXTENSIONS and not f.name.endswith(".bak")]


def clear_collection(chroma_host: str, collection_name: str):
    import chromadb
    host, port = chroma_host.replace("http://", "").replace("https://", "").split(":")
    client = chromadb.HttpClient(host=host, port=int(port))
    try:
        client.delete_collection(collection_name)
        log.info(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass
    col = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    log.info(f"Created collection: {collection_name}")
    return col


def main():
    parser = argparse.ArgumentParser(description="Re-index corpus with sentence-transformers")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--pattern", default="*Deleuze*")
    parser.add_argument("--collection", default="gutenberg-qwen3")
    parser.add_argument("--embed-model", default=os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--surya-dir", default="data/surya_corpus", help="Surya corpus directory")
    parser.add_argument("--build-bm25", default="data/bm25_index.json", help="Build and save BM25 index (empty to skip)")
    args = parser.parse_args()

    chroma_host = os.environ.get("CHROMA_HOST", "http://localhost:8001")
    processed_dir = Path(args.processed_dir)

    if not processed_dir.exists():
        log.error(f"Directory not found: {processed_dir}")
        sys.exit(1)

    files = get_files(processed_dir, args.pattern)
    log.info(f"Found {len(files)} files matching '{args.pattern}'")

    if args.dry_run:
        for f in files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name} ({size_mb:.1f} MB)")
        print(f"\n{len(files)} files would be re-indexed into '{args.collection}'")
        return

    # Set embed model and surya dir
    os.environ["EMBED_MODEL"] = args.embed_model
    os.environ["SURYA_CORPUS_DIR"] = args.surya_dir

    # Load embedder
    from shared.embedder import embed_texts
    log.info(f"Embedding model: {args.embed_model}")

    # Test embed to get dimension
    test_emb = embed_texts(["test"])
    embed_dim = len(test_emb[0])
    log.info(f"Embedding dimension: {embed_dim}")

    # Clear and recreate collection
    collection = clear_collection(chroma_host, args.collection)

    total_chunks = 0
    total_time = 0
    failed = []

    for i, f in enumerate(files, 1):
        t0 = time.time()
        log.info(f"[{i}/{len(files)}] {f.name}")

        try:
            doc_type = classify_document(f)
            text, metadata, page_segments = extract_text(f, doc_type)
            if not text.strip():
                raise ValueError("No text extracted")

            chunks = chunk_text(text, metadata, page_segments)
            log.info(f"  {len(text):,} chars → {len(chunks)} chunks")

            # Embed in batches
            texts = [c["text"] for c in chunks]
            all_embeddings = []
            for j in range(0, len(texts), args.batch_size):
                batch = texts[j:j + args.batch_size]
                embs = embed_texts(batch)
                all_embeddings.extend(embs)

            # Store in ChromaDB
            import uuid
            for j in range(0, len(chunks), args.batch_size):
                batch_chunks = chunks[j:j + args.batch_size]
                batch_embs = all_embeddings[j:j + args.batch_size]
                ids = [str(uuid.uuid4()) for _ in batch_chunks]
                documents = [c["text"] for c in batch_chunks]
                metadatas = [c["metadata"] for c in batch_chunks]
                collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=batch_embs,
                    metadatas=metadatas,
                )

            elapsed = time.time() - t0
            total_chunks += len(chunks)
            total_time += elapsed
            log.info(f"  {len(chunks)} chunks stored in {elapsed:.1f}s")

        except Exception:
            log.exception(f"  FAILED: {f.name}")
            failed.append(f.name)

    # Build BM25 index
    if args.build_bm25:
        log.info("Building BM25 index...")
        t0 = time.time()
        sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "api"))
        os.environ["CHROMA_HOST"] = chroma_host
        os.environ["CHROMA_COLLECTION"] = args.collection
        os.environ["BM25_PERSIST_PATH"] = args.build_bm25

        from core.rag import _build_bm25_index, _bm25_cache
        _bm25_cache.clear()
        _build_bm25_index(args.collection)
        bm25_time = time.time() - t0
        log.info(f"BM25 index built in {bm25_time:.1f}s → {args.build_bm25}")
    else:
        bm25_time = 0

    # Summary
    print(f"\n{'='*60}")
    print(f"Re-indexing complete")
    print(f"  Model: {args.embed_model} ({embed_dim}-dim)")
    print(f"  Collection: {args.collection}")
    print(f"  Files: {len(files) - len(failed)}/{len(files)} succeeded")
    print(f"  Total chunks: {total_chunks:,}")
    print(f"  Total time: {total_time:.1f}s (embed) + {bm25_time:.1f}s (bm25)")
    print(f"  Chunks in collection: {collection.count()}")
    if args.build_bm25:
        bm25_path = Path(args.build_bm25)
        if bm25_path.exists():
            size_mb = bm25_path.stat().st_size / 1024 / 1024
            print(f"  BM25 index: {args.build_bm25} ({size_mb:.1f} MB)")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
