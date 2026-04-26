#!/usr/bin/env python3
"""A/B test harness for preprocessing strategies.

Processes the same corpus with different extraction strategies into
separate ChromaDB collections, then compares timing and quality metrics.

Strategies:
  A — Current Docling (full, GPU)
  B — Optimized Docling (tables/pictures/formulas OFF, batch processing)
  C — OCRmyPDF preprocessing → PyMuPDF (Tesseract OCR, 10-50x faster)

Usage:
    python scripts/ab_test_pipeline.py --strategies A,B
    python scripts/ab_test_pipeline.py --strategies A,B,C --corpus data/processed
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add worker pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "worker"))

from pipeline.detector import classify_document
from pipeline.extractors import extract_text
from pipeline.chunker import chunk_text
from pipeline.embedder import embed_chunks
from pipeline.store import store_chunks, _get_collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ab_test")

CORPUS_DIR = os.environ.get("CORPUS_DIR", "data/processed")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "http://localhost:8200")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
COLLECTION_PREFIX = os.environ.get("COLLECTION_PREFIX", "gutenberg-qwen3")


def free_gpu_for_docling():
    """Unload Ollama models to free VRAM for Docling's layout models."""
    import httpx
    try:
        for model in ["qwen3-embedding:4b", "qwen3:8b", "mxbai-embed-large"]:
            httpx.post(f"{OLLAMA_HOST}/api/generate", json={"model": model, "keep_alive": 0}, timeout=10)
        log.info("Unloaded Ollama models to free GPU VRAM for Docling")
        time.sleep(2)
    except Exception:
        log.warning("Could not unload Ollama models — Docling may OOM on GPU")


def get_corpus_files(corpus_dir: str, pattern: str = "*.pdf") -> list[Path]:
    """Get all PDF files in the corpus directory."""
    d = Path(corpus_dir)
    files = sorted(d.glob(pattern))
    log.info(f"Found {len(files)} files in {corpus_dir}")
    return files


def clear_collection(name: str):
    """Delete a ChromaDB collection if it exists."""
    import chromadb
    host = CHROMA_HOST.replace("http://", "").replace("https://", "")
    parts = host.split(":")
    hostname = parts[0]
    port = int(parts[1]) if len(parts) > 1 else 8000
    client = chromadb.HttpClient(host=hostname, port=port)
    try:
        client.delete_collection(name)
        log.info(f"Cleared collection: {name}")
    except Exception:
        pass


def process_file_strategy_a(path: Path) -> dict:
    """Strategy A: Current Docling (full, default config)."""
    doc_type = classify_document(path)
    t0 = time.time()
    text, metadata, pages = extract_text(path, doc_type, strategy="default")
    extract_time = time.time() - t0
    return {"text": text, "metadata": metadata, "pages": pages, "extract_time": extract_time}


def process_file_strategy_b(path: Path) -> dict:
    """Strategy B: Optimized Docling (skip unused features, batch processing)."""
    doc_type = classify_document(path)
    t0 = time.time()
    text, metadata, pages = extract_text(path, doc_type, strategy="optimized")
    extract_time = time.time() - t0
    return {"text": text, "metadata": metadata, "pages": pages, "extract_time": extract_time}


def process_file_strategy_c(path: Path, tmp_dir: Path) -> dict:
    """Strategy C: OCRmyPDF preprocessing → PyMuPDF."""
    from pipeline.ocrmypdf_preprocess import preprocess, ocrmypdf_available

    doc_type = classify_document(path)

    if doc_type == "pdf_scanned" and ocrmypdf_available():
        t0 = time.time()
        preprocessed = preprocess(path, tmp_dir)
        # Now classify again — should be pdf_digital after OCR
        doc_type = classify_document(preprocessed)
        text, metadata, pages = extract_text(preprocessed, doc_type)
        extract_time = time.time() - t0
        # Keep original source name
        metadata["source"] = path.name
    else:
        # Digital PDFs go through normal path
        t0 = time.time()
        text, metadata, pages = extract_text(path, doc_type)
        extract_time = time.time() - t0

    return {"text": text, "metadata": metadata, "pages": pages, "extract_time": extract_time}


def run_strategy(
    strategy: str,
    files: list[Path],
    collection_name: str,
    base_collection: str | None = None,
) -> list[dict]:
    """Run a single strategy on all files and store results."""
    clear_collection(collection_name)
    results = []
    tmp_dir = Path("/tmp/ab_test_ocrmypdf")

    # Free GPU VRAM before strategies that use Docling (A, B)
    if strategy in ("A", "B"):
        free_gpu_for_docling()

    for i, path in enumerate(files):
        log.info(f"[{strategy}] [{i+1}/{len(files)}] {path.name}")
        file_result = {"file": path.name, "strategy": strategy}

        try:
            # Extract
            if strategy == "A":
                extracted = process_file_strategy_a(path)
            elif strategy == "B":
                extracted = process_file_strategy_b(path)
            elif strategy == "C":
                extracted = process_file_strategy_c(path, tmp_dir)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            file_result["extract_time"] = round(extracted["extract_time"], 2)
            file_result["doc_type"] = extracted["metadata"].get("doc_type", "unknown")
            file_result["text_length"] = len(extracted["text"])

            # Chunk
            t0 = time.time()
            chunks = chunk_text(extracted["text"], extracted["metadata"], extracted["pages"])
            chunk_time = time.time() - t0
            file_result["chunk_count"] = len(chunks)
            file_result["chunk_time"] = round(chunk_time, 2)

            # Embed
            t0 = time.time()
            embeddings = embed_chunks([c["text"] for c in chunks])
            embed_time = time.time() - t0
            file_result["embed_time"] = round(embed_time, 2)

            # Store
            t0 = time.time()
            store_chunks(chunks, embeddings, collection_name=collection_name)
            store_time = time.time() - t0
            file_result["store_time"] = round(store_time, 2)

            file_result["total_time"] = round(
                extracted["extract_time"] + chunk_time + embed_time + store_time
                , 2
            )
            file_result["status"] = "OK"

        except Exception as e:
            log.exception(f"Failed: {path.name}")
            file_result["status"] = "FAILED"
            file_result["error"] = str(e)

        results.append(file_result)

    return results


def print_comparison(all_results: dict[str, list[dict]]):
    """Print comparison table across strategies."""
    print(f"\n{'='*80}")
    print(f"  A/B TEST RESULTS")
    print(f"{'='*80}")

    # Summary per strategy
    print(f"\n  {'Strategy':<12} {'Files':>6} {'Chunks':>7} {'Ext(s)':>8} {'Total(s)':>9} {'Status':>8}")
    print(f"  {'─'*55}")

    for strat, results in all_results.items():
        ok = [r for r in results if r["status"] == "OK"]
        total_chunks = sum(r.get("chunk_count", 0) for r in ok)
        total_extract = sum(r.get("extract_time", 0) for r in ok)
        total_time = sum(r.get("total_time", 0) for r in ok)
        failed = len(results) - len(ok)
        status = f"{len(ok)} OK" + (f", {failed} FAIL" if failed else "")

        print(f"  {strat:<12} {len(results):>6} {total_chunks:>7} {total_extract:>7.0f}s {total_time:>8.0f}s {status:>8}")

    # Per-file breakdown for scanned PDFs (where strategies differ most)
    print(f"\n{'='*80}")
    print(f"  SCANNED PDF DETAIL")
    print(f"{'='*80}")

    for strat, results in all_results.items():
        scanned = [r for r in results if r.get("doc_type") == "pdf_scanned"]
        if scanned:
            print(f"\n  Strategy {strat}:")
            for r in scanned:
                ext = r.get("extract_time", 0)
                total = r.get("total_time", 0)
                chunks = r.get("chunk_count", 0)
                print(f"    {r['file'][:50]:<52} ext={ext:.0f}s  total={total:.0f}s  chunks={chunks}")


def main():
    parser = argparse.ArgumentParser(description="A/B test preprocessing strategies")
    parser.add_argument("--strategies", default="A,B", help="Comma-separated strategy letters (A,B,C,D)")
    parser.add_argument("--corpus", default=CORPUS_DIR, help="Path to corpus directory")
    parser.add_argument("--pattern", default="*Deleuze*.pdf", help="Glob pattern for files")
    parser.add_argument("--prefix", default=COLLECTION_PREFIX, help="Collection name prefix")
    parser.add_argument("--output", default=None, help="Save results JSON to file")
    args = parser.parse_args()

    strategies = [s.strip().upper() for s in args.strategies.split(",")]
    files = get_corpus_files(args.corpus, args.pattern)

    if not files:
        log.error(f"No files found matching {args.pattern} in {args.corpus}")
        sys.exit(1)

    all_results = {}
    for strat in strategies:
        collection_name = f"{args.prefix}-strat{strat}"
        log.info(f"\n{'='*60}")
        log.info(f"RUNNING STRATEGY {strat} → collection: {collection_name}")
        log.info(f"{'='*60}")
        all_results[strat] = run_strategy(strat, files, collection_name)

    print_comparison(all_results)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        log.info(f"Results saved to {out}")


if __name__ == "__main__":
    main()
