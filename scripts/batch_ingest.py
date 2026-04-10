"""Batch ingest PDFs into ChromaDB using the cleaned pipeline.

Stores two collections:
  - {name}         : 384-token chunks with 96-token overlap
  - {name}-windows : 3-sentence sliding windows (fine-grained passage retrieval)

Usage:
    uv run scripts/batch_ingest.py [--fresh]  # --fresh deletes existing collections first
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))

import chromadb
import fitz  # PyMuPDF
import httpx
import tiktoken

from shared.text_normalize import clean_for_ingestion, strip_headers_footers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch_ingest")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ── Config ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PDF_DIR = DATA_DIR / "processed"
STATE_FILE = DATA_DIR / "state" / "documents_v3.jsonl"
CACHE_DIR = DATA_DIR / "cache" / "embeddings"

CHROMA_HOST = os.environ.get("CHROMA_HOST", "http://localhost:8200")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "gutenberg-qwen3-v3")
WINDOW_COLLECTION = COLLECTION_NAME + "-windows"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "384"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "96"))
WINDOW_SENTENCES = int(os.environ.get("WINDOW_SENTENCES", "3"))

EMBED_BATCH_SIZE = 64  # GPU-saturating batch size (benchmarked: 53 texts/s vs 25 at bs=8)
CHROMA_BATCH_SIZE = 500  # ChromaDB handles large batches well over HTTP
EXTRACT_WORKERS = 4  # Parallel PDF extraction (CPU-bound, PyMuPDF releases GIL)
EMBED_WORKERS = 1  # Ollama serializes GPU work; >1 just adds queue overhead

# PDFs to ingest (Deleuze + Macy, excluding D&D)
TARGET_PDFS = [
    p for p in sorted(PDF_DIR.glob("*.pdf"))
    if "Deleuze" in p.name or "Macy" in p.name
]

# ── Helpers ─────────────────────────────────────────────────────────
_enc = tiktoken.get_encoding("cl100k_base")
_SENT_END_RE = re.compile(r"(?<=[.!?])\s+")

# Persistent httpx client with connection pooling
_http_client = httpx.Client(
    timeout=120,
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _text_hash(text: str) -> str:
    """Fast hash for embedding cache keys."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _already_ingested(path: Path) -> bool:
    sha = _file_sha256(path)
    if not STATE_FILE.exists():
        return False
    with open(STATE_FILE) as f:
        for line in f:
            try:
                if json.loads(line).get("sha256") == sha:
                    return True
            except json.JSONDecodeError:
                continue
    return False


def _record_document(path: Path, chunk_count: int, window_count: int):
    import datetime
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "filename": path.name,
        "sha256": _file_sha256(path),
        "chunks": chunk_count,
        "windows": window_count,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(STATE_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


def _trim_to_sentence_start(text: str) -> str:
    m = _SENT_END_RE.search(text)
    if m:
        trimmed = text[m.end():]
        if trimmed.strip():
            return trimmed.strip()
    return text.strip()


# ── Embedding Cache ────────────────────────────────────────────────
# On-disk cache: data/cache/embeddings/<hash>.json
# Avoids re-embedding on re-runs or --fresh (embeddings don't change).

def _load_cache_batch(hashes: list[str]) -> dict[str, list[float]]:
    """Load cached embeddings for a batch of text hashes."""
    results = {}
    for h in hashes:
        cache_file = CACHE_DIR / f"{h}.bin"
        if cache_file.exists():
            import struct
            data = cache_file.read_bytes()
            n_floats = len(data) // 4
            results[h] = list(struct.unpack(f"{n_floats}f", data))
    return results


def _save_cache_batch(items: dict[str, list[float]]):
    """Save embeddings to disk cache."""
    import struct
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for h, vec in items.items():
        cache_file = CACHE_DIR / f"{h}.bin"
        if not cache_file.exists():
            cache_file.write_bytes(struct.pack(f"{len(vec)}f", *vec))


# ── Extract ─────────────────────────────────────────────────────────
def _is_scanned(path: Path) -> bool:
    doc = fitz.open(str(path))
    text_pages = sum(1 for i in range(min(5, len(doc))) if doc[i].get_text("text").strip())
    doc.close()
    return text_pages == 0


def _ocr_pdf(path: Path) -> Path:
    import subprocess
    import tempfile
    out_path = Path(tempfile.mktemp(suffix=".pdf"))
    cmd = [
        "ocrmypdf", "--force-ocr", "--optimize", "1",
        "--jobs", "4", "--deskew", "--clean",
        str(path), str(out_path),
    ]
    log.info(f"  Running OCRmyPDF...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"OCRmyPDF failed: {result.stderr[:200]}")
    return out_path


def extract_pdf(path: Path) -> tuple[str, list[dict]]:
    """Extract text from PDF. Called in worker processes for parallelism."""
    ocr_path = None
    if _is_scanned(path):
        log.info(f"  Scanned PDF, running OCR...")
        ocr_path = _ocr_pdf(path)
        extract_path = ocr_path
    else:
        extract_path = path

    doc = fitz.open(str(extract_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": i + 1, "text": clean_for_ingestion(text)})
    doc.close()

    if ocr_path and ocr_path.exists():
        ocr_path.unlink()

    pages = strip_headers_footers(pages)
    full_text = "\n\n".join(p["text"] for p in pages)
    return full_text, pages


# ── Chunk ───────────────────────────────────────────────────────────
def chunk_text(text: str, source: str, page_segments: list[dict]) -> list[dict]:
    """384-token sliding window chunks with 96-token overlap."""
    tokens = _enc.encode(text)
    total = len(tokens)
    if total == 0:
        return []

    breakpoints = []
    offset = 0
    for i, seg in enumerate(page_segments):
        breakpoints.append((offset, seg["page"]))
        offset += len(seg["text"])
        if i < len(page_segments) - 1:
            offset += 2

    def pages_for(start: int, length: int) -> tuple[int, int]:
        if not breakpoints:
            return (0, 0)
        end = start + length
        ps = pe = breakpoints[0][1]
        for bp_off, bp_page in breakpoints:
            if bp_off <= start:
                ps = bp_page
            if bp_off <= end:
                pe = bp_page
        return ps, pe

    chunks = []
    start = 0
    idx = 0
    stride = CHUNK_SIZE - CHUNK_OVERLAP

    while start < total:
        end = min(start + CHUNK_SIZE, total)
        ct = _enc.decode(tokens[start:end]).strip()
        if not ct:
            start += stride
            continue

        if start > 0 and CHUNK_OVERLAP > 0:
            overlap_text = _enc.decode(tokens[start:start + CHUNK_OVERLAP])
            new_text = _enc.decode(tokens[start + CHUNK_OVERLAP:end]).strip()
            trimmed = _trim_to_sentence_start(overlap_text)
            if trimmed and new_text:
                ct = f"{trimmed} {new_text}"
            elif new_text:
                ct = new_text

        char_start = len(_enc.decode(tokens[:start]))
        ps, pe = pages_for(char_start, len(ct))

        chunks.append({
            "text": ct,
            "metadata": {
                "source": source, "heading": "", "chunk_index": idx,
                "doc_type": "pdf_digital", "page_start": ps, "page_end": pe,
            },
        })
        idx += 1
        if end >= total:
            break
        start += stride

    return chunks


# ── Sentence Windows ────────────────────────────────────────────────
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])')


def build_sentence_windows(text: str, source: str, page_segments: list[dict]) -> list[dict]:
    """Build overlapping N-sentence sliding windows across the full text."""
    sentences = _SENTENCE_SPLIT.split(text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) < WINDOW_SENTENCES:
        return []

    breakpoints = []
    offset = 0
    for i, seg in enumerate(page_segments):
        breakpoints.append((offset, seg["page"]))
        offset += len(seg["text"])
        if i < len(page_segments) - 1:
            offset += 2

    def page_for_pos(pos: int) -> int:
        if not breakpoints:
            return 0
        pg = breakpoints[0][1]
        for bp_off, bp_page in breakpoints:
            if bp_off <= pos:
                pg = bp_page
        return pg

    windows = []
    for i in range(len(sentences) - WINDOW_SENTENCES + 1):
        window_text = " ".join(sentences[i:i + WINDOW_SENTENCES])

        tc = _token_count(window_text)
        if tc < 15 or tc > 512:
            continue

        pos = text.find(sentences[i][:40])
        page = page_for_pos(pos) if pos >= 0 else 0

        windows.append({
            "text": window_text,
            "metadata": {
                "source": source,
                "window_index": i,
                "doc_type": "pdf_digital",
                "page_start": page,
                "page_end": page,
                "granularity": "sentence_window",
            },
        })

    return windows


# ── Embed (cached + batched) ───────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts with on-disk caching and large batches.

    1. Hash all texts, check cache
    2. Embed only uncached texts in GPU-saturating batches of 64
    3. Save new embeddings to cache
    4. Return all embeddings in original order
    """
    hashes = [_text_hash(t) for t in texts]

    # Check cache
    cached = _load_cache_batch(hashes)
    cache_hits = len(cached)

    # Find uncached indices
    uncached_indices = [i for i, h in enumerate(hashes) if h not in cached]

    if uncached_indices:
        uncached_texts = [texts[i] for i in uncached_indices]

        # Embed in large batches, using thread pool for concurrent requests
        new_embeddings = []

        def _embed_batch(batch: list[str]) -> list[list[float]]:
            resp = _http_client.post(
                f"{OLLAMA_HOST}/api/embed",
                json={"model": EMBED_MODEL, "input": batch},
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Embedding failed: {resp.text[:500]}")
            return resp.json()["embeddings"]

        # Split into batches of EMBED_BATCH_SIZE
        batches = [
            uncached_texts[i:i + EMBED_BATCH_SIZE]
            for i in range(0, len(uncached_texts), EMBED_BATCH_SIZE)
        ]

        # Submit concurrent embedding requests to keep GPU pipeline full
        if len(batches) > 1 and EMBED_WORKERS > 1:
            with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
                futures = [pool.submit(_embed_batch, b) for b in batches]
                for f in futures:
                    new_embeddings.extend(f.result())
        else:
            for batch in batches:
                new_embeddings.extend(_embed_batch(batch))

        # Save to cache
        new_cache = {}
        for idx, emb in zip(uncached_indices, new_embeddings):
            h = hashes[idx]
            cached[h] = emb
            new_cache[h] = emb

        if new_cache:
            _save_cache_batch(new_cache)

    if cache_hits > 0:
        log.info(f"    embed cache: {cache_hits}/{len(texts)} hits")

    # Reconstruct in order
    return [cached[h] for h in hashes]


# ── Store ───────────────────────────────────────────────────────────
_chroma_client = None


def _get_client():
    global _chroma_client
    if _chroma_client is None:
        host = CHROMA_HOST.replace("http://", "").replace("https://", "")
        parts = host.split(":")
        _chroma_client = chromadb.HttpClient(
            host=parts[0], port=int(parts[1]) if len(parts) > 1 else 8000
        )
    return _chroma_client


def store_to_collection(name: str, chunks: list[dict], embeddings: list[list[float]]) -> int:
    client = _get_client()
    col = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    ids = [str(uuid.uuid4()) for _ in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    for i in range(0, len(ids), CHROMA_BATCH_SIZE):
        end = i + CHROMA_BATCH_SIZE
        col.add(
            ids=ids[i:end], embeddings=embeddings[i:end],
            documents=documents[i:end], metadatas=metadatas[i:end],
        )
    return col.count()


def delete_collections():
    client = _get_client()
    for name in [COLLECTION_NAME, WINDOW_COLLECTION]:
        try:
            client.delete_collection(name)
            log.info(f"Deleted collection: {name}")
        except Exception:
            pass


# ── Pipeline: per-book processing ──────────────────────────────────
def _process_one_book(pdf: Path, idx: int, total: int) -> dict:
    """Process a single book: extract → chunk + window → embed → store.

    Returns summary dict with counts and timing.
    """
    log.info(f"[{idx}/{total}] {pdf.name}")
    t0 = time.time()

    # Extract
    full_text, pages = extract_pdf(pdf)
    t_extract = time.time() - t0
    log.info(f"  {len(pages)} pages, {len(full_text):,} chars ({t_extract:.1f}s extract)")

    # Chunk + window in parallel threads (both CPU-bound but fast)
    with ThreadPoolExecutor(max_workers=2) as pool:
        chunk_future = pool.submit(chunk_text, full_text, pdf.name, pages)
        window_future = pool.submit(build_sentence_windows, full_text, pdf.name, pages)
        chunks = chunk_future.result()
        windows = window_future.result()

    t_chunk = time.time() - t0 - t_extract
    log.info(f"  {len(chunks)} chunks + {len(windows)} windows ({t_chunk:.1f}s)")

    # Embed chunks and windows concurrently
    chunk_texts = [c["text"] for c in chunks]
    win_texts = [w["text"] for w in windows] if windows else []

    t_emb_start = time.time()
    if win_texts:
        with ThreadPoolExecutor(max_workers=2) as pool:
            chunk_emb_future = pool.submit(embed_texts, chunk_texts)
            win_emb_future = pool.submit(embed_texts, win_texts)
            chunk_embeddings = chunk_emb_future.result()
            win_embeddings = win_emb_future.result()
    else:
        chunk_embeddings = embed_texts(chunk_texts)
        win_embeddings = []
    t_embed = time.time() - t_emb_start
    log.info(f"  embedded in {t_embed:.1f}s")

    # Store chunks and windows concurrently
    t_store_start = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        store_futures = [pool.submit(store_to_collection, COLLECTION_NAME, chunks, chunk_embeddings)]
        if windows and win_embeddings:
            store_futures.append(pool.submit(store_to_collection, WINDOW_COLLECTION, windows, win_embeddings))
        chunk_count = store_futures[0].result()
        win_count = store_futures[1].result() if len(store_futures) > 1 else 0
    t_store = time.time() - t_store_start

    log.info(f"  stored: {chunk_count} chunks, {win_count} windows ({t_store:.1f}s)")

    _record_document(pdf, len(chunks), len(windows))

    elapsed = time.time() - t0
    log.info(f"  DONE in {elapsed:.1f}s (extract={t_extract:.1f} chunk={t_chunk:.1f} embed={t_embed:.1f} store={t_store:.1f})")

    return {
        "name": pdf.name,
        "chunks": len(chunks),
        "windows": len(windows),
        "elapsed": elapsed,
    }


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Delete existing collections first")
    args = parser.parse_args()

    if args.fresh:
        delete_collections()
        if STATE_FILE.exists():
            STATE_FILE.unlink()
            log.info("Deleted state file")

    # Filter to un-ingested PDFs
    to_ingest = []
    skipped = 0
    for pdf in TARGET_PDFS:
        if _already_ingested(pdf):
            skipped += 1
        else:
            to_ingest.append(pdf)

    log.info(f"Target PDFs: {len(TARGET_PDFS)} ({skipped} already ingested, {len(to_ingest)} to process)")
    log.info(f"Collections: {COLLECTION_NAME} + {WINDOW_COLLECTION}")
    log.info(f"Chunk: {CHUNK_SIZE} tokens, {CHUNK_OVERLAP} overlap")
    log.info(f"Windows: {WINDOW_SENTENCES}-sentence sliding")
    log.info(f"Embed batch: {EMBED_BATCH_SIZE}, workers: {EMBED_WORKERS}")
    log.info(f"Cache dir: {CACHE_DIR}")

    if not to_ingest:
        log.info("Nothing to ingest.")
        return

    total_chunks = 0
    total_windows = 0
    total_time = 0
    errors = []

    # Process books sequentially (GPU is the bottleneck, not CPU).
    # Parallelism is within each book: concurrent embed + store.
    for i, pdf in enumerate(to_ingest, 1):
        try:
            result = _process_one_book(pdf, i, len(to_ingest))
            total_chunks += result["chunks"]
            total_windows += result["windows"]
            total_time += result["elapsed"]
        except Exception as e:
            log.error(f"  FAILED: {e}")
            errors.append((pdf.name, str(e)))

    log.info("=" * 60)
    log.info(f"Ingested: {len(to_ingest) - len(errors)} PDFs in {total_time:.0f}s")
    log.info(f"Skipped: {skipped}, Errors: {len(errors)}")
    log.info(f"Chunks: {total_chunks}, Windows: {total_windows}")
    if errors:
        for name, err in errors:
            log.error(f"  {name}: {err}")


if __name__ == "__main__":
    main()
