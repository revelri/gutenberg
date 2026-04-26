"""Build a sibling ChromaDB collection with contextual prefixes stripped (P0 ablation).

The source collection was indexed with ``feature_contextual_chunking=True``:
each chunk's embedding was computed over ``context_prefix + "\\n\\n" + text``
and the chunk's metadata carries ``context_prefix``. For a clean ablation
of P0 we need a sibling collection where embeddings reflect the raw
``text`` only (and ``context_prefix`` is stripped from metadata so BM25,
which reconstructs enriched text from metadata, also degrades).

This script copies the chunks 1:1 into ``{source}-noctx``, re-embedding with
``shared.embedder.embed_texts`` on the raw text. No reparse of PDFs.

Usage:
    docker compose exec api python scripts/reindex_noctx.py \
        --source gutenberg-deleuze-corpus
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services"))
sys.path.insert(0, str(ROOT / "services" / "api"))

log = logging.getLogger("scripts.reindex_noctx")


def _strip_context_prefix(metadata: dict) -> dict:
    m = dict(metadata or {})
    m.pop("context_prefix", None)
    return m


def clone_without_context(source: str, dest: str, batch_size: int) -> int:
    from core.chroma import get_collection
    from shared.embedder import embed_texts

    src = get_collection(source)
    total = src.count()
    log.info("source '%s' → %d chunks", source, total)
    if total == 0:
        raise SystemExit(f"source collection '{source}' is empty")

    dst = get_collection(dest)
    if dst.count() > 0:
        raise SystemExit(
            f"destination '{dest}' already has {dst.count()} items — "
            "delete the collection first to rebuild"
        )

    offset = 0
    written = 0
    while offset < total:
        limit = min(batch_size, total - offset)
        got = src.get(
            include=["documents", "metadatas"],
            limit=limit,
            offset=offset,
        )
        ids = got["ids"]
        docs = got["documents"]
        metas = [_strip_context_prefix(m) for m in got["metadatas"]]
        # Re-embed raw text only (no prefix). This is the heart of the ablation.
        embs = embed_texts(docs)
        dst.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        written += len(ids)
        offset += limit
        log.info("  wrote %d / %d", written, total)
    return written


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Existing enriched collection")
    ap.add_argument("--dest", default=None,
                    help="Destination collection (default: {source}-noctx)")
    ap.add_argument("--batch-size", type=int, default=200)
    args = ap.parse_args()

    dest = args.dest or f"{args.source}-noctx"
    written = clone_without_context(args.source, dest, args.batch_size)
    log.info("Done: %d chunks copied into '%s'", written, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
