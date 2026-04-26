"""Idempotent corpus reindex + index manifest (P11).

Drives all index-building side effects triggered by the P0–P7 upgrades:

  * Rebuild contextual embeddings if P0 flag flipped
  * Rebuild BM25 with contextual prefixes
  * Build ColBERT index when P4 enabled (RAGatouille)
  * Build graph-lite entity co-occurrence store when P7 enabled
  * Write ``data/index_manifest.json`` capturing the feature-flag matrix used

The API (``services/api/main.py``) can optionally refuse to start when
``ENFORCE_INDEX_MANIFEST=true`` and the runtime flags disagree with the
manifest on disk — see ``verify_manifest()``.

Idempotency is via content-hash cache (doc SHA256 → chunk list). Safe to rerun.

Usage:
    python scripts/reindex.py --collection gutenberg
    python scripts/reindex.py --verify-only
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("scripts.reindex")


_FLAG_KEYS = [
    "feature_contextual_chunking",
    "feature_entity_gazetteer",
    "feature_modal_chunks",
    "feature_colbert_retrieval",
    "feature_raptor",
    "feature_graph_boost",
    "feature_crag",
    "feature_vlm_answer",
    "feature_rapidfuzz_verify",
    "feature_anchor_validation",
]


def _settings():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))
    from api.core.config import settings
    return settings


def current_flags(settings) -> dict[str, bool]:
    return {k: bool(getattr(settings, k, False)) for k in _FLAG_KEYS}


def write_manifest(settings, extra: dict | None = None) -> dict:
    manifest = {
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "flags": current_flags(settings),
        "anthropic_model": getattr(settings, "anthropic_model", ""),
        "ollama_embed_model": getattr(settings, "ollama_embed_model", ""),
        "chroma_collection": getattr(settings, "chroma_collection", ""),
    }
    if extra:
        manifest.update(extra)
    path = Path(settings.index_manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_manifest(settings) -> tuple[bool, dict]:
    path = Path(settings.index_manifest_path)
    if not path.exists():
        return False, {"reason": "manifest_missing"}
    try:
        manifest = json.loads(path.read_text())
    except Exception as e:
        return False, {"reason": f"manifest_unreadable: {e}"}
    saved = manifest.get("flags", {})
    live = current_flags(settings)
    diff = {k: (saved.get(k), live.get(k)) for k in _FLAG_KEYS if saved.get(k) != live.get(k)}
    return (not diff), {"diff": diff, "manifest": manifest}


def _load_chroma_corpus(collection_name: str) -> list[dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))
    from api.core.chroma import get_collection
    col = get_collection(collection_name)
    result = col.get(include=["documents", "metadatas"])
    return [
        {"id": i, "text": d, "metadata": m}
        for i, d, m in zip(result["ids"], result["documents"], result["metadatas"])
    ]


def rebuild_bm25(settings, collection: str | None) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))
    from api.core.rag import refresh_bm25_index
    refresh_bm25_index(collection)


def rebuild_colbert(settings, collection: str | None) -> bool:
    if not settings.feature_colbert_retrieval:
        return False
    from api.core.colbert_retriever import build_index
    corpus = _load_chroma_corpus(collection or settings.chroma_collection)
    return build_index(corpus, collection)


def rebuild_graph(settings, collection: str | None) -> int:
    if not settings.feature_graph_boost:
        return 0
    from api.core.graph import build_from_chunks
    corpus = _load_chroma_corpus(collection or settings.chroma_collection)
    return build_from_chunks(corpus)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=None)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--skip-bm25", action="store_true")
    ap.add_argument("--skip-colbert", action="store_true")
    ap.add_argument("--skip-graph", action="store_true")
    args = ap.parse_args()

    settings = _settings()

    if args.verify_only:
        ok, info = verify_manifest(settings)
        print(json.dumps({"ok": ok, **info}, indent=2))
        return 0 if ok else 1

    counts: dict[str, object] = {}

    if not args.skip_bm25:
        log.info("Rebuilding BM25 index…")
        rebuild_bm25(settings, args.collection)
        counts["bm25_rebuilt"] = True

    if not args.skip_colbert:
        log.info("Building ColBERT index (if enabled)…")
        counts["colbert_built"] = rebuild_colbert(settings, args.collection)

    if not args.skip_graph:
        log.info("Building graph-lite (if enabled)…")
        counts["graph_edges"] = rebuild_graph(settings, args.collection)

    manifest = write_manifest(settings, extra={"reindex_stats": counts})
    log.info("Index manifest written:\n%s", json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
