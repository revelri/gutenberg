"""Graph-lite entity co-occurrence store.

Entities are the canonical_ids produced by the gazetteer (P1); an edge between
two canonical_ids means they appear in the same chunk. The edge weight is the
chunk count. Retrieval applies a bounded boost when a chunk's canonical_ids
overlap the 1-hop neighborhood of the query's canonical_ids.

Storage: SQLite, path from ``settings.graph_db_path``. Schema:

  nodes(canonical_id TEXT PRIMARY KEY, label TEXT)
  edges(src TEXT, dst TEXT, weight INTEGER, PRIMARY KEY (src, dst))

No community summarization, no global graph queries — just a sparse, fast
1-hop lookup that's cheap to rebuild from chunk metadata.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from functools import lru_cache

from core.config import settings

log = logging.getLogger("gutenberg.graph")

_LOCK = threading.Lock()


def _conn() -> sqlite3.Connection:
    path = settings.graph_db_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    c = sqlite3.connect(path, check_same_thread=False)
    c.execute("CREATE TABLE IF NOT EXISTS nodes (canonical_id TEXT PRIMARY KEY, label TEXT)")
    c.execute(
        "CREATE TABLE IF NOT EXISTS edges (src TEXT, dst TEXT, weight INTEGER, "
        "PRIMARY KEY (src, dst))"
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src)")
    return c


def build_from_chunks(chunks: list[dict]) -> int:
    """Rebuild the graph from the current chunk corpus.

    Expects each chunk's metadata to carry ``canonical_ids`` (comma-joined).
    Returns the number of edges written.
    """
    from shared.gazetteer import get_canonical_labels

    labels = get_canonical_labels()
    edge_counts: dict[tuple[str, str], int] = {}
    node_labels: dict[str, str] = {}
    for ch in chunks:
        cid_raw = (ch.get("metadata") or {}).get("canonical_ids", "")
        if not cid_raw:
            continue
        cids = sorted(set(cid_raw.split(",")))
        for cid in cids:
            node_labels.setdefault(cid, labels.get(cid, cid))
        for i, a in enumerate(cids):
            for b in cids[i + 1 :]:
                edge_counts[(a, b)] = edge_counts.get((a, b), 0) + 1

    with _LOCK:
        c = _conn()
        c.execute("DELETE FROM nodes")
        c.execute("DELETE FROM edges")
        c.executemany(
            "INSERT INTO nodes(canonical_id, label) VALUES (?, ?)",
            list(node_labels.items()),
        )
        c.executemany(
            "INSERT INTO edges(src, dst, weight) VALUES (?, ?, ?)",
            [(a, b, w) for (a, b), w in edge_counts.items()],
        )
        c.commit()
        c.close()
    log.info(f"Graph built: {len(node_labels)} nodes, {len(edge_counts)} edges")
    return len(edge_counts)


@lru_cache(maxsize=512)
def neighbors(canonical_id: str) -> frozenset[str]:
    """Return 1-hop neighbors of ``canonical_id`` (empty if unknown)."""
    try:
        with _LOCK:
            c = _conn()
            rows = c.execute(
                "SELECT dst FROM edges WHERE src = ? UNION SELECT src FROM edges WHERE dst = ?",
                (canonical_id, canonical_id),
            ).fetchall()
            c.close()
        return frozenset(r[0] for r in rows)
    except Exception as e:
        log.debug(f"graph neighbors lookup failed: {e}")
        return frozenset()


def expand(canonical_ids: list[str]) -> set[str]:
    """Return ``canonical_ids`` plus their 1-hop neighbors (bounded)."""
    out: set[str] = set(canonical_ids)
    for cid in canonical_ids:
        out |= neighbors(cid)
    return out
