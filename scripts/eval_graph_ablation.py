"""P7 graph-boost ablation on multi-entity queries.

Each query in ``data/eval/multi_entity_queries.json`` names ≥2 canonical
entities and carries ``expected_edges[]`` — pairs (a, b) that exist in
the co-occurrence graph and whose *both* endpoints appear in the query.

Ablation:

  * arm A — ``feature_graph_boost=False`` (baseline; passage scoring
    still uses direct entity overlap, just no 1-hop neighbor boost).
  * arm B — ``feature_graph_boost=True`` (P7 on).

Metrics:

  * ``entity_recall@k`` — fraction of ``expected_entities`` present in
    top-k chunks' ``canonical_ids``.
  * ``co_occurrence_recall@k`` — fraction of ``expected_edges`` such
    that some retrieved chunk in top-k contains *both* endpoints in its
    ``canonical_ids`` (the edge "materialized" at retrieval time).

Usage:
    docker compose exec api python scripts/eval_graph_ablation.py \
        --collection gutenberg-deleuze-corpus
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services"))
sys.path.insert(0, str(ROOT / "services" / "api"))
sys.path.insert(0, str(ROOT / "scripts"))

log = logging.getLogger("eval.graph_ablation")

TOP_K = 10


def _chunk_cids(ch: dict) -> set[str]:
    raw = (ch.get("metadata") or {}).get("canonical_ids", "") or ""
    return {c for c in raw.split(",") if c}


def _score(chunks: list[dict], sample: dict, k: int) -> dict:
    top = chunks[:k]
    expected_entities = set(sample.get("expected_entities") or [])
    expected_edges = [tuple(e) for e in (sample.get("expected_edges") or []) if len(e) == 2]

    got_cids: set[str] = set()
    per_chunk_cids: list[set[str]] = []
    for c in top:
        cids = _chunk_cids(c)
        per_chunk_cids.append(cids)
        got_cids |= cids

    entity_recall = (
        len(got_cids & expected_entities) / len(expected_entities)
        if expected_entities else 0.0
    )

    edge_hits = 0
    for a, b in expected_edges:
        if any({a, b} <= c for c in per_chunk_cids):
            edge_hits += 1
    edge_recall = edge_hits / len(expected_edges) if expected_edges else 0.0

    return {
        "entity_recall_at_k": round(entity_recall, 4),
        "co_occurrence_recall_at_k": round(edge_recall, 4),
    }


def _aggregate(scores: list[dict]) -> dict:
    if not scores:
        return {}
    keys = scores[0].keys()
    return {k: round(sum(s.get(k, 0.0) for s in scores) / len(scores), 4) for k in keys}


def _run_arm(samples: list[dict], collection: str | None, graph_on: bool) -> dict:
    from scripts.eval_feature_matrix import override_flags  # type: ignore
    from core.rag import retrieve

    scores: list[dict] = []
    latencies: list[float] = []
    errors = 0
    # P7 requires the gazetteer flag — co-ping them together so query resolution
    # surfaces canonical_ids during passage scoring.
    overrides = {
        "feature_graph_boost": graph_on,
        "feature_entity_gazetteer": True,
    }
    with override_flags(**overrides):
        for s in samples:
            t0 = time.perf_counter()
            try:
                _, chunks = retrieve(s["query"], collection=collection)
            except Exception as e:
                log.warning("retrieve failed for %r: %s", s["query"][:40], e)
                errors += 1
                chunks = []
            latencies.append((time.perf_counter() - t0) * 1000.0)
            scores.append(_score(chunks, s, TOP_K))

    agg = _aggregate(scores)
    agg["errors"] = errors
    agg["mean_latency_ms"] = round(sum(latencies) / max(len(latencies), 1), 1)
    return agg


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=None)
    ap.add_argument("--eval", default="data/eval/multi_entity_queries.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-json", default="data/eval/graph_ablation.json")
    args = ap.parse_args()

    samples = json.loads(Path(args.eval).read_text())
    if args.limit:
        samples = samples[: args.limit]
    log.info("Loaded %d multi-entity queries from %s", len(samples), args.eval)

    log.info("→ arm off (graph_boost=False)")
    off = _run_arm(samples, args.collection, graph_on=False)
    log.info("→ arm on  (graph_boost=True)")
    on = _run_arm(samples, args.collection, graph_on=True)

    deltas = {
        k: round(on.get(k, 0.0) - off.get(k, 0.0), 4)
        for k in set(on) & set(off)
        if isinstance(on[k], (int, float)) and k not in ("errors",)
    }

    out = {
        "eval": args.eval,
        "collection": args.collection,
        "n_samples": len(samples),
        "off": off,
        "on": on,
        "on_minus_off": deltas,
    }
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", path)
    log.info("Δ entity_recall@%d=%+.3f  Δ co_occurrence_recall@%d=%+.3f",
             TOP_K, deltas.get("entity_recall_at_k", 0.0),
             TOP_K, deltas.get("co_occurrence_recall_at_k", 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
