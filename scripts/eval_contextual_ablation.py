"""P0 contextual-chunking ablation: ctx vs. noctx collections.

Runs an exact-citation gold set against:

  * arm A — the enriched collection (embeddings include context_prefix,
    BM25 tokens include context_prefix).
  * arm B — a sibling collection built by ``scripts/reindex_noctx.py``
    (raw text embeddings, no context_prefix in metadata → BM25 also sees
    raw text only).

Both arms hit ``core.rag.retrieve`` in-process. Reports per-arm MRR,
page_hit@5, source_p@1 and signed deltas. Scoring reuses
``scripts.eval_feature_matrix`` so metric definitions are identical to
the feature matrix.

Usage:
    docker compose exec api python scripts/eval_contextual_ablation.py \
        --eval data/eval/deleuze_exact_citations.json \
        --ctx-collection gutenberg-deleuze-corpus
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

log = logging.getLogger("eval.contextual_ablation")


def _run_arm(samples: list[dict], collection: str) -> dict:
    from core.rag import retrieve
    from scripts.eval_feature_matrix import score_sample, aggregate  # type: ignore

    scores: list[dict] = []
    latencies: list[float] = []
    errors = 0
    for s in samples:
        t0 = time.perf_counter()
        try:
            _, chunks = retrieve(s["query"], collection=collection)
        except Exception as e:
            log.warning("retrieve failed for %r: %s", s["query"][:40], e)
            errors += 1
            chunks = []
        latencies.append((time.perf_counter() - t0) * 1000.0)
        scores.append(score_sample(chunks, s))

    agg = aggregate(scores)
    agg["errors"] = errors
    agg["mean_latency_ms"] = round(sum(latencies) / max(len(latencies), 1), 1)
    return agg


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/eval/deleuze_exact_citations.json")
    ap.add_argument("--ctx-collection", required=True,
                    help="Enriched collection (P0 on)")
    ap.add_argument("--noctx-collection", default=None,
                    help="Sibling collection (default: {ctx}-noctx)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-json", default="data/eval/contextual_ablation.json")
    args = ap.parse_args()

    from scripts.eval_feature_matrix import load_eval  # type: ignore

    samples = load_eval(Path(args.eval))
    if args.limit:
        samples = samples[: args.limit]
    log.info("Loaded %d samples from %s", len(samples), args.eval)

    ctx = args.ctx_collection
    noctx = args.noctx_collection or f"{ctx}-noctx"

    log.info("→ arm ctx  (collection=%s)", ctx)
    ctx_metrics = _run_arm(samples, ctx)
    log.info("→ arm noctx (collection=%s)", noctx)
    noctx_metrics = _run_arm(samples, noctx)

    deltas = {}
    for k in ("mrr", "page_hit_5", "source_p_1", "source_p_5", "quote_hit_5"):
        deltas[k] = round(ctx_metrics.get(k, 0.0) - noctx_metrics.get(k, 0.0), 4)

    out = {
        "eval": args.eval,
        "n_samples": len(samples),
        "ctx_collection": ctx,
        "noctx_collection": noctx,
        "ctx": ctx_metrics,
        "noctx": noctx_metrics,
        "ctx_minus_noctx": deltas,
    }
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", path)
    log.info("Δ MRR=%+.3f  Δ page_hit@5=%+.3f  Δ source_p@1=%+.3f",
             deltas["mrr"], deltas["page_hit_5"], deltas["source_p_1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
