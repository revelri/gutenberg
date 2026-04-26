"""P5 RAPTOR ablation: leaves-only vs leaves+summaries.

Uses the ``exclude_summary_nodes`` kwarg on ``core.rag.retrieve`` (added
for this eval) to drop ``metadata.level >= 1`` summary chunks from the
fused candidate set, producing a leaves-only arm without reindexing.

Two datasets, two metric schemas:

  * deleuze_exact_citations.json — factual/exact queries. Uses the
    standard page_hit / MRR / source_p metrics from eval_feature_matrix.
    RAPTOR summaries should *not* help here.

  * deleuze_abstractive.json — thematic queries (authored by
    mine_eval_golds.py). Uses ``entity_recall@k`` and
    ``concept_span_recall@k`` — no page anchors. This is where RAPTOR is
    expected to help.

Both arms additionally report ``summary_contribution@k`` — the fraction
of top-k chunks that are summary nodes, a sanity signal that summaries
are in fact being retrieved on abstractive queries.

Usage:
    docker compose exec api python scripts/eval_raptor_ablation.py \
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

log = logging.getLogger("eval.raptor_ablation")

TOP_K = 5


def _chunk_level(ch: dict) -> int:
    lvl = (ch.get("metadata") or {}).get("level", 0)
    try:
        return int(lvl)
    except (TypeError, ValueError):
        return 0


def _chunk_cids(ch: dict) -> set[str]:
    raw = (ch.get("metadata") or {}).get("canonical_ids", "") or ""
    return {c for c in raw.split(",") if c}


def _concat_text(chunks: list[dict]) -> str:
    return " ".join((c.get("text") or "") for c in chunks)


def _abstractive_score(chunks: list[dict], sample: dict, k: int) -> dict:
    from shared.text_normalize import normalize_for_matching

    top = chunks[:k]
    expected_entities = set(sample.get("expected_entities") or [])
    expected_spans = sample.get("expected_concept_spans") or []

    got_cids: set[str] = set()
    for c in top:
        got_cids |= _chunk_cids(c)

    entity_recall = (
        len(got_cids & expected_entities) / len(expected_entities)
        if expected_entities else 0.0
    )

    hay = normalize_for_matching(_concat_text(top))
    span_hits = sum(
        1 for s in expected_spans
        if s and normalize_for_matching(s) in hay
    )
    span_recall = span_hits / len(expected_spans) if expected_spans else 0.0

    summary_share = (
        sum(1 for c in top if _chunk_level(c) >= 1) / len(top)
        if top else 0.0
    )

    return {
        "entity_recall_at_k": round(entity_recall, 4),
        "concept_span_recall_at_k": round(span_recall, 4),
        "summary_contribution_at_k": round(summary_share, 4),
    }


def _exact_score(chunks: list[dict], sample: dict, k: int) -> dict:
    from scripts.eval_feature_matrix import score_sample  # type: ignore

    metrics = score_sample(chunks, sample)
    top = chunks[:k]
    metrics["summary_contribution_at_k"] = round(
        sum(1 for c in top if _chunk_level(c) >= 1) / len(top) if top else 0.0,
        4,
    )
    return metrics


def _aggregate(scores: list[dict]) -> dict:
    if not scores:
        return {}
    keys = scores[0].keys()
    return {k: round(sum(s.get(k, 0.0) for s in scores) / len(scores), 4) for k in keys}


def _run_arm(
    samples: list[dict],
    collection: str | None,
    exclude_summaries: bool,
    scorer,
) -> dict:
    from core.rag import retrieve

    scores: list[dict] = []
    latencies: list[float] = []
    errors = 0
    for s in samples:
        t0 = time.perf_counter()
        try:
            _, chunks = retrieve(
                s["query"],
                collection=collection,
                exclude_summary_nodes=exclude_summaries,
            )
        except Exception as e:
            log.warning("retrieve failed for %r: %s", s["query"][:40], e)
            errors += 1
            chunks = []
        latencies.append((time.perf_counter() - t0) * 1000.0)
        scores.append(scorer(chunks, s, TOP_K))

    agg = _aggregate(scores)
    agg["errors"] = errors
    agg["mean_latency_ms"] = round(sum(latencies) / max(len(latencies), 1), 1)
    return agg


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=None)
    ap.add_argument("--exact", default="data/eval/deleuze_exact_citations.json")
    ap.add_argument("--abstractive", default="data/eval/deleuze_abstractive.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-json", default="data/eval/raptor_ablation.json")
    args = ap.parse_args()

    from scripts.eval_feature_matrix import load_eval  # type: ignore

    exact_samples = load_eval(Path(args.exact))
    abstractive_samples = json.loads(Path(args.abstractive).read_text())
    if args.limit:
        exact_samples = exact_samples[: args.limit]
        abstractive_samples = abstractive_samples[: args.limit]
    log.info("exact=%d  abstractive=%d", len(exact_samples), len(abstractive_samples))

    out: dict = {"collection": args.collection}

    for name, samples, scorer in (
        ("exact", exact_samples, _exact_score),
        ("abstractive", abstractive_samples, _abstractive_score),
    ):
        log.info("── dataset=%s ─ arm=full (leaves+summaries)", name)
        full = _run_arm(samples, args.collection, False, scorer)
        log.info("── dataset=%s ─ arm=leaves_only", name)
        leaves = _run_arm(samples, args.collection, True, scorer)

        deltas = {
            k: round(full.get(k, 0.0) - leaves.get(k, 0.0), 4)
            for k in set(full) & set(leaves)
            if isinstance(full[k], (int, float)) and k not in ("errors",)
        }
        out[name] = {
            "n_samples": len(samples),
            "full": full,
            "leaves_only": leaves,
            "full_minus_leaves": deltas,
        }

    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
