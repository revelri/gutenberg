"""Weekly context-cite audit (P10).

Samples recent answered queries from the telemetry log, runs context-cite
span attribution, and compares against our 3-tier verification.

Surfaces:
  * False positives — sentences we marked verified but context-cite scores low
  * False negatives — sentences marked [unverified] that context-cite attributes

Offline-only. Cost is O(n_chunks × forward passes) so keep sample size bounded
(defaults to 50). Intended to be driven from ``/loop`` or cron.

Usage:
    python scripts/audit_contextcite.py --input data/telemetry/retrieval.jsonl \\
        --sample 50 --output data/eval/contextcite_reports/YYYY-MM-DD.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import random
import sys
from pathlib import Path

log = logging.getLogger("audit.contextcite")


def _load_samples(path: Path, n: int) -> list[dict]:
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    random.seed(42)
    if len(lines) <= n:
        return lines
    return random.sample(lines, n)


def _run_contextcite(query: str, answer: str, chunks: list[dict]) -> list[dict]:
    """Call context-cite. Returns list of {sentence, top_chunk_id, score}."""
    try:
        from context_cite import ContextCiter  # type: ignore
    except ImportError:
        log.warning("context-cite not installed — emitting empty attribution")
        return []
    try:
        context = "\n\n".join(c.get("text", "") for c in chunks)
        citer = ContextCiter.from_pretrained("gpt2", context=context, query=query)
        citer.response = answer
        attributions = citer.get_attributions(as_dataframe=False)
        out = []
        for attr in attributions or []:
            out.append(
                {
                    "sentence": attr.get("sentence", ""),
                    "top_chunk_index": attr.get("source_index"),
                    "score": float(attr.get("score", 0.0)),
                }
            )
        return out
    except Exception as e:
        log.warning(f"context-cite run failed: {e}")
        return []


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Telemetry JSONL with {query, answer, chunks}")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))
        from api.core.config import settings
        default_n = settings.contextcite_sample_size
    except Exception:
        default_n = 50

    n = args.sample or default_n
    samples = _load_samples(Path(args.input), n)

    report = {"generated_at": datetime.datetime.utcnow().isoformat(), "samples": []}
    for sample in samples:
        attrs = _run_contextcite(
            sample.get("query", ""), sample.get("answer", ""), sample.get("chunks", [])
        )
        report["samples"].append(
            {
                "query": sample.get("query"),
                "attributions": attrs,
                "our_verification": sample.get("verification", {}),
            }
        )

    out_path = Path(
        args.output
        or f"data/eval/contextcite_reports/{datetime.date.today().isoformat()}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    log.info(f"context-cite audit: {len(samples)} samples → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
