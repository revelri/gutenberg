"""Weekly retrieval telemetry report (P12).

Aggregates ``data/telemetry/retrieval.jsonl`` over the last N days into a
flag-matrix summary: per flag combination, count of queries, mean retrieval
latency, mean fused-list size, mean top-k size.

Intended for eval-driven rollout — each feature flag lands in report-only
mode first, then is flipped on after this report shows lift.

Usage:
    python scripts/weekly_report.py --days 7 --output data/telemetry/report.md
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import statistics
import sys
from pathlib import Path

log = logging.getLogger("scripts.weekly_report")


def _load(path: Path, since_ts: float) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
            if ev.get("ts", 0) >= since_ts:
                out.append(ev)
        except Exception:
            continue
    return out


def _flag_key(ev: dict) -> tuple:
    flags = ev.get("flags") or {}
    return tuple(sorted((k, bool(v)) for k, v in flags.items()))


def aggregate(events: list[dict]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = {}
    for ev in events:
        buckets.setdefault(_flag_key(ev), []).append(ev)
    rows: list[dict] = []
    for key, group in buckets.items():
        latencies = [
            sum((ev.get("timings_ms") or {}).values()) for ev in group
        ]
        fused = [ev.get("n_fused", 0) for ev in group]
        colbert = [ev.get("n_colbert", 0) for ev in group]
        rows.append(
            {
                "flags": dict(key),
                "n_queries": len(group),
                "mean_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0.0,
                "mean_fused": round(statistics.mean(fused), 1) if fused else 0.0,
                "mean_colbert": round(statistics.mean(colbert), 1) if colbert else 0.0,
            }
        )
    rows.sort(key=lambda r: -r["n_queries"])
    return rows


def render_markdown(rows: list[dict], since: datetime.datetime) -> str:
    lines = [
        f"# Retrieval telemetry report",
        f"_Since {since.isoformat()}, {sum(r['n_queries'] for r in rows)} queries_",
        "",
        "| N | mean latency (ms) | mean fused | mean ColBERT | flags |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        flag_str = ", ".join(
            k.replace("feature_", "") for k, v in sorted(r["flags"].items()) if v
        ) or "_baseline_"
        lines.append(
            f"| {r['n_queries']} | {r['mean_latency_ms']} | {r['mean_fused']} | {r['mean_colbert']} | {flag_str} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default="data/telemetry/report.md")
    args = ap.parse_args()

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))
        from api.core.config import settings
        default_path = settings.telemetry_log_path
    except Exception:
        default_path = "data/telemetry/retrieval.jsonl"

    input_path = Path(args.input or default_path)
    since = datetime.datetime.utcnow() - datetime.timedelta(days=args.days)
    events = _load(input_path, since.timestamp())
    rows = aggregate(events)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(rows, since))
    log.info(f"report: {len(events)} events, {len(rows)} flag buckets → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
