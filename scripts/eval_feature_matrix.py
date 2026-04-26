"""Feature-matrix evaluation for P0–P12 upgrades.

Replays existing eval sets (``data/eval/*.json``) across every new feature
flag — individually and combined — and produces a taxonomic report.

Features are taxonomized by when they take effect:

  * RETRIEVAL  — flag flip changes query-time behaviour only; A/B runs cheap.
                 Covered: P1 gazetteer, P2 rapidfuzz, P7 graph boost,
                          P9 CRAG-lite, P12 telemetry.
  * INDEX      — requires rebuilding the ChromaDB/BM25/ColBERT index before
                 flag effect is visible. We report what the flag would change
                 but do not silently reindex.
                 Covered: P0 contextual chunking, P3 modal chunks,
                          P4 ColBERT retrieval, P5 RAPTOR summaries.
  * ANSWER     — takes effect at generation time, not retrieval.
                 Covered: P8 VLM answer.
  * EVAL_ONLY  — offline scripts; not runtime feature toggles.
                 Covered: P6 ALCE, P10 contextcite audit, P11 reindex,
                          P12 weekly report.

Metrics computed per configuration:

  * source_p@1/@5/@10 — top-K contains a chunk whose ``source`` matches gold
  * page_hit@1/@5/@10 — top-K contains a chunk whose page span covers gold page
  * quote_hit@5/@10   — top-K contains the gold quote (fuzzy) verbatim
  * MRR                — 1/rank of first page-matching chunk
  * latency_ms         — mean retrieval latency

Output: ``data/eval/feature_matrix.md`` + ``data/eval/feature_matrix.json``.

Usage:
    # Dry run — print taxonomy, no retrieval calls
    python scripts/eval_feature_matrix.py --dry-run

    # Run only retrieval-time A/B against the default eval set
    python scripts/eval_feature_matrix.py --class retrieval

    # Full matrix against a specific eval file
    python scripts/eval_feature_matrix.py --eval data/eval/deleuze_exact_citations.json

    # Include index-dependent features (must have reindexed first)
    python scripts/eval_feature_matrix.py --include-index
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services"))
sys.path.insert(0, str(ROOT / "services" / "api"))

log = logging.getLogger("eval.feature_matrix")


# ── Feature taxonomy ───────────────────────────────────────────────────

FEATURES: list[dict] = [
    # (id, class, flag, human_name, depends_on_reindex)
    {"id": "P0", "cls": "INDEX",     "flag": "feature_contextual_chunking", "name": "Contextual chunking"},
    {"id": "P1", "cls": "RETRIEVAL", "flag": "feature_entity_gazetteer",    "name": "SpaCy gazetteer + alias map"},
    {"id": "P2", "cls": "RETRIEVAL", "flag": "feature_rapidfuzz_verify",    "name": "rapidfuzz + anchor validation"},
    {"id": "P3", "cls": "INDEX",     "flag": "feature_modal_chunks",        "name": "Modal chunks (tables, equations)"},
    {"id": "P4", "cls": "INDEX",     "flag": "feature_colbert_retrieval",   "name": "ColBERTv2 late-interaction"},
    {"id": "P5", "cls": "INDEX",     "flag": "feature_raptor",              "name": "RAPTOR summary tree"},
    {"id": "P6", "cls": "EVAL_ONLY", "flag": None,                          "name": "ALCE NLI citation eval"},
    {"id": "P7", "cls": "RETRIEVAL", "flag": "feature_graph_boost",         "name": "Graph-lite entity neighborhood"},
    {"id": "P8", "cls": "ANSWER",    "flag": "feature_vlm_answer",          "name": "VLM-enhanced answer"},
    {"id": "P9", "cls": "RETRIEVAL", "flag": "feature_crag",                "name": "CRAG-lite gate + rewrite"},
    {"id": "P10","cls": "EVAL_ONLY", "flag": None,                          "name": "Contextcite offline audit"},
    {"id": "P11","cls": "EVAL_ONLY", "flag": None,                          "name": "Reindex automation + manifest"},
    {"id": "P12","cls": "EVAL_ONLY", "flag": "telemetry_enabled",           "name": "Structured telemetry"},
]


# ── Runtime flag override helper ────────────────────────────────────────

@contextmanager
def override_flags(**overrides: Any):
    """Temporarily override attrs on the live ``settings`` singleton and
    clear the in-process query cache so flag changes are not masked by
    cached results from a previous configuration."""
    from core.config import settings
    old = {k: getattr(settings, k) for k in overrides}
    for k, v in overrides.items():
        setattr(settings, k, v)
    # Drop cached retrieval results — each config must re-run retrieval.
    # Keep the BM25 index warm; rebuilding it per config is ~30-60s of waste.
    try:
        from core import rag as _rag
        with _rag._query_lock:
            _rag._query_cache.clear()
    except Exception:
        pass
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(settings, k, v)


# ── Eval data loading ───────────────────────────────────────────────────

def load_eval(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    # Accept both top-level list and {"questions": [...]} shapes.
    if isinstance(data, dict):
        for k in ("questions", "items", "samples", "data"):
            if isinstance(data.get(k), list):
                data = data[k]
                break
    normalized: list[dict] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        # Tolerant field mapping.
        query = entry.get("query") or entry.get("question") or entry.get("q")
        source = entry.get("source") or entry.get("gold_source") or entry.get("title")
        page = (
            entry.get("pdf_page")
            or entry.get("page")
            or entry.get("gold_page")
            or entry.get("page_start")
        )
        gold_quote = (
            entry.get("ground_truth")
            or entry.get("quote")
            or entry.get("expected")
            or entry.get("answer")
            or ""
        )
        if not query:
            continue
        normalized.append(
            {
                "query": query,
                "source": source,
                "page": int(page) if page else None,
                "gold_quote": gold_quote,
                "label": entry.get("label", ""),
            }
        )
    return normalized


# ── Metric computation ──────────────────────────────────────────────────

_PAGE_RE = re.compile(r"\d+")


def _source_match(chunk: dict, gold_source: str | None) -> bool:
    if not gold_source:
        return False
    chunk_src = (chunk.get("metadata") or {}).get("source", "")
    if not chunk_src:
        return False
    gs = gold_source.lower()
    cs = chunk_src.lower()
    # Tolerate year/ext drift: match on the title stem.
    gs_core = re.sub(r"^\d{4}\s+|\.pdf$", "", gs).strip()
    cs_core = re.sub(r"^\d{4}\s+|\.pdf$", "", cs).strip()
    return gs_core[:20] in cs_core or cs_core[:20] in gs_core


def _page_match(chunk: dict, gold_page: int | None) -> bool:
    if not gold_page:
        return False
    meta = chunk.get("metadata") or {}
    ps = meta.get("page_start") or 0
    pe = meta.get("page_end") or ps
    return ps and gold_page - 1 <= pe and gold_page + 1 >= ps


def _quote_hit(chunk: dict, gold_quote: str) -> bool:
    if not gold_quote:
        return False
    needle = re.sub(r"\s+", " ", gold_quote[:40].lower()).strip()
    hay = re.sub(r"\s+", " ", (chunk.get("text") or "").lower())
    return needle in hay


def score_sample(chunks: list[dict], sample: dict) -> dict:
    src_hit = [_source_match(c, sample["source"]) for c in chunks]
    page_hit = [_page_match(c, sample["page"]) for c in chunks]
    quote_hit = [_quote_hit(c, sample["gold_quote"]) for c in chunks]

    def at(k: int, arr: list[bool]) -> float:
        return 1.0 if any(arr[:k]) else 0.0

    mrr = 0.0
    for i, hit in enumerate(page_hit, start=1):
        if hit:
            mrr = 1.0 / i
            break
    return {
        "source_p_1": at(1, src_hit),
        "source_p_5": at(5, src_hit),
        "source_p_10": at(10, src_hit),
        "page_hit_1": at(1, page_hit),
        "page_hit_5": at(5, page_hit),
        "page_hit_10": at(10, page_hit),
        "quote_hit_5": at(5, quote_hit),
        "quote_hit_10": at(10, quote_hit),
        "mrr": mrr,
    }


def aggregate(scores: list[dict]) -> dict:
    if not scores:
        return {}
    keys = scores[0].keys()
    return {k: round(sum(s[k] for s in scores) / len(scores), 4) for k in keys}


# ── Configuration matrix ────────────────────────────────────────────────

def build_matrix(include_index: bool, include_answer: bool, cls_filter: str | None) -> list[dict]:
    """Return the list of configurations to evaluate."""
    configs: list[dict] = [{"id": "baseline", "label": "baseline (no flags)", "overrides": {}}]

    selected_features = []
    for f in FEATURES:
        if f["flag"] is None:
            continue
        if cls_filter and f["cls"] != cls_filter.upper():
            continue
        if f["cls"] == "INDEX" and not include_index:
            continue
        if f["cls"] == "ANSWER" and not include_answer:
            continue
        if f["cls"] == "EVAL_ONLY":
            continue
        selected_features.append(f)

    for f in selected_features:
        configs.append(
            {
                "id": f["id"],
                "label": f"{f['id']} only — {f['name']}",
                "overrides": {f["flag"]: True},
                "class": f["cls"],
            }
        )

    if len(selected_features) > 1:
        configs.append(
            {
                "id": "all-retrieval",
                "label": "all retrieval-time flags on",
                "overrides": {
                    f["flag"]: True
                    for f in selected_features
                    if f["cls"] == "RETRIEVAL"
                },
            }
        )
        configs.append(
            {
                "id": "all-on",
                "label": "all flags on",
                "overrides": {f["flag"]: True for f in selected_features},
            }
        )
    return configs


# ── Retrieval driver ────────────────────────────────────────────────────

def _retrieve_inprocess():
    """Return the in-process ``retrieve`` callable or raise with a hint."""
    try:
        from core.rag import retrieve
        return retrieve
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"in-process retrieve() unavailable ({e}). Either:\n"
            "  1. Run inside the API container: "
            "`docker compose exec api python scripts/eval_feature_matrix.py …`\n"
            "  2. Use the remote API: add `--api-url http://localhost:8002`"
        )


def _retrieve_via_api(api_url: str, query: str, collection: str | None) -> list[dict]:
    """Call the running API's /v1/chat/completions with stream=False and harvest
    retrieved sources from the non-stream response pipeline. Falls back to a
    direct /api/search endpoint when present."""
    import httpx

    # Preferred: a dedicated search endpoint (services/api/routers/search.py).
    try:
        r = httpx.post(
            f"{api_url.rstrip('/')}/api/search",
            json={"query": query, "collection": collection},
            timeout=60,
        )
        if r.status_code == 200:
            payload = r.json()
            return payload.get("results") or payload.get("chunks") or []
    except Exception:
        pass
    # Fallback: use chat endpoint with tiny max_tokens (we only need retrieval side-effect).
    model = f"gutenberg-rag/{collection}" if collection else "gutenberg-rag"
    r = httpx.post(
        f"{api_url.rstrip('/')}/v1/chat/completions",
        json={
            "model": model,
            "stream": False,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": query}],
        },
        timeout=120,
    )
    r.raise_for_status()
    # The non-stream path does not publicly return sources; we approximate
    # by falling back to an empty list — caller should prefer /api/search.
    return []


def run_config(
    samples: list[dict],
    overrides: dict,
    collection: str | None,
    api_url: str | None = None,
) -> dict:
    sample_scores: list[dict] = []
    latencies: list[float] = []
    errors = 0

    if api_url:
        for s in samples:
            t0 = time.perf_counter()
            try:
                chunks = _retrieve_via_api(api_url, s["query"], collection)
            except Exception as e:
                log.warning(f"api retrieve failed: {e}")
                errors += 1
                chunks = []
            latencies.append((time.perf_counter() - t0) * 1000.0)
            sample_scores.append(score_sample(chunks, s))
    else:
        retrieve = _retrieve_inprocess()
        with override_flags(**overrides):
            for s in samples:
                t0 = time.perf_counter()
                try:
                    _, chunks = retrieve(s["query"], collection=collection)
                except Exception as e:
                    log.warning(f"retrieve failed for '{s['query'][:40]}…': {e}")
                    errors += 1
                    chunks = []
                latencies.append((time.perf_counter() - t0) * 1000.0)
                sample_scores.append(score_sample(chunks, s))
    agg = aggregate(sample_scores)
    agg["errors"] = errors
    agg["mean_latency_ms"] = round(sum(latencies) / max(len(latencies), 1), 1)
    return agg


# ── Report rendering ────────────────────────────────────────────────────

def render_taxonomy() -> str:
    lines = ["## Feature taxonomy", "", "| P | Class | Name | Flag | Reindex required |", "|---|---|---|---|---|"]
    for f in FEATURES:
        reindex = "yes" if f["cls"] == "INDEX" else "no"
        flag = f["flag"] or "_(script)_"
        lines.append(f"| {f['id']} | {f['cls']} | {f['name']} | `{flag}` | {reindex} |")
    return "\n".join(lines) + "\n"


def render_matrix(eval_name: str, n: int, baseline: dict, rows: list[dict]) -> str:
    cols = [
        "page_hit_1", "page_hit_5", "page_hit_10",
        "source_p_5", "quote_hit_5", "mrr", "mean_latency_ms",
    ]
    header = "| config | " + " | ".join(cols) + " | Δ page_hit_5 |"
    sep = "|---" * (len(cols) + 2) + "|"
    lines = [
        f"## Results — `{eval_name}` ({n} samples)",
        "",
        header,
        sep,
    ]
    base_phit5 = baseline.get("page_hit_5", 0.0)
    for row in rows:
        metrics = row["metrics"]
        delta = metrics.get("page_hit_5", 0.0) - base_phit5
        values = " | ".join(f"{metrics.get(c, 0):.3f}" if isinstance(metrics.get(c), float) else str(metrics.get(c, "-")) for c in cols)
        sign = "+" if delta > 0 else ""
        lines.append(f"| {row['id']} | {values} | {sign}{delta:+.3f} |")
    return "\n".join(lines) + "\n"


# ── Main ────────────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", default="data/eval/deleuze_exact_citations.json")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--include-index", action="store_true")
    ap.add_argument("--include-answer", action="store_true")
    ap.add_argument("--class", dest="cls_filter", default=None, help="Filter by class: RETRIEVAL|INDEX|ANSWER")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output-md", default="data/eval/feature_matrix.md")
    ap.add_argument("--output-json", default="data/eval/feature_matrix.json")
    ap.add_argument(
        "--api-url",
        default=None,
        help="Drive a running API instead of in-process retrieve() "
        "(required when deps aren't installed locally).",
    )
    ap.add_argument(
        "--ablate",
        default=None,
        help="Comma-separated list of ablation IDs to run after the matrix: "
        "P0 (contextual), P5 (RAPTOR), P7 (graph). Requires each ablation's "
        "prerequisites (sibling noctx collection, abstractive/multi-entity "
        "datasets, graph DB). Results appended to the output report.",
    )
    ap.add_argument(
        "--ctx-collection",
        default=None,
        help="Enriched collection for the P0 ablation (defaults to --collection).",
    )
    args = ap.parse_args()

    configs = build_matrix(args.include_index, args.include_answer, args.cls_filter)

    taxonomy_md = render_taxonomy()
    config_md = (
        "## Configurations to evaluate\n\n"
        + "\n".join(f"- **{c['id']}** — {c['label']}" for c in configs)
        + "\n"
    )

    if args.dry_run:
        print(taxonomy_md)
        print(config_md)
        print("Eval file:", args.eval)
        print("Samples would be replayed against each configuration.")
        return 0

    eval_path = Path(args.eval)
    samples = load_eval(eval_path)
    if args.limit:
        samples = samples[: args.limit]
    log.info(f"Loaded {len(samples)} samples from {eval_path}")

    results: list[dict] = []
    for cfg in configs:
        log.info(f"→ running config '{cfg['id']}' ({len(cfg['overrides'])} override(s))")
        metrics = run_config(samples, cfg["overrides"], args.collection, api_url=args.api_url)
        log.info(
            "  %-14s  page_hit@5=%.3f  source_p@5=%.3f  MRR=%.3f  lat=%.1fms",
            cfg["id"], metrics.get("page_hit_5", 0.0),
            metrics.get("source_p_5", 0.0), metrics.get("mrr", 0.0),
            metrics.get("mean_latency_ms", 0.0),
        )
        results.append({"id": cfg["id"], "label": cfg["label"], "overrides": cfg["overrides"], "metrics": metrics})

    baseline_metrics = results[0]["metrics"]

    md = "\n\n".join(
        [
            f"# Feature-matrix eval — `{eval_path.name}`",
            taxonomy_md,
            config_md,
            render_matrix(eval_path.name, len(samples), baseline_metrics, results),
            "Δ columns are signed lift vs baseline on the same eval set.\n",
        ]
    )
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"eval": eval_path.name, "n_samples": len(samples), "results": results}

    if args.ablate:
        ablation_md, ablation_results = _run_ablations(
            args.ablate.split(","),
            collection=args.collection,
            ctx_collection=args.ctx_collection or args.collection,
        )
        md += "\n\n" + ablation_md
        payload["ablations"] = ablation_results
        out_md.write_text(md)

    out_json.write_text(json.dumps(payload, indent=2))
    log.info(f"Wrote {out_md} and {out_json}")
    return 0


def _run_ablations(
    ids: list[str],
    collection: str | None,
    ctx_collection: str | None,
) -> tuple[str, dict]:
    """Run ablation scripts as subprocesses and collect their JSON outputs."""
    import subprocess

    ids = [x.strip().upper() for x in ids if x.strip()]
    root = Path(__file__).resolve().parent.parent
    artifacts: dict[str, dict] = {}
    md_lines: list[str] = ["## Artifact ablations", ""]

    for aid in ids:
        if aid == "P0":
            out = root / "data/eval/contextual_ablation.json"
            cmd = [
                sys.executable, str(root / "scripts/eval_contextual_ablation.py"),
                "--ctx-collection", ctx_collection or "",
                "--output-json", str(out),
            ]
        elif aid == "P5":
            out = root / "data/eval/raptor_ablation.json"
            cmd = [sys.executable, str(root / "scripts/eval_raptor_ablation.py"),
                   "--output-json", str(out)]
            if collection:
                cmd += ["--collection", collection]
        elif aid == "P7":
            out = root / "data/eval/graph_ablation.json"
            cmd = [sys.executable, str(root / "scripts/eval_graph_ablation.py"),
                   "--output-json", str(out)]
            if collection:
                cmd += ["--collection", collection]
        else:
            log.warning("skip unknown ablation id %r", aid)
            continue

        log.info("ablation %s: %s", aid, " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, cwd=str(root))
            data = json.loads(out.read_text())
        except Exception as e:
            log.warning("ablation %s failed: %s", aid, e)
            md_lines.append(f"- **{aid}** — failed: {e}")
            continue
        artifacts[aid] = data
        md_lines.append(f"### {aid}\n\n```json\n{json.dumps(data, indent=2)[:1200]}\n```\n")

    return "\n".join(md_lines), artifacts


if __name__ == "__main__":
    raise SystemExit(main())
