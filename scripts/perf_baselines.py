#!/usr/bin/env python3
"""Performance baselines for the Gutenberg RAG pipeline.

Usage:
    uv run scripts/perf_baselines.py
    uv run scripts/perf_baselines.py --iterations 10 --skip-llm --skip-ingest
    uv run scripts/perf_baselines.py --output results.json
"""

import argparse
import gzip
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

API_URL = os.environ.get("GUTENBERG_API_URL", "http://localhost:8002")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b")
FRONTEND_BUILD_DIR = Path("frontend/build")

TARGETS = {
    "retrieval_latency_ms": {"target": 500, "op": "lt", "label": "< 500ms"},
    "embedding_latency_ms": {"target": 200, "op": "lt", "label": "< 200ms"},
    "streaming_first_token_local_ms": {
        "target": 2000,
        "op": "lt",
        "label": "< 2s (local)",
    },
    "streaming_first_token_openrouter_ms": {
        "target": 5000,
        "op": "lt",
        "label": "< 5s (OpenRouter)",
    },
    "ingestion_pages_per_min": {"target": 50, "op": "gt", "label": "> 50 pages/min"},
    "frontend_gzip_kb": {"target": 500, "op": "lt", "label": "< 500KB gzipped"},
}

QUERY = "What is the relationship between the rhizome and the arborescent model?"
TIMEOUT = 30


def compute_stats(samples: list[float]) -> dict:
    if not samples:
        return {"min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0, "n": 0}
    s = sorted(samples)
    n = len(s)
    return {
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
        "mean": round(statistics.mean(s), 2),
        "p50": round(s[int(n * 0.50)], 2),
        "p95": round(s[min(int(n * 0.95), n - 1)], 2),
        "n": n,
    }


def check_target(metric: str, value: float) -> bool:
    t = TARGETS[metric]
    return value < t["target"] if t["op"] == "lt" else value > t["target"]


def print_metric(name: str, value: float, stats: dict | None = None):
    passed = check_target(name, value)
    icon = "PASS" if passed else "FAIL"
    print(f"  {icon}  {name}: {value:.1f}  (target {TARGETS[name]['label']})")
    if stats and stats["n"] > 0:
        print(
            f"        min={stats['min']:.0f}ms  max={stats['max']:.0f}ms  "
            f"mean={stats['mean']:.0f}ms  p50={stats['p50']:.0f}ms"
        )


def measure_embedding(n: int) -> tuple[dict, list[float]]:
    samples = []
    for _ in range(n):
        try:
            t0 = time.perf_counter()
            resp = httpx.post(
                f"{OLLAMA_HOST}/api/embed",
                json={"model": EMBED_MODEL, "input": [QUERY]},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            samples.append((time.perf_counter() - t0) * 1000)
        except Exception as e:
            print(f"  [WARN] embedding: {e}")
    return compute_stats(samples), samples


def measure_retrieval(n: int) -> tuple[dict, list[float]]:
    """Full retrieval pipeline via API (non-streaming, max_tokens=1 to minimize generation)."""
    samples = []
    for _ in range(n):
        try:
            t0 = time.perf_counter()
            resp = httpx.post(
                f"{API_URL}/v1/chat/completions",
                json={
                    "model": "gutenberg-rag",
                    "messages": [{"role": "user", "content": QUERY}],
                    "stream": False,
                    "temperature": 0.1,
                    "max_tokens": 1,
                },
                timeout=TIMEOUT * 4,
            )
            resp.raise_for_status()
            samples.append((time.perf_counter() - t0) * 1000)
        except Exception as e:
            print(f"  [WARN] retrieval: {e}")
    return compute_stats(samples), samples


def measure_streaming_first_token(n: int) -> tuple[dict, list[float]]:
    """Time from request to first SSE data token."""
    samples = []
    for _ in range(n):
        try:
            t0 = time.perf_counter()
            with httpx.stream(
                "POST",
                f"{API_URL}/v1/chat/completions",
                json={
                    "model": "gutenberg-rag",
                    "messages": [{"role": "user", "content": QUERY}],
                    "stream": True,
                    "temperature": 0.1,
                },
                timeout=TIMEOUT * 4,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line and line.startswith("data: ") and line != "data: [DONE]":
                        samples.append((time.perf_counter() - t0) * 1000)
                        break
        except Exception as e:
            print(f"  [WARN] streaming: {e}")
    return compute_stats(samples), samples


def measure_ingestion(n_pages: int = 50) -> tuple[dict, float]:
    """Generate a synthetic PDF, upload it, measure pages/min throughput."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        print("  [SKIP] reportlab not available")
        return compute_stats([]), 0.0

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    _, height = letter
    for i in range(n_pages):
        c.setFont("Helvetica", 11)
        y = height - 50
        c.drawString(50, y, f"Performance Baseline Test Document - Page {i + 1}")
        y -= 20
        for ln in range(40):
            c.drawString(
                50, y, f"Line {ln}: Synthetic test content for page {i + 1}."[:95]
            )
            y -= 14
            if y < 50:
                break
        c.showPage()
    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()

    try:
        t0 = time.perf_counter()
        resp = httpx.post(
            f"{API_URL}/v1/corpus/default/ingest",
            files={"files": ("perf_test.pdf", pdf_bytes, "application/pdf")},
            timeout=TIMEOUT * 20,
        )
        resp.raise_for_status()
        job_id = resp.json().get("job_id")
        if job_id:
            for _ in range(120):
                time.sleep(1)
                try:
                    sr = httpx.get(
                        f"{API_URL}/v1/corpus/default/ingest/status", timeout=10
                    )
                    if sr.status_code == 200:
                        break
                except Exception:
                    break
        elapsed = time.perf_counter() - t0
        ppm = (n_pages / elapsed) * 60 if elapsed > 0 else 0
        return compute_stats([ppm]), ppm
    except Exception as e:
        print(f"  [WARN] ingestion: {e}")
        return compute_stats([]), 0.0


def measure_frontend_size() -> tuple[float, dict]:
    if not FRONTEND_BUILD_DIR.exists():
        print("  [SKIP] frontend/build/ not found")
        return 0.0, {"files": 0, "details": "directory not found"}
    total = 0
    count = 0
    for f in FRONTEND_BUILD_DIR.rglob("*"):
        if f.is_file():
            total += len(gzip.compress(f.read_bytes(), compresslevel=9))
            count += 1
    return total / 1024, {"files": count}


def main():
    parser = argparse.ArgumentParser(description="Gutenberg performance baselines")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", default=None, help="Write JSON results to file")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()

    results: dict = {}
    all_pass = True

    def fail_if(ok: bool):
        nonlocal all_pass
        if not ok:
            all_pass = False

    print("=" * 64)
    print("  GUTENBORG PERFORMANCE BASELINES")
    print("=" * 64)

    print(f"\n[1/5] Embedding latency ({args.iterations} iterations)")
    emb_stats, _ = measure_embedding(args.iterations)
    results["embedding_latency_ms"] = emb_stats
    fail_if(check_target("embedding_latency_ms", emb_stats["p95"]))
    print_metric("embedding_latency_ms", emb_stats["p95"], emb_stats)

    print(f"\n[2/5] Retrieval latency ({args.iterations} iterations)")
    ret_stats, _ = measure_retrieval(args.iterations)
    results["retrieval_latency_ms"] = ret_stats
    fail_if(check_target("retrieval_latency_ms", ret_stats["p95"]))
    print_metric("retrieval_latency_ms", ret_stats["p95"], ret_stats)

    if not args.skip_llm:
        print(f"\n[3/5] Streaming first-token latency ({args.iterations} iterations)")
        stream_stats, _ = measure_streaming_first_token(args.iterations)
        results["streaming_first_token_ms"] = stream_stats
        fail_if(check_target("streaming_first_token_local_ms", stream_stats["p95"]))
        print_metric(
            "streaming_first_token_local_ms", stream_stats["p95"], stream_stats
        )
    else:
        print("\n[3/5] Streaming first-token latency - SKIPPED (--skip-llm)")

    if not args.skip_ingest:
        print("\n[4/5] Ingestion throughput (50-page synthetic PDF)")
        ingest_stats, throughput = measure_ingestion(50)
        results["ingestion_pages_per_min"] = ingest_stats
        if throughput > 0:
            fail_if(check_target("ingestion_pages_per_min", throughput))
            print_metric("ingestion_pages_per_min", throughput)
        else:
            print("  SKIP  ingestion: no measurement")
    else:
        print("\n[4/5] Ingestion throughput - SKIPPED (--skip-ingest)")

    print("\n[5/5] Frontend build size (gzipped)")
    total_kb, details = measure_frontend_size()
    results["frontend_gzip_kb"] = {"total_kb": round(total_kb, 1), **details}
    if total_kb > 0:
        fail_if(check_target("frontend_gzip_kb", total_kb))
        passed = check_target("frontend_gzip_kb", total_kb)
        print(
            f"  {'PASS' if passed else 'FAIL'}  frontend_gzip_kb: {total_kb:.1f}  "
            f"(target {TARGETS['frontend_gzip_kb']['label']})"
        )
        print(f"        {details.get('files', 0)} files in {FRONTEND_BUILD_DIR}/")
    else:
        print("  SKIP  frontend build: no measurement")

    print(f"\n{'=' * 64}")
    print(f"  Result: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print(f"{'=' * 64}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {
                    "targets": {k: v["target"] for k, v in TARGETS.items()},
                    "results": results,
                    "all_pass": all_pass,
                },
                f,
                indent=2,
            )
        print(f"\n  Results saved to {out}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
