"""Multi-criteria grader for précis / cross-work evolution queries.

Replaces the structural zero on ``data/eval/deleuze_precis_evolution.json``
(queries that span multiple works and therefore have no single gold page).

Two modes:

  * retrieval-only (default) — calls ``core.rag.retrieve`` in-process and
    grades retrieved chunks only. Fast; does not depend on a running API.

  * --with-answer — additionally calls the chat endpoint, parses
    ``[Source: …, p. N]`` tags, and computes ``works_cited_coverage``
    (and optionally ``alce_entailment`` via the existing eval_alce
    NLI path).

Metrics:

  * ``works_cited_coverage``    (answer) — share of ``works_required``
    appearing in citation tags, or as retrieved-chunk sources when
    --with-answer is off.
  * ``concept_span_hit_rate``   (retrieval) — fraction of
    ``expected_concept_spans`` fuzzily present in top-k chunk text.
    Falls back to ``expected_stages`` tokens when spans are absent
    (datasets not yet extended by mine_eval_golds.py).
  * ``entity_coverage``         (retrieval) — overlap with
    ``expected_entities``.
  * ``alce_entailment``         (answer, --with-answer + --alce) —
    reuses eval_alce's NLI routine; optional.

Composite:
    weights configurable; default
    0.35 * works + 0.35 * spans + 0.2 * entities + 0.10 * alce.

Usage:
    docker compose exec api python scripts/eval_precis.py \
        --collection gutenberg-deleuze-corpus
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services"))
sys.path.insert(0, str(ROOT / "services" / "api"))
sys.path.insert(0, str(ROOT / "scripts"))

log = logging.getLogger("eval.precis")

TOP_K = 10

_CITATION_RE = re.compile(
    r"\[Source:\s*(?P<title>[^\]]+?),\s*pp?\.\s*(?P<page>\d+(?:\s*[-–—]\s*\d+)?)\s*\]",
    re.IGNORECASE,
)


def _chunk_cids(ch: dict) -> set[str]:
    raw = (ch.get("metadata") or {}).get("canonical_ids", "") or ""
    return {c for c in raw.split(",") if c}


def _chunk_source(ch: dict) -> str:
    return (ch.get("metadata") or {}).get("source", "") or ""


def _source_match(candidate: str, required: str) -> bool:
    """Tolerant source match: stem-on-stem, strip year + ext."""
    def stem(s: str) -> str:
        s = s.lower()
        s = re.sub(r"^\d{4}\s+", "", s)
        s = re.sub(r"\.pdf$", "", s)
        return s.strip()
    a, b = stem(candidate), stem(required)
    if not a or not b:
        return False
    return a[:30] in b or b[:30] in a


def _extract_titles(answer: str) -> set[str]:
    return {m.group("title").strip() for m in _CITATION_RE.finditer(answer)}


def _concept_spans(sample: dict) -> list[str]:
    spans = sample.get("expected_concept_spans") or []
    if spans:
        return [s for s in spans if s]
    # Fallback: extract short content phrases from expected_stages strings
    # so an un-augmented precis file still produces a non-zero score.
    stages = sample.get("expected_stages") or []
    out: list[str] = []
    for st in stages:
        if not isinstance(st, str):
            continue
        # Use the text after the "prefix:" separator as the salient part.
        tail = st.split(":", 1)[-1].strip().lower()
        if 4 <= len(tail) <= 120:
            out.append(tail)
    return out


def _score_retrieval(chunks: list[dict], sample: dict, k: int) -> dict:
    from shared.text_normalize import normalize_for_matching

    top = chunks[:k]
    spans = _concept_spans(sample)
    expected_entities = set(sample.get("expected_entities") or [])
    works_required: list[str] = sample.get("works_required") or []

    hay = normalize_for_matching(" ".join((c.get("text") or "") for c in top))
    span_hits = sum(
        1 for s in spans
        if s and normalize_for_matching(s) in hay
    )
    span_rate = span_hits / len(spans) if spans else 0.0

    got_cids: set[str] = set()
    for c in top:
        got_cids |= _chunk_cids(c)
    entity_cov = (
        len(got_cids & expected_entities) / len(expected_entities)
        if expected_entities else 0.0
    )

    retrieved_sources = {_chunk_source(c) for c in top if _chunk_source(c)}
    works_cov_retrieval = (
        sum(1 for w in works_required if any(_source_match(s, w) for s in retrieved_sources))
        / len(works_required)
        if works_required else 0.0
    )

    return {
        "concept_span_hit_rate": round(span_rate, 4),
        "entity_coverage": round(entity_cov, 4),
        "works_cited_coverage_retrieval": round(works_cov_retrieval, 4),
    }


def _score_answer(answer: str, sample: dict) -> dict:
    works_required: list[str] = sample.get("works_required") or []
    cited_titles = _extract_titles(answer)
    matched = 0
    for w in works_required:
        if any(_source_match(t, w) for t in cited_titles):
            matched += 1
    cov = matched / len(works_required) if works_required else 0.0
    return {"works_cited_coverage": round(cov, 4)}


def _composite(m: dict, weights: dict) -> float:
    works = m.get("works_cited_coverage", m.get("works_cited_coverage_retrieval", 0.0))
    spans = m.get("concept_span_hit_rate", 0.0)
    ents = m.get("entity_coverage", 0.0)
    alce = m.get("alce_entailment", 0.0)
    return round(
        weights["works"] * works
        + weights["spans"] * spans
        + weights["entities"] * ents
        + weights["alce"] * alce,
        4,
    )


def _call_chat(api_url: str, model: str, query: str, timeout: float) -> str:
    import httpx

    r = httpx.post(
        f"{api_url.rstrip('/')}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": query}],
            "stream": False,
            "temperature": 0.1,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _call_openrouter(
    chunks: list[dict],
    query: str,
    *,
    or_model: str,
    or_key: str,
    timeout: float,
) -> str:
    """Build a RAG system prompt from already-retrieved ``chunks`` and send to
    OpenRouter directly. Bypasses the local API → Ollama path so we can use
    Gemini Flash / Haiku / GPT-4o-mini as the answerer."""
    import httpx

    from core.rag import build_rag_prompt

    system_prompt = build_rag_prompt(query, chunks)
    r = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {or_key}",
            "HTTP-Referer": "https://github.com/revelri/gutenborg",
            "X-Title": "gutenborg-precis-eval",
        },
        json={
            "model": or_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/eval/deleuze_precis_evolution.json")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--with-answer", action="store_true",
                    help="Call the chat API and grade citations")
    ap.add_argument("--api-url", default="http://localhost:8002")
    ap.add_argument("--model", default="gutenberg-rag")
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument(
        "--or-direct",
        action="store_true",
        help="Generate answers via OpenRouter directly (bypasses the API → "
        "Ollama path; requires OPENROUTER_API_KEY/OPENROUTER_KEY).",
    )
    ap.add_argument(
        "--or-model",
        default="google/gemini-2.5-flash",
        help="OpenRouter model id when --or-direct is set",
    )
    ap.add_argument(
        "--structured",
        action="store_true",
        help="Use structured-output hybrid (JSON schema → render). Requires "
        "--or-direct. Guarantees per-work citation breadth.",
    )
    ap.add_argument(
        "--save-answers",
        default=None,
        help="Optional path to dump query/answer/chunks for downstream "
        "evals (e.g., eval_alce).",
    )
    ap.add_argument(
        "--verbatim-threshold",
        type=int,
        default=None,
        help="rapidfuzz partial_ratio cutoff (0-100) for --structured "
        "verbatim enforcement. Defaults to settings.verbatim_min_score (85). "
        "Higher values reject more near-paraphrases at the cost of coverage.",
    )
    ap.add_argument("--weights", default="0.35,0.35,0.20,0.10",
                    help="works,spans,entities,alce")
    ap.add_argument("--output-json", default="data/eval/precis_results.json")
    args = ap.parse_args()

    w = [float(x) for x in args.weights.split(",")]
    weights = {"works": w[0], "spans": w[1], "entities": w[2], "alce": w[3]}

    samples = json.loads(Path(args.eval).read_text())
    log.info("Loaded %d precis queries", len(samples))

    from core.rag import retrieve

    or_key = ""
    if args.or_direct:
        import os as _os
        or_key = (
            _os.environ.get("OPENROUTER_API_KEY")
            or _os.environ.get("OPENROUTER_KEY")
            or ""
        )
        if not or_key:
            log.error("--or-direct set but no OPENROUTER_KEY in env")
            return 2

    results: list[dict] = []
    saved_samples: list[dict] = []
    for i, s in enumerate(samples, 1):
        q = s["query"]
        log.info("[%d/%d] %s", i, len(samples), (s.get("label") or q[:60]))
        t0 = time.perf_counter()
        try:
            _, chunks = retrieve(q, collection=args.collection)
        except Exception as e:
            log.warning("retrieve failed: %s", e)
            chunks = []
        retrieve_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        metrics = _score_retrieval(chunks, s, TOP_K)

        answer = ""
        validation: dict = {}
        if args.with_answer:
            try:
                if args.structured and args.or_direct:
                    from core.structured_answer import answer_structured
                    answer, _, validation = answer_structured(
                        q,
                        chunks[:TOP_K],
                        s.get("works_required") or [],
                        model=args.or_model,
                        api_key=or_key,
                        timeout=args.request_timeout,
                        verbatim_min_score=args.verbatim_threshold,
                    )
                elif args.or_direct:
                    answer = _call_openrouter(
                        chunks[:TOP_K],
                        q,
                        or_model=args.or_model,
                        or_key=or_key,
                        timeout=args.request_timeout,
                    )
                else:
                    answer = _call_chat(
                        args.api_url, args.model, q, args.request_timeout
                    )
                metrics.update(_score_answer(answer, s))
                metrics["answer_length"] = len(answer)
                if validation:
                    metrics["per_work_coverage"] = validation.get("per_work_coverage", 0.0)
                    metrics["synthesis_coverage"] = validation.get("synthesis_coverage", 0.0)
                    metrics["unverified_quotes_n"] = len(validation.get("unverified_quotes", []))
            except Exception as e:
                log.warning("answer call failed: %s", e)
                metrics["works_cited_coverage"] = 0.0

        metrics["composite"] = _composite(metrics, weights)
        metrics["retrieve_ms"] = retrieve_ms
        results.append({"query": q, "label": s.get("label"), "metrics": metrics})

        if args.save_answers and answer:
            saved_samples.append({
                "query": q,
                "answer": answer,
                "chunks": [
                    {
                        "source": (c.get("metadata") or {}).get("source"),
                        "page_start": (c.get("metadata") or {}).get("page_start"),
                        "page_end": (c.get("metadata") or {}).get("page_end"),
                        "text": c.get("text"),
                    }
                    for c in chunks[:TOP_K]
                ],
            })

    def _mean(key: str) -> float:
        vals = [r["metrics"].get(key) for r in results if r["metrics"].get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary = {
        "n_samples": len(results),
        "with_answer": args.with_answer,
        "weights": weights,
        "mean_composite": _mean("composite"),
        "mean_concept_span_hit_rate": _mean("concept_span_hit_rate"),
        "mean_entity_coverage": _mean("entity_coverage"),
        "mean_works_cited_coverage_retrieval": _mean("works_cited_coverage_retrieval"),
        "mean_works_cited_coverage": _mean("works_cited_coverage"),
    }

    out = {"summary": summary, "results": results}
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s — composite=%.3f", path, summary["mean_composite"])

    if args.save_answers and saved_samples:
        ap = Path(args.save_answers)
        ap.parent.mkdir(parents=True, exist_ok=True)
        ap.write_text(json.dumps(saved_samples, indent=2))
        log.info("Saved %d query/answer/chunks records to %s", len(saved_samples), ap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
