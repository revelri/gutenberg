#!/usr/bin/env python3
"""End-to-end citation accuracy test.

For each test case:
  1. Sends a query to the Gutenborg API
  2. Extracts quoted text and cited page numbers from the LLM response
  3. Retrieves the cited page from the source PDF via PyMuPDF (CPU)
  4. Checks: does the quoted text actually appear on the cited page?
  5. Scores character-level overlap between quote and source page

Primary metric: "quote grounding" — what % of the LLM's quotes are
actually present in the source PDF at the page it cited?

No GPU usage in this script. Ollama uses GPU remotely for inference.

Usage:
    python scripts/eval_citations.py
    python scripts/eval_citations.py --api-url http://localhost:8002
"""

import argparse
import json
import logging
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import fitz  # PyMuPDF — CPU only, no VRAM
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_citations")

API_URL = os.environ.get("GUTENBERG_API_URL", "http://localhost:8002")
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "data/processed")


# ── PDF page extraction (CPU only, ~1ms per page) ──────────────────

def extract_page_text(source_filename: str, pdf_page: int) -> str:
    """Extract text from a specific PDF page. Pure CPU, no VRAM."""
    path = Path(PROCESSED_DIR) / source_filename
    if not path.exists():
        return ""
    doc = fitz.open(str(path))
    if pdf_page < 1 or pdf_page > len(doc):
        doc.close()
        return ""
    text = doc[pdf_page - 1].get_text("text")
    doc.close()
    return text


def extract_page_range_text(source_filename: str, start: int, end: int) -> str:
    """Extract text from a range of PDF pages. Handles ±1 tolerance."""
    texts = []
    for p in range(max(1, start - 1), end + 2):  # ±1 page tolerance
        t = extract_page_text(source_filename, p)
        if t:
            texts.append(t)
    return "\n".join(texts)


def extract_full_pdf_text(source_filename: str) -> str:
    """Extract text from the entire PDF. For checking quote authenticity."""
    path = Path(PROCESSED_DIR) / source_filename
    if not path.exists():
        return ""
    doc = fitz.open(str(path))
    texts = []
    for page in doc:
        t = page.get_text("text")
        if t.strip():
            texts.append(t)
    doc.close()
    return "\n".join(texts)


# ── Quote and citation parsing ──────────────────────────────────────

def extract_quotes_with_citations(text: str) -> list[dict]:
    """Extract quotes and their associated [Source: ..., p. N] citations.

    Returns list of {"quote": str, "cited_page": int|None, "cited_source": str|None}
    """
    results = []

    # Find quoted text with nearby citations
    # Pattern: "quote text" ... [Source: title, p. N]
    quote_patterns = [
        re.compile(r'"([^"]{15,})"'),           # straight quotes
        re.compile(r'\u201c([^\u201d]{15,})\u201d'),  # smart quotes
    ]

    for pat in quote_patterns:
        for m in pat.finditer(text):
            quote = m.group(1)
            # Look for [Source: ...] within 300 chars after the quote
            after = text[m.end():m.end() + 300]
            page_match = re.search(r'p\.?\s*(\d+)', after)
            source_match = re.search(r'\[Source[^]]*?:\s*([^],]+)', after)

            results.append({
                "quote": quote,
                "cited_page": int(page_match.group(1)) if page_match else None,
                "cited_source": source_match.group(1).strip().strip('*_') if source_match else None,
            })

    # Blockquotes
    bq_lines = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith(">"):
            bq_lines.append(s.lstrip("> ").strip())
        else:
            if bq_lines:
                bq = " ".join(bq_lines)
                if len(bq) >= 15:
                    # Look for citation after blockquote
                    after_idx = text.find(bq_lines[-1]) + len(bq_lines[-1])
                    after = text[after_idx:after_idx + 300] if after_idx > 0 else ""
                    page_match = re.search(r'p\.?\s*(\d+)', after)
                    results.append({
                        "quote": bq,
                        "cited_page": int(page_match.group(1)) if page_match else None,
                        "cited_source": None,
                    })
                bq_lines = []
    if bq_lines:
        bq = " ".join(bq_lines)
        if len(bq) >= 15:
            results.append({"quote": bq, "cited_page": None, "cited_source": None})

    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r["quote"] not in seen:
            seen.add(r["quote"])
            unique.append(r)
    return unique


# ── Comparison scoring (CPU only) ──────────────────────────────────

def normalize(text: str) -> str:
    """Strip markdown, lowercase, collapse whitespace."""
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def quote_in_page(quote: str, page_text: str) -> float:
    """Check if a quote appears in page text. Returns best overlap ratio.

    1.0 = exact substring match, 0.85+ = near-verbatim, <0.7 = paraphrase/hallucination.
    """
    q = normalize(quote)
    t = normalize(page_text)
    if not q or not t:
        return 0.0
    if q in t:
        return 1.0
    # Sliding window for near-matches
    qlen = len(q)
    if qlen >= len(t):
        return SequenceMatcher(None, q, t).ratio()
    best = 0.0
    step = max(1, qlen // 6)
    for start in range(0, len(t) - qlen + 1, step):
        window = t[start:start + qlen]
        ratio = SequenceMatcher(None, q, window).ratio()
        if ratio > best:
            best = ratio
            if best >= 0.98:
                return best
    return best


# ── API interaction (GPU used remotely by Ollama, not by us) ───────

def query_api(query: str, api_url: str) -> str:
    resp = httpx.post(
        f"{api_url}/v1/chat/completions",
        json={
            "model": "gutenberg-rag",
            "messages": [{"role": "user", "content": query}],
            "stream": False,
            "temperature": 0.1,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── Main evaluation ────────────────────────────────────────────────

def run_eval(test_cases: list[dict], api_url: str) -> list[dict]:
    results = []

    for i, tc in enumerate(test_cases):
        query = tc["query"]
        source = tc["source"]
        expected_page = tc["pdf_page"]
        ground_truth = tc["ground_truth"]

        log.info(f"[{i+1}/{len(test_cases)}] {query[:70]}...")

        # 1. Query the API
        try:
            response = query_api(query, api_url)
        except Exception as e:
            log.error(f"  API error: {e}")
            results.append({"query": query, "status": "API_ERROR", "scores": {}})
            continue

        # 2. Extract quotes with their citations
        quote_cites = extract_quotes_with_citations(response)

        if not quote_cites:
            abstained = "could not find" in response.lower()
            status = "ABSTAINED" if abstained else "NO_QUOTES"
            results.append({"query": query, "status": status, "scores": {}})
            log.info(f"  {status}")
            continue

        # 3. Get full PDF text for authenticity check (once per query)
        full_pdf_text = extract_full_pdf_text(source)

        # 4. For each quote, verify against cited page AND full PDF
        quote_scores = []
        for qc in quote_cites:
            quote = qc["quote"]
            cited_page = qc["cited_page"]

            # Check against the page the LLM actually cited (±1)
            if cited_page:
                cited_page_text = extract_page_range_text(source, cited_page, cited_page)
                cited_overlap = quote_in_page(quote, cited_page_text)
            else:
                cited_overlap = 0.0

            # Check against the expected page (ground truth location, ±1)
            expected_page_text = extract_page_range_text(source, expected_page, expected_page)
            expected_overlap = quote_in_page(quote, expected_page_text)

            # Check against FULL PDF — does this quote exist anywhere in the source?
            pdf_overlap = quote_in_page(quote, full_pdf_text)

            # Ground truth snippet match
            gt_overlap = SequenceMatcher(
                None, normalize(quote), normalize(ground_truth)
            ).ratio()

            quote_scores.append({
                "quote_preview": quote[:80],
                "cited_page": cited_page,
                "grounded_on_cited_page": round(cited_overlap, 3),
                "grounded_on_expected_page": round(expected_overlap, 3),
                "grounded_in_pdf": round(pdf_overlap, 3),
                "ground_truth_match": round(gt_overlap, 3),
            })

        # 5. Aggregate: best score across all quotes
        best_cited = max(s["grounded_on_cited_page"] for s in quote_scores)
        best_expected = max(s["grounded_on_expected_page"] for s in quote_scores)
        best_pdf = max(s["grounded_in_pdf"] for s in quote_scores)
        best_gt = max(s["ground_truth_match"] for s in quote_scores)
        any_page_correct = any(s["cited_page"] == expected_page for s in quote_scores)
        any_page_near = any(
            s["cited_page"] and abs(s["cited_page"] - expected_page) <= 1
            for s in quote_scores
        )

        result = {
            "query": query,
            "status": "QUOTED",
            "num_quotes": len(quote_scores),
            "scores": {
                "grounded_cited": round(best_cited, 3),
                "grounded_expected": round(best_expected, 3),
                "grounded_pdf": round(best_pdf, 3),
                "ground_truth": round(best_gt, 3),
                "page_exact": any_page_correct,
                "page_near": any_page_near or any_page_correct,
            },
            "quotes": quote_scores,
        }
        results.append(result)

        icon = "✓" if best_pdf >= 0.90 else "≈" if best_pdf >= 0.70 else "✗"
        pg = "✓" if any_page_correct else "≈" if any_page_near else "✗"
        log.info(f"  {icon} pdf={best_pdf:.0%} cited_pg={best_cited:.0%} page={pg}")

    return results


def print_report(results: list[dict]):
    n = len(results)
    quoted = [r for r in results if r["status"] == "QUOTED"]
    abstained = [r for r in results if r["status"] == "ABSTAINED"]
    no_quotes = [r for r in results if r["status"] == "NO_QUOTES"]
    errors = [r for r in results if r["status"] == "API_ERROR"]

    print(f"\n{'='*72}")
    print(f"  CITATION ACCURACY TEST — {n} queries")
    print(f"{'='*72}")
    print(f"  Quoted: {len(quoted)}  Abstained: {len(abstained)}  "
          f"No quotes: {len(no_quotes)}  Errors: {len(errors)}")

    if not quoted:
        print("  No quotes to evaluate.")
        return

    pdf_scores = [r["scores"]["grounded_pdf"] for r in quoted]
    cited_scores = [r["scores"]["grounded_cited"] for r in quoted]
    gts = [r["scores"]["ground_truth"] for r in quoted]
    pages_exact = sum(1 for r in quoted if r["scores"]["page_exact"])
    pages_near = sum(1 for r in quoted if r["scores"]["page_near"])

    avg_pdf = sum(pdf_scores) / len(pdf_scores)
    avg_cited = sum(cited_scores) / len(cited_scores)
    avg_gt = sum(gts) / len(gts)
    pdf_verified = sum(1 for g in pdf_scores if g >= 0.90)
    pdf_approx = sum(1 for g in pdf_scores if 0.70 <= g < 0.90)
    pdf_failed = sum(1 for g in pdf_scores if g < 0.70)

    print(f"\n  {'METRIC':<40} {'SCORE':>8}")
    print(f"  {'─'*50}")
    print(f"  {'Quote exists in source PDF (mean)':<40} {avg_pdf:>7.1%}")
    print(f"  {'  ✓ Verified (≥90%)':<40} {pdf_verified:>5}/{len(quoted)}")
    print(f"  {'  ≈ Approximate (70-89%)':<40} {pdf_approx:>5}/{len(quoted)}")
    print(f"  {'  ✗ Failed (<70%)':<40} {pdf_failed:>5}/{len(quoted)}")
    print(f"  {'Quote on cited page (mean)':<40} {avg_cited:>7.1%}")
    print(f"  {'Page number exact':<40} {pages_exact:>5}/{len(quoted)}")
    print(f"  {'Page number ±1':<40} {pages_near:>5}/{len(quoted)}")

    print(f"\n{'='*72}")
    print(f"  PER-QUERY DETAIL")
    print(f"{'='*72}")
    print(f"  {'#':<3} {'Query':<34} {'PDF':>5} {'Cited':>6} {'GT':>5} {'Pg':>3} {'Q':>2}")
    print(f"  {'─'*62}")

    for i, r in enumerate(results):
        q = r["query"][:31] + "..." if len(r["query"]) > 34 else r["query"]
        if r["status"] == "QUOTED":
            s = r["scores"]
            pdf = f"{s['grounded_pdf']:.0%}"
            ci = f"{s['grounded_cited']:.0%}"
            gt = f"{s['ground_truth']:.0%}"
            pg = "✓" if s["page_exact"] else "≈" if s["page_near"] else "✗"
            nq = str(r["num_quotes"])
            print(f"  {i+1:<3} {q:<34} {pdf:>5} {ci:>6} {gt:>5} {pg:>3} {nq:>2}")
        else:
            print(f"  {i+1:<3} {q:<34} {'—':>5} {'—':>6} {'—':>5} {'—':>3} {'—':>2}  [{r['status']}]")

    print(f"\n{'='*72}")
    target = 0.85
    status = "PASS" if avg_pdf >= target else "FAIL"
    print(f"  TARGET: ≥{target:.0%} mean quote authenticity (exists in source PDF)")
    print(f"  ACTUAL: {avg_pdf:.1%}  [{status}]")
    print(f"{'='*72}\n")


def main():
    parser = argparse.ArgumentParser(description="Citation accuracy test against source PDFs")
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--ground-truth", default="data/eval/citation_ground_truth.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        log.error(f"Ground truth not found: {gt_path}")
        sys.exit(1)

    with open(gt_path) as f:
        test_cases = json.load(f)
    log.info(f"Loaded {len(test_cases)} test cases")

    results = run_eval(test_cases, args.api_url)
    print_report(results)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Results saved to {out}")


if __name__ == "__main__":
    main()
