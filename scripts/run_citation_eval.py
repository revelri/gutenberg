import httpx, json, re, subprocess, sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))
from shared.text_normalize import normalize_for_comparison, strip_markdown_formatting

API_URL = "http://localhost:8002"
PROCESSED_DIR = "data/processed"


def extract_page_text_pdftotext(source_filename, page_num):
    path = f"{PROCESSED_DIR}/{source_filename}"
    try:
        result = subprocess.run(
            [
                "pdftotext",
                "-f",
                str(page_num),
                "-l",
                str(page_num),
                "-layout",
                path,
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout
    except:
        return ""


def extract_page_range_text(source_filename, start, end):
    texts = []
    for p in range(max(1, start - 1), end + 2):
        t = extract_page_text_pdftotext(source_filename, p)
        if t:
            texts.append(t)
    return "\n".join(texts)


def extract_full_pdf_text(source_filename):
    path = f"{PROCESSED_DIR}/{source_filename}"
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", path, "-"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.stdout
    except:
        return ""


def normalize(text):
    return normalize_for_comparison(strip_markdown_formatting(text))


def quote_in_page(quote, page_text):
    q = normalize(quote)
    t = normalize(page_text)
    if not q or not t:
        return 0.0
    if q in t:
        return 1.0
    qlen = len(q)
    if qlen >= len(t):
        return SequenceMatcher(None, q, t).ratio()
    best = 0.0
    step = max(1, qlen // 6)
    for start in range(0, len(t) - qlen + 1, step):
        window = t[start : start + qlen]
        ratio = SequenceMatcher(None, q, window).ratio()
        if ratio > best:
            best = ratio
        if best >= 0.98:
            return best
    return best


def extract_quotes_with_citations(text):
    results = []
    quote_patterns = [
        re.compile(r'"([^"]{15,})"'),
        re.compile(r"\u201c([^\u201d]{15,})\u201d"),
    ]
    for pat in quote_patterns:
        for m in pat.finditer(text):
            quote = m.group(1)
            after = text[m.end() : m.end() + 300]
            page_match = re.search(r"p\.?\s*(\d+)", after)
            source_match = re.search(r"\[Source[^]]*?:\s*([^],]+)", after)
            results.append(
                {
                    "quote": quote,
                    "cited_page": int(page_match.group(1)) if page_match else None,
                    "cited_source": source_match.group(1).strip().strip("*_")
                    if source_match
                    else None,
                }
            )
    bq_lines = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith(">"):
            bq_lines.append(s.lstrip("> ").strip())
        else:
            if bq_lines:
                bq = " ".join(bq_lines)
                if len(bq) >= 15:
                    after_idx = text.find(bq_lines[-1]) + len(bq_lines[-1])
                    after = text[after_idx : after_idx + 300] if after_idx > 0 else ""
                    page_match = re.search(r"p\.?\s*(\d+)", after)
                    results.append(
                        {
                            "quote": bq,
                            "cited_page": int(page_match.group(1))
                            if page_match
                            else None,
                            "cited_source": None,
                        }
                    )
                bq_lines = []
    seen = set()
    unique = []
    for r in results:
        if r["quote"] not in seen:
            seen.add(r["quote"])
            unique.append(r)
    return unique


with open("data/eval/ao_eval_v2.json") as f:
    test_cases = json.load(f)

results = []
for i, tc in enumerate(test_cases):
    query = tc["query"]
    source = tc["source"]
    expected_page = tc["pdf_page"]
    ground_truth = tc["ground_truth"]
    print(f"[{i + 1}/{len(test_cases)}] {query[:70]}...")
    try:
        resp = httpx.post(
            f"{API_URL}/v1/chat/completions",
            json={
                "model": "gutenberg-rag",
                "messages": [{"role": "user", "content": query}],
                "stream": False,
                "temperature": 0.1,
            },
            timeout=300,
        )
        resp.raise_for_status()
        response = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  API error: {e}")
        results.append({"query": query, "status": "API_ERROR", "scores": {}})
        continue

    quote_cites = extract_quotes_with_citations(response)
    if not quote_cites:
        abstained = "could not find" in response.lower()
        status = "ABSTAINED" if abstained else "NO_QUOTES"
        results.append({"query": query, "status": status, "scores": {}})
        print(f"  {status}")
        continue

    full_pdf_text = extract_full_pdf_text(source)
    quote_scores = []
    for qc in quote_cites:
        quote = qc["quote"]
        cited_page = qc["cited_page"]
        if cited_page:
            cited_page_text = extract_page_range_text(source, cited_page, cited_page)
            cited_overlap = quote_in_page(quote, cited_page_text)
        else:
            cited_overlap = 0.0
        expected_page_text = extract_page_range_text(
            source, expected_page, expected_page
        )
        expected_overlap = quote_in_page(quote, expected_page_text)
        pdf_overlap = quote_in_page(quote, full_pdf_text)
        gt_overlap = SequenceMatcher(
            None, normalize(quote), normalize(ground_truth)
        ).ratio()
        quote_scores.append(
            {
                "quote_preview": quote[:80],
                "cited_page": cited_page,
                "grounded_on_cited_page": round(cited_overlap, 3),
                "grounded_on_expected_page": round(expected_overlap, 3),
                "grounded_in_pdf": round(pdf_overlap, 3),
                "ground_truth_match": round(gt_overlap, 3),
            }
        )

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
    icon = "Y" if best_pdf >= 0.90 else "~" if best_pdf >= 0.70 else "X"
    pg = "Y" if any_page_correct else "~" if any_page_near else "X"
    print(
        f"  {icon} pdf={best_pdf:.0%} cited_pg={best_cited:.0%} page={pg} gt={best_gt:.0%}"
    )

n = len(results)
quoted = [r for r in results if r["status"] == "QUOTED"]
abstained = [r for r in results if r["status"] == "ABSTAINED"]
no_quotes = [r for r in results if r["status"] == "NO_QUOTES"]
errors = [r for r in results if r["status"] == "API_ERROR"]

sep = "=" * 72
print(f"\n{sep}")
print(f"  CITATION ACCURACY TEST - {n} queries")
print(f"{sep}")
print(
    f"  Quoted: {len(quoted)}  Abstained: {len(abstained)}  No quotes: {len(no_quotes)}  Errors: {len(errors)}"
)

if quoted:
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
    dash = "-" * 50
    print(f"  {dash}")
    print(f"  {'Quote exists in source PDF (mean)':<40} {avg_pdf:>7.1%}")
    print(f"  {'  Y Verified (>=90%)':<40} {pdf_verified:>5}/{len(quoted)}")
    print(f"  {'  ~ Approximate (70-89%)':<40} {pdf_approx:>5}/{len(quoted)}")
    print(f"  {'  X Failed (<70%)':<40} {pdf_failed:>5}/{len(quoted)}")
    print(f"  {'Quote on cited page (mean)':<40} {avg_cited:>7.1%}")
    print(f"  {'Page number exact':<40} {pages_exact:>5}/{len(quoted)}")
    print(f"  {'Page number +/-1':<40} {pages_near:>5}/{len(quoted)}")
    print(f"  {'Ground truth match (mean)':<40} {avg_gt:>7.1%}")

    print(f"\n{sep}")
    print("  PER-QUERY DETAIL")
    print(f"{sep}")
    print(
        f"  {'#':<3} {'Query':<34} {'PDF':>5} {'Cited':>6} {'GT':>5} {'Pg':>3} {'Q':>2}"
    )
    print(f"  {dash}")
    for i, r in enumerate(results):
        q = r["query"][:31] + "..." if len(r["query"]) > 34 else r["query"]
        if r["status"] == "QUOTED":
            s = r["scores"]
            pdf = f"{s['grounded_pdf']:.0%}"
            ci = f"{s['grounded_cited']:.0%}"
            gt = f"{s['ground_truth']:.0%}"
            pg = "Y" if s["page_exact"] else "~" if s["page_near"] else "X"
            nq = str(r["num_quotes"])
            print(f"  {i + 1:<3} {q:<34} {pdf:>5} {ci:>6} {gt:>5} {pg:>3} {nq:>2}")
        else:
            print(
                f"  {i + 1:<3} {q:<34} {'---':>5} {'---':>6} {'---':>5} {'---':>3} {'---':>2}  [{r['status']}]"
            )

    print(f"\n{sep}")
    target = 0.85
    status = "PASS" if avg_pdf >= target else "FAIL"
    print(f"  TARGET: >={target:.0%} mean quote authenticity (exists in source PDF)")
    print(f"  ACTUAL: {avg_pdf:.1%}  [{status}]")
    print(f"{sep}")

with open("data/eval/ao_citation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to data/eval/ao_citation_results.json")
