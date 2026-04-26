"""
Benchmark B — Metadata extraction head-to-head.

Compares four extractors on title / copyright / combined pages from the
Deleuze corpus:

    1. donut    — p786/donut-base-finetuned-docvqa, image + question → answer
    2. regex    — hand-written regex over PyMuPDF (digital) or Surya md (scanned)
    3. spacy    — spaCy en_core_web_sm NER over the same text
    4. ollama   — llama3.1:8b-instruct-q4_K_M structured-extraction over same text

Fields: title, author, translator, publisher, year, isbn.
Metrics: normalized-match accuracy (primary), exact-match accuracy,
latency p50/p95, peak CUDA memory.

Writes:
    data/eval/donut_benchmark_b_metadata.json   — raw predictions + aggregates
    data/eval/donut_benchmark_b_metadata.md     — human-readable comparison table

Sidecar only: no imports from services/.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path
from statistics import median
from typing import Any

import fitz
import requests
import spacy
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "surya_all_input"
SURYA_CORPUS = ROOT / "data" / "surya_corpus"
OUT_JSON = ROOT / "data" / "eval" / "donut_benchmark_b_metadata.json"
OUT_MD = ROOT / "data" / "eval" / "donut_benchmark_b_metadata.md"

FIELDS = ["title", "author", "translator", "publisher", "year", "isbn"]

FIELD_QUESTIONS = {
    "title": "What is the title of this book?",
    "author": "Who is the author?",
    "translator": "Who is the translator?",
    "publisher": "Who is the publisher?",
    "year": "What year was this published?",
    "isbn": "What is the ISBN?",
}

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b-instruct-q4_K_M"


def ollama_unload() -> None:
    """Force Ollama to release the model from GPU memory (important when sharing
    an 8 GB GPU with Donut)."""
    try:
        requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "keep_alive": 0, "prompt": ""},
            timeout=10,
        )
        time.sleep(2)
    except Exception:
        pass


# ---------------------------------------------------------------- normalize

_LEADING_ARTICLE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s-]")
_WS = re.compile(r"\s+")


def _norm(value: str | None) -> str:
    if value is None:
        return ""
    v = str(value).strip().lower()
    v = _PUNCT.sub(" ", v)
    v = _WS.sub(" ", v).strip()
    v = _LEADING_ARTICLE.sub("", v)
    return v.strip()


def _exact(value: str | None) -> str:
    if value is None:
        return ""
    return _WS.sub(" ", str(value).strip().lower())


def field_match(pred: Any, gold: Any) -> tuple[bool, bool]:
    """Returns (exact_match, normalized_match). Both handle None on either side."""
    pred_s = "" if pred is None else str(pred).strip()
    gold_s = "" if gold is None else str(gold).strip()
    # Both empty (or null) counts as a match — extractor correctly said "nothing".
    if not pred_s and not gold_s:
        return True, True
    if not pred_s or not gold_s:
        return False, False
    em = _exact(pred_s) == _exact(gold_s)
    np, ng = _norm(pred_s), _norm(gold_s)
    # Substring symmetric: pred contains gold or vice-versa, to credit partial answers
    # (e.g. "Gilles Deleuze" vs "Deleuze"), but only when the shorter side is >=5 chars.
    nm = em or np == ng or (len(ng) >= 5 and ng in np) or (len(np) >= 5 and np in ng)
    return em, nm


# --------------------------------------------------------- text extraction

_SURYA_CACHE: dict[str, str] = {}


def surya_text(doc_name: str) -> str:
    if doc_name in _SURYA_CACHE:
        return _SURYA_CACHE[doc_name]
    stem = doc_name[:-4] if doc_name.lower().endswith(".pdf") else doc_name
    md = SURYA_CORPUS / stem / f"{stem}.md"
    text = md.read_text() if md.exists() else ""
    _SURYA_CACHE[doc_name] = text
    return text


def page_text(pdf_path: Path, page_no: int) -> str:
    """Match the pipeline's actual extraction: PyMuPDF for digital; Surya md for scanned."""
    doc = fitz.open(str(pdf_path))
    try:
        text = doc[page_no - 1].get_text("text")
    finally:
        doc.close()
    if len(text.strip()) >= 50:
        return text
    # Scanned — use Surya markdown for the whole doc (no per-page split available;
    # that's also how the real pipeline treats Surya output).
    return surya_text(pdf_path.name)[:4000]


def page_image(pdf_path: Path, page_no: int, dpi: int = 200) -> Image.Image:
    doc = fitz.open(str(pdf_path))
    try:
        pix = doc[page_no - 1].get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


# --------------------------------------------------------- extractor: regex

_ISBN_RE = re.compile(r"\bISBN[\s:]*([\d][\d\s\-Xx]{8,})")
_YEAR_COPYRIGHT_RE = re.compile(r"©\s*(?:[^\d\n]*?)(\d{4})")
_YEAR_BARE_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
_ALL_RIGHTS_RE = re.compile(r"all rights reserved", re.IGNORECASE)
_TRANSLATED_RE = re.compile(
    r"[Tt]ranslated\s+(?:from\s+the\s+\w+\s+)?(?:and\s+edited\s+)?by\s+([A-Z][\w.\-]*(?:\s+(?:[A-Z][\w.\-]*|and|&|,))*\s*[A-Z][\w.\-]*)"
)
_BY_AUTHOR_RE = re.compile(
    r"\bby\s+((?:[A-Z][\w.\-]+)(?:\s+[A-Z][\w.\-]+)+(?:\s+and\s+(?:[A-Z][\w.\-]+)(?:\s+[A-Z][\w.\-]+)+)?)"
)
_PUBLISHER_HINTS = [
    "Columbia University Press",
    "University of Minnesota Press",
    "Presses Universitaires de France",
    "Zone Books",
    "The Athlone Press",
    "Athlone Press",
    "Continuum",
    "City Lights Books",
    "Les Editions de Minuit",
]


def extract_regex(text: str) -> dict[str, str | None]:
    out: dict[str, str | None] = {f: None for f in FIELDS}

    m = _ISBN_RE.search(text)
    if m:
        out["isbn"] = m.group(1).strip()

    m = _YEAR_COPYRIGHT_RE.search(text)
    if m:
        out["year"] = m.group(1)
    elif (y := _YEAR_BARE_RE.search(text)):
        out["year"] = y.group(1)

    for pub in _PUBLISHER_HINTS:
        if pub.lower() in text.lower():
            out["publisher"] = pub
            break

    if (m := _TRANSLATED_RE.search(text)):
        out["translator"] = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(",")

    if "Deleuze" in text and "Guattari" in text:
        out["author"] = "Gilles Deleuze and Felix Guattari"
    elif "Deleuze" in text and "Parnet" in text:
        out["author"] = "Gilles Deleuze and Claire Parnet"
    elif "Deleuze" in text:
        out["author"] = "Gilles Deleuze"

    # Title: heuristic — the longest line in the first 600 chars with no digits and
    # reasonable length is a plausible candidate. Weak; that's the point.
    head = text[:600]
    candidates = [ln.strip() for ln in head.splitlines() if ln.strip()]
    candidates = [
        c for c in candidates
        if 4 <= len(c) <= 80 and not re.search(r"\d", c)
        and not any(k in c.lower() for k in ("translated", "edited", "press", "by ", "copyright", "deleuze", "guattari"))
    ]
    if candidates:
        out["title"] = max(candidates, key=len)

    return out


# ----------------------------------------------------- extractor: spacy NER

_SPACY_NLP = None


def spacy_nlp():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        _SPACY_NLP = spacy.load("en_core_web_sm")
    return _SPACY_NLP


def extract_spacy(text: str) -> dict[str, str | None]:
    out: dict[str, str | None] = {f: None for f in FIELDS}
    doc = spacy_nlp()(text[:4000])
    persons, orgs, dates = [], [], []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            persons.append(ent.text.strip())
        elif ent.label_ in ("ORG", "GPE"):
            orgs.append(ent.text.strip())
        elif ent.label_ == "DATE":
            dates.append(ent.text.strip())

    # Author: first PERSON that looks like "Gilles Deleuze" or similar.
    deleuze_like = [p for p in persons if "deleuze" in p.lower()]
    if deleuze_like:
        out["author"] = deleuze_like[0]

    # Translator: PERSON appearing after the word "translated"
    low = text.lower()
    t_idx = low.find("translated")
    if t_idx >= 0:
        tail = text[t_idx : t_idx + 400]
        trans_doc = spacy_nlp()(tail)
        t_persons = [e.text for e in trans_doc.ents if e.label_ == "PERSON"]
        if t_persons:
            out["translator"] = t_persons[0]

    # Publisher: longest ORG that contains "Press" / "Books" / "Minuit".
    press_orgs = [o for o in orgs if re.search(r"\bpress\b|\bbooks\b|minuit|continuum|universitaires", o, re.I)]
    if press_orgs:
        out["publisher"] = max(press_orgs, key=len)
    elif orgs:
        out["publisher"] = max(orgs, key=len)

    # Year: first 4-digit year from DATE ents
    for d in dates:
        m = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", d)
        if m:
            out["year"] = m.group(1)
            break
    if not out["year"]:
        m = _YEAR_COPYRIGHT_RE.search(text) or _YEAR_BARE_RE.search(text)
        if m:
            out["year"] = m.group(1)

    # ISBN: NER doesn't help; fall back to regex on same text (baseline is "NER + cheap helpers").
    m = _ISBN_RE.search(text)
    if m:
        out["isbn"] = m.group(1).strip()

    # Title: NER rarely labels book titles; leave null (honest — NER doesn't solve this field).
    return out


# ---------------------------------------------------- extractor: ollama llm

_OLLAMA_AVAILABLE: bool | None = None


def ollama_available() -> bool:
    global _OLLAMA_AVAILABLE
    if _OLLAMA_AVAILABLE is None:
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            names = [m["name"] for m in r.json().get("models", [])]
            _OLLAMA_AVAILABLE = OLLAMA_MODEL in names
        except Exception:
            _OLLAMA_AVAILABLE = False
    return _OLLAMA_AVAILABLE


OLLAMA_SYSTEM = (
    "You extract bibliographic metadata from a book's title or copyright page. "
    "Return ONLY a JSON object with these six keys: title, author, translator, "
    "publisher, year, isbn. Use null for fields not present on the page. "
    "Do not invent values. Do not include any explanation."
)


def extract_ollama(text: str) -> dict[str, str | None]:
    if not ollama_available():
        return {f: None for f in FIELDS}
    prompt = f"PAGE TEXT:\n---\n{text[:3500]}\n---\nJSON:"
    body = {
        "model": OLLAMA_MODEL,
        "system": OLLAMA_SYSTEM,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 300},
        "format": "json",
    }
    try:
        r = requests.post(OLLAMA_URL, json=body, timeout=120)
        r.raise_for_status()
        raw = r.json().get("response", "").strip()
        data = json.loads(raw) if raw else {}
    except Exception as e:
        return {f: None for f in FIELDS} | {"_error": f"ollama: {e}"}  # type: ignore
    out: dict[str, str | None] = {}
    for f in FIELDS:
        v = data.get(f)
        out[f] = None if v in (None, "", "null", "N/A") else str(v).strip()
    return out


# ----------------------------------------------------------- extractor: donut

_DONUT: dict[str, Any] = {}


def donut_load():
    if _DONUT:
        return
    print("[donut] loading p786/donut-base-finetuned-docvqa...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = DonutProcessor.from_pretrained("p786/donut-base-finetuned-docvqa")
    model = VisionEncoderDecoderModel.from_pretrained("p786/donut-base-finetuned-docvqa")
    model = model.to(device).eval()
    _DONUT.update(processor=processor, model=model, device=device)
    print(f"[donut] loaded on {device}")


def donut_answer(image: Image.Image, question: str) -> str:
    processor = _DONUT["processor"]
    model = _DONUT["model"]
    device = _DONUT["device"]
    prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    decoder_input_ids = processor.tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    with torch.inference_mode():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
    seq = processor.batch_decode(outputs.sequences)[0]
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    # Donut returns something like <s_answer>VALUE</s_answer>... extract the answer span.
    m = re.search(r"<s_answer>(.*?)(</s_answer>|<|$)", seq)
    if m:
        ans = m.group(1).strip()
    else:
        ans = seq.replace(prompt, "").strip()
    return ans


def extract_donut(image: Image.Image) -> dict[str, str | None]:
    donut_load()
    out: dict[str, str | None] = {}
    for f in FIELDS:
        ans = donut_answer(image, FIELD_QUESTIONS[f]).strip()
        # Donut is trained to answer in short spans; empty / refusal → null.
        if not ans or ans.lower() in ("none", "n/a", "unknown", "no", "no answer"):
            out[f] = None
        else:
            out[f] = ans
    return out


# ---------------------------------------------------------- benchmark loop

EXTRACTORS: list[tuple[str, str]] = [
    ("regex", "text"),
    ("spacy", "text"),
    ("ollama", "text"),
    ("donut", "image"),
]


def run_benchmark(gold_path: Path) -> dict[str, Any]:
    gold_doc = json.loads(gold_path.read_text())
    pages = gold_doc["pages"]
    results = {
        "gold_path": str(gold_path),
        "extractors": [e[0] for e in EXTRACTORS],
        "pages": [],
        "ollama_available": ollama_available(),
    }

    # Seed per-page result scaffolds.
    for entry in pages:
        pdf = CORPUS / entry["doc"]
        if not pdf.exists():
            print(f"[skip] missing pdf: {pdf}")
            continue
        results["pages"].append({
            "doc": entry["doc"],
            "page": entry["page"],
            "type": entry["type"],
            "scan_layout": entry.get("scan_layout", "single"),
            "gold": entry["gold"],
            "predictions": {},
            "latency_sec": {},
            "cuda_peak_mb": {},
        })

    text_extractors = {"regex": extract_regex, "spacy": extract_spacy, "ollama": extract_ollama}

    # Phase 1 — all text-based extractors, per-page.
    for per_page in results["pages"]:
        pdf = CORPUS / per_page["doc"]
        text = page_text(pdf, per_page["page"])
        for name in ("regex", "spacy", "ollama"):
            if name == "ollama" and not ollama_available():
                per_page["predictions"][name] = {f: None for f in FIELDS} | {"_error": "ollama unavailable"}
                per_page["latency_sec"][name] = None
                per_page["cuda_peak_mb"][name] = None
                continue
            t0 = time.perf_counter()
            try:
                pred = text_extractors[name](text)
            except Exception as e:
                pred = {f: None for f in FIELDS} | {"_error": f"{type(e).__name__}: {e}"}
            dt = time.perf_counter() - t0
            per_page["predictions"][name] = pred
            per_page["latency_sec"][name] = dt
            per_page["cuda_peak_mb"][name] = 0.0
            print(
                f"[{per_page['doc'][:40]:40s} p{per_page['page']:>2} {per_page['type'][:9]:9s}] "
                f"{name:7s} {dt:6.2f}s  title={str(pred.get('title'))[:30]!r}"
            )

    # Unload Ollama from GPU before loading Donut — 8 GB card can't hold both.
    if ollama_available():
        print("[ollama] unloading model from GPU...")
        ollama_unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 2 — Donut on page images.
    for per_page in results["pages"]:
        pdf = CORPUS / per_page["doc"]
        image = page_image(pdf, per_page["page"])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            pred = extract_donut(image)
        except Exception as e:
            pred = {f: None for f in FIELDS} | {"_error": f"{type(e).__name__}: {e}"}
        dt = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
        per_page["predictions"]["donut"] = pred
        per_page["latency_sec"]["donut"] = dt
        per_page["cuda_peak_mb"]["donut"] = peak_mb
        print(
            f"[{per_page['doc'][:40]:40s} p{per_page['page']:>2} {per_page['type'][:9]:9s}] "
            f"donut   {dt:6.2f}s  {peak_mb:6.0f}MB  title={str(pred.get('title'))[:40]!r}"
        )

    # ---- aggregate
    agg: dict[str, dict[str, dict[str, float]]] = {}
    for name, _ in EXTRACTORS:
        per_field: dict[str, dict[str, float]] = {}
        for f in FIELDS:
            em_ct = nm_ct = total = 0
            for p in results["pages"]:
                gold_v = p["gold"].get(f)
                pred_v = p["predictions"].get(name, {}).get(f)
                em, nm = field_match(pred_v, gold_v)
                em_ct += int(em)
                nm_ct += int(nm)
                total += 1
            per_field[f] = {
                "exact": em_ct / total if total else 0.0,
                "normalized": nm_ct / total if total else 0.0,
                "n": total,
            }
        overall_nm = (
            sum(per_field[f]["normalized"] for f in FIELDS) / len(FIELDS)
        )
        latencies = [
            p["latency_sec"][name] for p in results["pages"]
            if p["latency_sec"].get(name) is not None
        ]
        per_field["_overall_normalized"] = overall_nm  # type: ignore
        per_field["_latency_p50"] = median(latencies) if latencies else None  # type: ignore
        per_field["_latency_p95"] = (
            sorted(latencies)[int(0.95 * (len(latencies) - 1))]
            if len(latencies) > 1 else (latencies[0] if latencies else None)
        )  # type: ignore
        peaks = [
            p["cuda_peak_mb"][name] for p in results["pages"]
            if p["cuda_peak_mb"].get(name) is not None
        ]
        per_field["_cuda_peak_mb_max"] = max(peaks) if peaks else None  # type: ignore
        agg[name] = per_field

    results["aggregate"] = agg
    return results


# ------------------------------------------------------------- md rendering


def render_md(results: dict[str, Any]) -> str:
    agg = results["aggregate"]
    lines = []
    lines.append("# Benchmark B — Metadata Extraction\n")
    lines.append(
        f"Gold: `{results['gold_path']}` — {len(results['pages'])} pages, "
        f"{len(FIELDS)} fields each. Ollama available: {results['ollama_available']}.\n"
    )
    lines.append("## Normalized-match accuracy by field\n")
    header = "| field | " + " | ".join(results["extractors"]) + " |"
    sep = "|" + "---|" * (1 + len(results["extractors"]))
    lines.append(header)
    lines.append(sep)
    for f in FIELDS:
        row = [f]
        for ex in results["extractors"]:
            v = agg[ex][f]["normalized"]
            row.append(f"{v*100:.0f}%")
        lines.append("| " + " | ".join(row) + " |")
    # overall
    row = ["**overall**"]
    for ex in results["extractors"]:
        v = agg[ex]["_overall_normalized"]  # type: ignore
        row.append(f"**{v*100:.0f}%**")
    lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Latency (seconds per page)\n")
    lines.append("| extractor | p50 | p95 | peak CUDA (MB) |")
    lines.append("|---|---|---|---|")
    for ex in results["extractors"]:
        p50 = agg[ex]["_latency_p50"]  # type: ignore
        p95 = agg[ex]["_latency_p95"]  # type: ignore
        peak = agg[ex]["_cuda_peak_mb_max"]  # type: ignore
        lines.append(
            f"| {ex} | {p50:.2f} | {p95:.2f} | {peak:.0f} |"
            if p50 is not None else f"| {ex} | — | — | — |"
        )

    lines.append("\n## Decision rule\n")
    lines.append(
        "Donut wins a field if its normalized-match accuracy exceeds the best "
        "non-Donut extractor by ≥10 percentage points **and** its latency p95 is "
        "within 2× of that extractor.\n"
    )
    winners = []
    for f in FIELDS:
        donut_acc = agg["donut"][f]["normalized"]
        best_other = max(agg[ex][f]["normalized"] for ex in results["extractors"] if ex != "donut")
        best_other_name = max(
            (ex for ex in results["extractors"] if ex != "donut"),
            key=lambda ex: agg[ex][f]["normalized"],
        )
        donut_p95 = agg["donut"]["_latency_p95"]  # type: ignore
        other_p95 = agg[best_other_name]["_latency_p95"]  # type: ignore
        lat_ok = donut_p95 is None or other_p95 is None or donut_p95 <= 2 * max(other_p95, 0.01)
        delta_pp = (donut_acc - best_other) * 100
        status = "✓" if (delta_pp >= 10 and lat_ok) else "✗"
        winners.append((f, status, delta_pp, best_other_name, best_other))
        lines.append(
            f"- `{f}`: donut {donut_acc*100:.0f}% vs best-other ({best_other_name}) "
            f"{best_other*100:.0f}% — Δ = {delta_pp:+.0f}pp {status}"
        )
    n_wins = sum(1 for _, s, *_ in winners if s == "✓")
    lines.append(f"\n**Donut wins {n_wins} / {len(FIELDS)} fields.** "
                 f"Integration threshold: ≥2. "
                 f"Recommendation: {'INTEGRATE' if n_wins >= 2 else 'DO NOT INTEGRATE'}.\n")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- cli


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/eval/donut_metadata_gold.json")
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--out-md", default=str(OUT_MD))
    args = ap.parse_args()

    results = run_benchmark(Path(args.gold))
    Path(args.out_json).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    Path(args.out_md).write_text(render_md(results))
    print(f"\nwrote {args.out_json}")
    print(f"wrote {args.out_md}")

    # Print the md summary so the run surfaces the numbers.
    print("\n" + "=" * 78)
    print(render_md(results))


if __name__ == "__main__":
    main()
