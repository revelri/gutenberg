#!/usr/bin/env python3
"""Canonical ground truth extraction for evaluation.

Extracts quotation passages from a PDF using PyMuPDF and applies the same
text cleaning pipeline (clean_for_ingestion) used during document ingestion.
This ensures eval ground truth matches the text that the retrieval pipeline
actually indexes.

Usage:
    python scripts/extract_eval_ground_truth.py <pdf_path> [--pages 7,9,14,18]
    python scripts/extract_eval_ground_truth.py <pdf_path> --existing data/eval/ao_ground_truth.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))
from shared.text_normalize import clean_for_ingestion, normalize_for_comparison


def extract_page_text(pdf_path: Path, page_num: int) -> str:
    """Extract and clean text from a single PDF page (1-indexed)."""
    doc = fitz.open(str(pdf_path))
    page = doc[page_num - 1]
    text = page.get_text("text")
    doc.close()
    return clean_for_ingestion(text)


def find_best_passage(page_text: str, min_words: int = 20, max_words: int = 80) -> str:
    """Find the best substantive passage on a page.

    Prefers multi-sentence passages with philosophical content,
    skipping headers, footnotes, and bibliographic entries.
    """
    # Split into paragraphs
    paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]

    candidates = []
    for para in paragraphs:
        words = para.split()
        if len(words) < min_words:
            continue
        # Skip likely headers, footnotes, bibliographic entries
        if para.startswith(("#", "[", "**")) or re.match(r"^\d+\.", para):
            continue
        if len(words) > max_words:
            # Take first max_words words, ending at a sentence boundary
            truncated = " ".join(words[:max_words])
            last_period = truncated.rfind(". ")
            if last_period > len(truncated) // 2:
                truncated = truncated[: last_period + 1]
            candidates.append(truncated)
        else:
            candidates.append(para)

    # Prefer longer, more substantive passages
    candidates.sort(key=lambda c: len(c.split()), reverse=True)
    return candidates[0] if candidates else ""


def reprocess_existing(pdf_path: Path, existing_path: Path) -> list[dict]:
    """Re-extract ground truth from an existing eval file through the ingestion pipeline."""
    with open(existing_path) as f:
        entries = json.load(f)

    updated = []
    for entry in entries:
        page_num = entry["pdf_page"]
        page_text = extract_page_text(pdf_path, page_num)

        # Try to find the original ground truth in the cleaned page text
        old_gt_norm = normalize_for_comparison(entry["ground_truth"])
        page_norm = normalize_for_comparison(page_text)

        if old_gt_norm[:50] in page_norm:
            # Original quote still findable — re-extract from cleaned text
            # Find the matching region in the cleaned (non-lowercased) page text
            start = page_norm.find(old_gt_norm[:50])
            # Extract from cleaned page text at the approximate position
            # Use word boundaries for cleaner extraction
            words = page_text.split()
            clean_norm = normalize_for_comparison(page_text)
            # Find approximate word position
            char_pos = clean_norm.find(old_gt_norm[:50])
            if char_pos >= 0:
                # Count words up to this position
                prefix = clean_norm[:char_pos]
                word_start = len(prefix.split()) - 1
                gt_word_count = len(entry["ground_truth"].split())
                new_gt = " ".join(words[max(0, word_start) : word_start + gt_word_count])
                updated.append({**entry, "ground_truth": new_gt})
            else:
                updated.append({**entry, "ground_truth": entry["ground_truth"]})
        else:
            # Original quote not found — extract best passage from cleaned page
            passage = find_best_passage(page_text)
            if passage:
                print(f"  WARNING: Page {page_num} — original GT not found, using best passage")
                updated.append({**entry, "ground_truth": passage})
            else:
                print(f"  WARNING: Page {page_num} — no suitable passage found, keeping original")
                updated.append(entry)

    return updated


def extract_fresh(pdf_path: Path, pages: list[int]) -> list[dict]:
    """Extract fresh ground truth passages from specified pages."""
    entries = []
    for page_num in pages:
        page_text = extract_page_text(pdf_path, page_num)
        passage = find_best_passage(page_text)
        if passage:
            entries.append({
                "query": f"What does the text discuss on page {page_num}?",
                "source": pdf_path.name,
                "pdf_page": page_num,
                "ground_truth": passage,
            })
        else:
            print(f"  WARNING: Page {page_num} — no suitable passage found")
    return entries


def main():
    parser = argparse.ArgumentParser(description="Extract eval ground truth from PDF")
    parser.add_argument("pdf_path", type=Path, help="Path to source PDF")
    parser.add_argument("--pages", type=str, help="Comma-separated page numbers")
    parser.add_argument("--existing", type=Path, help="Existing eval JSON to reprocess")
    parser.add_argument("--output", type=Path, help="Output JSON path")
    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"ERROR: PDF not found: {args.pdf_path}")
        sys.exit(1)

    if args.existing:
        print(f"Reprocessing {args.existing} through ingestion pipeline...")
        entries = reprocess_existing(args.pdf_path, args.existing)
    elif args.pages:
        pages = [int(p.strip()) for p in args.pages.split(",")]
        print(f"Extracting from pages: {pages}")
        entries = extract_fresh(args.pdf_path, pages)
    else:
        print("ERROR: Specify --pages or --existing")
        sys.exit(1)

    output = args.output or Path("data/eval/ground_truth_cleaned.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(entries)} entries to {output}")


if __name__ == "__main__":
    main()
