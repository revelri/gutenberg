#!/usr/bin/env python
"""
Extract 20 test cases from Anti-Oedipus PDF for ao_eval_v2.json
- Spread across different page ranges
- At least 20 words per quotation
- Substantive philosophical content (not headers, TOC, etc.)
- No duplicates from existing ao_ground_truth.json
"""

import json
import fitz  # PyMuPDF
from pathlib import Path

# Load existing ground truth to avoid duplicates
existing_gt_path = Path("data/eval/ao_ground_truth.json")
with open(existing_gt_path) as f:
    existing_gt = json.load(f)

existing_quotations = {entry["ground_truth"] for entry in existing_gt}
existing_pages = {entry["pdf_page"] for entry in existing_gt}

# Open the PDF
pdf_path = Path(
    "data/pdfs/1972 Anti-Oedipus Capitalism and Schizophr - Deleuze, Gilles.pdf"
)
doc = fitz.open(pdf_path)
total_pages = len(doc)

print(f"PDF has {total_pages} pages")

# Target page ranges (PyMuPDF uses 0-index, but we store 1-indexed)
# We want to cover: 1-50, 50-100, 100-200, 200-300, 300-400, 400+
# But PDF only has 213 pages, so: 1-50, 50-100, 100-150, 150-213

# Strategy: Sample pages from different sections
# Pages to check (0-indexed for PyMuPDF)
page_ranges = [
    range(40, 50),  # Pages 41-50
    range(55, 65),  # Pages 56-65
    range(70, 80),  # Pages 71-80
    range(85, 95),  # Pages 86-95
    range(100, 110),  # Pages 101-110
    range(115, 125),  # Pages 116-125
    range(130, 140),  # Pages 131-140
    range(145, 155),  # Pages 146-155
    range(160, 170),  # Pages 161-170
    range(175, 185),  # Pages 176-185
    range(190, 200),  # Pages 191-200
    range(200, 213),  # Pages 201-213
]

candidates = []


def is_good_quotation(text):
    """Check if text is a good quotation candidate"""
    # At least 20 words
    words = text.split()
    if len(words) < 20:
        return False

    # Skip headers, TOC, etc.
    lower = text.lower()
    skip_phrases = [
        "contents",
        "chapter",
        "part i",
        "part ii",
        "part iii",
        "bibliography",
        "index",
        "notes",
        "page",
        "deleuze and guattari",
    ]
    for phrase in skip_phrases:
        if lower.startswith(phrase) and len(text) < 100:
            return False

    # Should have substantive content (look for philosophical terms)
    philosophical_terms = [
        "desire",
        "production",
        "machine",
        "schizo",
        "capital",
        "body",
        "organ",
        "social",
        "process",
        "flow",
        "code",
        "socius",
        "territorial",
        "oedipus",
        "unconscious",
        "libido",
        "representation",
        "real",
        "virtual",
    ]

    has_philosophical = any(term in lower for term in philosophical_terms)

    return has_philosophical


def extract_quotations_from_page(page, page_num):
    """Extract good quotations from a page"""
    text = page.get_text()

    # Split into paragraphs
    paragraphs = text.split("\n\n")

    results = []
    for para in paragraphs:
        para = para.strip()
        if is_good_quotation(para) and para not in existing_quotations:
            results.append((page_num, para))

    return results


# Extract from all target pages
for page_range in page_ranges:
    for page_idx in page_range:
        if page_idx >= total_pages:
            continue
        page = doc[page_idx]
        page_num = page_idx + 1  # Convert to 1-indexed

        found = extract_quotations_from_page(page, page_num)
        candidates.extend(found)

print(f"\nFound {len(candidates)} candidate quotations")

# Select 20 diverse quotations
# Group by page range to ensure spread
from collections import defaultdict

by_range = defaultdict(list)
for page_num, quote in candidates:
    if page_num <= 50:
        by_range["1-50"].append((page_num, quote))
    elif page_num <= 100:
        by_range["51-100"].append((page_num, quote))
    elif page_num <= 150:
        by_range["101-150"].append((page_num, quote))
    elif page_num <= 213:
        by_range["151-213"].append((page_num, quote))

print("\nCandidates by range:")
for r, items in sorted(by_range.items()):
    print(f"  {r}: {len(items)} candidates")

# Select quotations trying to get ~5 from each range
selected = []
for r in ["1-50", "51-100", "101-150", "151-213"]:
    items = by_range.get(r, [])
    # Take up to 5 from each range
    selected.extend(items[:5])

# If we don't have enough, take more from ranges that have extras
if len(selected) < 20:
    for r in ["1-50", "51-100", "101-150", "151-213"]:
        items = by_range.get(r, [])
        remaining = [x for x in items if x not in selected]
        needed = 20 - len(selected)
        selected.extend(remaining[:needed])
        if len(selected) >= 20:
            break

# Trim to exactly 20
selected = selected[:20]

print(f"\nSelected {len(selected)} quotations")

# Create ground truth entries
entries = []
for i, (page_num, quote) in enumerate(selected, 1):
    # Create a natural language query (not verbatim)
    # Extract key concepts from the quote
    words = quote.lower().split()

    # Find significant words for the query
    query_templates = [
        f"Find the passage in Anti-Oedipus that discusses {words[5]} and {words[10]}",
        f"Locate where Deleuze and Guattari talk about {words[8]}",
        f"Find the discussion of {words[7]} in relation to {words[12]}",
    ]

    import random

    query = random.choice(query_templates)

    entry = {
        "query": query,
        "source": "1972 Anti-Oedipus Capitalism and Schizophr - Deleuze, Gilles.pdf",
        "pdf_page": page_num,
        "ground_truth": quote,
        "label": f"Eval v2 entry {i}",
    }
    entries.append(entry)

# Save to file
output_path = Path("data/eval/ao_eval_v2.json")
with open(output_path, "w") as f:
    json.dump(entries, f, indent=2)

print(f"\nSaved to {output_path}")

# Verify all quotations
print("\nVerifying quotations...")
verification_failures = []

for entry in entries:
    page_num = entry["pdf_page"]
    quote = entry["ground_truth"]
    page_idx = page_num - 1

    if page_idx >= total_pages:
        verification_failures.append((page_num, "Page out of range"))
        continue

    page = doc[page_idx]
    text = page.get_text()

    if quote not in text:
        verification_failures.append((page_num, "Quotation not found on page"))

if verification_failures:
    print(f"\n{len(verification_failures)} verification failures:")
    for page_num, reason in verification_failures:
        print(f"  Page {page_num}: {reason}")
else:
    print("\nAll 20 quotations verified successfully!")

doc.close()
