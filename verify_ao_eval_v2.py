#!/usr/bin/env python3
import fitz
import json


def normalize(text):
    return " ".join(text.split())


pdf_path = (
    "data/processed/1972 Anti-Oedipus Capitalism and Schizophr - Deleuze, Gilles.pdf"
)
doc = fitz.open(pdf_path)

with open("data/eval/ao_eval_v2.json", "r") as f:
    data = json.load(f)

verified_count = 0
errors = []

for i, entry in enumerate(data):
    page_num = entry["pdf_page"]
    page = doc[page_num - 1]
    page_text = page.get_text()

    normalized_page = normalize(page_text)
    normalized_gt = normalize(entry["ground_truth"])

    if normalized_gt in normalized_page:
        verified_count += 1
        print(f"✓ Entry {i + 1}: Page {page_num} verified")
    else:
        print(f"❌ Entry {i + 1}: FAILED - ground truth not found on page {page_num}")
        errors.append(
            {
                "entry_num": i + 1,
                "page": page_num,
                "query": entry["query"],
                "ground_truth": entry["ground_truth"][:100],
            }
        )

doc.close()

print(f"\n{'=' * 60}")
print(f"VERIFICATION RESULTS")
print(f"{'=' * 60}")
print(f"Total entries: {len(data)}")
print(f"Verified: {verified_count}")
print(f"Failed: {len(errors)}")

if errors:
    print(f"\n❌ {len(errors)} entries failed verification:")
    for err in errors:
        print(f"\n  Entry {err['entry_num']} (Page {err['page']}):")
        print(f"    Query: {err['query']}")
        print(f"    Ground truth: {err['ground_truth']}...")
else:
    print(f"\n✓ All {len(data)} entries verified successfully!")
