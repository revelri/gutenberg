#!/usr/bin/env python3
import fitz
import json
import re


def normalize_whitespace(text):
    return " ".join(text.split())


pdf_path = (
    "data/processed/1972 Anti-Oedipus Capitalism and Schizophr - Deleuze, Gilles.pdf"
)
doc = fitz.open(pdf_path)

print(f"PDF has {doc.page_count} pages\n")

# Read pages across different ranges
ranges_to_check = [
    (10, 30),
    (30, 60),
    (60, 100),
    (100, 150),
    (150, 200),
    (200, 213),
]

for start, end in ranges_to_check:
    print(f"\n=== Pages {start}-{end} ===")
    for page_num in range(start, min(end + 1, doc.page_count)):
        page = doc[page_num - 1]
        text = normalize_whitespace(page.get_text())
        # Print first 500 chars
        preview = text[:800]
        if len(preview.strip()) > 50:
            print(f"\nPage {page_num}:")
            print(preview[:600])
            print("...")

doc.close()
