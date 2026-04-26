#!/usr/bin/env python3
import fitz
import json
import re


def normalize(text):
    return " ".join(text.split())


pdf_path = (
    "data/processed/1972 Anti-Oedipus Capitalism and Schizophr - Deleuze, Gilles.pdf"
)
doc = fitz.open(pdf_path)

quotations = []

pages_to_read = [
    10,
    11,
    12,
    20,
    25,
    26,
    28,
    29,
    30,
    31,
    32,
    33,
    35,
    37,
    39,
    46,
    50,
    52,
    55,
    58,
    61,
    65,
    70,
    72,
    76,
    78,
    80,
    82,
    85,
    86,
    88,
    90,
]

for page_num in pages_to_read:
    page = doc[page_num - 1]
    text = page.get_text()
    normalized = normalize(text)

    # Skip very short pages
    if len(normalized) < 100:
        continue

    # Find passages with    # Look for substantive philosophical content (at least 30 words)
    words = normalized.split()

    # Find a good starting sentence
    for i in range(len(words)):
        if words[i][0].isupper() and (i == 0 or words[i - 1][-1] in '.!?"'):
            # Found start of sentence
            start_idx = i
            break

    if start_idx is None:
        continue

    # Find end of sentence (look for 3-4 sentences)
    end_idx = start_idx
    sentence_count = 0
    for i in range(start_idx + 1, len(words)):
        if i < len(words) and sentence_count < 4:
            break
        if words[i][-1] in ".!?":
            end_idx = i
            sentence_count += 1

    if sentence_count >= 2:
        passage = " ".join(words[start_idx:end_idx])

        # Check if passage is substantive (not just "Contents" or "Introduction")
        if (
            "contents" in passage.lower()
            or "bibliography" in passage.lower()
            or "index" in passage.lower()
        ):
            continue

        if len(passage.split()) >= 20:
            quotations.append(
                {
                    "query": f"What do Deleuze and Guattari say about the content on page {page_num}?",
                    "source": "1972 Anti-Oedipus Capitalism and Schizophr - Deleuze, Gilles.pdf",
                    "pdf_page": page_num,
                    "ground_truth": passage,
                }
            )
            print(f"✓ Page {page_num}: Found passage ({len(passage.split())} words)")

            if len(quotations) >= 20:
                break

doc.close()

print(f"\nExtracted {len(quotations)} quotations")

with open("data/eval/ao_eval_v2.json", "w") as f:
    json.dump(quotations, f, indent=2)

print(f"Wrote {len(quotations)} entries to data/eval/ao_eval_v2.json")
