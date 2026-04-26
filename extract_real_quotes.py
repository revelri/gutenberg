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

# Extract real passages from specific pages
pages_to_extract = [
    (10, "political economy"),
    (11, "anthropomorphic"),
    (12, "revolutionary actions"),
    (20, "Production is not recorded"),
    (25, "delirium"),
    (26, "great discovery"),
    (28, "Lack is created"),
    (29, "desire produces reality"),
    (30, "work of art"),
    (31, "decoding of flows"),
    (32, "A machine may be defined"),
    (33, "code built into"),
    (35, "desiring-machines"),
    (37, "Partial objects"),
    (39, "Foucault has noted"),
    (46, "subject-groups"),
    (52, "exclusive, restrictive"),
    (55, "orphan unconscious"),
    (58, "autistic"),
    (61, "eccentric, decentered"),
    (85, "primitive territorial"),
    (86, "despotic machine"),
]

results = []

for page_num, search_phrase in pages_to_extract:
    page = doc[page_num - 1]
    text = page.get_text()
    normalized = normalize(text)

    # Find the search phrase
    idx = normalized.lower().find(search_phrase.lower())
    if idx == -1:
        print(f"✗ Page {page_num}: Could not find '{search_phrase}'")
        continue

    # Extract context around it
    start = max(0, idx - 100)
    end = min(len(normalized), idx + 400)
    passage = normalized[start:end]

    # Find sentence boundaries
    sentences = []
    for match in re.finditer(r"[.!?]+\s+", passage):
        if match:
            sentences.append(match.group().strip())

    if len(sentences) >= 2:
        result = " ".join(sentences[:3])  # Take first 3 sentences

        # Check length
        word_count = len(result.split())
        if word_count >= 20:
            results.append(
                {
                    "query": f"What do Deleuze and Guattari say about {search_phrase} in Anti-Oedipus?",
                    "source": "1972 Anti-Oedipus Capitalism and Schizophr - Deleuze, Gilles.pdf",
                    "pdf_page": page_num,
                    "ground_truth": result,
                }
            )
            print(f"✓ Page {page_num}: Extracted passage ({word_count} words)")
        else:
            print(f"✗ Page {page_num}: Passage too short ({word_count} words)")
    else:
        print(f"✗ Page {page_num}: Could not extract complete sentences")

doc.close()

# Write to file
with open("data/eval/ao_eval_v2.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Wrote {len(results)} entries to data/eval/ao_eval_v2.json")
