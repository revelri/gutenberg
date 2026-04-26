#!/usr/bin/env python3
"""Extract 20 high-quality quotations from Anti-Oedipus PDF for eval ground truth."""

import fitz
import json
import re


def normalize_whitespace(text):
    """Normalize whitespace for comparison."""
    return " ".join(text.split())


def extract_page_text(pdf_path, page_num):
    """Extract text from a specific page."""
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]  # PyMuPDF is 0-indexed
    text = page.get_text()
    doc.close()
    return text


def find_quotation_in_page(text, quotation):
    """Check if quotation appears in page text."""
    normalized_text = normalize_whitespace(text)
    normalized_quote = normalize_whitespace(quotation)
    return normalized_quote in normalized_text


def extract_clean_passage(text, start_marker, length=200):
    """Extract a clean passage starting from a marker."""
    idx = text.find(start_marker)
    if idx == -1:
        return None
    passage = text[idx : idx + length]
    # Clean up hyphenation and whitespace
    passage = re.sub(r"-\s+", "", passage)
    passage = normalize_whitespace(passage)
    return passage


# PDF path
pdf_path = (
    "data/processed/1972 Anti-Oedipus Capitalism and Schizophr - Deleuze, Gilles.pdf"
)

# Page ranges to sample from (1-indexed PDF pages)
# Target distribution across the book
page_ranges = [
    (1, 50, 4),  # Early pages: 4 quotes
    (51, 100, 4),  # First quarter: 4 quotes
    (101, 200, 5),  # Second quarter: 5 quotes
    (201, 300, 4),  # Third quarter: 4 quotes
    (301, 400, 3),  # Fourth quarter: 3 quotes
]

# Quotations to extract - these are NEW passages not in ao_ground_truth.json
# I'll extract them page by page and verify they exist
target_quotations = [
    # Range 1-50
    {
        "page": 25,
        "search": "synthesis of connection",
        "query": "How do Deleuze and Guattari define the connective synthesis in Anti-Oedipus?",
    },
    {
        "page": 26,
        "search": "synthesis of conjunction",
        "query": "What is the conjunctive synthesis of consumption and its relationship to desire?",
    },
    {
        "page": 39,
        "search": "partial objects",
        "query": "How do Deleuze and Guattari describe partial objects in relation to the body without organs?",
    },
    {
        "page": 42,
        "search": "the full BwO",
        "query": "What distinguishes the full body without organs from the empty body without organs?",
    },
    # Range 51-100
    {
        "page": 77,
        "search": "primitive machine",
        "query": "How is the primitive territorial machine characterized in Anti-Oedipus?",
    },
    {
        "page": 85,
        "search": "cruelty is not",
        "query": "What do Deleuze and Guattari say about cruelty in relation to inscription?",
    },
    {
        "page": 90,
        "search": "despotic machine",
        "query": "How does the despotic machine differ from the primitive territorial machine?",
    },
    {
        "page": 97,
        "search": "Alliance and filiation",
        "query": "What is the relationship between alliance and filiation in primitive societies?",
    },
    # Range 101-200
    {
        "page": 115,
        "search": "capitalist machine",
        "query": "What defines the capitalist machine according to Deleuze and Guattari?",
    },
    {
        "page": 140,
        "search": "State apparatus",
        "query": "How do Deleuze and Guattari analyze the State apparatus?",
    },
    {
        "page": 160,
        "search": "primitive repression",
        "query": "What is primitive repression and how does it differ from capitalist repression?",
    },
    {
        "page": 175,
        "search": "the despotic signifier",
        "query": "How do Deleuze and Guattari characterize the despotic signifier?",
    },
    {
        "page": 190,
        "search": "decoding of flows",
        "query": "What role does the decoding of flows play in the emergence of capitalism?",
    },
    # Range 201-300
    {
        "page": 210,
        "search": "Oedipus is always",
        "query": "In what sense is Oedipus always a colonization or imperialism according to Anti-Oedipus?",
    },
    {
        "page": 235,
        "search": "Freud discovers",
        "query": "What did Freud discover about the nature of desire according to Deleuze and Guattari?",
    },
    {
        "page": 260,
        "search": "schizoanalysis",
        "query": "What is the goal of schizoanalysis as a critical and revolutionary practice?",
    },
    {
        "page": 280,
        "search": "micro-politics",
        "query": "How do Deleuze and Guattari distinguish micro-politics from macro-politics?",
    },
    # Range 301-400
    {
        "page": 320,
        "search": "molecular revolution",
        "query": "What is the relationship between molecular desire and revolutionary politics?",
    },
    {
        "page": 350,
        "search": "collective assemblages",
        "query": "How do collective assemblages of enunciation relate to desire?",
    },
    {
        "page": 380,
        "search": "line of flight",
        "query": "What is a line of flight and how does it relate to deterritorialization?",
    },
]

# Extract actual quotations
results = []
doc = fitz.open(pdf_path)

for target in target_quotations:
    page_num = target["page"]
    search_text = target["search"]
    query = target["query"]

    # Get page text
    page = doc[page_num - 1]
    page_text = page.get_text()

    # Find the search text and extract surrounding passage
    normalized_page = normalize_whitespace(page_text)
    idx = normalized_page.lower().find(search_text.lower())

    if idx == -1:
        print(f"WARNING: Could not find '{search_text}' on page {page_num}")
        continue

    # Extract ~300 characters around the found text
    start = max(0, idx - 50)
    end = min(len(normalized_page), idx + 350)
    passage = normalized_page[start:end]

    # Clean up passage boundaries - try to start/end on sentence boundaries
    # Find first capital letter after start
    first_cap = 0
    for i, c in enumerate(passage):
        if c.isupper() and (i == 0 or passage[i - 1] in ".!?: "):
            first_cap = i
            break

    # Find last sentence end
    last_end = len(passage)
    for i in range(len(passage) - 1, 0, -1):
        if passage[i] in ".!?" and i < len(passage) - 1:
            last_end = i + 1
            break

    clean_passage = passage[first_cap:last_end].strip()

    # Check length
    word_count = len(clean_passage.split())
    if word_count < 20:
        print(f"WARNING: Passage on page {page_num} too short ({word_count} words)")
        continue

    results.append(
        {
            "query": query,
            "source": "1972 Anti-Oedipus Capitalism and Schizophr - Deleuze, Gilles.pdf",
            "pdf_page": page_num,
            "ground_truth": clean_passage,
        }
    )
    print(f"✓ Page {page_num}: Found passage ({word_count} words)")

doc.close()

print(f"\nExtracted {len(results)} quotations")

# Write to file
with open("data/eval/ao_eval_v2.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nWrote {len(results)} entries to data/eval/ao_eval_v2.json")

# Print first few entries for review
print("\n=== Sample entries ===")
for entry in results[:3]:
    print(f"\nPage {entry['pdf_page']}: {entry['query']}")
    print(f"  {entry['ground_truth'][:150]}...")
