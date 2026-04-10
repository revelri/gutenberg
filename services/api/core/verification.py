"""Post-generation quote verification for citation accuracy.

Extracts quoted text from LLM responses and verifies each quote against
the retrieved context chunks using exact and fuzzy matching.
"""

import logging
import os
import re
from difflib import SequenceMatcher

from shared.text_normalize import normalize_for_matching

log = logging.getLogger("gutenberg.verification")

# Minimum similarity ratio for fuzzy match to count as "approximate"
FUZZY_THRESHOLD = 0.85

# Minimum quote length to bother verifying (very short quotes are often
# common phrases that match everywhere)
MIN_QUOTE_LENGTH = 20


def extract_quotes(text: str) -> list[str]:
    """Extract quoted text from an LLM response.

    Finds text in:
    - Double quotation marks: "..." or \u201c...\u201d
    - Markdown blockquotes: > ...
    """
    quotes = []

    # Standard double quotes
    for match in re.finditer(r'"([^"]{10,})"', text):
        quotes.append(match.group(1))

    # Curly/smart quotes
    for match in re.finditer(r'\u201c([^\u201d]{10,})\u201d', text):
        quotes.append(match.group(1))

    # Markdown blockquotes (lines starting with >)
    blockquote_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(">"):
            blockquote_lines.append(stripped.lstrip("> ").strip())
        else:
            if blockquote_lines:
                bq_text = " ".join(blockquote_lines)
                if len(bq_text) >= 10:
                    quotes.append(bq_text)
                blockquote_lines = []
    # Flush remaining blockquote
    if blockquote_lines:
        bq_text = " ".join(blockquote_lines)
        if len(bq_text) >= 10:
            quotes.append(bq_text)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in quotes:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    return unique


def verify_quotes(quotes: list[str], chunks: list[dict]) -> list[dict]:
    """Verify each quote against retrieved context chunks.

    Returns list of verification results, one per quote:
    {
        "quote": str (truncated for display),
        "status": "verified" | "approximate" | "unverified",
        "source": str | None,
        "page": int | None,
        "similarity": float,
    }
    """
    if not quotes or not chunks:
        return []

    results = []
    for quote in quotes:
        result = _verify_single_quote(quote, chunks)
        results.append(result)

    return results


def _verify_single_quote(quote: str, chunks: list[dict]) -> dict:
    """Verify a single quote against all chunks."""
    best_match = {
        "quote": quote[:80] + "..." if len(quote) > 80 else quote,
        "_full_quote": quote,
        "status": "unverified",
        "source": None,
        "page": None,
        "similarity": 0.0,
    }

    # Skip very short quotes — they're likely common phrases
    if len(quote) < MIN_QUOTE_LENGTH:
        best_match["status"] = "too_short"
        return best_match

    quote_normalized = _normalize(quote)

    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        chunk_normalized = _normalize(chunk_text)
        meta = chunk.get("metadata", {})

        # Try exact substring match first (fast path)
        if quote_normalized in chunk_normalized:
            return {
                "quote": best_match["quote"],
                "_full_quote": quote,
                "status": "verified",
                "source": meta.get("source"),
                "page": meta.get("page_start"),
                "similarity": 1.0,
            }

        # Fuzzy match (slower — only for longer quotes)
        if len(quote) >= MIN_QUOTE_LENGTH:
            ratio = SequenceMatcher(None, quote_normalized, chunk_normalized).ratio()
            # For substring-like matching, also check if the quote is a
            # near-match of any window of similar length in the chunk
            if ratio < FUZZY_THRESHOLD and len(quote_normalized) < len(chunk_normalized):
                ratio = _best_window_ratio(quote_normalized, chunk_normalized)

            if ratio > best_match["similarity"]:
                best_match["similarity"] = ratio
                if ratio >= FUZZY_THRESHOLD:
                    best_match["status"] = "approximate"
                    best_match["source"] = meta.get("source")
                    best_match["page"] = meta.get("page_start")

    # Third tier: lemma-normalized matching (catches "produces" vs "produced")
    if best_match["status"] == "unverified":
        try:
            from shared.nlp import get_nlp, is_available
            if is_available():
                nlp = get_nlp()
                quote_lemmas = " ".join(t.lemma_ for t in nlp(_normalize(quote)) if t.is_alpha)
                for chunk in chunks:
                    chunk_lemmas = " ".join(t.lemma_ for t in nlp(_normalize(chunk.get("text", ""))) if t.is_alpha)
                    if len(quote_lemmas) > 20 and quote_lemmas[:40] in chunk_lemmas:
                        meta = chunk.get("metadata", {})
                        best_match["status"] = "approximate"
                        best_match["source"] = meta.get("source")
                        best_match["page"] = meta.get("page_start")
                        best_match["similarity"] = 0.85
                        break
        except ImportError:
            pass

    return best_match


def _best_window_ratio(quote: str, text: str) -> float:
    """Find the best fuzzy match ratio for a quote within sliding windows of text."""
    quote_len = len(quote)
    if quote_len >= len(text):
        return SequenceMatcher(None, quote, text).ratio()

    best = 0.0
    # Step through text in chunks (not char by char — too slow for large text)
    step = max(1, quote_len // 4)
    for start in range(0, len(text) - quote_len + 1, step):
        window = text[start:start + quote_len]
        ratio = SequenceMatcher(None, quote, window).ratio()
        if ratio > best:
            best = ratio
            if best >= FUZZY_THRESHOLD:
                return best  # early exit
    return best


def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip markdown, collapse whitespace."""
    return normalize_for_matching(text, strip_markdown=True)


def verify_against_source(
    results: list[dict],
    pdf_dirs: list[str] | None = None,
) -> list[dict]:
    """Cross-check verified quotes against the raw PDF page text.

    For each quote marked 'verified' or 'approximate', attempt to find
    the source PDF and check the quote against the actual page text.
    Adds 'source_verified' field: True if found in PDF, False if not, None if PDF unavailable.
    """
    if not pdf_dirs:
        pdf_dirs = []
        for d in ["data/processed", "data/inbox", "data/surya_corpus"]:
            if os.path.isdir(d):
                pdf_dirs.append(d)

    for result in results:
        result["source_verified"] = None
        if result["status"] not in ("verified", "approximate"):
            continue
        source = result.get("source")
        page = result.get("page")
        if not source or not page:
            continue

        # Find the PDF file
        pdf_path = _find_pdf(source, pdf_dirs)
        if not pdf_path:
            continue

        # Check the quote against the actual page text
        try:
            import fitz
            doc = fitz.open(pdf_path)
            # Page numbers are 1-indexed in our metadata, 0-indexed in PyMuPDF
            page_idx = page - 1
            if 0 <= page_idx < len(doc):
                page_text = doc[page_idx].get_text()
                page_normalized = _normalize(page_text)
                # Extract the original quote from the truncated display
                quote_text = result.get("_full_quote", result["quote"].rstrip("..."))
                quote_normalized = _normalize(quote_text)
                # Check: does the quote appear on this page (or adjacent pages)?
                found = quote_normalized[:60] in page_normalized
                if not found and page_idx + 1 < len(doc):
                    adj_text = doc[page_idx + 1].get_text()
                    found = quote_normalized[:60] in _normalize(adj_text)
                if not found and page_idx > 0:
                    adj_text = doc[page_idx - 1].get_text()
                    found = quote_normalized[:60] in _normalize(adj_text)
                result["source_verified"] = found
            doc.close()
        except Exception as e:
            log.warning(f"Source verification failed for '{source}': {e}")

    return results


def _find_pdf(source_name: str, pdf_dirs: list[str]) -> str | None:
    """Find a PDF file matching the source name in the given directories."""
    import os
    for d in pdf_dirs:
        for f in os.listdir(d):
            if not f.lower().endswith(".pdf"):
                continue
            # Match by source name prefix (source names may have year prefix like "1980 A Thousand Plateaus")
            f_lower = f.lower().replace("_", " ").replace("-", " ")
            source_lower = source_name.lower()
            # Try matching the title part (without year)
            title = re.sub(r"^\d{4}\s+", "", source_lower).strip()
            if title and title in f_lower:
                return os.path.join(d, f)
            if source_lower in f_lower:
                return os.path.join(d, f)
    return None


def format_verification_footer(results: list[dict]) -> str:
    """Format verification results as a markdown footer.

    Returns empty string if no quotes were found to verify.
    """
    if not results:
        return ""

    # Filter out too_short
    meaningful = [r for r in results if r["status"] != "too_short"]
    if not meaningful:
        return ""

    verified = sum(1 for r in meaningful if r["status"] == "verified")
    approximate = sum(1 for r in meaningful if r["status"] == "approximate")
    unverified = sum(1 for r in meaningful if r["status"] == "unverified")
    total = len(meaningful)

    lines = [f"\n---\n**Citation verification:** {verified}/{total} verified"]
    if approximate:
        lines[0] += f", {approximate} approximate"
    if unverified:
        lines[0] += f", {unverified} unverified"

    # Count source-verified results
    source_checked = [r for r in meaningful if r.get("source_verified") is not None]
    if source_checked:
        src_ok = sum(1 for r in source_checked if r["source_verified"])
        lines[0] += f" | {src_ok}/{len(source_checked)} source-confirmed"

    # Detail lines for non-verified quotes
    for r in meaningful:
        if r["status"] == "verified" and r.get("source_verified") is not False:
            continue
        if r["status"] == "verified" and r.get("source_verified") is False:
            tag = "\u26a0"
            lines.append(f"- {tag} \"{r['quote']}\" — verified in chunk but NOT found on cited page")
        elif r["status"] != "verified":
            tag = "\u2248" if r["status"] == "approximate" else "\u2717"
            lines.append(f"- {tag} \"{r['quote']}\" — {r['status']} (similarity: {r['similarity']:.0%})")

    return "\n".join(lines)
