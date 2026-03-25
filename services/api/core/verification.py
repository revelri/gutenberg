"""Post-generation quote verification for citation accuracy.

Extracts quoted text from LLM responses and verifies each quote against
the retrieved context chunks using exact and fuzzy matching.
"""

import logging
import re
from difflib import SequenceMatcher

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
    text = text.lower()
    # Strip markdown bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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

    # Detail lines for non-verified quotes
    for r in meaningful:
        if r["status"] == "verified":
            continue
        tag = "\u2248" if r["status"] == "approximate" else "\u2717"
        lines.append(f"- {tag} \"{r['quote']}\" — {r['status']} (similarity: {r['similarity']:.0%})")

    return "\n".join(lines)
