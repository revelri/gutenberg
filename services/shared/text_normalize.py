"""Canonical text normalization for ingestion and evaluation.

Two main entry points:
  clean_for_ingestion(text)       — full pipeline cleaning for extracted text
  normalize_for_comparison(text)  — lightweight normalization for eval matching
"""

import re
import unicodedata

# ── Unicode normalization ────────────────────────────────────────────

# Smart quotes → ASCII
_QUOTE_MAP = str.maketrans({
    "\u2018": "'",   # left single
    "\u2019": "'",   # right single / apostrophe
    "\u201a": "'",   # single low-9
    "\u201b": "'",   # single high-reversed-9
    "\u201c": '"',   # left double
    "\u201d": '"',   # right double
    "\u201e": '"',   # double low-9
    "\u201f": '"',   # double high-reversed-9
    "\u2039": "'",   # single left-pointing angle
    "\u203a": "'",   # single right-pointing angle
    "\u00ab": '"',   # left-pointing double angle (guillemet)
    "\u00bb": '"',   # right-pointing double angle
})

# Zero-width and invisible characters
_INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad]")


def normalize_unicode(text: str) -> str:
    """NFKC normalization + smart quote/dash/ellipsis canonicalization."""
    # NFKC decomposes ligatures (fi→fi, fl→fl, ff→ff) and normalizes fullwidth chars
    text = unicodedata.normalize("NFKC", text)
    # Smart quotes → ASCII
    text = text.translate(_QUOTE_MAP)
    # Em-dash / en-dash → --
    text = text.replace("\u2014", "--")   # em-dash
    text = text.replace("\u2013", "--")   # en-dash
    # Ellipsis character → three dots
    text = text.replace("\u2026", "...")
    # Non-breaking space → regular space
    text = text.replace("\u00a0", " ")
    # Strip zero-width / invisible chars
    text = _INVISIBLE_RE.sub("", text)
    return text


# ── OCR deconfusion ─────────────────────────────────────────────────

# Common OCR confusion patterns in serif fonts (e.g., scanned philosophy texts).
# These substitutions are applied after Unicode normalization.
_OCR_REPLACEMENTS = [
    # rn → m confusion (very common in serif fonts: "rnachine" → "machine")
    (re.compile(r"\brn(?=achine|ade|an|atter|eans|ent|ethod|ind|odern|ode|oment|oral|ore|ost|ove|uch|ulti|ust|yth)"), "m"),
    (re.compile(r"\brn(?=ach|ake|ark|atch|aterial|ax|edic|eet|ember|emor|ental|erge|essage|etal|icro|iddl|ilieu|inor|irror|istak|ix|olecul|onar|onstr|ontag|oral|ov|urdoch|usic)"), "m"),
    # cl → d confusion
    (re.compile(r"\bcl(?=eath|esire|esirous|elirium|espotic|estroy|etermin|evelop|iagram|ifferenc|irect|iscours|istinct|ivision|omin|omain|ouble)"), "d"),
    # li → h confusion (less common, only apply to very specific patterns)
    (re.compile(r"(?<=t)li(?=e\b|at\b|is\b|ey\b|em\b|eir\b|ere\b|en\b|ings?\b|ink|ose|ough|ousand|rough|us)"), "h"),
]

# Isolated tilde artifacts from OCR (e.g., "mouth ~ machine" → "mouth a machine")
_TILDE_ARTIFACT_RE = re.compile(r"(?<=\s)~(?=\s)")

# Broken words across line boundaries (e.g., "inter rupts" → "interrupts")
_BROKEN_WORD_PAIRS = [
    ("inter rupts", "interrupts"),
    ("con nections", "connections"),
    ("pro duction", "production"),
    ("ex periment", "experiment"),
    ("de siring", "desiring"),
]


def repair_ocr_artifacts(text: str) -> str:
    """Fix common OCR artifacts in extracted text.

    Targets serif-font confusions: rn→m, cl→d, tilde artifacts,
    and broken words from line-boundary extraction.
    """
    for pattern, replacement in _OCR_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    # Remove tilde artifacts (standalone ~ between spaces)
    text = _TILDE_ARTIFACT_RE.sub("", text)

    # Fix known broken word pairs
    for broken, fixed in _BROKEN_WORD_PAIRS:
        text = text.replace(broken, fixed)

    return text


# ── Hyphenation repair ───────────────────────────────────────────────

# Match word-ending hyphen followed by newline + optional whitespace + lowercase letter
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w+)-\s*\n\s*([a-z])")

# Common word-ending suffixes that suggest the first part is a complete word
# (and the hyphen is part of a compound, not a syllable break).
_WORD_SUFFIXES = re.compile(
    r"(?:ing|tion|sion|ment|ness|ous|ive|ful|less|able|ible|ated|ized|"
    r"ling|ism|ist|ent|ant|ary|ory|ure|ite|ous|al|er|ed|ly|or|us)$",
    re.IGNORECASE,
)


def repair_hyphenation(text: str) -> str:
    """Rejoin line-break hyphenation, preserving legitimate compounds.

    Two-pass approach:
    1. Collapse the newline in all word-hyphen-newline-word patterns
    2. Remove the hyphen only when the first part looks like a broken syllable
       (not a complete English word), keeping it for compounds like "desiring-machine"
    """
    def _repair(m: re.Match) -> str:
        before, after = m.group(1), m.group(2)
        # If the first part ends with a common word suffix, it's likely a compound
        # like "desiring-machine" — keep the hyphen, just remove the linebreak
        if _WORD_SUFFIXES.search(before) and len(before) >= 4:
            return f"{before}-{after}"
        # Otherwise it's likely a syllable break — join without hyphen
        return f"{before}{after}"

    return _HYPHEN_LINEBREAK_RE.sub(_repair, text)


# ── Whitespace normalization ─────────────────────────────────────────

def normalize_whitespace(text: str) -> str:
    """Normalize structural whitespace from PDF extraction.

    - Form feeds / vertical tabs → newline
    - Collapse 3+ consecutive newlines → 2
    - Collapse multiple spaces within lines → single space
    - Strip trailing whitespace per line
    """
    # Form feeds and vertical tabs to newlines
    text = text.replace("\f", "\n").replace("\v", "\n")
    # Collapse 3+ newlines to 2 (preserve paragraph boundaries)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace per line and collapse inline multi-spaces
    lines = []
    for line in text.split("\n"):
        line = re.sub(r" {2,}", " ", line).rstrip()
        lines.append(line)
    return "\n".join(lines)


# ── Repeated punctuation collapse ────────────────────────────────────

def collapse_repeated_punctuation(text: str) -> str:
    """Collapse repeated dots/dashes/underscores/spaces from TOC formatting."""
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"-{3,}", "---", text)
    text = re.sub(r"_{3,}", "___", text)
    text = re.sub(r" {3,}", "  ", text)
    return text


# ── Header/footer stripping ─────────────────────────────────────────

import logging

_log = logging.getLogger("gutenberg.text_normalize")


def strip_headers_footers(pages: list[dict], threshold: float = 0.5) -> list[dict]:
    """Remove repeated page headers and footers.

    Detects strings appearing as the first or last non-empty line on
    more than `threshold` fraction of pages and strips them.

    Args:
        pages: list of {"page": int, "text": str} dicts
        threshold: fraction of pages a line must appear on to be stripped (default 50%)

    Returns:
        New list of page dicts with headers/footers removed.
    """
    if len(pages) < 4:
        return pages

    # Count first-line and last-line occurrences
    first_lines: dict[str, int] = {}
    last_lines: dict[str, int] = {}
    for p in pages:
        lines = [l.strip() for l in p["text"].split("\n") if l.strip()]
        if not lines:
            continue
        norm_first = lines[0].lower().strip()
        norm_last = lines[-1].lower().strip()
        if len(norm_first) < 80:  # only short lines are likely headers
            first_lines[norm_first] = first_lines.get(norm_first, 0) + 1
        if len(norm_last) < 80:
            last_lines[norm_last] = last_lines.get(norm_last, 0) + 1

    min_count = int(len(pages) * threshold)
    strip_firsts = {s for s, c in first_lines.items() if c >= min_count}
    strip_lasts = {s for s, c in last_lines.items() if c >= min_count}

    if strip_firsts:
        _log.debug(f"Stripping repeated headers: {strip_firsts}")
    if strip_lasts:
        _log.debug(f"Stripping repeated footers: {strip_lasts}")

    if not strip_firsts and not strip_lasts:
        return pages

    result = []
    for p in pages:
        lines = p["text"].split("\n")
        # Strip matching first lines
        while lines:
            stripped = lines[0].strip().lower()
            if stripped and stripped in strip_firsts:
                lines.pop(0)
            else:
                break
        # Strip matching last lines
        while lines:
            stripped = lines[-1].strip().lower()
            if stripped and stripped in strip_lasts:
                lines.pop()
            else:
                break
        result.append({"page": p["page"], "text": "\n".join(lines)})

    return result


# ── Markdown formatting removal ──────────────────────────────────────

def strip_markdown_formatting(text: str) -> str:
    """Remove markdown bold/italic markers for clean text comparison."""
    # Bold+italic: ***text*** or ___text___
    text = re.sub(r"\*{3}([^*]+)\*{3}", r"\1", text)
    text = re.sub(r"_{3}([^_]+)_{3}", r"\1", text)
    # Bold: **text** or __text__
    text = re.sub(r"\*{2}([^*]+)\*{2}", r"\1", text)
    text = re.sub(r"_{2}([^_]+)_{2}", r"\1", text)
    # Italic: *text* or _text_
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    # Skip single underscore — too many false positives in identifiers
    return text


# ── Composite entry points ───────────────────────────────────────────

def clean_for_ingestion(text: str) -> str:
    """Full cleaning pipeline for extracted document text.

    Applied post-extraction, before chunking. Order matters:
    1. Unicode normalization (ligatures, smart quotes, dashes)
    2. OCR deconfusion (rn→m, tilde artifacts, broken words)
    3. Hyphenation repair (must happen before whitespace collapse)
    4. Whitespace normalization
    5. Repeated punctuation collapse
    """
    if not text:
        return text
    text = normalize_unicode(text)
    text = repair_ocr_artifacts(text)
    text = repair_hyphenation(text)
    text = normalize_whitespace(text)
    text = collapse_repeated_punctuation(text)
    return text


def normalize_for_comparison(text: str) -> str:
    """Lightweight normalization for eval/retrieval matching.

    Canonical form: Unicode-normalized, OCR-deconfused, lowercased, single-space-separated.
    Use this instead of ad-hoc normalize() functions in eval scripts.
    """
    if not text:
        return text
    text = normalize_unicode(text)
    text = repair_ocr_artifacts(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_matching(text: str, strip_markdown: bool = False) -> str:
    """Normalize text for quote matching and passage scoring.

    Lowercase, collapse whitespace, normalize dashes and quotes.
    Designed to maximize matching across OCR variants, different
    editions/translations, and typographic conventions.
    """
    if not text:
        return text
    if strip_markdown:
        text = strip_markdown_formatting(text)
    text = text.lower()
    # Normalize all dash variants to single hyphen (em-dash, en-dash, double-dash)
    text = re.sub(r"[\u2013\u2014\u2015\u2212]", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    # Strip quotation marks entirely — they vary between editions and OCR
    text = re.sub(r'["\'\u201c\u201d\u2018\u2019\u00ab\u00bb]', "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
