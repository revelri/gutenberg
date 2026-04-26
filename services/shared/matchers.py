"""Citation anchor patterns.

Extracts structured page/section anchors from both LLM output and scholarly
chunk metadata so citation tags can be validated against the chunk they were
actually drawn from. Supports common conventions:

  * Plain page: ``p. 47``, ``pp. 47-49``, ``p 47``
  * Section: ``§ 12``, ``§12``
  * Chapter: ``ch. 3``, ``chapter 3``
  * Kant A/B pagination: ``A51/B75`` or ``B75``
  * Heidegger Sein und Zeit: ``SZ 42``, ``SZ:42``
  * Plateau numbers: ``plateau 3``, ``ATP 3``

Returns ``list[dict]`` with ``{"kind": str, "value": str}``. Kinds are
stable identifiers that callers can match against chunk metadata.
"""

from __future__ import annotations

import re

_ANCHOR_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("page", re.compile(r"\bpp?\.?\s*(\d+(?:\s*[-–—]\s*\d+)?)\b", re.IGNORECASE)),
    ("section", re.compile(r"(?:§|§|section)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)),
    ("chapter", re.compile(r"\b(?:ch\.?|chapter)\s*(\d+)\b", re.IGNORECASE)),
    ("kant_ab", re.compile(r"\b([AB]\s*\d{1,4}(?:\s*/\s*[AB]\s*\d{1,4})?)\b")),
    ("heidegger_sz", re.compile(r"\bSZ[:\s]+(\d{1,4})\b", re.IGNORECASE)),
    ("plateau", re.compile(r"\b(?:plateau|ATP)\s+(\d+)\b", re.IGNORECASE)),
]


def extract_anchors(text: str) -> list[dict]:
    """Return every anchor found in ``text``."""
    if not text:
        return []
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for kind, pattern in _ANCHOR_PATTERNS:
        for m in pattern.finditer(text):
            value = re.sub(r"\s+", "", m.group(1))
            key = (kind, value.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append({"kind": kind, "value": value})
    return out


def page_in_range(
    anchor_value: str, page_start: int | None, page_end: int | None
) -> bool:
    """True iff the numeric page anchor overlaps the chunk's page span."""
    if not page_start:
        return False
    end = page_end or page_start
    # anchor_value may be "47" or "47-49"
    nums = re.findall(r"\d+", anchor_value)
    if not nums:
        return False
    a_start = int(nums[0])
    a_end = int(nums[-1]) if len(nums) > 1 else a_start
    return not (a_end < page_start or a_start > end)
