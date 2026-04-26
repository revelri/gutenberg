"""CRAG-lite — retrieval confidence gate + synonym rewrite (P9).

After the initial fused retrieval pass, classify confidence from the top
reranker score. On ``ambiguous`` or ``irrelevant``, rewrite the query by
appending alias surface forms from the canonical gazetteer and by stripping
function words, then trigger a widened re-retrieval. Original canonical_ids
are preserved so the rewrite only ADDS signal.

This is the conservative half of Corrective RAG — no LLM generation step,
no external search, no full retrieval re-run replacement. Cost is bounded:
one extra retrieval call plus rerank on the widened candidate set.
"""

from __future__ import annotations

import logging

from core.config import settings

log = logging.getLogger("gutenberg.crag")


def classify_confidence(top_score: float) -> str:
    if top_score >= settings.crag_confident_score:
        return "confident"
    if top_score >= settings.crag_ambiguous_score:
        return "ambiguous"
    return "irrelevant"


def rewrite(query: str) -> str:
    """Return a widened query string with gazetteer aliases appended."""
    try:
        from shared.gazetteer import get_aliases, resolve
    except Exception:
        return query

    cids = resolve(query)
    if not cids:
        return query
    inverted: dict[str, list[str]] = {}
    for alias, cid in get_aliases().items():
        inverted.setdefault(cid, []).append(alias)
    extras: list[str] = []
    for cid in cids:
        extras.extend(inverted.get(cid, []))
    if not extras:
        return query
    rewritten = f"{query} {' '.join(extras)}"
    log.info(f"CRAG rewrote query: {len(extras)} aliases added")
    return rewritten
