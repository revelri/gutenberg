"""Entity gazetteer + canonical alias resolution.

Loads hand-curated YAML gazetteers (one file per author/corpus) and builds:

  * A spaCy ``EntityRuler`` patterns list for deterministic entity detection.
  * A lowercase alias → canonical_id map for resolving translation variants
    and surface-form differences at both index and query time.

Used at:
  * Ingest — tag each chunk's metadata with ``entities[]`` and ``canonical_ids[]``.
  * Query — expand BM25 terms and produce a boost signal during reranking.

Failures are non-fatal: if PyYAML is missing, the directory is empty, or a
file is malformed, ``get_aliases()`` returns an empty dict and the system
behaves as if the flag were off.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("gutenberg.gazetteer")


@lru_cache(maxsize=1)
def _load_raw() -> list[dict]:
    try:
        from core.config import settings  # API path
    except Exception:
        try:
            from services.api.core.config import settings  # type: ignore
        except Exception:
            class _S:  # last-resort default
                gazetteer_dir = "/app/data/gazetteer"
            settings = _S()  # type: ignore

    candidates = [
        settings.gazetteer_dir,
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "gazetteer"),
        "data/gazetteer",
    ]
    gdir = next((Path(c) for c in candidates if Path(c).is_dir()), None)
    if gdir is None:
        return []

    try:
        import yaml
    except ImportError:
        log.warning("PyYAML not installed — gazetteer disabled")
        return []

    entries: list[dict] = []
    for yml in sorted(gdir.glob("*.yml")):
        try:
            data = yaml.safe_load(yml.read_text()) or {}
        except Exception as e:
            log.warning(f"Failed to parse {yml}: {e}")
            continue
        for section in data.values():
            if not isinstance(section, list):
                continue
            for entry in section:
                if isinstance(entry, dict) and entry.get("canonical_id"):
                    entries.append(entry)
    log.info(f"Gazetteer loaded: {len(entries)} entities from {gdir}")
    return entries


@lru_cache(maxsize=1)
def get_aliases() -> dict[str, str]:
    """Map lowercase alias → canonical_id."""
    out: dict[str, str] = {}
    for entry in _load_raw():
        cid = entry["canonical_id"]
        for alias in entry.get("aliases", []) or []:
            out[alias.lower().strip()] = cid
        out[entry.get("label", "").lower().strip()] = cid
        out.pop("", None)
    return out


@lru_cache(maxsize=1)
def get_canonical_labels() -> dict[str, str]:
    """Map canonical_id → preferred display label."""
    return {
        e["canonical_id"]: e.get("label", e["canonical_id"])
        for e in _load_raw()
    }


@lru_cache(maxsize=1)
def get_patterns() -> list[dict]:
    """spaCy EntityRuler pattern list.

    Uses lowercase LOWER match so case-folded queries still resolve.
    Adds a ``CANONICAL`` entity type for downstream filtering.
    """
    patterns: list[dict] = []
    for entry in _load_raw():
        cid = entry["canonical_id"]
        label = f"CANONICAL_{entry.get('type', 'TERM')}"
        for alias in entry.get("aliases", []) or []:
            tokens = [{"LOWER": t.lower()} for t in alias.split() if t]
            if tokens:
                patterns.append({"label": label, "pattern": tokens, "id": cid})
    return patterns


def resolve(text: str) -> list[str]:
    """Find all canonical_ids whose aliases appear as substrings in ``text``."""
    if not text:
        return []
    t = text.lower()
    hits: list[str] = []
    seen: set[str] = set()
    # Sort aliases by length descending so multi-word aliases win.
    for alias, cid in sorted(get_aliases().items(), key=lambda kv: -len(kv[0])):
        if len(alias) < 3:
            continue
        if alias in t and cid not in seen:
            hits.append(cid)
            seen.add(cid)
    return hits
