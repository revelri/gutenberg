"""Structured per-query telemetry (P12).

Writes one JSON line per retrieval call to ``settings.telemetry_log_path``.
Captures:

  * normalized query (optionally SHA-256 hashed for privacy)
  * feature-flag matrix active for this request
  * retriever contributions (counts per channel)
  * rerank scores, verification outcome, latency, token cost (when available)

Consumed by ``scripts/weekly_report.py`` and ``scripts/audit_contextcite.py``.
Failures are swallowed — telemetry never breaks the response path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any

log = logging.getLogger("gutenberg.telemetry")

_LOCK = threading.Lock()

_FLAG_KEYS = (
    "feature_contextual_chunking",
    "feature_entity_gazetteer",
    "feature_modal_chunks",
    "feature_colbert_retrieval",
    "feature_raptor",
    "feature_graph_boost",
    "feature_crag",
    "feature_vlm_answer",
    "feature_rapidfuzz_verify",
    "feature_anchor_validation",
)


def _get_settings():
    try:
        from core.config import settings
        return settings
    except Exception:
        try:
            from services.api.core.config import settings  # type: ignore
            return settings
        except Exception:
            return None


def _flag_snapshot(settings) -> dict[str, bool]:
    return {k: bool(getattr(settings, k, False)) for k in _FLAG_KEYS}


def record(event: dict[str, Any]) -> None:
    settings = _get_settings()
    if settings is None or not getattr(settings, "telemetry_enabled", True):
        return
    try:
        path = settings.telemetry_log_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        query = event.get("query")
        if query and getattr(settings, "telemetry_hash_queries", False):
            event = dict(event)
            event["query_hash"] = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
            event.pop("query", None)

        event.setdefault("ts", time.time())
        event.setdefault("flags", _flag_snapshot(settings))

        with _LOCK:
            with open(path, "a") as f:
                f.write(json.dumps(event) + "\n")
    except Exception as e:
        log.debug(f"telemetry write failed: {e}")
