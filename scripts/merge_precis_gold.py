"""Merge generated candidates + hand-curated additions + existing into a single
50-entry precis gold file. Augments any entries lacking
``expected_entities`` / ``expected_concept_spans`` via the same retrieval+
spaCy logic ``build_precis_gold.py`` uses, so all entries share schema.

Usage:
    docker run --rm --network host -v "$PWD:/app" -w /app \\
        -e CHROMA_HOST=http://127.0.0.1:8001 \\
        -e PYTHONPATH=/app:/app/services:/app/services/api \\
        gutenborg-api:latest python scripts/merge_precis_gold.py \\
            --collection gutenberg-deleuze-corpus \\
            --existing data/eval/deleuze_precis_evolution.json \\
            --candidates data/eval/deleuze_precis_evolution_candidates.json \\
            --handcurated data/eval/_precis_handcurated.json \\
            --target 50 \\
            --output data/eval/deleuze_precis_evolution_v2.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services"))
sys.path.insert(0, str(ROOT / "services" / "api"))

log = logging.getLogger("scripts.merge_precis_gold")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--existing", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--handcurated", default=None)
    ap.add_argument("--target", type=int, default=50)
    ap.add_argument("--output", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-spans", type=int, default=10)
    args = ap.parse_args()

    existing = json.loads(Path(args.existing).read_text())
    candidates = json.loads(Path(args.candidates).read_text())
    handcurated: list[dict] = []
    if args.handcurated and Path(args.handcurated).exists():
        handcurated = json.loads(Path(args.handcurated).read_text())

    log.info(
        "existing=%d  candidates=%d  handcurated=%d",
        len(existing), len(candidates), len(handcurated),
    )

    # De-dup hand-curated against existing+candidates by (concept, sources).
    seen: set[tuple] = set()
    def _key(e: dict) -> tuple:
        return (
            (e.get("concept_id") or e.get("concept") or "").lower(),
            tuple(sorted(e.get("works_required") or [])),
        )
    for e in existing + candidates:
        seen.add(_key(e))
    deduped_hc = [e for e in handcurated if _key(e) not in seen]
    log.info("hand-curated after de-dup: %d", len(deduped_hc))

    # Augment any entry lacking expected_entities/spans.
    needs_aug = [e for e in deduped_hc if not e.get("expected_entities")]
    if needs_aug:
        from scripts.build_precis_gold import augment as _aug
        log.info("augmenting %d hand-curated entries", len(needs_aug))
        augmented = _aug(
            needs_aug, args.collection,
            top_k_chunks=args.top_k, top_spans=args.top_spans,
        )
        # Replace in deduped_hc (preserves order)
        i = 0
        for j, e in enumerate(deduped_hc):
            if not e.get("expected_entities"):
                deduped_hc[j] = augmented[i]
                i += 1

    final = existing + candidates + deduped_hc
    if len(final) > args.target:
        final = final[: args.target]
    log.info("final entry count: %d", len(final))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final, indent=2))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
