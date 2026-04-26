"""Generate precis-evolution gold candidates by mining the corpus.

Strategy: a precis-evolution query needs a (concept, work_pair_or_triple)
tuple where each work has substantive treatment of the concept. We
already have those signals materialised in chunk metadata:

  * `canonical_ids` (P1 gazetteer tags) on every chunk → counts how
     many chunks per (concept, source) pair the corpus contains.
  * `level == 0` filter → only count leaves, not RAPTOR summaries.

Process:

  1. Read all level-0 chunks from the target collection.
  2. Build a (canonical_id → {source: chunk_count}) histogram.
  3. For each concept with ≥2 sources at ≥``min_chunks_per_source``,
     enumerate all year-ordered pairs (W_early, W_late) and the most
     informative triples.
  4. Drop pairs already covered by the existing precis JSON to avoid
     duplicates.
  5. Emit a candidate JSON via templated query strings; populate
     ``works_required``, ``concept``, ``label``, and the basic
     ``min_works_cited``/``min_quotes`` fields.
  6. Run the same retrieval + spaCy noun-chunk extraction
     ``mine_eval_golds.py`` uses to populate ``expected_entities``
     and ``expected_concept_spans``.
  7. Write the candidate file. The user manually trims/edits the
     output before promoting.

This is split from ``mine_eval_golds.py`` because that script targets
abstractive seeds and multi-entity queries, both of which are
hand-written. Precis evolution requires generation from corpus state,
not curated seeds.

Usage:
    docker run --rm --network host -v "$PWD:/app" -w /app \\
        -e CHROMA_HOST=http://127.0.0.1:8001 \\
        -e PYTHONPATH=/app:/app/services:/app/services/api \\
        gutenborg-api:latest python scripts/build_precis_gold.py \\
            --collection gutenberg-deleuze-corpus \\
            --target 50 \\
            --output data/eval/deleuze_precis_evolution_v2_candidates.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services"))
sys.path.insert(0, str(ROOT / "services" / "api"))

log = logging.getLogger("scripts.build_precis_gold")


# ── Concept presentation ────────────────────────────────────────────────

CONCEPT_LABEL: dict[str, str] = {
    "body_without_organs": "body without organs",
    "rhizome": "rhizome",
    "deterritorialization": "deterritorialization",
    "virtual_actual": "the virtual",
    "assemblage": "assemblage",
    "multiplicity": "multiplicity",
    "becoming": "becoming",
    "desiring_machines": "desiring-machines",
    "plane_of_immanence": "plane of immanence",
    "anti_oedipus": "anti-oedipus / desiring-production",  # work-as-concept proxy
    "a_thousand_plateaus": "thousand plateaus / nomad thought",
    "difference_and_repetition": "difference-in-itself",
    "baruch_spinoza": "Spinoza's affect",
    "felix_guattari": "machinic / Guattarian",
    "gilles_deleuze": "Deleuze's signature",
}

# Concepts that are people/works and don't make great evolution-query
# subjects on their own. We keep them in counts but skip query generation
# when the concept is itself a work or person.
SKIP_AS_CONCEPT: set[str] = {
    "anti_oedipus",
    "a_thousand_plateaus",
    "difference_and_repetition",
    "gilles_deleuze",
    "felix_guattari",
}


# ── Source utilities ───────────────────────────────────────────────────

def _short_title(source: str) -> str:
    s = re.sub(r"^\d{4}\s+", "", source)
    s = s.replace(".pdf", "")
    s = s.split(" - ")[0].strip()
    return s


def _year(source: str) -> int:
    m = re.match(r"^(\d{4})\s+", source)
    return int(m.group(1)) if m else 9999


def _label(concept: str, sources: list[str]) -> str:
    abbr = {
        "1962 Nietzsche and Philosophy - Deleuze, Gilles.pdf": "NP",
        "1964 Proust and Signs - Deleuze, Gilles.pdf": "PS",
        "1966 Bergsonism - Deleuze, Gilles.pdf": "B",
        "1968 Difference and repetition - Deleuze, Gilles.pdf": "DR",
        "1969 The Logic of Sense - Deleuze, Gilles.pdf": "LS",
        "1970 Spinoza Practical Philosophy - Deleuze, Gilles.pdf": "SPP",
        "1972 Anti-Oedipus Capitalism and Schizophr - Deleuze, Gilles.pdf": "AO",
        "1977 Dialogues - Deleuze, Gilles.pdf": "Dial",
        "1980 A Thousand Plateaus - Deleuze, Gilles.pdf": "ATP",
        "1981 Francis Bacon The Logic of Sensation - Deleuze, Gilles.pdf": "FB",
        "1986 Cinema 1 The Movement-Image - Deleuze, Gilles.pdf": "C1",
        "1988 Foucault - Deleuze, Gilles.pdf": "Fou",
        "1988 The Fold Leibniz and the Baroque - Deleuze, Gilles.pdf": "Fold",
        "1989 Cinema 2 The Time-Image - Deleuze, Gilles.pdf": "C2",
        "1991 What Is Philosophy - Deleuze, Gilles.pdf": "WIP",
    }
    arrow = "→".join(abbr.get(s, _short_title(s)[:5]) for s in sources)
    return f"{CONCEPT_LABEL.get(concept, concept)} {arrow}"


# ── Query templates ────────────────────────────────────────────────────

PAIR_TEMPLATES = [
    "Trace the evolution of '{concept}' from {t1} ({y1}) to {t2} ({y2}). Show how the concept's function or scope shifts between the two works. Cite specific passages from each.",
    "Compare Deleuze's treatment of '{concept}' in {t1} ({y1}) with its treatment in {t2} ({y2}). Identify points of continuity and divergence and quote passages that exhibit the shift.",
    "Show how '{concept}' is reworked between {t1} and {t2}. What problem is each work using the concept to solve? Cite passages from both.",
]

TRIPLE_TEMPLATES = [
    "Trace the development of '{concept}' across {t1} ({y1}), {t2} ({y2}), and {t3} ({y3}). Show how the concept's role changes through the sequence. Cite specific passages from each work.",
    "How does '{concept}' shift across {t1}, {t2}, and {t3}? Quote passages from each that capture the difference in how the concept is mobilised.",
]


def _make_pair_query(concept: str, sources: list[str], rotation: int) -> str:
    s1, s2 = sources[0], sources[1]
    tpl = PAIR_TEMPLATES[rotation % len(PAIR_TEMPLATES)]
    return tpl.format(
        concept=CONCEPT_LABEL.get(concept, concept),
        t1=_short_title(s1), y1=_year(s1),
        t2=_short_title(s2), y2=_year(s2),
    )


def _make_triple_query(concept: str, sources: list[str], rotation: int) -> str:
    s1, s2, s3 = sources[0], sources[1], sources[2]
    tpl = TRIPLE_TEMPLATES[rotation % len(TRIPLE_TEMPLATES)]
    return tpl.format(
        concept=CONCEPT_LABEL.get(concept, concept),
        t1=_short_title(s1), y1=_year(s1),
        t2=_short_title(s2), y2=_year(s2),
        t3=_short_title(s3), y3=_year(s3),
    )


# ── Mining ──────────────────────────────────────────────────────────────

def collect_counts(collection: str) -> tuple[Counter, dict[str, dict[str, int]]]:
    import chromadb
    from core.config import settings

    host = settings.chroma_host.replace("http://", "").replace("https://", "")
    parts = host.split(":")
    hostname, port = parts[0], int(parts[1]) if len(parts) > 1 else 8000
    c = chromadb.HttpClient(host=hostname, port=port)
    col = c.get_collection(collection)
    n = col.count()
    got = col.get(include=["metadatas"], limit=n)

    src_total: Counter = Counter()
    by_concept: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for m in got["metadatas"]:
        if (m.get("level") or 0) > 0:
            continue
        src = m.get("source") or ""
        if not src:
            continue
        src_total[src] += 1
        for cid in (m.get("canonical_ids") or "").split(","):
            if cid and cid not in SKIP_AS_CONCEPT:
                by_concept[cid][src] += 1
    return src_total, by_concept


def enumerate_candidates(
    by_concept: dict[str, dict[str, int]],
    *,
    min_chunks: int,
    max_per_concept_pairs: int = 4,
    max_per_concept_triples: int = 1,
) -> list[dict]:
    """Generate candidate seeds (no augmentation yet)."""
    candidates: list[dict] = []
    rotation = 0
    for concept, srcs in by_concept.items():
        elig = sorted(
            [(src, cnt) for src, cnt in srcs.items() if cnt >= min_chunks],
            key=lambda x: _year(x[0]),
        )
        if len(elig) < 2:
            continue

        # Pairs (year-ordered, all combinations)
        n_pairs = 0
        for i in range(len(elig)):
            for j in range(i + 1, len(elig)):
                if n_pairs >= max_per_concept_pairs:
                    break
                s1, s2 = elig[i][0], elig[j][0]
                cnt1, cnt2 = elig[i][1], elig[j][1]
                seed = {
                    "query": _make_pair_query(concept, [s1, s2], rotation),
                    "concept": CONCEPT_LABEL.get(concept, concept),
                    "concept_id": concept,
                    "works_required": [s1, s2],
                    "min_works_cited": 2,
                    "min_quotes": 2,
                    "label": _label(concept, [s1, s2]),
                    "_chunk_counts": {s1: cnt1, s2: cnt2},
                }
                candidates.append(seed)
                rotation += 1
                n_pairs += 1
            if n_pairs >= max_per_concept_pairs:
                break

        # One triple per concept (the three top-coverage works)
        if len(elig) >= 3 and max_per_concept_triples > 0:
            top3 = sorted(elig, key=lambda x: -x[1])[:3]
            top3 = sorted(top3, key=lambda x: _year(x[0]))
            ss = [s for s, _ in top3]
            seed = {
                "query": _make_triple_query(concept, ss, rotation),
                "concept": CONCEPT_LABEL.get(concept, concept),
                "concept_id": concept,
                "works_required": ss,
                "min_works_cited": 3,
                "min_quotes": 3,
                "label": _label(concept, ss),
                "_chunk_counts": {s: c for s, c in top3},
            }
            candidates.append(seed)
            rotation += 1
    return candidates


def drop_existing(
    candidates: list[dict], existing: list[dict]
) -> list[dict]:
    """Skip candidates whose (concept, works_required) overlaps an
    existing entry to avoid trivial duplicates."""
    sigs = {
        (
            (e.get("concept_id") or e.get("concept") or "").lower(),
            tuple(sorted(e.get("works_required") or [])),
        )
        for e in existing
    }
    out = []
    for c in candidates:
        key = (
            (c.get("concept_id") or c.get("concept") or "").lower(),
            tuple(sorted(c.get("works_required") or [])),
        )
        if key in sigs:
            continue
        out.append(c)
    return out


# ── Augmentation (entities + concept spans) ────────────────────────────

def augment(
    candidates: list[dict],
    collection: str,
    *,
    top_k_chunks: int,
    top_spans: int,
) -> list[dict]:
    from core.rag import retrieve
    from shared.nlp import get_nlp_full, is_available

    if not is_available():
        log.warning("spaCy unavailable — skipping span augmentation")
        nlp = None
    else:
        nlp = get_nlp_full()

    def _noun_chunks(text: str) -> list[str]:
        if nlp is None:
            return []
        doc = nlp(text)
        out: list[str] = []
        for ch in doc.noun_chunks:
            span = ch.text.strip().lower()
            span = re.sub(r"\s+", " ", span)
            if 2 <= len(span.split()) <= 4 and len(span) > 3:
                out.append(span)
        return out

    def _tfidf(chunks: list[list[str]]) -> list[str]:
        df: Counter = Counter()
        for spans in chunks:
            for s in set(spans):
                df[s] += 1
        keep = [(s, c) for s, c in df.items() if c >= 2]
        keep.sort(key=lambda t: (-t[1], t[0]))
        return [s for s, _ in keep[:top_spans]]

    out: list[dict] = []
    for i, seed in enumerate(candidates, 1):
        q = seed["query"]
        try:
            _, retrieved = retrieve(q, collection=collection)
        except Exception as e:
            log.warning("retrieve failed for %s: %s", seed["label"], e)
            retrieved = []
        top = retrieved[:top_k_chunks]

        cid_counter: Counter = Counter()
        span_docs: list[list[str]] = []
        l0_count = l1_count = 0
        for ch in top:
            meta = ch.get("metadata") or {}
            level = (meta.get("level") or 0)
            if level == 0:
                l0_count += 1
            else:
                l1_count += 1
            for cid in (meta.get("canonical_ids") or "").split(","):
                if cid:
                    cid_counter[cid] += 1
            span_docs.append(_noun_chunks(ch.get("text") or ""))

        seed["expected_entities"] = [
            cid for cid, cnt in cid_counter.most_common() if cnt >= 2
        ]
        seed["expected_concept_spans"] = _tfidf(span_docs)
        seed["min_level_mix"] = {"L0": max(l0_count, 1), "L1+": max(l1_count, 0)}
        out.append(seed)
        if i % 5 == 0:
            log.info("augmented %d/%d candidates", i, len(candidates))
    return out


# ── Main ────────────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--existing", default="data/eval/deleuze_precis_evolution.json")
    ap.add_argument("--target", type=int, default=40,
                    help="Number of new candidates to keep after augment")
    ap.add_argument("--min-chunks", type=int, default=10,
                    help="Min chunks per (concept, source) to consider")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-spans", type=int, default=10)
    ap.add_argument("--output", default="data/eval/deleuze_precis_evolution_candidates.json")
    args = ap.parse_args()

    src_total, by_concept = collect_counts(args.collection)
    log.info("level-0 leaves: %d sources, %d concepts present",
             len(src_total), len(by_concept))

    existing = json.loads(Path(args.existing).read_text())
    log.info("existing precis entries: %d", len(existing))

    candidates = enumerate_candidates(by_concept, min_chunks=args.min_chunks)
    log.info("raw candidates: %d", len(candidates))

    candidates = drop_existing(candidates, existing)
    log.info("after de-dup vs existing: %d", len(candidates))

    candidates.sort(
        key=lambda c: -min((c.get("_chunk_counts") or {1: 0}).values()),
    )
    candidates = candidates[: args.target]
    log.info("top-%d by min-chunks signal", args.target)

    augmented = augment(
        candidates, args.collection,
        top_k_chunks=args.top_k, top_spans=args.top_spans,
    )

    # Strip the internal _chunk_counts before writing.
    final: list[dict] = []
    for s in augmented:
        ent = {k: v for k, v in s.items() if not k.startswith("_")}
        final.append(ent)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final, indent=2))
    log.info("wrote %d candidates → %s", len(final), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
