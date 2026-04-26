"""One-shot gold-dataset authoring via retrieval-log mining.

Generates two datasets that the new artifact-attribution evals need:

  * ``data/eval/deleuze_abstractive.json`` — thematic/conceptual queries with
    ``expected_entities`` (most-frequent canonical_ids in top-k) and
    ``expected_concept_spans`` (tf-idf-ranked noun chunks from retrieved text).
    Drives the RAPTOR ablation.

  * ``data/eval/multi_entity_queries.json`` — queries that explicitly name
    ≥2 canonical entities. ``expected_entities`` is derived from the query
    itself via ``gazetteer.resolve``. ``expected_edges`` is derived from the
    co-occurrence graph (``graph.expand`` → keep edges where both endpoints
    are in the query). Drives the graph ablation.

No LLM drafting — the grader never grades itself. Expect a human trim pass
afterward to drop obviously noisy concept spans.

Usage:
    docker compose exec api python scripts/mine_eval_golds.py \
        --collection gutenberg-deleuze-corpus
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services"))
sys.path.insert(0, str(ROOT / "services" / "api"))

log = logging.getLogger("scripts.mine_eval_golds")


# Seed queries. Abstractive = thematic; multi-entity = names ≥2 canonicals.
ABSTRACTIVE_SEEDS: list[dict] = [
    {"query": "How does Deleuze characterize the body without organs across his work?",
     "label": "bwo_across_oeuvre"},
    {"query": "What is Deleuze's account of difference in itself?",
     "label": "difference_in_itself"},
    {"query": "How does Deleuze understand the relation between the virtual and the actual?",
     "label": "virtual_actual"},
    {"query": "What does Deleuze mean by a line of flight?",
     "label": "line_of_flight"},
    {"query": "How does Deleuze conceive of immanence?",
     "label": "immanence"},
    {"query": "How does Deleuze read Spinoza's ethics of affect?",
     "label": "spinoza_affect"},
    {"query": "What is the role of intensity in Deleuze's ontology?",
     "label": "intensity"},
    {"query": "How does Deleuze describe the rhizome versus the tree?",
     "label": "rhizome_vs_tree"},
    {"query": "How does Deleuze treat the eternal return in Nietzsche?",
     "label": "eternal_return"},
    {"query": "What is the function of the refrain in A Thousand Plateaus?",
     "label": "refrain"},
    {"query": "How does Deleuze distinguish molar and molecular?",
     "label": "molar_molecular"},
    {"query": "What does Deleuze mean by the plane of consistency?",
     "label": "plane_of_consistency"},
    {"query": "How does Deleuze characterize the crystal-image in cinema?",
     "label": "crystal_image"},
    {"query": "What is Deleuze's concept of assemblage?",
     "label": "assemblage"},
    {"query": "How does Deleuze treat the relation between desire and production?",
     "label": "desire_production"},
    {"query": "What does Deleuze mean by smooth versus striated space?",
     "label": "smooth_striated"},
    {"query": "How does Deleuze characterize the event?",
     "label": "event"},
    {"query": "How does Deleuze treat the disjunctive synthesis?",
     "label": "disjunctive_synthesis"},
    {"query": "What role does Bergson's duration play in Deleuze's philosophy?",
     "label": "bergson_duration"},
    {"query": "How does Deleuze read Kant's transcendental aesthetic?",
     "label": "kant_aesthetic"},
    {"query": "What is the abstract machine in Deleuze and Guattari?",
     "label": "abstract_machine"},
    {"query": "How does Deleuze understand the simulacrum?",
     "label": "simulacrum"},
]

MULTI_ENTITY_SEEDS: list[dict] = [
    {"query": "How does Deleuze relate Nietzsche and Spinoza on affect?",
     "label": "nietzsche_spinoza_affect"},
    {"query": "How does Deleuze connect Bergson's duration with Nietzsche's eternal return?",
     "label": "bergson_nietzsche_time"},
    {"query": "What is the role of Kant and Hume in Deleuze's early philosophy?",
     "label": "kant_hume"},
    {"query": "How does Deleuze read Spinoza through Leibniz on expression?",
     "label": "spinoza_leibniz_expression"},
    {"query": "How do Deleuze and Guattari read Freud and Lacan in Anti-Oedipus?",
     "label": "freud_lacan_AO"},
    {"query": "How does Deleuze's Nietzsche diverge from Heidegger's?",
     "label": "nietzsche_heidegger"},
    {"query": "How do Deleuze and Foucault differ on power?",
     "label": "deleuze_foucault_power"},
    {"query": "How does Deleuze connect Artaud's body without organs to Spinoza's affect?",
     "label": "artaud_spinoza_bwo"},
    {"query": "What is the relation between Bergson and Riemann in Deleuze's work?",
     "label": "bergson_riemann"},
    {"query": "How does Deleuze read Hume alongside Kant on empiricism?",
     "label": "hume_kant_empiricism"},
    {"query": "How do Deleuze and Guattari deploy Marx and Nietzsche together?",
     "label": "marx_nietzsche_DG"},
    {"query": "How does Deleuze relate Proust's signs to Bergson's memory?",
     "label": "proust_bergson"},
    {"query": "How does Deleuze read Kafka with Nietzsche on minor politics?",
     "label": "kafka_nietzsche_minor"},
    {"query": "How does Deleuze link Spinoza and Leibniz on the fold?",
     "label": "spinoza_leibniz_fold"},
    {"query": "How do Deleuze and Guattari read Freud and Marx on family?",
     "label": "freud_marx_family"},
    {"query": "How does Deleuze compare Nietzsche and Bergson on the image of thought?",
     "label": "nietzsche_bergson_thought"},
    {"query": "What is the relation between Foucault and Nietzsche in Deleuze's reading?",
     "label": "foucault_nietzsche"},
    {"query": "How does Deleuze weave Spinoza and Nietzsche into Difference and Repetition?",
     "label": "spinoza_nietzsche_DR"},
]


def _noun_chunks(text: str, nlp) -> list[str]:
    doc = nlp(text)
    out: list[str] = []
    for ch in doc.noun_chunks:
        span = ch.text.strip().lower()
        span = re.sub(r"\s+", " ", span)
        if 2 <= len(span.split()) <= 4 and len(span) > 3:
            out.append(span)
    return out


def _tfidf_rank(chunks: list[list[str]], top_k: int) -> list[str]:
    """Rank spans by document frequency * inverse rarity within this query's
    top-k. We want spans that show up across *multiple* retrieved chunks
    (discriminative for the query) but are not single-chunk noise.
    """
    df: Counter = Counter()
    for spans in chunks:
        for s in set(spans):
            df[s] += 1
    # Keep spans appearing in ≥2 chunks; rank by df desc.
    keep = [(s, c) for s, c in df.items() if c >= 2]
    keep.sort(key=lambda t: (-t[1], t[0]))
    return [s for s, _ in keep[:top_k]]


def mine_abstractive(
    queries: list[dict],
    retrieve_fn,
    collection: str | None,
    top_k_chunks: int,
    top_spans: int,
) -> list[dict]:
    from shared.nlp import get_nlp_full, is_available
    if not is_available():
        raise SystemExit("spaCy unavailable — cannot mine noun chunks")
    nlp = get_nlp_full()

    out: list[dict] = []
    for seed in queries:
        q = seed["query"]
        _, chunks = retrieve_fn(q, collection=collection)
        top = chunks[:top_k_chunks]

        cid_counter: Counter = Counter()
        span_docs: list[list[str]] = []
        for ch in top:
            meta = ch.get("metadata") or {}
            cids = [c for c in (meta.get("canonical_ids") or "").split(",") if c]
            cid_counter.update(cids)
            text = ch.get("text") or ""
            span_docs.append(_noun_chunks(text, nlp))

        expected_entities = [cid for cid, cnt in cid_counter.most_common() if cnt >= 2]
        expected_concept_spans = _tfidf_rank(span_docs, top_spans)

        out.append({
            "query": q,
            "label": seed["label"],
            "expected_entities": expected_entities,
            "expected_concept_spans": expected_concept_spans,
        })
        log.info("mined '%s' → %d entities, %d spans", seed["label"],
                 len(expected_entities), len(expected_concept_spans))
    return out


def mine_multi_entity(queries: list[dict], min_entities: int) -> list[dict]:
    from shared.gazetteer import resolve
    from core.graph import neighbors

    out: list[dict] = []
    for seed in queries:
        q = seed["query"]
        entities = resolve(q)
        if len(entities) < min_entities:
            log.warning("skip '%s': only %d canonical entities resolved (need %d)",
                        seed["label"], len(entities), min_entities)
            continue
        # Edges = pairs (a,b) in the graph where both endpoints are in the query.
        ent_set = set(entities)
        expected_edges: list[list[str]] = []
        for a in entities:
            for b in neighbors(a):
                if b in ent_set and a < b:
                    expected_edges.append([a, b])
        out.append({
            "query": q,
            "label": seed["label"],
            "expected_entities": entities,
            "expected_edges": expected_edges,
        })
        log.info("mined '%s' → %d entities, %d edges", seed["label"],
                 len(entities), len(expected_edges))
    return out


def augment_precis(path: Path, retrieve_fn, collection: str | None,
                   top_k_chunks: int, top_spans: int) -> int:
    """Augment each entry of precis_evolution.json with expected_entities,
    expected_concept_spans, and min_level_mix — mined from retrieval, not
    LLM-drafted. Idempotent: overwrites existing mined fields, leaves
    human-authored fields (query, works_required, expected_stages) alone.
    """
    from shared.nlp import get_nlp_full, is_available
    if not is_available():
        raise SystemExit("spaCy unavailable — cannot mine precis noun chunks")
    nlp = get_nlp_full()

    data = json.loads(path.read_text())
    for entry in data:
        q = entry.get("query")
        if not q:
            continue
        _, chunks = retrieve_fn(q, collection=collection)
        top = chunks[:top_k_chunks]

        cid_counter: Counter = Counter()
        span_docs: list[list[str]] = []
        level_counts: Counter = Counter()
        for ch in top:
            meta = ch.get("metadata") or {}
            cid_counter.update(c for c in (meta.get("canonical_ids") or "").split(",") if c)
            span_docs.append(_noun_chunks(ch.get("text") or "", nlp))
            try:
                lvl = int(meta.get("level", 0) or 0)
            except (TypeError, ValueError):
                lvl = 0
            level_counts["L0" if lvl == 0 else "L1+"] += 1

        entry["expected_entities"] = [cid for cid, c in cid_counter.most_common() if c >= 2]
        entry["expected_concept_spans"] = _tfidf_rank(span_docs, top_spans)
        # min_level_mix: require at least one summary node if the current
        # retrieval already surfaces one (otherwise require zero — no
        # artificial pressure on queries that don't benefit from RAPTOR).
        entry["min_level_mix"] = {
            "L0": max(1, level_counts.get("L0", 0) // 2),
            "L1+": 1 if level_counts.get("L1+", 0) >= 1 else 0,
        }
        log.info("augmented '%s' → %d entities, %d spans, %d L1+ in top-%d",
                 entry.get("label") or q[:40],
                 len(entry["expected_entities"]),
                 len(entry["expected_concept_spans"]),
                 level_counts.get("L1+", 0), top_k_chunks)

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return len(data)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=None)
    ap.add_argument("--top-k-chunks", type=int, default=20,
                    help="Chunks per query to mine for entity/span frequencies")
    ap.add_argument("--top-spans", type=int, default=10,
                    help="Concept spans to keep per abstractive query")
    ap.add_argument("--min-entities", type=int, default=2,
                    help="Minimum canonical entities a multi-entity query must resolve")
    ap.add_argument("--output-dir", default="data/eval")
    ap.add_argument("--skip-abstractive", action="store_true")
    ap.add_argument("--skip-multi-entity", action="store_true")
    ap.add_argument("--augment-precis", default=None,
                    help="Path to precis_evolution.json to augment with mined fields")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_abstractive:
        from core.rag import retrieve
        abstractive = mine_abstractive(
            ABSTRACTIVE_SEEDS, retrieve, args.collection,
            args.top_k_chunks, args.top_spans,
        )
        (out_dir / "deleuze_abstractive.json").write_text(
            json.dumps(abstractive, indent=2, ensure_ascii=False)
        )
        log.info("Wrote %d abstractive queries", len(abstractive))

    if not args.skip_multi_entity:
        # feature_entity_gazetteer must be on for gazetteer.resolve to be useful,
        # but resolve() itself doesn't check the flag. Graph lookup needs the
        # graph DB to exist — built via scripts/reindex.py with P7 on.
        multi = mine_multi_entity(MULTI_ENTITY_SEEDS, args.min_entities)
        (out_dir / "multi_entity_queries.json").write_text(
            json.dumps(multi, indent=2, ensure_ascii=False)
        )
        log.info("Wrote %d multi-entity queries", len(multi))

    if args.augment_precis:
        from core.rag import retrieve
        n = augment_precis(
            Path(args.augment_precis), retrieve, args.collection,
            args.top_k_chunks, args.top_spans,
        )
        log.info("Augmented %d precis entries in %s", n, args.augment_precis)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
