"""ALCE-style citation faithfulness eval (P6).

For every sentence in an answer that carries an inline ``[Source: …, p. N]``
tag, run NLI entailment on (cited_chunk, sentence) and compute:

  * citation recall     = fraction of citation-bearing sentences with at least
                          one entailing cited chunk
  * citation precision  = fraction of individual citations whose cited chunk
                          entails the sentence

Input JSON format:
    [
      {
        "query": "...",
        "answer": "<LLM response with inline citation tags>",
        "chunks": [
          {"source": "...", "page_start": 12, "page_end": 14, "text": "..."},
          ...
        ]
      },
      ...
    ]

Output: summary + per-sample breakdown. Write results to
``data/eval/alce_results.json``.

Offline-only — no runtime cost. Intended to be driven from scripts/reindex.py
or a weekly report.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger("eval.alce")

_CITATION_RE = re.compile(
    r"\[Source:\s*(?P<title>[^\]]+?),\s*pp?\.\s*(?P<page>\d+(?:\s*[-–—]\s*\d+)?)\s*\]",
    re.IGNORECASE,
)


def _sentences(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, text) for each sentence. Uses spaCy if available."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))
        from shared.nlp import get_nlp_with_sentencizer, is_available
        if is_available():
            doc = get_nlp_with_sentencizer()(text)
            return [(s.start_char, s.end_char, s.text) for s in doc.sents]
    except Exception:
        pass
    # Fallback: split on end-punctuation.
    spans: list[tuple[int, int, str]] = []
    pos = 0
    for m in re.finditer(r"[.!?](?:\s+|$)", text):
        end = m.end()
        spans.append((pos, end, text[pos:end]))
        pos = end
    if pos < len(text):
        spans.append((pos, len(text), text[pos:]))
    return spans


def _find_chunks_for_citation(
    title: str, page_spec: str, chunks: list[dict]
) -> list[dict]:
    nums = re.findall(r"\d+", page_spec)
    if not nums:
        return []
    low, high = int(nums[0]), int(nums[-1])
    title_lower = (title or "").lower().strip()
    hits = []
    for ch in chunks:
        if title_lower and title_lower not in (ch.get("source") or "").lower():
            continue
        ps = ch.get("page_start") or 0
        pe = ch.get("page_end") or ps
        if not ps:
            continue
        if not (pe < low or ps > high):
            hits.append(ch)
    return hits


class _NLI:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def entails(self, premise: str, hypothesis: str) -> float:
        # NLI cross-encoders typically output 3-class logits:
        # [contradiction, neutral, entailment]. Return entailment probability.
        scores = self.model.predict([(premise, hypothesis)])
        arr = np.asarray(scores).reshape(-1)
        if arr.size >= 3:
            exp = np.exp(arr - arr.max())
            probs = exp / exp.sum()
            return float(probs[-1])
        return float(arr[0])


def evaluate(
    samples: list[dict],
    model_name: str,
    threshold: float,
) -> dict:
    nli = _NLI(model_name)
    total_sentences = 0
    recall_hits = 0
    total_citations = 0
    precision_hits = 0
    per_sample: list[dict] = []

    for sample in samples:
        answer = sample.get("answer", "")
        chunks = sample.get("chunks", [])
        sent_results: list[dict] = []
        for _, _, sentence in _sentences(answer):
            tags = list(_CITATION_RE.finditer(sentence))
            if not tags:
                continue
            total_sentences += 1
            tag_premises: list[tuple[str, float]] = []
            for tag in tags:
                total_citations += 1
                cited = _find_chunks_for_citation(
                    tag.group("title"), tag.group("page"), chunks
                )
                if not cited:
                    tag_premises.append((tag.group(0), 0.0))
                    continue
                best = 0.0
                for ch in cited:
                    score = nli.entails(ch.get("text", ""), sentence)
                    if score > best:
                        best = score
                tag_premises.append((tag.group(0), best))
                if best >= threshold:
                    precision_hits += 1
            if any(score >= threshold for _, score in tag_premises):
                recall_hits += 1
            sent_results.append(
                {"sentence": sentence, "tags": tag_premises}
            )
        per_sample.append(
            {"query": sample.get("query"), "sentences": sent_results}
        )

    recall = recall_hits / total_sentences if total_sentences else 0.0
    precision = precision_hits / total_citations if total_citations else 0.0
    return {
        "summary": {
            "threshold": threshold,
            "total_sentences": total_sentences,
            "total_citations": total_citations,
            "citation_recall": recall,
            "citation_precision": precision,
            "model": model_name,
        },
        "samples": per_sample,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to samples JSON (list of {query, answer, chunks})")
    ap.add_argument("--output", default="data/eval/alce_results.json")
    ap.add_argument("--model", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))
        from api.core.config import settings
    except Exception:
        settings = None  # type: ignore

    model = args.model or (getattr(settings, "alce_nli_model", None) or "cross-encoder/nli-deberta-v3-base")
    threshold = args.threshold or (getattr(settings, "alce_entail_threshold", None) or 0.5)

    samples = json.loads(Path(args.input).read_text())
    results = evaluate(samples, model, threshold)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    log.info(
        "ALCE — sentences=%d citations=%d recall=%.3f precision=%.3f",
        results["summary"]["total_sentences"],
        results["summary"]["total_citations"],
        results["summary"]["citation_recall"],
        results["summary"]["citation_precision"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
