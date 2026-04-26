"""Shared SpaCy NLP module — singleton loader with graceful fallback.

Provides lemmatization, sentence segmentation, and POS tagging for both
the API (query-time BM25) and worker (ingestion-time chunking) containers.

Uses en_core_web_sm (12MB) with selective pipeline components for speed.
Falls back gracefully if SpaCy is not installed.
"""

import logging

log = logging.getLogger("gutenberg.nlp")

_nlp = None
_nlp_sent = None
_nlp_full = None
_available = None


def is_available() -> bool:
    """Check if SpaCy and en_core_web_sm are installed."""
    global _available
    if _available is None:
        try:
            import spacy
            spacy.load("en_core_web_sm")
            _available = True
        except (ImportError, OSError):
            _available = False
            log.warning("SpaCy not available — falling back to NLTK for tokenization")
    return _available


def get_nlp():
    """Get SpaCy model with tagger+lemmatizer only (fast, no parser/NER).

    ~2ms per query at 10K tokens/sec. Used for BM25 tokenization and
    quote verification at query time.
    """
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        log.info("SpaCy loaded (tagger+lemmatizer, no parser/NER)")
    return _nlp


def get_nlp_with_sentencizer():
    """Get SpaCy model with dependency parser for sentence boundary detection.

    Slower than get_nlp() but needed for accurate sentence splitting.
    Used at ingestion time in the chunker.
    """
    global _nlp_sent
    if _nlp_sent is None:
        import spacy
        _nlp_sent = spacy.load("en_core_web_sm", disable=["ner"])
        log.info("SpaCy loaded (tagger+lemmatizer+parser for sentencizer)")
    return _nlp_sent


def get_nlp_full():
    """Get SpaCy model with parser + NER enabled.

    Heavier than :func:`get_nlp` (~5-10ms/query). Needed for entity extraction
    and noun-chunk expansion. Loaded lazily on first use.

    If ``feature_entity_gazetteer`` is enabled and a gazetteer is available, an
    ``EntityRuler`` is attached before the NER component so curated entities
    (authors, works, concepts) are matched deterministically regardless of
    whether the statistical NER would pick them up.
    """
    global _nlp_full
    if _nlp_full is None:
        import spacy
        _nlp_full = spacy.load("en_core_web_sm")

        try:
            from core.config import settings
        except Exception:
            try:
                from services.api.core.config import settings  # type: ignore
            except Exception:
                settings = None  # type: ignore

        if settings is not None and getattr(settings, "feature_entity_gazetteer", False):
            try:
                from shared.gazetteer import get_patterns

                patterns = get_patterns()
                if patterns and "entity_ruler" not in _nlp_full.pipe_names:
                    ruler = _nlp_full.add_pipe("entity_ruler", before="ner")
                    ruler.add_patterns(patterns)
                    log.info(f"EntityRuler attached with {len(patterns)} patterns")
            except Exception as e:
                log.warning(f"EntityRuler setup failed: {e}")

        log.info("SpaCy loaded (full pipeline: tagger+parser+NER)")
    return _nlp_full


def lemmatize(text: str) -> list[str]:
    """Lemmatize text, returning list of lemmas (alpha tokens only, len > 1)."""
    doc = get_nlp()(text)
    return [t.lemma_ for t in doc if t.is_alpha and len(t.text) > 1]


def sentencize(text: str) -> list[str]:
    """Split text into sentences using SpaCy's statistical sentencizer."""
    doc = get_nlp_with_sentencizer()(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
