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


def lemmatize(text: str) -> list[str]:
    """Lemmatize text, returning list of lemmas (alpha tokens only, len > 1)."""
    doc = get_nlp()(text)
    return [t.lemma_ for t in doc if t.is_alpha and len(t.text) > 1]


def sentencize(text: str) -> list[str]:
    """Split text into sentences using SpaCy's statistical sentencizer."""
    doc = get_nlp_with_sentencizer()(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
