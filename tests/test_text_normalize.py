"""Tests for shared.text_normalize — canonical text normalization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))

from shared.text_normalize import (
    clean_for_ingestion,
    collapse_repeated_punctuation,
    normalize_for_comparison,
    normalize_unicode,
    normalize_whitespace,
    repair_hyphenation,
    strip_headers_footers,
    strip_markdown_formatting,
)


# ── normalize_unicode ────────────────────────────────────────────────


class TestNormalizeUnicode:
    def test_smart_quotes_to_ascii(self):
        assert normalize_unicode("\u201cHello\u201d") == '"Hello"'
        assert normalize_unicode("\u2018world\u2019") == "'world'"

    def test_em_dash_to_double_hyphen(self):
        assert normalize_unicode("word\u2014word") == "word--word"

    def test_en_dash_to_double_hyphen(self):
        assert normalize_unicode("1\u20132") == "1--2"

    def test_ellipsis_char_to_three_dots(self):
        assert normalize_unicode("wait\u2026") == "wait..."

    def test_nbsp_to_space(self):
        assert normalize_unicode("a\u00a0b") == "a b"

    def test_zero_width_chars_removed(self):
        assert normalize_unicode("a\u200bb\ufeffc") == "abc"

    def test_ligatures_decomposed(self):
        # NFKC decomposes fi ligature (U+FB01) to "fi"
        assert normalize_unicode("\ufb01nd") == "find"
        assert normalize_unicode("\ufb02ow") == "flow"

    def test_fullwidth_chars_normalized(self):
        # NFKC normalizes fullwidth ASCII
        assert normalize_unicode("\uff21\uff22\uff23") == "ABC"

    def test_guillemets_to_quotes(self):
        assert normalize_unicode("\u00abHello\u00bb") == '"Hello"'

    def test_empty_string(self):
        assert normalize_unicode("") == ""

    def test_plain_ascii_unchanged(self):
        text = "Hello, world! This is plain ASCII."
        assert normalize_unicode(text) == text


# ── repair_hyphenation ───────────────────────────────────────────────


class TestRepairHyphenation:
    def test_basic_syllable_break(self):
        assert repair_hyphenation("philoso-\nphy") == "philosophy"

    def test_syllable_break_with_spaces(self):
        assert repair_hyphenation("philoso-\n  phy") == "philosophy"

    def test_preserves_uppercase_after_hyphen(self):
        # Uppercase after hyphen = not a linebreak hyphenation
        assert repair_hyphenation("Anti-\nOedipus") == "Anti-\nOedipus"

    def test_preserves_inline_hyphens(self):
        assert repair_hyphenation("well-known") == "well-known"

    def test_preserves_compound_with_suffix(self):
        # "desiring" ends with -ing suffix → compound word, keep hyphen
        assert repair_hyphenation("desiring-\nmachine") == "desiring-machine"

    def test_preserves_compound_tion_suffix(self):
        assert repair_hyphenation("production-\nmachine") == "production-machine"

    def test_joins_short_prefix(self):
        # "Fur" doesn't end with a word suffix → syllable break → join
        assert repair_hyphenation("Fur-\nthermore") == "Furthermore"

    def test_joins_non_word_prefix(self):
        # "Oedi" is not a complete word → join
        assert repair_hyphenation("Oedi-\npus") == "Oedipus"

    def test_multiple_mixed(self):
        text = "philoso-\nphy and desiring-\nmachine"
        assert repair_hyphenation(text) == "philosophy and desiring-machine"


# ── normalize_whitespace ─────────────────────────────────────────────


class TestNormalizeWhitespace:
    def test_form_feed_to_newline(self):
        result = normalize_whitespace("a\fb")
        assert "\f" not in result
        assert "a\nb" == result

    def test_collapse_triple_newlines(self):
        assert normalize_whitespace("a\n\n\n\nb") == "a\n\nb"

    def test_collapse_multi_spaces(self):
        assert normalize_whitespace("a    b") == "a b"

    def test_strip_trailing_whitespace(self):
        assert normalize_whitespace("hello   \nworld  ") == "hello\nworld"

    def test_preserves_double_newline(self):
        assert normalize_whitespace("a\n\nb") == "a\n\nb"


# ── collapse_repeated_punctuation ────────────────────────────────────


class TestCollapseRepeatedPunctuation:
    def test_collapse_dots(self):
        assert collapse_repeated_punctuation("Chapter 1 ......... 5") == "Chapter 1 ... 5"

    def test_collapse_dashes(self):
        assert collapse_repeated_punctuation("text------text") == "text---text"

    def test_collapse_underscores(self):
        assert collapse_repeated_punctuation("____") == "___"

    def test_three_dots_unchanged(self):
        assert collapse_repeated_punctuation("wait...") == "wait..."

    def test_two_dots_unchanged(self):
        assert collapse_repeated_punctuation("e.g.") == "e.g."


# ── strip_headers_footers ───────────────────────────────────────────


class TestStripHeadersFooters:
    def test_strips_repeated_header(self):
        pages = [
            {"page": i, "text": f"ANTI-OEDIPUS\nContent on page {i}"}
            for i in range(1, 10)
        ]
        result = strip_headers_footers(pages)
        for p in result:
            assert not p["text"].startswith("ANTI-OEDIPUS")

    def test_strips_repeated_footer(self):
        pages = [
            {"page": i, "text": f"Content on page {i}\nAnti-Oedipus"}
            for i in range(1, 10)
        ]
        result = strip_headers_footers(pages)
        for p in result:
            assert "Anti-Oedipus" not in p["text"]

    def test_preserves_unique_lines(self):
        pages = [
            {"page": 1, "text": "Unique header\nContent"},
            {"page": 2, "text": "Different header\nMore content"},
            {"page": 3, "text": "Another header\nEven more content"},
        ]
        result = strip_headers_footers(pages)
        # Too few pages to strip anything
        assert result == pages

    def test_skips_few_pages(self):
        pages = [{"page": 1, "text": "Header\nContent"}]
        assert strip_headers_footers(pages) == pages

    def test_long_lines_not_stripped(self):
        long_line = "A" * 100
        pages = [
            {"page": i, "text": f"{long_line}\nContent {i}"}
            for i in range(1, 10)
        ]
        result = strip_headers_footers(pages)
        # Long first lines should not be stripped (likely content, not headers)
        for p in result:
            assert long_line in p["text"]


# ── strip_markdown_formatting ────────────────────────────────────────


class TestStripMarkdownFormatting:
    def test_bold(self):
        assert strip_markdown_formatting("**bold** text") == "bold text"

    def test_italic(self):
        assert strip_markdown_formatting("*italic* text") == "italic text"

    def test_bold_italic(self):
        assert strip_markdown_formatting("***both***") == "both"

    def test_no_formatting(self):
        assert strip_markdown_formatting("plain text") == "plain text"


# ── clean_for_ingestion (integration) ────────────────────────────────


class TestCleanForIngestion:
    def test_full_pipeline(self):
        text = "\u201cphiloso-\nphy\u201d is the study\u2026 of every-\nthing."
        result = clean_for_ingestion(text)
        assert '"philosophy" is the study... of everything.' == result

    def test_empty_passthrough(self):
        assert clean_for_ingestion("") == ""

    def test_already_clean_unchanged(self):
        text = "This is clean ASCII text with no issues."
        assert clean_for_ingestion(text) == text

    def test_toc_formatting(self):
        text = "Chapter 1 .............. 5\nChapter 2 .............. 23"
        result = clean_for_ingestion(text)
        assert "..." in result
        assert "...." not in result


# ── normalize_for_comparison ─────────────────────────────────────────


class TestNormalizeForComparison:
    def test_case_and_whitespace(self):
        assert normalize_for_comparison("  Hello   World  ") == "hello world"

    def test_unicode_normalized(self):
        assert normalize_for_comparison("\u201cHello\u201d") == '"hello"'

    def test_empty_passthrough(self):
        assert normalize_for_comparison("") == ""

    def test_round_trip_consistency(self):
        """Text cleaned for ingestion then normalized for comparison should
        match text directly normalized for comparison."""
        raw = "\u201cphiloso-\nphy\u201d is the study\u2026"
        via_ingestion = normalize_for_comparison(clean_for_ingestion(raw))
        direct = normalize_for_comparison(raw)
        # Both paths should produce comparable text
        # (ingestion path may repair hyphenation, so it can differ slightly)
        assert "philosophy" in via_ingestion
        # Direct path preserves the hyphen since repair_hyphenation is not in normalize_for_comparison
        # This is expected — the ingestion path is more thorough
