"""Tests for citation formatter — verified against style guide exemplars."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "api"))

import pytest
from core.citation_formatter import Citation, Style, format_inline, format_bibliography


@pytest.fixture
def single():
    return Citation(
        quote="x",
        author="Deleuze, Gilles",
        title="Difference and Repetition",
        year=1968,
        page=1,
        publisher="Columbia University Press",
        city="New York",
    )


@pytest.fixture
def multi():
    return Citation(
        quote="x",
        author="Deleuze, Gilles and Felix Guattari",
        title="A Thousand Plateaus",
        year=1980,
        page=24,
        publisher="University of Minnesota Press",
        city="Minneapolis",
    )


class TestMLA:
    def test_inline_single(self, single):
        assert format_inline(single, Style.MLA) == "(Deleuze 1)"

    def test_inline_multi(self, multi):
        assert format_inline(multi, Style.MLA) == "(Deleuze and Guattari 24)"

    def test_bib_single(self, single):
        assert (
            format_bibliography(single, Style.MLA)
            == "Deleuze, Gilles. *Difference and Repetition*. Columbia University Press, 1968."
        )

    def test_bib_multi(self, multi):
        assert "Deleuze, Gilles, and Felix Guattari" in format_bibliography(
            multi, Style.MLA
        )


class TestAPA:
    def test_inline_single(self, single):
        assert format_inline(single, Style.APA) == "(Deleuze, 1968, p. 1)"

    def test_inline_multi(self, multi):
        assert format_inline(multi, Style.APA) == "(Deleuze & Guattari, 1980, p. 24)"

    def test_bib_single(self, single):
        assert (
            format_bibliography(single, Style.APA)
            == "Deleuze, G. (1968). *Difference and Repetition*. Columbia University Press."
        )

    def test_bib_multi(self, multi):
        assert "Deleuze, G., & Guattari, F." in format_bibliography(multi, Style.APA)


class TestChicago:
    def test_inline_single(self, single):
        r = format_inline(single, Style.CHICAGO)
        assert "Gilles Deleuze" in r and "*Difference and Repetition*" in r

    def test_bib_single(self, single):
        assert "New York: Columbia University Press, 1968." in format_bibliography(
            single, Style.CHICAGO
        )


class TestHarvard:
    def test_inline_single(self, single):
        assert format_inline(single, Style.HARVARD) == "(Deleuze 1968, p. 1)"

    def test_inline_multi(self, multi):
        assert format_inline(multi, Style.HARVARD) == "(Deleuze & Guattari 1980, p. 24)"


class TestASA:
    def test_inline_single(self, single):
        assert format_inline(single, Style.ASA) == "(Deleuze 1968:1)"

    def test_inline_multi(self, multi):
        assert format_inline(multi, Style.ASA) == "(Deleuze and Guattari 1980:24)"

    def test_no_double_period(self, single):
        assert ".." not in format_bibliography(single, Style.ASA)


class TestSAGE:
    def test_inline_single(self, single):
        assert format_inline(single, Style.SAGE) == "(Deleuze, 1968, p. 1)"

    def test_bib_matches_apa(self, single):
        assert format_bibliography(single, Style.APA) == format_bibliography(
            single, Style.SAGE
        )


class TestEdgeCases:
    def test_epub_page(self):
        c = Citation(
            quote="x", author="Deleuze, Gilles", title="x", year=2000, page="ch. 3"
        )
        assert "ch. 3" in format_inline(c, Style.APA)


class TestCitationEdgeCases:
    def test_empty_author(self):
        c = Citation(quote="x", author="", title="Test", year=2000, page=1)
        result = format_inline(c, Style.APA)
        assert "2000" in result
        assert "(, " not in result

    def test_three_authors(self):
        c = Citation(
            quote="x",
            author="Deleuze, Gilles and Felix Guattari and Claire Parnet",
            title="Test",
            year=2000,
            page=1,
        )
        result = format_inline(c, Style.APA)
        assert "et al." in result

    def test_page_zero(self):
        c = Citation(
            quote="x", author="Deleuze, Gilles", title="Test", year=2000, page=0
        )
        result = format_inline(c, Style.MLA)
        assert "(Deleuze 0)" == result

    def test_missing_year_zero(self):
        c = Citation(quote="x", author="Deleuze, Gilles", title="Test", year=0, page=1)
        result = format_inline(c, Style.APA)
        assert "Deleuze" in result

    def test_missing_year_none(self):
        c = Citation(
            quote="x", author="Deleuze, Gilles", title="Test", year=None, page=1
        )
        result = format_inline(c, Style.APA)
        assert "Deleuze" in result
        assert "None" not in result

    def test_very_long_title(self):
        long_title = "A" * 300
        c = Citation(
            quote="x", author="Deleuze, Gilles", title=long_title, year=2000, page=1
        )
        result = format_inline(c, Style.CHICAGO)
        assert "Deleuze" in result
        assert long_title in result

    def test_special_characters_in_author(self):
        c = Citation(
            quote="x",
            author="Müller-Lüdenscheidt, Hans-Jürgen",
            title="Test",
            year=2000,
            page=1,
        )
        result = format_inline(c, Style.MLA)
        assert "Müller-Lüdenscheidt" in result

    def test_page_as_string(self):
        c = Citation(
            quote="x", author="Deleuze, Gilles", title="Test", year=2000, page="ch. 3"
        )
        result = format_inline(c, Style.APA)
        assert "ch. 3" in result
        result_mla = format_inline(c, Style.MLA)
        assert "ch. 3" in result_mla

    def test_all_styles_with_empty_author(self):
        c = Citation(quote="x", author="", title="Test", year=2000, page=1)
        for style in Style:
            result = format_inline(c, style)
            assert isinstance(result, str)
            assert len(result) > 0
