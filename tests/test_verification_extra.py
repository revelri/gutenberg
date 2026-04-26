"""Extended coverage for services/api/core/verification.py.

Targets the gaps left by tests/test_verification.py and
tests/test_verification_repair.py:

- trailing-blockquote flush in extract_quotes
- too-short and stripped-match paths of _verify_single_quote
- third-tier lemma fallback path
- difflib fallback when rapidfuzz is unavailable
- verify_against_source PDF cross-check (PyMuPDF mocked)
- _find_pdf directory scan
- format_verification_footer source-confirmed and warning branches
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "api"))
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))

from core import verification as v


# ── extract_quotes trailing blockquote flush ────────────────────────────

class TestExtractQuotesTrailingBlockquote:
    def test_blockquote_at_eof_is_flushed(self):
        text = "intro\n> the body without organs\n> is not a fantasy"
        quotes = v.extract_quotes(text)
        # Single blockquote spanning two lines at end of input
        assert any("body without organs" in q for q in quotes)

    def test_short_trailing_blockquote_dropped(self):
        text = "intro\n> hi"
        quotes = v.extract_quotes(text)
        assert quotes == []


# ── _verify_single_quote: too_short and stripped paths ──────────────────

class TestVerifySingleQuoteEdges:
    def test_too_short_quote_marked(self):
        # Below MIN_QUOTE_LENGTH (20)
        out = v._verify_single_quote("short", [{"text": "anything", "metadata": {}}])
        assert out["status"] == "too_short"

    def test_stripped_match_path(self):
        # Exact substring won't match because of compound hyphenation;
        # _strip_for_comparison removes the hyphen so quote matches.
        chunk_text = "the line-strokes form a nomadic distribution across the page"
        quote = "the linestrokes form a nomadic distribution"  # hyphen removed
        out = v._verify_single_quote(
            quote,
            [{"text": chunk_text, "metadata": {"source": "X.pdf", "page_start": 12}}],
        )
        assert out["status"] == "verified"
        assert out["page"] == 12

    def test_unverified_when_no_chunk_matches(self):
        out = v._verify_single_quote(
            "this passage definitely does not appear anywhere",
            [{"text": "totally different content here", "metadata": {}}],
        )
        assert out["status"] == "unverified"


# ── Third-tier lemma fallback ───────────────────────────────────────────

class TestLemmaFallback:
    def test_lemma_match_when_substring_fails(self):
        # The quote uses a different inflection than the chunk; lemma
        # reduction should make them match. We mock the spaCy hook so the
        # test stays deterministic and doesn't require the model.
        chunk = {
            "text": "the machine produces desire continuously across the social field",
            "metadata": {"source": "AO.pdf", "page_start": 7},
        }
        # Quote that won't match exact, stripped, OR fuzzy paths because
        # the wording diverges enough.
        quote = "machines created longing through the body politic"

        fake_doc = lambda text: [
            MagicMock(lemma_=tok, is_alpha=True) for tok in text.lower().split()
        ]
        # Build lemma-equivalent prefix between the two so the
        # `quote_lemmas[:40] in chunk_lemmas` check succeeds.
        chunk_lemmas = "machine create longing through the body politic across the social field"
        quote_lemmas = "machine create longing through the body politic"

        nlp_calls = {"i": 0}

        def fake_nlp(text):
            nlp_calls["i"] += 1
            # First call after _normalize is for the quote, subsequent for chunks
            lemmas = quote_lemmas if nlp_calls["i"] == 1 else chunk_lemmas
            return [MagicMock(lemma_=t, is_alpha=True) for t in lemmas.split()]

        nlp_module = MagicMock()
        nlp_module.is_available = MagicMock(return_value=True)
        nlp_module.get_nlp = MagicMock(return_value=fake_nlp)

        with patch.dict(sys.modules, {"shared.nlp": nlp_module}):
            out = v._verify_single_quote(quote, [chunk])

        assert out["status"] == "approximate"
        assert out["source"] == "AO.pdf"
        assert out["page"] == 7
        assert out["similarity"] == pytest.approx(0.85)

    def test_lemma_path_swallows_import_error(self):
        # Force ImportError so we land in the except branch (line 303)
        chunk = {"text": "alpha bravo charlie", "metadata": {}}
        quote = "delta echo foxtrot golf hotel india juliet"

        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "shared.nlp":
                raise ImportError("forced")
            return real_import(name, *a, **kw)

        with patch.object(builtins, "__import__", side_effect=fake_import):
            out = v._verify_single_quote(quote, [chunk])
        assert out["status"] == "unverified"


# ── _best_window_ratio difflib fallback ─────────────────────────────────

class TestBestWindowRatioDifflibFallback:
    def test_difflib_used_when_rapidfuzz_missing(self):
        # Disable feature flag so we skip the rapidfuzz branch entirely
        with patch.object(v._settings, "feature_rapidfuzz_verify", False, create=True):
            r = v._best_window_ratio("hello world", "hello world there")
        assert r > 0.7

    def test_difflib_quote_longer_than_text(self):
        with patch.object(v._settings, "feature_rapidfuzz_verify", False, create=True):
            r = v._best_window_ratio("a very long quote string", "tiny")
        # Should not crash; returns SequenceMatcher ratio of full strings
        assert 0.0 <= r <= 1.0

    def test_difflib_early_exit_on_threshold(self):
        text = "x" * 50 + "match here exactly" + "x" * 50
        with patch.object(v._settings, "feature_rapidfuzz_verify", False, create=True):
            r = v._best_window_ratio("match here exactly", text)
        assert r >= v.FUZZY_THRESHOLD


# ── verify_against_source ───────────────────────────────────────────────

class TestVerifyAgainstSource:
    def test_skips_unverified_results(self, tmp_path):
        results = [{"status": "unverified", "source": "X.pdf", "page": 1}]
        out = v.verify_against_source(results, pdf_dirs=[str(tmp_path)])
        assert out[0]["source_verified"] is None

    def test_skips_when_no_source_or_page(self, tmp_path):
        results = [{"status": "verified", "source": None, "page": None, "_full_quote": "x", "quote": "x"}]
        out = v.verify_against_source(results, pdf_dirs=[str(tmp_path)])
        assert out[0]["source_verified"] is None

    def test_skips_when_pdf_not_found(self, tmp_path):
        results = [{"status": "verified", "source": "missing.pdf", "page": 1, "_full_quote": "x", "quote": "x"}]
        out = v.verify_against_source(results, pdf_dirs=[str(tmp_path)])
        assert out[0]["source_verified"] is None

    def test_default_pdf_dirs_used_when_none(self, tmp_path, monkeypatch):
        # Just exercise the os.path.isdir branch with no real dirs
        monkeypatch.chdir(tmp_path)
        results = [{"status": "verified", "source": "X.pdf", "page": 1, "_full_quote": "x", "quote": "x"}]
        out = v.verify_against_source(results, pdf_dirs=None)
        assert out[0]["source_verified"] is None

    def test_quote_found_on_cited_page(self, tmp_path):
        # Create a fake PDF so _find_pdf returns a path, then mock fitz.
        # Use non-hyphenated source so _find_pdf's normalization matches.
        pdf = tmp_path / "1972 antioedipus.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        results = [{
            "status": "verified",
            "source": "antioedipus.pdf",
            "page": 1,
            "_full_quote": "the body without organs is not a fantasy at all",
            "quote": "the body without organs",
        }]
        page0 = MagicMock()
        page0.get_text = MagicMock(return_value="THE BODY WITHOUT ORGANS IS NOT A FANTASY AT ALL — surrounding text")
        doc = MagicMock()
        doc.__len__ = MagicMock(return_value=1)
        doc.__getitem__ = MagicMock(return_value=page0)
        doc.close = MagicMock()
        fitz_mod = MagicMock()
        fitz_mod.open = MagicMock(return_value=doc)
        with patch.dict(sys.modules, {"fitz": fitz_mod}):
            out = v.verify_against_source(results, pdf_dirs=[str(tmp_path)])
        assert out[0]["source_verified"] is True

    def test_quote_found_on_adjacent_page(self, tmp_path):
        pdf = tmp_path / "1972 antioedipus.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        results = [{
            "status": "verified",
            "source": "antioedipus.pdf",
            "page": 2,
            "_full_quote": "the body without organs is not a fantasy at all",
            "quote": "the body without organs",
        }]
        # Page 1 (idx 1) doesn't contain it; page 2 (idx 2) does
        page_target = MagicMock(); page_target.get_text = MagicMock(return_value="unrelated")
        page_next = MagicMock(); page_next.get_text = MagicMock(return_value="the body without organs is not a fantasy at all and so on")
        doc = MagicMock()
        doc.__len__ = MagicMock(return_value=3)
        doc.__getitem__ = MagicMock(side_effect=lambda i: [MagicMock(get_text=MagicMock(return_value="")), page_target, page_next][i])
        doc.close = MagicMock()
        fitz_mod = MagicMock(); fitz_mod.open = MagicMock(return_value=doc)
        with patch.dict(sys.modules, {"fitz": fitz_mod}):
            out = v.verify_against_source(results, pdf_dirs=[str(tmp_path)])
        assert out[0]["source_verified"] is True

    def test_pymupdf_exception_swallowed(self, tmp_path):
        pdf = tmp_path / "anti-oedipus.pdf"
        pdf.write_bytes(b"x")
        results = [{
            "status": "verified",
            "source": "anti-oedipus.pdf",
            "page": 1,
            "_full_quote": "x",
            "quote": "x",
        }]
        fitz_mod = MagicMock(); fitz_mod.open = MagicMock(side_effect=RuntimeError("boom"))
        with patch.dict(sys.modules, {"fitz": fitz_mod}):
            out = v.verify_against_source(results, pdf_dirs=[str(tmp_path)])
        # source_verified stays None because of the exception
        assert out[0]["source_verified"] is None


# ── _find_pdf ────────────────────────────────────────────────────────────

class TestFindPdf:
    def test_matches_title_after_year_prefix(self, tmp_path):
        # _find_pdf normalizes filename with replace("-", " ") but not source,
        # so use a no-hyphen title to verify the year-prefix-strip path.
        (tmp_path / "1972 antioedipus.pdf").write_bytes(b"x")
        out = v._find_pdf("antioedipus", [str(tmp_path)])
        assert out and out.endswith("1972 antioedipus.pdf")

    def test_matches_full_source_lowercase(self, tmp_path):
        (tmp_path / "Some_File-Name.pdf").write_bytes(b"x")
        out = v._find_pdf("some file name.pdf", [str(tmp_path)])
        assert out is not None

    def test_returns_none_when_no_match(self, tmp_path):
        (tmp_path / "other.pdf").write_bytes(b"x")
        assert v._find_pdf("nonexistent", [str(tmp_path)]) is None

    def test_skips_non_pdf_files(self, tmp_path):
        (tmp_path / "anti-oedipus.txt").write_text("x")
        assert v._find_pdf("anti-oedipus", [str(tmp_path)]) is None


# ── format_verification_footer edges ────────────────────────────────────

class TestFooterEdges:
    def test_empty_results_returns_empty(self):
        assert v.format_verification_footer([]) == ""

    def test_only_too_short_returns_empty(self):
        assert v.format_verification_footer([{"status": "too_short", "quote": "x"}]) == ""

    def test_source_confirmed_count_rendered(self):
        results = [
            {"status": "verified", "quote": "q1", "similarity": 1.0, "source_verified": True},
            {"status": "verified", "quote": "q2", "similarity": 1.0, "source_verified": False},
        ]
        out = v.format_verification_footer(results)
        assert "1/2 source-confirmed" in out
        # The False one should get the warning emoji line
        assert "verified in chunk but NOT found on cited page" in out

    def test_approximate_and_unverified_lines(self):
        results = [
            {"status": "approximate", "quote": "q1", "similarity": 0.9},
            {"status": "unverified", "quote": "q2", "similarity": 0.4},
        ]
        out = v.format_verification_footer(results)
        assert "approximate" in out
        assert "unverified" in out
