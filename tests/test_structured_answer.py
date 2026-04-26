"""Unit tests for services/api/core/structured_answer.py.

Covers schema construction, prompt assembly, JSON parsing, evidence
rendering, verbatim enforcement, validation, and the end-to-end
``answer_structured`` orchestration with the OpenRouter call mocked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

API_PATH = str(Path(__file__).resolve().parent.parent / "services" / "api")
SERVICES_PATH = str(Path(__file__).resolve().parent.parent / "services")
sys.path.insert(0, API_PATH)
sys.path.insert(0, SERVICES_PATH)

from core import structured_answer as sa


# ── Schema ──────────────────────────────────────────────────────────────

class TestSchema:
    def test_min_items_matches_required_works(self):
        schema = sa._schema(["A", "B", "C"])
        assert schema["properties"]["per_work"]["minItems"] == 3

    def test_min_items_defaults_to_one_when_no_required(self):
        schema = sa._schema(None)
        assert schema["properties"]["per_work"]["minItems"] == 1

    def test_min_items_one_for_empty_list(self):
        schema = sa._schema([])
        assert schema["properties"]["per_work"]["minItems"] == 1

    def test_additional_properties_locked_off(self):
        schema = sa._schema(["A"])
        assert schema["additionalProperties"] is False
        assert schema["properties"]["per_work"]["items"]["additionalProperties"] is False

    def test_required_keys_present(self):
        schema = sa._schema(["A"])
        assert set(schema["required"]) == {"synthesis", "per_work"}
        item_required = schema["properties"]["per_work"]["items"]["required"]
        assert set(item_required) == {"work", "quote", "page", "gloss"}


# ── Prompt ──────────────────────────────────────────────────────────────

class TestBuildSystemPrompt:
    def _chunk(self, source="x.pdf", ps=10, pe=12, text="lorem ipsum"):
        return {
            "metadata": {"source": source, "page_start": ps, "page_end": pe},
            "text": text,
        }

    def test_context_block_includes_source_and_pages(self):
        prompt = sa._build_system_prompt(
            "q", [self._chunk(ps=5, pe=7)], None
        )
        assert "[Source 1: x.pdf, pp. 5-7]" in prompt
        assert "lorem ipsum" in prompt

    def test_single_page_uses_p_label(self):
        prompt = sa._build_system_prompt(
            "q", [self._chunk(ps=5, pe=5)], None
        )
        assert "p. 5" in prompt
        assert "pp. 5-5" not in prompt

    def test_missing_page_meta_omits_label(self):
        prompt = sa._build_system_prompt(
            "q", [self._chunk(ps=0, pe=0)], None
        )
        # No "p." or "pp." should appear when both are zero
        assert "[Source 1: x.pdf]" in prompt

    def test_required_works_block_present(self):
        prompt = sa._build_system_prompt("q", [self._chunk()], ["A", "B"])
        assert "Required Works" in prompt
        assert "- A" in prompt
        assert "- B" in prompt

    def test_required_works_block_absent_when_none(self):
        prompt = sa._build_system_prompt("q", [self._chunk()], None)
        assert "Required Works" not in prompt

    def test_chunk_with_missing_metadata(self):
        prompt = sa._build_system_prompt(
            "q", [{"text": "raw"}], None
        )
        assert "[Source 1: unknown]" in prompt


# ── _parse_loose_json ───────────────────────────────────────────────────

class TestParseLooseJson:
    def test_plain_json(self):
        assert sa._parse_loose_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        text = '```json\n{"a": 1}\n```'
        assert sa._parse_loose_json(text) == {"a": 1}

    def test_fenced_no_lang(self):
        text = "```\n{\"a\": 1}\n```"
        assert sa._parse_loose_json(text) == {"a": 1}

    def test_extracts_balanced_object_from_trailing_text(self):
        text = 'preamble {"a": {"b": 2}} trailing chatter'
        assert sa._parse_loose_json(text) == {"a": {"b": 2}}

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            sa._parse_loose_json("")

    def test_no_brace_raises(self):
        with pytest.raises(json.JSONDecodeError):
            sa._parse_loose_json("just words")

    def test_unbalanced_raises(self):
        with pytest.raises(json.JSONDecodeError):
            sa._parse_loose_json('{"a": 1')


# ── _call_openrouter ────────────────────────────────────────────────────

class TestCallOpenrouter:
    def _ok_response(self, payload: dict):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(payload)}}]
        }
        resp.raise_for_status = MagicMock()
        resp.text = ""
        return resp

    def test_happy_path_returns_parsed(self):
        payload = {"synthesis": "s", "per_work": []}
        with patch.object(sa.httpx, "post", return_value=self._ok_response(payload)) as mock_post:
            out = sa._call_openrouter("sys", "q", {}, model="m", api_key="k")
        assert out == payload
        body = mock_post.call_args.kwargs["json"]
        assert body["model"] == "m"
        assert body["response_format"]["type"] == "json_schema"

    def test_falls_back_to_json_object_on_400(self):
        bad = MagicMock(spec=httpx.Response)
        bad.status_code = 400
        bad.text = "model does not support json_schema"
        good = self._ok_response({"synthesis": "s", "per_work": []})
        with patch.object(sa.httpx, "post", side_effect=[bad, good]) as mock_post:
            out = sa._call_openrouter("sys", "q", {}, model="m", api_key="k")
        assert out == {"synthesis": "s", "per_work": []}
        # Second call should have json_object response_format
        retry_body = mock_post.call_args_list[1].kwargs["json"]
        assert retry_body["response_format"] == {"type": "json_object"}

    def test_400_unrelated_raises(self):
        bad = MagicMock(spec=httpx.Response)
        bad.status_code = 400
        bad.text = "rate limit"
        bad.raise_for_status.side_effect = httpx.HTTPStatusError("400", request=MagicMock(), response=bad)
        with patch.object(sa.httpx, "post", return_value=bad):
            with pytest.raises(httpx.HTTPStatusError):
                sa._call_openrouter("sys", "q", {}, model="m", api_key="k")


# ── render_evidence_line ────────────────────────────────────────────────

class TestRenderEvidenceLine:
    def test_full_entry(self):
        line = sa.render_evidence_line(
            {"work": "A-O", "quote": "hello", "page": "47", "gloss": "g"}
        )
        assert line.startswith("- **A-O:** \"hello\"")
        assert "[Source: A-O, p. 47]" in line
        assert line.endswith("— g")

    def test_quote_only_no_page(self):
        line = sa.render_evidence_line(
            {"work": "A-O", "quote": "hello", "page": "", "gloss": ""}
        )
        assert "[Source:" not in line
        assert "\"hello\"" in line

    def test_no_quote_renders_placeholder(self):
        line = sa.render_evidence_line({"work": "A-O", "quote": "", "page": ""})
        assert "no verbatim passage" in line

    def test_strips_smart_quotes(self):
        line = sa.render_evidence_line(
            {"work": "A-O", "quote": "“hi”", "page": "1", "gloss": ""}
        )
        assert "\"hi\"" in line

    def test_missing_work_returns_empty(self):
        assert sa.render_evidence_line({"work": "", "quote": "x", "page": "1"}) == ""

    def test_no_corpus_chunks_branch(self):
        line = sa.render_evidence_line(
            {"work": "A-O", "gloss": "tried but absent"},
            no_corpus_chunks=True,
        )
        assert "no chunk for this work" in line
        assert "tried but absent" in line


# ── chunks_per_work ─────────────────────────────────────────────────────

class TestChunksPerWork:
    def test_counts_via_short_title_match(self):
        chunks = [
            {"metadata": {"source": "1972 Anti-Oedipus - Deleuze.pdf"}},
            {"metadata": {"source": "1980 A Thousand Plateaus - Deleuze.pdf"}},
            {"metadata": {"source": "1972 Anti-Oedipus - Deleuze.pdf"}},
        ]
        out = sa.chunks_per_work(chunks, ["Anti-Oedipus", "A Thousand Plateaus"])
        assert out == {"Anti-Oedipus": 2, "A Thousand Plateaus": 1}

    def test_zero_count_for_unmatched_work(self):
        out = sa.chunks_per_work(
            [{"metadata": {"source": "1972 Anti-Oedipus.pdf"}}],
            ["What Is Philosophy"],
        )
        assert out == {"What Is Philosophy": 0}

    def test_empty_inputs(self):
        assert sa.chunks_per_work([], []) == {}
        assert sa.chunks_per_work(None, ["A"]) == {"A": 0}

    def test_chunk_with_no_source_skipped(self):
        out = sa.chunks_per_work(
            [{"metadata": {}}, {"metadata": {"source": ""}}],
            ["Anti-Oedipus"],
        )
        assert out == {"Anti-Oedipus": 0}


# ── _render_markdown ────────────────────────────────────────────────────

class TestRenderMarkdown:
    def test_synthesis_only(self):
        out = sa._render_markdown(
            {"synthesis": "syn", "per_work": []}, None
        )
        assert out == "syn"

    def test_evidence_block_rendered(self):
        parsed = {
            "synthesis": "syn",
            "per_work": [
                {"work": "A-O", "quote": "x", "page": "1", "gloss": ""}
            ],
        }
        out = sa._render_markdown(parsed, ["A-O"])
        assert "### Evidence" in out
        assert "- **A-O:**" in out

    def test_works_without_chunks_synthesised(self):
        # Required work missing from per_work but flagged as no-chunks gets
        # an injected gap line.
        out = sa._render_markdown(
            {"synthesis": "syn", "per_work": []},
            ["Anti-Oedipus"],
            works_without_chunks={"Anti-Oedipus"},
        )
        assert "no chunk for this work" in out

    def test_works_without_chunks_dedup_against_existing_entry(self):
        parsed = {
            "synthesis": "",
            "per_work": [
                {"work": "Anti-Oedipus", "quote": "", "page": "", "gloss": "g"}
            ],
        }
        out = sa._render_markdown(
            parsed,
            ["Anti-Oedipus"],
            works_without_chunks={"Anti-Oedipus"},
        )
        # Only one Anti-Oedipus line should render (not duplicated)
        assert out.count("Anti-Oedipus") == 1


# ── stem / short_title / name_match ─────────────────────────────────────

class TestTitleHelpers:
    def test_stem_strips_year_and_pdf(self):
        assert sa._stem("1972 Anti-Oedipus.pdf") == "anti-oedipus"

    def test_stem_handles_none(self):
        assert sa._stem("") == ""

    def test_short_title_drops_author_tail(self):
        # _short_title caps at 4 tokens AND 30 chars; 4 tokens here = 32 chars,
        # so it's then sliced to 30 with trailing punct stripped.
        out = sa._short_title("anti-oedipus capitalism and schizophr - deleuze, gilles")
        assert "deleuze" not in out
        assert out.startswith("anti-oedipus")

    def test_short_title_caps_at_30_chars(self):
        out = sa._short_title("a" * 50)
        assert len(out) <= 30

    def test_name_match_subset(self):
        assert sa._name_match("Anti-Oedipus.pdf", "Anti-Oedipus")

    def test_name_match_superset(self):
        assert sa._name_match("Anti-Oedipus", "Anti-Oedipus")

    def test_name_match_negative(self):
        assert not sa._name_match("Anti-Oedipus", "What Is Philosophy")

    def test_name_match_empty(self):
        assert not sa._name_match("", "x")


# ── validate_coverage ───────────────────────────────────────────────────

class TestValidateCoverage:
    def test_full_coverage(self):
        parsed = {
            "synthesis": "anti-oedipus and a thousand plateaus together...",
            "per_work": [
                {"work": "Anti-Oedipus", "quote": "", "page": "", "gloss": ""},
                {"work": "A Thousand Plateaus", "quote": "", "page": "", "gloss": ""},
            ],
        }
        out = sa.validate_coverage(parsed, ["Anti-Oedipus", "A Thousand Plateaus"])
        assert out["per_work_coverage"] == 1.0
        assert out["synthesis_coverage"] == 1.0
        assert out["missing_in_per_work"] == []

    def test_partial_coverage(self):
        parsed = {
            "synthesis": "anti-oedipus only",
            "per_work": [
                {"work": "Anti-Oedipus", "quote": "", "page": "", "gloss": ""},
            ],
        }
        out = sa.validate_coverage(parsed, ["Anti-Oedipus", "What Is Philosophy"])
        assert out["per_work_coverage"] == 0.5
        assert "What Is Philosophy" in out["missing_in_per_work"]

    def test_no_required_works(self):
        out = sa.validate_coverage({"synthesis": "", "per_work": []}, None)
        assert out["per_work_coverage"] == 1.0
        assert out["synthesis_coverage"] == 1.0
        assert out["required_works_n"] == 0

    def test_unverified_quotes_via_rapidfuzz(self):
        parsed = {
            "synthesis": "",
            "per_work": [
                {
                    "work": "A-O",
                    "quote": "totally fabricated passage that is not in chunks",
                    "page": "1",
                    "gloss": "",
                }
            ],
        }
        chunks = [{"text": "completely unrelated content"}]
        out = sa.validate_coverage(parsed, ["A-O"], chunks=chunks)
        assert len(out["unverified_quotes"]) == 1

    def test_short_quote_skipped_in_unverified(self):
        parsed = {
            "synthesis": "",
            "per_work": [
                {"work": "A-O", "quote": "tiny", "page": "1", "gloss": ""}
            ],
        }
        out = sa.validate_coverage(parsed, ["A-O"], chunks=[{"text": "x"}])
        assert out["unverified_quotes"] == []


# ── _enforce_verbatim ───────────────────────────────────────────────────

class TestEnforceVerbatim:
    def test_drops_paraphrased_quote(self):
        parsed = {
            "per_work": [
                {
                    "work": "A-O",
                    "quote": "this is definitely not in the chunk text at all",
                    "page": "47",
                    "gloss": "",
                }
            ]
        }
        chunks = [{"text": "completely different content"}]
        dropped = sa._enforce_verbatim(parsed, chunks, min_score=85)
        assert dropped == 1
        assert parsed["per_work"][0]["quote"] == ""
        assert parsed["per_work"][0]["page"] == ""

    def test_keeps_verbatim_quote(self):
        chunk_text = "the body without organs is not a fantasy at all"
        parsed = {
            "per_work": [
                {
                    "work": "A-O",
                    "quote": "the body without organs is not a fantasy",
                    "page": "47",
                    "gloss": "",
                }
            ]
        }
        dropped = sa._enforce_verbatim(parsed, [{"text": chunk_text}], min_score=85)
        assert dropped == 0
        assert parsed["per_work"][0]["quote"]

    def test_no_chunks_returns_zero(self):
        assert sa._enforce_verbatim({"per_work": []}, []) == 0

    def test_short_quote_skipped(self):
        parsed = {"per_work": [{"work": "A", "quote": "abc", "page": "1"}]}
        dropped = sa._enforce_verbatim(parsed, [{"text": "xyz"}])
        assert dropped == 0

    def test_rapidfuzz_unavailable_returns_zero(self):
        # Patch the import inside the function — _enforce_verbatim does
        # `from rapidfuzz import fuzz` lazily.
        with patch.dict(sys.modules, {"rapidfuzz": None}):
            # Force ImportError by stashing a shim module that raises
            import builtins
            real_import = builtins.__import__

            def fake_import(name, *a, **kw):
                if name == "rapidfuzz":
                    raise ImportError("forced")
                return real_import(name, *a, **kw)

            with patch.object(builtins, "__import__", side_effect=fake_import):
                parsed = {"per_work": [{"work": "A", "quote": "x" * 20, "page": "1"}]}
                dropped = sa._enforce_verbatim(parsed, [{"text": "y"}])
                assert dropped == 0


# ── _settings_default_min_score ─────────────────────────────────────────

class TestSettingsDefaultMinScore:
    def test_returns_85_when_settings_unavailable(self, monkeypatch):
        # Force the import to fail
        import builtins
        real_import = builtins.__import__

        def fake(name, *a, **kw):
            if name == "core.config":
                raise ImportError("forced")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake)
        assert sa._settings_default_min_score() == 85

    def test_returns_settings_value_when_available(self, monkeypatch):
        from core import config
        monkeypatch.setattr(config.settings, "verbatim_min_score", 90, raising=False)
        assert sa._settings_default_min_score() == 90


# ── answer_structured (orchestration) ───────────────────────────────────

class TestAnswerStructured:
    def _chunk(self, source, text, ps=1, pe=1):
        return {
            "metadata": {"source": source, "page_start": ps, "page_end": pe},
            "text": text,
        }

    def test_full_path_with_required_works(self):
        chunks = [
            self._chunk("1972 Anti-Oedipus - Deleuze.pdf", "the body without organs"),
        ]
        parsed_response = {
            "synthesis": "Anti-Oedipus discusses the BwO.",
            "per_work": [
                {
                    "work": "Anti-Oedipus",
                    "quote": "the body without organs",
                    "page": "47",
                    "gloss": "core concept",
                }
            ],
        }

        with patch.object(sa, "_call_openrouter", return_value=parsed_response):
            rendered, parsed, validation = sa.answer_structured(
                "what is the BwO?",
                chunks,
                ["Anti-Oedipus"],
                model="m",
                api_key="k",
            )

        assert "Anti-Oedipus" in rendered
        assert parsed == parsed_response
        assert validation["per_work_coverage"] == 1.0
        assert validation["works_without_chunks"] == []

    def test_corpus_gap_surfaced(self):
        # Required: two works, only one has chunks
        chunks = [self._chunk("1972 Anti-Oedipus.pdf", "lorem ipsum")]
        parsed_response = {
            "synthesis": "Only A-O found.",
            "per_work": [
                {"work": "Anti-Oedipus", "quote": "lorem ipsum", "page": "1", "gloss": "g"}
            ],
        }
        with patch.object(sa, "_call_openrouter", return_value=parsed_response):
            rendered, _, validation = sa.answer_structured(
                "compare",
                chunks,
                ["Anti-Oedipus", "What Is Philosophy"],
                model="m",
                api_key="k",
            )
        assert "What Is Philosophy" in validation["works_without_chunks"]
        assert "no chunk for this work" in rendered

    def test_verbatim_enforcement_drops_paraphrase(self):
        chunks = [self._chunk("1972 Anti-Oedipus.pdf", "actual chunk text content")]
        parsed_response = {
            "synthesis": "syn",
            "per_work": [
                {
                    "work": "Anti-Oedipus",
                    "quote": "totally fabricated quote nowhere in chunks",
                    "page": "47",
                    "gloss": "g",
                }
            ],
        }
        with patch.object(sa, "_call_openrouter", return_value=parsed_response):
            rendered, parsed, _ = sa.answer_structured(
                "q", chunks, ["Anti-Oedipus"],
                model="m", api_key="k",
                enforce_verbatim=True, verbatim_min_score=85,
            )
        # Quote should have been blanked, so render shows "no verbatim passage"
        assert parsed["per_work"][0]["quote"] == ""
        assert "no verbatim passage" in rendered

    def test_enforce_verbatim_disabled(self):
        chunks = [self._chunk("1972 X.pdf", "alpha")]
        parsed_response = {
            "synthesis": "",
            "per_work": [
                {"work": "X", "quote": "totally elsewhere passage", "page": "1", "gloss": ""}
            ],
        }
        with patch.object(sa, "_call_openrouter", return_value=parsed_response):
            _, parsed, _ = sa.answer_structured(
                "q", chunks, ["X"],
                model="m", api_key="k",
                enforce_verbatim=False,
            )
        assert parsed["per_work"][0]["quote"] == "totally elsewhere passage"
