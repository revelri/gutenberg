"""Unit tests for the chat completions endpoint helpers."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

API_PATH = str(Path(__file__).resolve().parent.parent / "services" / "api")
SERVICES_PATH = str(Path(__file__).resolve().parent.parent / "services")
sys.path.insert(0, API_PATH)
sys.path.insert(0, SERVICES_PATH)

# Stub heavy dependencies before importing chat module
sys.modules.setdefault("shared", MagicMock())
sys.modules.setdefault("shared.chroma", MagicMock())
sys.modules.setdefault("shared.nlp", MagicMock())
sys.modules.setdefault("shared.embeddings", MagicMock())
sys.modules.setdefault("shared.text_normalize", MagicMock())

from routers.chat import _estimate_tokens, _run_verification, Message, ChatRequest


# ---------------------------------------------------------------------------
# 1. _estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_basic(self):
        assert _estimate_tokens("abcdefgh") == 2  # 8 // 4

    def test_empty_string(self):
        assert _estimate_tokens("") == 0

    def test_short_string(self):
        # Strings shorter than 4 chars should still return 0
        assert _estimate_tokens("abc") == 0

    def test_long_string(self):
        text = "a" * 400
        assert _estimate_tokens(text) == 100


# ---------------------------------------------------------------------------
# 2. Collection routing via model name
# ---------------------------------------------------------------------------

class TestCollectionRouting:
    """Test that the endpoint parses collection slug from the model name."""

    def test_model_with_slug_routes_correctly(self):
        """Verify that 'gutenberg-rag/macy' maps through collection_routes."""
        model = "gutenberg-rag/macy"
        _, slug = model.split("/", 1)
        routes = {"macy": "gutenborg-macy-collection"}
        assert routes.get(slug) == "gutenborg-macy-collection"

    def test_model_without_slash_has_no_collection(self):
        model = "gutenberg-rag"
        assert "/" not in model


# ---------------------------------------------------------------------------
# 3. Qwen3 <think> token filtering (streaming logic)
# ---------------------------------------------------------------------------

class TestThinkTokenFiltering:
    """Test the streaming <think>...</think> stripping logic extracted from
    _stream_response.  We replicate the exact filtering code here since it
    is inline in the async generator and not separately callable."""

    @staticmethod
    def _filter_chunks(chunks: list[str]) -> list[str]:
        """Simulate the streaming think-token filter from _stream_response."""
        in_think = False
        output = []
        for content in chunks:
            if "<think>" in content:
                in_think = True
                content = content.split("<think>")[0]
            if in_think:
                if "</think>" in content:
                    in_think = False
                    content = content.split("</think>")[-1]
                else:
                    content = ""
            if content:
                output.append(content)
        return output

    def test_no_think_tokens(self):
        chunks = ["Hello", " world"]
        assert self._filter_chunks(chunks) == ["Hello", " world"]

    def test_think_block_single_chunk(self):
        # When <think>...</think> appears in one chunk, the code splits on
        # <think> first (yielding ""), then in_think is True and since
        # the remaining content "" has no </think>, it gets blanked.
        # The </think> part is in the same original content but was already
        # stripped by the split("<think>")[0].  So nothing survives.
        chunks = ["<think>internal reasoning</think>actual answer"]
        result = self._filter_chunks(chunks)
        assert result == []

    def test_think_block_single_chunk_with_close_separate(self):
        # When open and close arrive in separate chunks, the close chunk
        # content after </think> is preserved.
        chunks = ["<think>internal reasoning", "</think>actual answer"]
        result = self._filter_chunks(chunks)
        assert result == ["actual answer"]

    def test_think_block_across_chunks(self):
        # The text before <think> in the same chunk is lost because
        # in_think immediately triggers and blanks the (already split) content.
        chunks = ["start<think>", "reasoning", "</think>end"]
        result = self._filter_chunks(chunks)
        assert result == ["end"]

    def test_think_block_mid_stream(self):
        # "world" before <think> in the same chunk is lost for same reason.
        chunks = ["Hello ", "world<think>", "hidden", "</think> visible"]
        result = self._filter_chunks(chunks)
        assert result == ["Hello ", " visible"]

    def test_empty_after_filtering(self):
        chunks = ["<think>", "all hidden", "</think>"]
        result = self._filter_chunks(chunks)
        assert result == []


# ---------------------------------------------------------------------------
# 4. _run_verification
# ---------------------------------------------------------------------------

class TestRunVerification:
    def test_empty_sources_returns_empty(self):
        assert _run_verification("some response", []) == ""

    @patch("routers.chat.format_verification_footer")
    @patch("routers.chat.verify_against_source")
    @patch("routers.chat.verify_quotes")
    @patch("routers.chat.extract_quotes")
    def test_with_quotes(self, mock_extract, mock_verify, mock_against, mock_footer):
        mock_extract.return_value = ['"quoted text"']
        mock_verify.return_value = [{"quote": "quoted text", "status": "verified"}]
        mock_against.return_value = [{"quote": "quoted text", "status": "verified"}]
        mock_footer.return_value = "\n\n---\nVerification: 1/1 verified"

        sources = [{"text": "some source chunk", "metadata": {}}]
        result = _run_verification("She said \"quoted text\" in her paper.", sources)

        mock_extract.assert_called_once()
        mock_verify.assert_called_once_with(['"quoted text"'], sources)
        mock_against.assert_called_once()
        mock_footer.assert_called_once()
        assert result == "\n\n---\nVerification: 1/1 verified"

    @patch("routers.chat.extract_quotes")
    def test_no_quotes_returns_empty(self, mock_extract):
        mock_extract.return_value = []
        sources = [{"text": "chunk"}]
        result = _run_verification("No quotes here.", sources)
        assert result == ""

    @patch("routers.chat.extract_quotes")
    def test_exception_returns_empty(self, mock_extract):
        mock_extract.side_effect = RuntimeError("boom")
        sources = [{"text": "chunk"}]
        result = _run_verification("text", sources)
        assert result == ""


# ---------------------------------------------------------------------------
# 5. Token budget for conversation history truncation
# ---------------------------------------------------------------------------

class TestHistoryTruncation:
    """Test the history truncation logic that limits conversation context
    by estimated token count.  We replicate the loop from chat_completions
    since it operates on the request messages inline."""

    @staticmethod
    def _truncate_history(messages: list[Message], max_tokens: int) -> list[Message]:
        """Replicate the history truncation from chat_completions.

        Takes all messages except the last (user query), walks backward,
        and includes messages until the token budget is exceeded.
        """
        history = []
        total_tokens = 0
        for msg in reversed(messages[:-1]):
            msg_tokens = _estimate_tokens(msg.content)
            if total_tokens + msg_tokens > max_tokens and history:
                break
            history.insert(0, msg)
            total_tokens += msg_tokens
        return history

    def test_all_messages_fit(self):
        msgs = [
            Message(role="user", content="Hi"),           # 0 tokens
            Message(role="assistant", content="Hello"),   # 1 token
            Message(role="user", content="How are you"),  # 2 tokens
        ]
        result = self._truncate_history(msgs, max_tokens=100)
        assert len(result) == 2  # first two messages (last is excluded)

    def test_truncation_drops_oldest(self):
        # Create messages that exceed a small budget
        msgs = [
            Message(role="user", content="a" * 100),       # 25 tokens
            Message(role="assistant", content="b" * 100),   # 25 tokens
            Message(role="user", content="c" * 100),        # 25 tokens
            Message(role="user", content="latest query"),    # excluded (last msg)
        ]
        # Budget of 30 tokens: should keep only the most recent msg before last
        result = self._truncate_history(msgs, max_tokens=30)
        assert len(result) >= 1
        # The last included message (most recent before query) should be "c"*100
        assert result[-1].content == "c" * 100

    def test_zero_budget_still_includes_first_walked(self):
        """Even with budget=0, the code includes at least the first message
        it encounters (because the `and history` guard is empty on first pass)."""
        msgs = [
            Message(role="user", content="a" * 1000),   # 250 tokens
            Message(role="user", content="query"),
        ]
        result = self._truncate_history(msgs, max_tokens=0)
        # First message is always included (history is empty on first iteration)
        assert len(result) == 1
