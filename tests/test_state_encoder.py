"""Unit tests for state encoders (src/state_encoder.py)."""

from src.state_encoder import lowercase_encoder

# Test constants
RAW_STATE = "Summarize This Article"
LOWERCASE_STATE = "summarize this article"
PADDED_STATE = "  Summarize This Article  "
EMPTY_STATE = ""
WHITESPACE_ONLY = "   "


class TestLowercaseEncoder:
    """Tests for lowercase_encoder function."""

    def test_lowercases(self) -> None:
        """lowercase_encoder should lowercase the input."""
        assert lowercase_encoder(RAW_STATE) == LOWERCASE_STATE

    def test_strips_whitespace(self) -> None:
        """lowercase_encoder should strip leading/trailing whitespace."""
        assert lowercase_encoder(PADDED_STATE) == LOWERCASE_STATE

    def test_already_lowercase(self) -> None:
        """lowercase_encoder should be idempotent on lowercase input."""
        assert lowercase_encoder(LOWERCASE_STATE) == LOWERCASE_STATE

    def test_empty_string(self) -> None:
        """lowercase_encoder should handle empty string."""
        assert lowercase_encoder(EMPTY_STATE) == EMPTY_STATE

    def test_whitespace_only(self) -> None:
        """lowercase_encoder should return empty for whitespace-only."""
        assert lowercase_encoder(WHITESPACE_ONLY) == EMPTY_STATE

    def test_consistency(self) -> None:
        """Same input should always produce same output."""
        result1 = lowercase_encoder(RAW_STATE)
        result2 = lowercase_encoder(RAW_STATE)
        assert result1 == result2
