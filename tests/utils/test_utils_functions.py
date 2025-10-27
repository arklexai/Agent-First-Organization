"""Tests for utility functions module.

This module tests the utility functions used throughout the Arklex framework,
including text processing, JSON handling, and chat history formatting.
"""

from unittest.mock import patch

import arklex.utils.utils as utils


class TestStrSimilarity:
    """Test cases for str_similarity function."""

    def test_str_similarity_identical_strings(self) -> None:
        """Test similarity of identical strings."""
        result = utils.str_similarity("hello", "hello")
        assert result == 1.0

    def test_str_similarity_different_strings(self) -> None:
        """Test similarity of different strings."""
        result = utils.str_similarity("hello", "world")
        assert 0.0 <= result < 1.0

    def test_str_similarity_empty_strings(self) -> None:
        """Test similarity of empty strings."""
        result = utils.str_similarity("", "")
        assert result == 0.0  # Actual behavior returns 0 for empty strings

    def test_str_similarity_one_empty_string(self) -> None:
        """Test similarity with one empty string."""
        result = utils.str_similarity("hello", "")
        assert 0.0 <= result < 1.0

    def test_str_similarity_similar_strings(self) -> None:
        """Test similarity of similar strings."""
        result = utils.str_similarity("hello", "helo")
        assert 0.0 < result < 1.0

    def test_str_similarity_case_sensitive(self) -> None:
        """Test that similarity is case sensitive."""
        result = utils.str_similarity("Hello", "hello")
        assert 0.0 < result < 1.0

    def test_str_similarity_unicode_strings(self) -> None:
        """Test similarity with unicode strings."""
        result = utils.str_similarity("hello", "hëllo")
        assert 0.0 < result < 1.0

    def test_str_similarity_exception_handling(self) -> None:
        """Test exception handling in similarity calculation."""
        with patch("arklex.utils.utils.Levenshtein") as mock_levenshtein:
            mock_levenshtein.distance.side_effect = Exception("Test error")
            result = utils.str_similarity("hello", "world")
            assert result == 0.0


class TestFormatChatHistory:
    """Test cases for format_chat_history function."""

    def test_format_chat_history_basic(self) -> None:
        """Test basic chat history formatting."""
        chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = utils.format_chat_history(chat_history)
        expected = "user: Hello\nassistant: Hi there"
        assert result == expected

    def test_format_chat_history_empty_list(self) -> None:
        """Test formatting empty chat history."""
        chat_history = []
        result = utils.format_chat_history(chat_history)
        assert result == ""

    def test_format_chat_history_single_message(self) -> None:
        """Test formatting single message."""
        chat_history = [{"role": "user", "content": "Hello"}]
        result = utils.format_chat_history(chat_history)
        assert result == "user: Hello"

    def test_format_chat_history_with_empty_content(self) -> None:
        """Test formatting with empty content."""
        chat_history = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "Response"},
        ]
        result = utils.format_chat_history(chat_history)
        expected = "user: \nassistant: Response"
        assert result == expected

    def test_format_chat_history_multiple_messages(self) -> None:
        """Test formatting multiple messages."""
        chat_history = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
        ]
        result = utils.format_chat_history(chat_history)
        expected = "user: First\nassistant: Second\nuser: Third"
        assert result == expected
