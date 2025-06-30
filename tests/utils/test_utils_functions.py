"""Tests for utility functions module.

This module tests the utility functions used throughout the Arklex framework,
including text processing, JSON handling, and chat history formatting.
"""

from unittest.mock import patch

import pytest

import arklex.utils.utils as utils


class TestChunkString:
    """Test cases for chunk_string function."""

    def test_chunk_string_from_end(self) -> None:
        """Test chunking from end of string."""
        text = "This is a test string with multiple words"
        result = utils.chunk_string(text, "cl100k_base", 10, from_end=True)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_chunk_string_from_start(self) -> None:
        """Test chunking from start of string."""
        text = "This is a test string with multiple words"
        result = utils.chunk_string(text, "cl100k_base", 10, from_end=False)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_chunk_string_short_text(self) -> None:
        """Test chunking short text that doesn't need truncation."""
        text = "Short text"
        result = utils.chunk_string(text, "cl100k_base", 100, from_end=True)
        assert result == text

    def test_chunk_string_empty_text(self) -> None:
        """Test chunking empty text."""
        text = ""
        result = utils.chunk_string(text, "cl100k_base", 10, from_end=True)
        assert result == ""

    def test_chunk_string_zero_max_length(self) -> None:
        """Test chunking with zero max length."""
        text = "This is a test string"
        result = utils.chunk_string(text, "cl100k_base", 0, from_end=True)
        assert result == text  # Returns original text when max_length is 0

    def test_chunk_string_different_tokenizer(self) -> None:
        """Test chunking with different tokenizer."""
        text = "This is a test string"
        result = utils.chunk_string(text, "gpt2", 10, from_end=True)
        assert isinstance(result, str)

    def test_chunk_string_unicode_text(self) -> None:
        """Test chunking unicode text."""
        text = "Hello 世界 🌍"
        result = utils.chunk_string(text, "cl100k_base", 10, from_end=True)
        assert isinstance(result, str)


class TestNormalize:
    """Test cases for normalize function."""

    def test_normalize_positive_numbers(self) -> None:
        """Test normalizing positive numbers."""
        numbers = [1.0, 2.0, 3.0, 4.0]
        result = utils.normalize(numbers)
        assert len(result) == 4
        assert sum(result) == pytest.approx(1.0, rel=1e-10)
        assert all(0 <= x <= 1 for x in result)

    def test_normalize_mixed_numbers(self) -> None:
        """Test normalizing mixed positive and negative numbers."""
        numbers = [1.0, -2.0, 3.0, -4.0]
        result = utils.normalize(numbers)
        assert len(result) == 4
        assert sum(result) == pytest.approx(1.0, rel=1e-10)

    def test_normalize_single_number(self) -> None:
        """Test normalizing single number."""
        numbers = [5.0]
        result = utils.normalize(numbers)
        assert result == [1.0]

    def test_normalize_empty_list(self) -> None:
        """Test normalizing empty list."""
        numbers = []
        result = utils.normalize(numbers)
        assert result == []

    def test_normalize_all_zeros(self) -> None:
        """Test normalizing list of zeros."""
        numbers = [0.0, 0.0, 0.0]
        with pytest.raises(ZeroDivisionError):
            utils.normalize(numbers)

    def test_normalize_integers(self) -> None:
        """Test normalizing integers."""
        numbers = [1, 2, 3, 4]
        result = utils.normalize(numbers)
        assert all(isinstance(x, float) for x in result)
        assert sum(result) == pytest.approx(1.0, rel=1e-10)


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


class TestPostprocessJson:
    """Test cases for postprocess_json function."""

    def test_postprocess_json_valid_json(self) -> None:
        """Test processing valid JSON."""
        raw_code = '{"key": "value", "number": 42}'
        result = utils.postprocess_json(raw_code)
        assert result == {"key": "value", "number": 42}

    def test_postprocess_json_with_comments(self) -> None:
        """Test processing JSON with comments."""
        raw_code = """
        // This is a comment
        {
            "key": "value"
        }
        // Another comment
        """
        result = utils.postprocess_json(raw_code)
        assert result == {"key": "value"}

    def test_postprocess_json_with_extra_text(self) -> None:
        """Test processing JSON with extra text."""
        raw_code = """
        Some text before
        {"key": "value"}
        Some text after
        """
        result = utils.postprocess_json(raw_code)
        assert result == {"key": "value"}

    def test_postprocess_json_invalid_json(self) -> None:
        """Test processing invalid JSON."""
        raw_code = '{"key": "value"'  # Missing closing brace
        result = utils.postprocess_json(raw_code)
        assert result is None

    def test_postprocess_json_empty_string(self) -> None:
        """Test processing empty string."""
        raw_code = ""
        result = utils.postprocess_json(raw_code)
        assert result is None

    def test_postprocess_json_no_valid_lines(self) -> None:
        """Test processing with no valid JSON lines."""
        raw_code = "This is not JSON at all"
        result = utils.postprocess_json(raw_code)
        assert result is None

    def test_postprocess_json_array(self) -> None:
        """Test processing JSON array."""
        raw_code = '[1, 2, 3, "test"]'
        result = utils.postprocess_json(raw_code)
        assert result == [1, 2, 3, "test"]

    def test_postprocess_json_nested_structure(self) -> None:
        """Test processing nested JSON structure."""
        raw_code = """
        {
            "outer": {
                "inner": "value",
                "array": [1, 2, 3]
            }
        }
        """
        result = utils.postprocess_json(raw_code)
        assert result == {"outer": {"inner": "value", "array": [1, 2, 3]}}


class TestTruncateString:
    """Test cases for truncate_string function."""

    def test_truncate_string_no_truncation_needed(self) -> None:
        """Test string that doesn't need truncation."""
        text = "Short text"
        result = utils.truncate_string(text, max_length=20)
        assert result == text

    def test_truncate_string_truncation_needed(self) -> None:
        """Test string that needs truncation."""
        text = "This is a very long text that needs to be truncated"
        result = utils.truncate_string(text, max_length=20)
        assert len(result) == 23  # 20 chars + "..."
        assert result.endswith("...")

    def test_truncate_string_exact_length(self) -> None:
        """Test string exactly at max length."""
        text = "Exactly twenty chars"
        result = utils.truncate_string(text, max_length=20)
        assert result == text

    def test_truncate_string_custom_max_length(self) -> None:
        """Test truncation with custom max length."""
        text = "This is a test"
        result = utils.truncate_string(text, max_length=10)
        assert len(result) == 13  # 10 chars + "..."
        assert result.endswith("...")

    def test_truncate_string_empty_string(self) -> None:
        """Test truncating empty string."""
        text = ""
        result = utils.truncate_string(text, max_length=10)
        assert result == ""

    def test_truncate_string_zero_max_length(self) -> None:
        """Test truncating with zero max length."""
        text = "Any text"
        result = utils.truncate_string(text, max_length=0)
        assert result == "..."

    def test_truncate_string_unicode_text(self) -> None:
        """Test truncating unicode text."""
        text = "Hello 世界 🌍"
        result = utils.truncate_string(text, max_length=10)
        # The text is shorter than max_length, so it shouldn't be truncated
        assert result == text


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


class TestFormatTruncatedChatHistory:
    """Test cases for format_truncated_chat_history function."""

    def test_format_truncated_chat_history_basic(self) -> None:
        """Test basic truncated chat history formatting."""
        chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = utils.format_truncated_chat_history(chat_history, max_length=10)
        assert "user: Hello" in result
        assert "assistant: Hi there" in result

    def test_format_truncated_chat_history_long_content(self) -> None:
        """Test formatting with long content that needs truncation."""
        chat_history = [
            {
                "role": "user",
                "content": "This is a very long message that should be truncated",
            },
            {"role": "assistant", "content": "Short response"},
        ]
        result = utils.format_truncated_chat_history(chat_history, max_length=20)
        assert result.count("...") >= 1

    def test_format_truncated_chat_history_empty_list(self) -> None:
        """Test formatting empty chat history."""
        chat_history = []
        result = utils.format_truncated_chat_history(chat_history, max_length=10)
        assert result == ""

    def test_format_truncated_chat_history_custom_max_length(self) -> None:
        """Test formatting with custom max length."""
        chat_history = [
            {"role": "user", "content": "Long message here"},
            {"role": "assistant", "content": "Another long message"},
        ]
        result = utils.format_truncated_chat_history(chat_history, max_length=5)
        assert result.count("...") >= 2

    def test_format_truncated_chat_history_none_content(self) -> None:
        """Test formatting with None content."""
        chat_history = [
            {"role": "user", "content": None},
            {"role": "assistant", "content": "Response"},
        ]
        result = utils.format_truncated_chat_history(chat_history, max_length=10)
        assert "user: " in result
        assert "assistant: Response" in result

    def test_format_truncated_chat_history_empty_content(self) -> None:
        """Test formatting with empty content."""
        chat_history = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "Response"},
        ]
        result = utils.format_truncated_chat_history(chat_history, max_length=10)
        assert "user: " in result
        assert "assistant: Response" in result
