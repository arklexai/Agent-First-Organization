"""Tests for utility functions module.

This module tests the utility functions used throughout the Arklex framework,
including text processing, JSON handling, and chat history formatting.
"""

import arklex.utils.utils as utils


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
