#!/usr/bin/env python3
"""Simple test runner for chatgpt_utils module."""

import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arklex.evaluation.chatgpt_utils import (
    _convert_messages_to_gemini_format,
    chatgpt_chatbot,
    filter_convo,
    flip_hist,
    flip_hist_content_only,
    format_chat_history_str,
)


def test_convert_messages_to_gemini_format_basic() -> None:
    """Test _convert_messages_to_gemini_format with basic messages."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"},
    ]

    result = _convert_messages_to_gemini_format(messages)

    assert len(result) == 3
    assert result[0]["role"] == "user"
    assert result[0]["parts"][0]["text"] == "Hello"
    assert result[1]["role"] == "model"
    assert result[1]["parts"][0]["text"] == "Hi there"
    assert result[2]["role"] == "user"
    assert result[2]["parts"][0]["text"] == "How are you?"


def test_convert_messages_to_gemini_format_with_system_message() -> None:
    """Test _convert_messages_to_gemini_format with system message."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    result = _convert_messages_to_gemini_format(messages)

    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[0]["parts"][0]["text"] == "You are a helpful assistant\n\nHello"
    assert result[1]["role"] == "model"
    assert result[1]["parts"][0]["text"] == "Hi there"


def test_convert_messages_to_gemini_format_empty() -> None:
    """Test _convert_messages_to_gemini_format with empty messages."""
    messages = []
    result = _convert_messages_to_gemini_format(messages)
    assert result == []


def test_convert_messages_to_gemini_format_only_system() -> None:
    """Test _convert_messages_to_gemini_format with only system message."""
    messages = [{"role": "system", "content": "You are a helpful assistant"}]
    result = _convert_messages_to_gemini_format(messages)
    assert result == []


def test_chatgpt_chatbot_google_provider() -> None:
    """Test chatgpt_chatbot with Google Gemini provider."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.text = "gemini result"
    mock_client.generate_content.return_value = mock_response

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    # Mock MODEL to use google provider
    with patch(
        "arklex.evaluation.chatgpt_utils.MODEL",
        {"llm_provider": "google", "model_type_or_path": "gemini-pro"},
    ):
        result = chatgpt_chatbot(messages, client=mock_client)

        assert result == "gemini result"
        mock_client.generate_content.assert_called_once()
        call_args = mock_client.generate_content.call_args
        assert len(call_args[0]) == 1  # First argument is the messages
        assert call_args[1]["generation_config"]["temperature"] == 0.1
        assert call_args[1]["generation_config"]["max_output_tokens"] == 1024


def test_chatgpt_chatbot_google_provider_with_system_message() -> None:
    """Test chatgpt_chatbot with Google Gemini provider and system message."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.text = "gemini result with system"
    mock_client.generate_content.return_value = mock_response

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    # Mock MODEL to use google provider
    with patch(
        "arklex.evaluation.chatgpt_utils.MODEL",
        {"llm_provider": "google", "model_type_or_path": "gemini-pro"},
    ):
        result = chatgpt_chatbot(messages, client=mock_client)

        assert result == "gemini result with system"
        mock_client.generate_content.assert_called_once()


def test_chatgpt_chatbot_unsupported_provider() -> None:
    """Test chatgpt_chatbot with unsupported provider raises ValueError."""
    mock_client = Mock()
    messages = [{"role": "user", "content": "Hello"}]

    with (
        patch(
            "arklex.evaluation.chatgpt_utils.MODEL",
            {"llm_provider": "unsupported", "model_type_or_path": "test-model"},
        ),
        pytest.raises(ValueError, match="Unsupported LLM provider: unsupported"),
    ):
        chatgpt_chatbot(messages, client=mock_client)


def test_flip_hist_content_only() -> None:
    """Test flip_hist_content_only function."""
    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm good"},
    ]

    result = flip_hist_content_only(conversation)

    assert len(result) == 4
    assert result[0]["role"] == "assistant"
    assert result[1]["role"] == "user"
    assert result[2]["role"] == "assistant"
    assert result[3]["role"] == "user"


def test_flip_hist() -> None:
    """Test flip_hist function."""
    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm good"},
    ]

    result = flip_hist(conversation)

    assert len(result) == 4
    assert result[0]["role"] == "assistant"
    assert result[1]["role"] == "user"
    assert result[2]["role"] == "assistant"
    assert result[3]["role"] == "user"


def test_format_chat_history_str() -> None:
    """Test format_chat_history_str function."""
    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    result = format_chat_history_str(conversation)

    assert isinstance(result, str)
    assert "USER: Hello" in result
    assert "ASSISTANT: Hi there" in result


def test_filter_convo() -> None:
    """Test filter_convo function."""
    conversation = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?\nThis is extra content"},
        {"role": "assistant", "content": "I'm good"},
    ]

    result = filter_convo(conversation)

    assert len(result) == 3
    assert result[0]["role"] == "assistant"
    assert result[0]["content"] == "Hi there"
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "How are you?"
    assert result[2]["role"] == "assistant"
    assert result[2]["content"] == "I'm good"


if __name__ == "__main__":
    # Run all tests
    test_functions = [
        test_convert_messages_to_gemini_format_basic,
        test_convert_messages_to_gemini_format_with_system_message,
        test_convert_messages_to_gemini_format_empty,
        test_convert_messages_to_gemini_format_only_system,
        test_chatgpt_chatbot_google_provider,
        test_chatgpt_chatbot_google_provider_with_system_message,
        test_chatgpt_chatbot_unsupported_provider,
        test_flip_hist_content_only,
        test_flip_hist,
        test_format_chat_history_str,
        test_filter_convo,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
