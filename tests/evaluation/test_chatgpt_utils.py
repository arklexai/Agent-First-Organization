"""Tests for chatgpt_utils module."""

import pytest
from unittest.mock import patch, MagicMock
from arklex.evaluation.chatgpt_utils import (
    create_client,
    chatgpt_chatbot,
    filter_convo,
    flip_hist,
    flip_hist_content_only,
    format_chat_history_str,
    query_chatbot,
)


class TestCreateClient:
    """Test cases for create_client function."""

    @patch.dict("os.environ", {"OPENAI_ORG_ID": "test-org-id"})
    @patch("arklex.evaluation.chatgpt_utils.MODEL", {"llm_provider": "openai"})
    @patch("arklex.evaluation.chatgpt_utils.OpenAI")
    def test_create_client_openai(self, mock_openai):
        """Test creating OpenAI client."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        result = create_client()
        
        assert result == mock_client
        mock_openai.assert_called_once()

    @patch.dict("os.environ", {}, clear=True)
    @patch("arklex.evaluation.chatgpt_utils.MODEL", {"llm_provider": "openai"})
    @patch("arklex.evaluation.chatgpt_utils.OpenAI")
    def test_create_client_openai_no_org_id(self, mock_openai):
        """Test creating OpenAI client without org ID."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        result = create_client()
        
        assert result == mock_client
        mock_openai.assert_called_once()

    @patch("arklex.evaluation.chatgpt_utils.MODEL", {"llm_provider": "anthropic"})
    @patch("arklex.evaluation.chatgpt_utils.anthropic.Anthropic")
    def test_create_client_anthropic(self, mock_anthropic):
        """Test creating Anthropic client."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        result = create_client()
        
        assert result == mock_client
        mock_anthropic.assert_called_once()

    @patch("arklex.evaluation.chatgpt_utils.MODEL", {"llm_provider": "gemini"})
    @patch("arklex.evaluation.chatgpt_utils.GenerativeModel")
    def test_create_client_gemini(self, mock_gemini):
        """Test creating Gemini client."""
        mock_client = MagicMock()
        mock_gemini.return_value = mock_client
        
        result = create_client()
        
        assert result == mock_client
        mock_gemini.assert_called_once()

    @patch("arklex.evaluation.chatgpt_utils.MODEL", {"llm_provider": None})
    def test_create_client_no_provider(self):
        """Test creating client with no provider specified."""
        with pytest.raises(ValueError, match="llm_provider must be explicitly specified"):
            create_client()

    @patch("arklex.evaluation.chatgpt_utils.MODEL", {"llm_provider": "unsupported"})
    def test_create_client_unsupported_provider(self):
        """Test creating client with unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_client()


class TestChatgptChatbot:
    """Test cases for chatgpt_chatbot function."""

    @patch("arklex.evaluation.chatgpt_utils.create_client")
    def test_chatgpt_chatbot_success(self, mock_create_client):
        """Test successful chatbot interaction."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = "Test response"
        mock_create_client.return_value = mock_client
        
        result = chatgpt_chatbot("Test message", "test-model")
        
        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch("arklex.evaluation.chatgpt_utils.create_client")
    def test_chatgpt_chatbot_with_parameters(self, mock_create_client):
        """Test chatbot interaction with parameters."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = "Test response"
        mock_create_client.return_value = mock_client
        
        params = {"temperature": 0.7, "max_tokens": 100}
        result = chatgpt_chatbot("Test message", "test-model", params)
        
        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch("arklex.evaluation.chatgpt_utils.create_client")
    def test_chatgpt_chatbot_api_error(self, mock_create_client):
        """Test chatbot interaction with API error."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_create_client.return_value = mock_client
        
        result = chatgpt_chatbot("Test message", "test-model")
        
        assert result is None


class TestFilterConvo:
    """Test cases for filter_convo function."""

    def test_filter_convo_valid_conversation(self):
        """Test filtering valid conversation."""
        convo = {
            "conversation": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }
        result = filter_convo(convo)
        assert result is True

    def test_filter_convo_missing_conversation_key(self):
        """Test filtering conversation missing conversation key."""
        convo = {"other_key": "value"}
        result = filter_convo(convo)
        assert result is False

    def test_filter_convo_empty_conversation(self):
        """Test filtering empty conversation."""
        convo = {"conversation": []}
        result = filter_convo(convo)
        assert result is False

    def test_filter_convo_single_message(self):
        """Test filtering conversation with single message."""
        convo = {
            "conversation": [
                {"role": "user", "content": "Hello"},
            ]
        }
        result = filter_convo(convo)
        assert result is False

    def test_filter_convo_missing_role_or_content(self):
        """Test filtering conversation with missing role or content."""
        convo = {
            "conversation": [
                {"role": "user"},  # Missing content
                {"role": "assistant", "content": "Hi there"},
            ]
        }
        result = filter_convo(convo)
        assert result is False


class TestFlipHist:
    """Test cases for flip_hist function."""

    def test_flip_hist_basic(self):
        """Test basic history flipping."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = flip_hist(history)
        
        expected = [
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "Hello"},
        ]
        assert result == expected

    def test_flip_hist_empty_history(self):
        """Test flipping empty history."""
        history = []
        result = flip_hist(history)
        assert result == []

    def test_flip_hist_single_message(self):
        """Test flipping single message."""
        history = [{"role": "user", "content": "Hello"}]
        result = flip_hist(history)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_flip_hist_multiple_messages(self):
        """Test flipping multiple messages."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good"},
        ]
        result = flip_hist(history)
        
        expected = [
            {"role": "user", "content": "I'm good"},
            {"role": "assistant", "content": "How are you?"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        assert result == expected


class TestFlipHistContentOnly:
    """Test cases for flip_hist_content_only function."""

    def test_flip_hist_content_only_basic(self):
        """Test basic content-only flipping."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = flip_hist_content_only(history)
        
        expected = [
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "Hello"},
        ]
        assert result == expected

    def test_flip_hist_content_only_empty_history(self):
        """Test flipping empty history."""
        history = []
        result = flip_hist_content_only(history)
        assert result == []

    def test_flip_hist_content_only_single_message(self):
        """Test flipping single message."""
        history = [{"role": "user", "content": "Hello"}]
        result = flip_hist_content_only(history)
        assert result == [{"role": "user", "content": "Hello"}]


class TestFormatChatHistoryStr:
    """Test cases for format_chat_history_str function."""

    def test_format_chat_history_str_basic(self):
        """Test basic chat history formatting."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = format_chat_history_str(history)
        
        assert "Hello" in result
        assert "Hi there" in result
        assert isinstance(result, str)

    def test_format_chat_history_str_empty_history(self):
        """Test formatting empty history."""
        history = []
        result = format_chat_history_str(history)
        assert result == ""

    def test_format_chat_history_str_single_message(self):
        """Test formatting single message."""
        history = [{"role": "user", "content": "Hello"}]
        result = format_chat_history_str(history)
        
        assert "Hello" in result
        assert isinstance(result, str)


class TestQueryChatbot:
    """Test cases for query_chatbot function."""

    @patch("arklex.evaluation.chatgpt_utils.chatgpt_chatbot")
    def test_query_chatbot_success(self, mock_chatbot):
        """Test successful chatbot query."""
        mock_chatbot.return_value = "Test response"
        
        result = query_chatbot("Test message", "test-model", {})
        
        assert result == "Test response"
        mock_chatbot.assert_called_once()

    @patch("arklex.evaluation.chatgpt_utils.chatgpt_chatbot")
    def test_query_chatbot_with_parameters(self, mock_chatbot):
        """Test chatbot query with parameters."""
        mock_chatbot.return_value = "Test response"
        
        params = {"temperature": 0.7}
        result = query_chatbot("Test message", "test-model", params)
        
        assert result == "Test response"
        mock_chatbot.assert_called_once()

    @patch("arklex.evaluation.chatgpt_utils.chatgpt_chatbot")
    def test_query_chatbot_failure(self, mock_chatbot):
        """Test chatbot query failure."""
        mock_chatbot.return_value = None
        
        result = query_chatbot("Test message", "test-model", {})
        
        assert result is None
        mock_chatbot.assert_called_once()
