"""Tests for the chatgpt_utils module."""

import pytest
from unittest.mock import Mock, patch

from arklex.evaluation.chatgpt_utils import (
    chatgpt_chatbot,
    query_chatbot,
    filter_convo,
    flip_hist,
    format_chat_history_str,
    flip_hist_content_only,
)


class TestChatGPTUtils:
    """Test cases for chatgpt_utils module."""

    @pytest.fixture
    def mock_client(self) -> Mock:
        """Create a mock client for testing."""
        client = Mock()
        # Simulate OpenAI API response structure
        mock_choice = Mock()
        mock_choice.message = Mock(content="Test response")
        client.chat.completions.create.return_value = Mock(choices=[mock_choice])
        return client

    @patch("arklex.evaluation.chatgpt_utils.create_client")
    def test_chatgpt_chatbot_with_messages(
        self, mock_create_client: Mock, mock_client: Mock
    ) -> None:
        """Test chatgpt_chatbot with messages list."""
        # Setup
        mock_create_client.return_value = mock_client
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        # Execute
        result = chatgpt_chatbot(messages, client=mock_client)

        # Assert
        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch("arklex.evaluation.chatgpt_utils.create_client")
    def test_chatgpt_chatbot_with_string(
        self, mock_create_client: Mock, mock_client: Mock
    ) -> None:
        """Test chatgpt_chatbot with string input."""
        # Setup
        mock_create_client.return_value = mock_client
        prompt = "Hello, how are you?"

        # Execute
        result = chatgpt_chatbot(prompt, client=mock_client)

        # Assert
        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch("arklex.evaluation.chatgpt_utils.create_client")
    def test_chatgpt_chatbot_with_model_parameter(
        self, mock_create_client: Mock, mock_client: Mock
    ) -> None:
        """Test chatgpt_chatbot with model parameter."""
        # Setup
        mock_create_client.return_value = mock_client
        messages = [{"role": "user", "content": "Hello"}]

        # Execute
        result = chatgpt_chatbot(messages, model="gpt-3.5-turbo", client=mock_client)

        # Assert
        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch("arklex.evaluation.chatgpt_utils.create_client")
    def test_query_chatbot(self, mock_create_client: Mock, mock_client: Mock) -> None:
        """Test query_chatbot function."""
        # Setup
        mock_create_client.return_value = mock_client
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?", "intent": "question"},
            {"role": "assistant", "content": "I'm good"},
        ]

        # Mock the requests.post call
        with patch("arklex.evaluation.chatgpt_utils.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": "Test response"}
            mock_post.return_value = mock_response

            # Execute - query_chatbot requires model_api, history, params, env_config
            result = query_chatbot(
                model_api="http://test-api.com",
                history=conversation,
                params={"param1": "value1"},
                env_config={"workers": [], "tools": []},
            )

            # Assert
            assert isinstance(result, dict)
            assert "response" in result

    def test_filter_convo(self) -> None:
        """Test filter_convo function."""
        # Setup
        conversation = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?\nThis is extra content"},
            {"role": "assistant", "content": "I'm good"},
        ]

        # Execute
        result = filter_convo(conversation)

        # Assert - filter_convo skips first 2 turns and truncates user messages at \n
        assert len(result) == 3  # The last 3 turns (after skipping first 2)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hi there"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "How are you?"  # Truncated at \n
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "I'm good"

    def test_flip_hist(self) -> None:
        """Test flip_hist function."""
        # Setup
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good"},
        ]

        # Execute
        result = flip_hist(conversation)

        # Assert
        assert len(result) == 4
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "user"

    def test_format_chat_history_str(self) -> None:
        """Test format_chat_history_str function."""
        # Setup
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good"},
        ]

        # Execute
        result = format_chat_history_str(conversation)

        # Assert - format uses uppercase role names
        assert isinstance(result, str)
        assert "USER: Hello" in result
        assert "ASSISTANT: Hi there" in result

    def test_flip_hist_content_only(self) -> None:
        """Test flip_hist_content_only function."""
        # Setup
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good"},
        ]

        # Execute
        result = flip_hist_content_only(conversation)

        # Assert
        assert len(result) == 4
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "user"

    @patch("arklex.evaluation.chatgpt_utils.create_client")
    def test_chatgpt_chatbot_error_handling(
        self, mock_create_client: Mock, mock_client: Mock
    ) -> None:
        """Test chatgpt_chatbot error handling."""
        # Setup
        mock_create_client.return_value = mock_client
        # Simulate an exception in the OpenAI client
        mock_client.chat.completions.create.side_effect = Exception("API error")
        conversation = [
            {"content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        # Execute and assert
        with pytest.raises(Exception):
            chatgpt_chatbot(conversation, client=mock_client)

    def test_filter_convo_empty_conversation(self) -> None:
        """Test filter_convo with empty conversation."""
        # Setup
        conversation = []

        # Execute
        result = filter_convo(conversation)

        # Assert
        assert result == []

    def test_format_chat_history_str_empty_conversation(self) -> None:
        """Test format_chat_history_str with empty conversation."""
        # Setup
        conversation = []

        # Execute
        result = format_chat_history_str(conversation)

        # Assert
        assert result == ""

    def test_flip_hist_empty_conversation(self) -> None:
        """Test flip_hist with empty conversation."""
        # Setup
        conversation = []

        # Execute
        result = flip_hist(conversation)

        # Assert
        assert result == []

    def test_flip_hist_content_only_empty_conversation(self) -> None:
        """Test flip_hist_content_only with empty conversation."""
        # Setup
        conversation = []

        # Execute
        result = flip_hist_content_only(conversation)

        # Assert
        assert result == []

    def test_filter_convo_with_missing_role(self) -> None:
        """Test filter_convo with messages missing role field."""
        # Setup
        conversation = [
            {"content": "Hello"},  # Missing role
            {"role": "assistant", "content": "Hi there"},
        ]

        # Execute
        result = filter_convo(conversation)

        # Assert - messages without role are filtered out
        assert len(result) == 0  # No messages with intent

    @patch("arklex.evaluation.chatgpt_utils.OpenAI")
    @patch("arklex.evaluation.chatgpt_utils.anthropic.Anthropic")
    @patch(
        "arklex.evaluation.chatgpt_utils.MODEL",
        {"llm_provider": "openai", "model_type_or_path": "gpt-3.5-turbo"},
    )
    def test_create_client_openai(
        self, mock_anthropic: Mock, mock_openai: Mock
    ) -> None:
        """Test create_client returns OpenAI client when provider is openai."""
        from arklex.evaluation.chatgpt_utils import create_client
        import os

        os.environ["OPENAI_API_KEY"] = "test"
        client = create_client()
        assert client == mock_openai.return_value

    @patch("arklex.evaluation.chatgpt_utils.OpenAI")
    @patch("arklex.evaluation.chatgpt_utils.anthropic.Anthropic")
    @patch(
        "arklex.evaluation.chatgpt_utils.MODEL",
        {"llm_provider": "anthropic", "model_type_or_path": "claude"},
    )
    def test_create_client_anthropic(
        self, mock_anthropic: Mock, mock_openai: Mock
    ) -> None:
        """Test create_client returns Anthropic client when provider is anthropic."""
        from arklex.evaluation.chatgpt_utils import create_client

        client = create_client()
        assert client == mock_anthropic.return_value

    def test_flip_hist_content_only_empty_conversation(self) -> None:
        """Test flip_hist_content_only with empty conversation returns empty list."""
        from arklex.evaluation.chatgpt_utils import flip_hist_content_only

        result = flip_hist_content_only([])
        assert result == []

    def test_flip_hist_empty_conversation(self) -> None:
        """Test flip_hist with empty conversation returns empty list."""
        from arklex.evaluation.chatgpt_utils import flip_hist

        result = flip_hist([])
        assert result == []

    def test_format_chat_history_str_empty_conversation(self) -> None:
        """Test format_chat_history_str with empty conversation returns empty string."""
        from arklex.evaluation.chatgpt_utils import format_chat_history_str

        result = format_chat_history_str([])
        assert result == ""

    def test_filter_convo_empty_conversation(self) -> None:
        """Test filter_convo with empty conversation returns empty list."""
        from arklex.evaluation.chatgpt_utils import filter_convo

        result = filter_convo([])
        assert result == []

    def test_filter_convo_with_missing_role(self) -> None:
        """Test filter_convo with missing role in conversation."""
        from arklex.evaluation.chatgpt_utils import filter_convo

        conversation = [{"content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        result = filter_convo(conversation)
        assert isinstance(result, list)

    @patch("arklex.evaluation.chatgpt_utils.chatgpt_chatbot")
    def test_adjust_goal(self, mock_chatgpt_chatbot: Mock) -> None:
        """Test adjust_goal returns a string."""
        from arklex.evaluation.chatgpt_utils import adjust_goal

        mock_chatgpt_chatbot.return_value = "adjusted goal"
        result = adjust_goal("doc content", "goal")
        assert isinstance(result, str)
        assert result == "adjusted goal"

    @patch("arklex.evaluation.chatgpt_utils.OpenAI")
    @patch("arklex.evaluation.chatgpt_utils.anthropic.Anthropic")
    @patch(
        "arklex.evaluation.chatgpt_utils.MODEL",
        {"llm_provider": "openai", "model_type_or_path": "gpt-3.5-turbo"},
    )
    def test_generate_goal(self, mock_anthropic: Mock, mock_openai: Mock) -> None:
        """Test generate_goal returns a string."""
        from arklex.evaluation.chatgpt_utils import generate_goal

        client = mock_openai.return_value
        client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="goal"))]
        )
        result = generate_goal("doc content", client)
        assert isinstance(result, str)

    @patch("arklex.evaluation.chatgpt_utils.OpenAI")
    @patch("arklex.evaluation.chatgpt_utils.anthropic.Anthropic")
    @patch(
        "arklex.evaluation.chatgpt_utils.MODEL",
        {"llm_provider": "openai", "model_type_or_path": "gpt-3.5-turbo"},
    )
    def test_generate_goals(self, mock_anthropic: Mock, mock_openai: Mock) -> None:
        """Test generate_goals returns a list of strings."""
        from arklex.evaluation.chatgpt_utils import generate_goals

        client = mock_openai.return_value
        client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="goal"))]
        )
        documents = [{"content": "doc1"}, {"content": "doc2"}]
        params = {"num_goals": 2}
        result = generate_goals(documents, params, client)
        assert isinstance(result, list)
        assert all(isinstance(goal, str) for goal in result)

    @patch("arklex.evaluation.chatgpt_utils.OpenAI")
    @patch("arklex.evaluation.chatgpt_utils.anthropic.Anthropic")
    @patch(
        "arklex.evaluation.chatgpt_utils.MODEL",
        {"llm_provider": "gemini", "model_type_or_path": "gemini-pro"},
    )
    def test_create_client_gemini(
        self, mock_anthropic: Mock, mock_openai: Mock
    ) -> None:
        """Test create_client with Gemini provider."""
        # Mock environment variables
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"}):
            from arklex.evaluation.chatgpt_utils import create_client

            client = create_client()

            # Verify OpenAI client was created with Gemini base URL
            mock_openai.assert_called_once_with(
                api_key="test_key",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                organization=None,
            )

    @patch("arklex.evaluation.chatgpt_utils.create_client")
    def test_chatgpt_chatbot_anthropic_with_system_message(
        self, mock_create_client: Mock
    ) -> None:
        """Test chatgpt_chatbot with Anthropic provider and system message."""
        # Mock the MODEL to use anthropic
        with patch(
            "arklex.evaluation.chatgpt_utils.MODEL",
            {"llm_provider": "anthropic", "model_type_or_path": "claude-3-sonnet"},
        ):
            # Create mock Anthropic client
            mock_client = Mock()
            mock_response = Mock()
            mock_response.content = [Mock(text="Test response")]
            mock_client.messages.create.return_value = mock_response
            mock_create_client.return_value = mock_client

            # Test with system message
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ]

            from arklex.evaluation.chatgpt_utils import chatgpt_chatbot

            result = chatgpt_chatbot(messages, client=mock_client)

            # Verify the correct kwargs were passed to Anthropic
            mock_client.messages.create.assert_called_once_with(
                model="claude-3-sonnet",
                messages=[
                    {"role": "user", "content": "Hello"}
                ],  # System message filtered out
                temperature=0.1,
                max_tokens=1024,
                system="You are a helpful assistant",
            )
            assert result == "Test response"

    @patch("arklex.evaluation.chatgpt_utils.create_client")
    def test_chatgpt_chatbot_anthropic_without_system_message(
        self, mock_create_client: Mock
    ) -> None:
        """Test chatgpt_chatbot with Anthropic provider without system message."""
        # Mock the MODEL to use anthropic
        with patch(
            "arklex.evaluation.chatgpt_utils.MODEL",
            {"llm_provider": "anthropic", "model_type_or_path": "claude-3-sonnet"},
        ):
            # Create mock Anthropic client
            mock_client = Mock()
            mock_response = Mock()
            mock_response.content = [Mock(text="Test response")]
            mock_client.messages.create.return_value = mock_response
            mock_create_client.return_value = mock_client

            # Test without system message
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]

            from arklex.evaluation.chatgpt_utils import chatgpt_chatbot

            result = chatgpt_chatbot(messages, client=mock_client)

            # Verify the correct kwargs were passed to Anthropic
            mock_client.messages.create.assert_called_once_with(
                model="claude-3-sonnet",
                messages=messages,  # All messages included
                temperature=0.1,
                max_tokens=1024,
            )
            assert result == "Test response"

    def test_filter_convo_with_filter_turns_false(self) -> None:
        """Test filter_convo with filter_turns=False."""
        conversation = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?\nThis is extra content"},
            {"role": "assistant", "content": "I'm good"},
            {"no_role": "This message has no role"},  # Message without role
        ]

        from arklex.evaluation.chatgpt_utils import filter_convo

        result = filter_convo(conversation, filter_turns=False)

        # Should include the message without role when filter_turns=False
        assert len(result) == 4  # The last 4 turns (after skipping first 2)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hi there"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "How are you?"  # Truncated at \n
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "I'm good"
        assert result[3]["no_role"] == "This message has no role"

    def test_filter_convo_with_delimiter_not_found(self) -> None:
        """Test filter_convo when delimiter is not found in user message."""
        conversation = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you? No delimiter here"},
            {"role": "assistant", "content": "I'm good"},
        ]

        from arklex.evaluation.chatgpt_utils import filter_convo

        result = filter_convo(conversation, delim="\n")

        # Should keep the full message when delimiter is not found
        assert len(result) == 3  # The last 3 turns (after skipping first 2)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hi there"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "How are you? No delimiter here"  # Full message
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "I'm good"

    def test_filter_convo_with_custom_delimiter(self) -> None:
        """Test filter_convo with custom delimiter."""
        conversation = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?|This is extra content"},
            {"role": "assistant", "content": "I'm good"},
        ]

        from arklex.evaluation.chatgpt_utils import filter_convo

        result = filter_convo(conversation, delim="|")

        # Should truncate at custom delimiter
        assert len(result) == 3  # The last 3 turns (after skipping first 2)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hi there"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "How are you?"  # Truncated at |
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "I'm good"

    def test_flip_hist_with_message_without_role(self) -> None:
        """Test flip_hist with message that has no role field."""
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"no_role": "This message has no role"},  # Message without role
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good"},
        ]

        from arklex.evaluation.chatgpt_utils import flip_hist

        result = flip_hist(conversation)

        # Should preserve messages without role and flip others
        assert len(result) == 5
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "user"
        assert result[2]["no_role"] == "This message has no role"  # Preserved
        assert result[3]["role"] == "assistant"
        assert result[4]["role"] == "user"

    def test_flip_hist_content_only_with_system_message(self) -> None:
        """Test flip_hist_content_only with system message."""
        conversation = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good"},
        ]

        from arklex.evaluation.chatgpt_utils import flip_hist_content_only

        result = flip_hist_content_only(conversation)

        # Should skip system message and flip others
        assert len(result) == 4  # System message skipped
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hello"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Hi there"
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "How are you?"
        assert result[3]["role"] == "user"
        assert result[3]["content"] == "I'm good"

    def test_create_client_with_openai_org_id(self) -> None:
        """Test create_client with OpenAI organization ID."""
        with patch.dict(
            "os.environ", {"OPENAI_API_KEY": "test_key", "OPENAI_ORG_ID": "org_test"}
        ):
            with patch(
                "arklex.evaluation.chatgpt_utils.MODEL",
                {"llm_provider": "openai", "model_type_or_path": "gpt-3.5-turbo"},
            ):
                with patch("arklex.evaluation.chatgpt_utils.OpenAI") as mock_openai:
                    from arklex.evaluation.chatgpt_utils import create_client

                    client = create_client()

                    # Verify OpenAI client was created with organization
                    mock_openai.assert_called_once_with(
                        api_key="test_key",
                        base_url=None,
                        organization="org_test",
                    )

    def test_create_client_without_openai_org_id(self) -> None:
        """Test create_client without OpenAI organization ID."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}, clear=True):
            with patch(
                "arklex.evaluation.chatgpt_utils.MODEL",
                {"llm_provider": "openai", "model_type_or_path": "gpt-3.5-turbo"},
            ):
                with patch("arklex.evaluation.chatgpt_utils.OpenAI") as mock_openai:
                    from arklex.evaluation.chatgpt_utils import create_client

                    client = create_client()

                    # Verify OpenAI client was created without organization
                    mock_openai.assert_called_once_with(
                        api_key="test_key",
                        base_url=None,
                        organization=None,
                    )

    def test_main_function(self):
        import sys
        import types
        from unittest.mock import patch, Mock
        from arklex.evaluation import chatgpt_utils

        # Mock the get_documents module and its functions
        mock_documents = [{"content": "test content"}]
        mock_get_documents = types.ModuleType("get_documents")
        mock_get_documents.get_all_documents = lambda: mock_documents
        mock_get_documents.filter_documents = lambda docs: docs
        sys.modules["get_documents"] = mock_get_documents

        with (
            patch(
                "arklex.evaluation.chatgpt_utils.create_client", return_value=Mock()
            ) as mock_create_client,
            patch(
                "arklex.evaluation.chatgpt_utils.generate_goals",
                return_value=["test goal"],
            ) as mock_generate_goals,
            patch("builtins.print") as mock_print,
        ):
            chatgpt_utils.main()

        mock_create_client.assert_called_once()
        mock_generate_goals.assert_called_once_with(
            mock_documents, {"num_goals": 1}, mock_create_client.return_value
        )
        mock_print.assert_called_once_with(["test goal"])
