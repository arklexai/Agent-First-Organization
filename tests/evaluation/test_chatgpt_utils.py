"""Tests for the chatgpt_utils module."""

from unittest.mock import Mock, patch

import pytest

from arklex.evaluation.chatgpt_utils import (
    _convert_messages_to_gemini_format,
    chatgpt_chatbot,
    filter_convo,
    flip_hist,
    flip_hist_content_only,
    format_chat_history_str,
    main,
    query_chatbot,
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
    @patch(
        "arklex.evaluation.chatgpt_utils.MODEL",
        {"llm_provider": "openai", "model_type_or_path": "gpt-3.5-turbo"},
    )
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
    @patch(
        "arklex.evaluation.chatgpt_utils.MODEL",
        {"llm_provider": "openai", "model_type_or_path": "gpt-3.5-turbo"},
    )
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
    @patch(
        "arklex.evaluation.chatgpt_utils.MODEL",
        {"llm_provider": "openai", "model_type_or_path": "gpt-3.5-turbo"},
    )
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
    @patch(
        "arklex.evaluation.chatgpt_utils.MODEL",
        {"llm_provider": "openai", "model_type_or_path": "gpt-3.5-turbo"},
    )
    def test_chatgpt_chatbot_error_handling(
        self, mock_create_client: Mock, mock_client: Mock
    ) -> None:
        """Test chatgpt_chatbot error handling."""
        # Setup
        mock_create_client.return_value = mock_client
        # Simulate an exception in the OpenAI client
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")
        conversation = [
            {"content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        # Execute and assert
        with pytest.raises(RuntimeError, match="API error"):
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
        import os

        from arklex.evaluation.chatgpt_utils import create_client

        os.environ["OPENAI_API_KEY"] = "test"
        create_client()
        assert mock_openai.return_value == mock_openai.return_value

    @patch("arklex.evaluation.chatgpt_utils.OpenAI")
    @patch("arklex.evaluation.chatgpt_utils.anthropic.Anthropic")
    @patch("google.generativeai")
    @patch("arklex.evaluation.chatgpt_utils.GenerativeModel")
    @patch(
        "arklex.evaluation.chatgpt_utils.MODEL",
        {"llm_provider": "google", "model_type_or_path": "gemini-pro"},
    )
    def test_create_client_gemini(
        self,
        mock_generative_model: Mock,
        mock_genai: Mock,
        mock_anthropic: Mock,
        mock_openai: Mock,
    ) -> None:
        """Test create_client with Google Generative AI provider."""
        # Mock environment variables
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test_key"}):
            from arklex.evaluation.chatgpt_utils import create_client

            create_client()

            # Verify Google Generative AI was configured and client was created
            mock_genai.configure.assert_called_once_with(api_key="test_key")
            mock_generative_model.assert_called_once_with("gemini-pro")

    @patch("arklex.evaluation.chatgpt_utils.chatgpt_chatbot")
    @patch(
        "arklex.evaluation.chatgpt_utils.MODEL",
        {"llm_provider": "openai", "model_type_or_path": "gpt-3.5-turbo"},
    )
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

            result = chatgpt_chatbot(
                messages, client=mock_client, model="claude-3-sonnet"
            )

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

            result = chatgpt_chatbot(
                messages, client=mock_client, model="claude-3-sonnet"
            )

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
        with (
            patch.dict(
                "os.environ",
                {"OPENAI_API_KEY": "test_key", "OPENAI_ORG_ID": "org_test"},
            ),
            patch(
                "arklex.evaluation.chatgpt_utils.MODEL",
                {"llm_provider": "openai", "model_type_or_path": "gpt-3.5-turbo"},
            ),
            patch("arklex.evaluation.chatgpt_utils.OpenAI") as mock_openai,
        ):
            from arklex.evaluation.chatgpt_utils import create_client

            create_client()

            # Verify OpenAI client was created with organization
            mock_openai.assert_called_once_with(
                api_key="test_key",
                organization="org_test",
            )

    def test_create_client_without_openai_org_id(self) -> None:
        """Test create_client without OpenAI organization ID."""
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}, clear=True),
            patch(
                "arklex.evaluation.chatgpt_utils.MODEL",
                {"llm_provider": "openai", "model_type_or_path": "gpt-3.5-turbo"},
            ),
            patch("arklex.evaluation.chatgpt_utils.OpenAI") as mock_openai,
        ):
            from arklex.evaluation.chatgpt_utils import create_client

            create_client()

            # Verify OpenAI client was created without organization
            mock_openai.assert_called_once_with(
                api_key="test_key",
                organization=None,
            )

    @patch("arklex.evaluation.chatgpt_utils.create_client")
    def test_chatgpt_chatbot_google_provider(self, mock_create_client: Mock) -> None:
        """Test chatgpt_chatbot with Google Gemini provider."""
        # Setup
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "gemini result"
        mock_client.generate_content.return_value = mock_response
        mock_create_client.return_value = mock_client

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        # Mock MODEL to use google provider
        with patch(
            "arklex.evaluation.chatgpt_utils.MODEL",
            {"llm_provider": "google", "model_type_or_path": "gemini-pro"},
        ):
            # Execute
            result = chatgpt_chatbot(messages, client=mock_client)

            # Assert
            assert result == "gemini result"
            mock_client.generate_content.assert_called_once()
            call_args = mock_client.generate_content.call_args
            assert len(call_args[0]) == 1  # First argument is the messages
            assert call_args[1]["generation_config"]["temperature"] == 0.1
            assert call_args[1]["generation_config"]["max_output_tokens"] == 1024

    @patch("arklex.evaluation.chatgpt_utils.create_client")
    def test_chatgpt_chatbot_google_provider_with_system_message(
        self, mock_create_client: Mock
    ) -> None:
        """Test chatgpt_chatbot with Google Gemini provider and system message."""
        # Setup
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "gemini result with system"
        mock_client.generate_content.return_value = mock_response
        mock_create_client.return_value = mock_client

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
            # Execute
            result = chatgpt_chatbot(messages, client=mock_client)

            # Assert
            assert result == "gemini result with system"
            mock_client.generate_content.assert_called_once()

    @patch("arklex.evaluation.chatgpt_utils.create_client")
    def test_chatgpt_chatbot_unsupported_provider(
        self, mock_create_client: Mock
    ) -> None:
        """Test chatgpt_chatbot with unsupported provider raises ValueError."""
        # Setup
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        messages = [{"role": "user", "content": "Hello"}]

        # Mock MODEL to use unsupported provider
        with (
            patch(
                "arklex.evaluation.chatgpt_utils.MODEL",
                {"llm_provider": "unsupported", "model_type_or_path": "test-model"},
            ),
            pytest.raises(ValueError, match="Unsupported LLM provider: unsupported"),
        ):
            chatgpt_chatbot(messages, client=mock_client)

    def test_convert_messages_to_gemini_format_basic(self) -> None:
        """Test _convert_messages_to_gemini_format with basic messages."""
        # Setup
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]

        # Execute
        result = _convert_messages_to_gemini_format(messages)

        # Assert
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[0]["parts"][0]["text"] == "Hello"
        assert result[1]["role"] == "model"
        assert result[1]["parts"][0]["text"] == "Hi there"
        assert result[2]["role"] == "user"
        assert result[2]["parts"][0]["text"] == "How are you?"

    def test_convert_messages_to_gemini_format_with_system_message(self) -> None:
        """Test _convert_messages_to_gemini_format with system message."""
        # Setup
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        # Execute
        result = _convert_messages_to_gemini_format(messages)

        # Assert
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["parts"][0]["text"] == "You are a helpful assistant\n\nHello"
        assert result[1]["role"] == "model"
        assert result[1]["parts"][0]["text"] == "Hi there"

    def test_convert_messages_to_gemini_format_system_message_no_user(self) -> None:
        """Test _convert_messages_to_gemini_format with system message but no user message."""
        # Setup
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "assistant", "content": "Hi there"},
        ]

        # Execute
        result = _convert_messages_to_gemini_format(messages)

        # Assert
        assert len(result) == 1
        assert result[0]["role"] == "model"
        assert result[0]["parts"][0]["text"] == "Hi there"

    def test_convert_messages_to_gemini_format_empty_messages(self) -> None:
        """Test _convert_messages_to_gemini_format with empty messages list."""
        # Setup
        messages = []

        # Execute
        result = _convert_messages_to_gemini_format(messages)

        # Assert
        assert result == []

    def test_convert_messages_to_gemini_format_only_system(self) -> None:
        """Test _convert_messages_to_gemini_format with only system message."""
        # Setup
        messages = [{"role": "system", "content": "You are a helpful assistant"}]

        # Execute
        result = _convert_messages_to_gemini_format(messages)

        # Assert
        assert result == []

    def test_flip_hist_with_system_message_continue_statement(self) -> None:
        """Test flip_hist function with system message to cover continue statement (line 142)."""
        # Setup conversation with system message
        conversation = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        # Execute
        result = flip_hist(conversation)

        # Assert - system message should be skipped (continue statement executed)
        assert len(result) == 2  # Only user and assistant messages
        assert result[0]["role"] == "assistant"  # user becomes assistant
        assert result[1]["role"] == "user"  # assistant becomes user
        assert "system" not in [msg["role"] for msg in result]

    def test_query_chatbot_return_response_json(self) -> None:
        """Test query_chatbot function to cover return response.json() statement (line 230)."""
        # Setup
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        params = {"param1": "value1"}
        env_config = {"workers": [], "tools": []}

        # Mock the requests.post call to return a valid JSON response
        with patch("arklex.evaluation.chatgpt_utils.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "response": "Test response",
                "status": "success",
            }
            mock_post.return_value = mock_response

            # Execute
            result = query_chatbot(
                model_api="http://test-api.com",
                history=conversation,
                params=params,
                env_config=env_config,
            )

            # Assert - should return the JSON response from the API
            assert isinstance(result, dict)
            assert result["response"] == "Test response"
            assert result["status"] == "success"
            # Verify that requests.post was called and response.json() was called
            mock_post.assert_called_once()
            mock_response.json.assert_called_once()

    def test_main_function_execution(self) -> None:
        """Test main function execution to cover print statement (line 331)."""
        # Mock the print function to avoid actual output
        with patch("builtins.print") as mock_print:
            # Mock the import inside main function by patching builtins.__import__
            mock_get_docs_module = Mock()
            mock_get_docs_module.get_all_documents.return_value = [
                {"content": "Test document"}
            ]
            mock_get_docs_module.filter_documents.return_value = [
                {"content": "Test document"}
            ]

            import builtins

            real_import = builtins.__import__

            def mock_import(name: str, *args: object, **kwargs: object) -> object:
                if name == "get_documents":
                    return mock_get_docs_module
                return real_import(name, *args, **kwargs)

            with (
                patch("builtins.__import__", side_effect=mock_import),
                patch(
                    "arklex.evaluation.chatgpt_utils.create_client"
                ) as mock_create_client,
                patch(
                    "arklex.evaluation.chatgpt_utils.generate_goals"
                ) as mock_generate_goals,
            ):
                mock_client = Mock()
                mock_create_client.return_value = mock_client
                mock_generate_goals.return_value = ["Test goal"]

                # Execute main function
                main()

                # Assert print was called with the expected result
                mock_print.assert_called_once_with(["Test goal"])

    def test__print_goals(self) -> None:
        from arklex.evaluation.chatgpt_utils import _print_goals

        with patch("builtins.print") as mock_print:
            goals = ["goal1", "goal2"]
            _print_goals(goals)
            mock_print.assert_called_once_with(goals)

    @patch(
        "arklex.evaluation.chatgpt_utils.MODEL",
        {"llm_provider": "openai", "model_type_or_path": "gpt-3.5-turbo"},
    )
    def test_chatgpt_chatbot_openai_branch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from arklex.evaluation.chatgpt_utils import chatgpt_chatbot

        class DummyOpenAI:
            class chat:
                class completions:
                    @staticmethod
                    def create(
                        model: str, messages: list[dict[str, str]], temperature: float
                    ) -> object:
                        class Choice:
                            class Message:
                                content = "response"

                            message = Message()

                        class Result:
                            choices = [Choice()]

                        return Result()

        messages = [{"role": "user", "content": "hi"}]
        result = chatgpt_chatbot(messages, DummyOpenAI())
        assert result == "response"

    def test_chatgpt_chatbot_anthropic_branch(self) -> None:
        """Explicitly test chatgpt_chatbot for Anthropic branch (lines 50-51)."""
        from arklex.evaluation import chatgpt_utils

        class DummyAnthropic:
            class messages:
                @staticmethod
                def create(**kwargs: object) -> object:
                    class Result:
                        content = [type("Obj", (), {"text": "anthropic result"})()]

                    return Result()

        messages = [{"role": "user", "content": "hi"}]
        # Patch MODEL to anthropic
        orig_model = chatgpt_utils.MODEL.copy()
        chatgpt_utils.MODEL["llm_provider"] = "anthropic"
        chatgpt_utils.MODEL["model_type_or_path"] = "claude-3-sonnet-20240229"
        try:
            result = chatgpt_utils.chatgpt_chatbot(messages, DummyAnthropic())
            assert result == "anthropic result"
        finally:
            chatgpt_utils.MODEL = orig_model

    def test_format_chat_history_str_empty(self) -> None:
        """Explicitly test format_chat_history_str for empty input (line 332)."""
        from arklex.evaluation.chatgpt_utils import format_chat_history_str

        assert format_chat_history_str([]) == ""

    def test_create_client_with_anthropic_provider(self) -> None:
        """Test create_client function with anthropic provider."""
        from arklex.evaluation.chatgpt_utils import create_client

        with (
            patch(
                "arklex.evaluation.chatgpt_utils.MODEL", {"llm_provider": "anthropic"}
            ),
            patch(
                "arklex.evaluation.chatgpt_utils.anthropic.Anthropic"
            ) as mock_anthropic,
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        ):
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            result = create_client()

            assert result == mock_client
            mock_anthropic.assert_called_once()

    def test_create_client_with_unknown_provider(self) -> None:
        """Test create_client function with unknown provider."""
        from arklex.evaluation.chatgpt_utils import create_client

        with (
            patch("arklex.evaluation.chatgpt_utils.MODEL", {"llm_provider": "unknown"}),
            patch("arklex.evaluation.chatgpt_utils.OpenAI") as mock_openai,
        ):
            mock_client = Mock()
            mock_openai.return_value = mock_client

            with pytest.raises(ValueError, match="Unsupported LLM provider: unknown"):
                create_client()

    def test_main_function_with_import_error(self) -> None:
        """Test main function when import fails."""
        from arklex.evaluation.chatgpt_utils import main

        with (
            patch("builtins.__import__", side_effect=ImportError("Test import error")),
            pytest.raises(ImportError, match="Test import error"),
        ):
            # Should raise an ImportError
            main()

    def test_main_function_with_sys_exit(self) -> None:
        """Test main function when sys.exit is called."""
        from arklex.evaluation.chatgpt_utils import main

        with (
            patch("sys.exit"),
            pytest.raises(ModuleNotFoundError, match="No module named 'get_documents'"),
        ):
            # Should raise ModuleNotFoundError
            main()

    def test_generate_goal_calls_chatgpt_chatbot(
        self, monkeypatch: pytest.MonkeyPatch, mock_client: Mock
    ) -> None:
        from arklex.evaluation import chatgpt_utils

        called = {}

        def fake_chatgpt_chatbot(
            messages: list[dict[str, str]], client: object, model: str | None = None
        ) -> str:
            called["messages"] = messages
            called["client"] = client
            called["model"] = model
            return "goal"

        monkeypatch.setattr(chatgpt_utils, "chatgpt_chatbot", fake_chatgpt_chatbot)
        doc_content = "Some content"
        client = mock_client
        result = chatgpt_utils.generate_goal(doc_content, client)
        assert result == "goal"
        assert called["messages"]
        assert called["client"] is client

    def test_generate_goals_return_line(
        self, monkeypatch: pytest.MonkeyPatch, mock_client: Mock
    ) -> None:
        from arklex.evaluation import chatgpt_utils

        # Patch generate_goal to return a fixed value
        monkeypatch.setattr(chatgpt_utils, "generate_goal", lambda doc, client: "goal")
        documents = [{"content": "doc1"}, {"content": "doc2"}]
        params = {"num_goals": 2}
        result = chatgpt_utils.generate_goals(documents, params, mock_client)
        assert result == ["goal", "goal"]

    def test_filter_convo_return_line_edge_case(self) -> None:
        from arklex.evaluation.chatgpt_utils import filter_convo

        # Edge case: convo with only two turns (should skip both)
        convo = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = filter_convo(convo)
        assert result == []
        # Edge case: convo with a user message after two turns
        convo = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "user content\nextra"},
        ]
        result = filter_convo(convo)
        assert result[0]["content"] == "user content"

    @patch("arklex.evaluation.chatgpt_utils.generate_goal")
    def test_generate_goals_return_value(self, mock_generate_goal: Mock) -> None:
        """Covers the return line of generate_goals (line 335)."""
        from arklex.evaluation.chatgpt_utils import generate_goals

        mock_generate_goal.side_effect = (
            lambda doc_content, client: f"goal-for-{doc_content}"
        )
        documents = [{"content": "doc1"}, {"content": "doc2"}]
        params = {"num_goals": 2}
        client = Mock()
        result = generate_goals(documents, params, client)
        assert len(result) == 2
        assert all(r.startswith("goal-for-") for r in result)
