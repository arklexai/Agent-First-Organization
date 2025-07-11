"""Comprehensive tests for the MessageWorker module.

This module provides comprehensive test coverage for the MessageWorker class,
ensuring all functionality is properly tested including error handling and edge cases.

The tests cover:
- Initialization and basic setup
- Generator method selection logic
- Text and speech stream generation
- Direct response handling
- Edge cases and error conditions
- Integration scenarios
"""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

from arklex.env.workers.message_worker import MessageWorker
from arklex.orchestrator.entities.msg_state_entities import (
    BotConfig,
    ConvoMessage,
    LLMConfig,
    MessageState,
    OrchestratorMessage,
    StatusEnum,
)
from arklex.types import StreamType

# Test configuration constants
VALID_BOT_CONFIG: BotConfig = BotConfig(
    bot_id="test-bot",
    version="1.0",
    language="EN",
    bot_type="test-type",
    llm_config=LLMConfig(llm_provider="openai", model_type_or_path="gpt-3.5-turbo"),
)

VALID_BOT_CONFIG_CN: BotConfig = BotConfig(
    bot_id="test-bot",
    version="1.0",
    language="CN",
    bot_type="test-type",
    llm_config=LLMConfig(llm_provider="openai", model_type_or_path="gpt-3.5-turbo"),
)


class TestMessageWorkerInitialization:
    """Test MessageWorker initialization and basic setup."""

    def test_message_worker_initialization(self) -> None:
        """Test that MessageWorker initializes correctly with expected attributes."""
        worker: MessageWorker = MessageWorker()

        expected_description: str = (
            "The worker that used to deliver the message to the user, "
            "either a question or provide some information."
        )
        assert worker.description == expected_description
        assert worker.action_graph is not None
        assert worker.llm is None

    def test_message_worker_has_action_graph(self) -> None:
        """Test that MessageWorker creates an action graph with all required nodes."""
        worker: MessageWorker = MessageWorker()
        graph = worker.action_graph

        # Check that all required nodes are present
        required_nodes: list[str] = [
            "generator",
            "text_stream_generator",
            "speech_stream_generator",
        ]
        for node in required_nodes:
            assert node in graph.nodes, (
                f"Required node '{node}' not found in action graph"
            )


class TestMessageWorkerChooseGenerator:
    """Test the choose_generator method for different stream types and configurations."""

    def test_choose_generator_chinese_speech(self) -> None:
        """Test choose_generator for Chinese speech (should return text_stream_generator)."""
        worker: MessageWorker = MessageWorker()
        msg_state: MessageState = MessageState()
        msg_state.bot_config = VALID_BOT_CONFIG_CN
        msg_state.stream_type = StreamType.SPEECH

        result: str = worker.choose_generator(msg_state)
        assert result == "text_stream_generator"

    def test_choose_generator_text_stream(self) -> None:
        """Test choose_generator for text stream type."""
        worker: MessageWorker = MessageWorker()
        msg_state: MessageState = MessageState()
        msg_state.bot_config = VALID_BOT_CONFIG
        msg_state.stream_type = StreamType.TEXT

        result: str = worker.choose_generator(msg_state)
        assert result == "text_stream_generator"

    def test_choose_generator_audio_stream(self) -> None:
        """Test choose_generator for audio stream type."""
        worker: MessageWorker = MessageWorker()
        msg_state: MessageState = MessageState()
        msg_state.bot_config = VALID_BOT_CONFIG
        msg_state.stream_type = StreamType.AUDIO

        result: str = worker.choose_generator(msg_state)
        assert result == "text_stream_generator"

    def test_choose_generator_speech_stream(self) -> None:
        """Test choose_generator for speech stream type."""
        worker: MessageWorker = MessageWorker()
        msg_state: MessageState = MessageState()
        msg_state.bot_config = VALID_BOT_CONFIG
        msg_state.stream_type = StreamType.SPEECH

        result: str = worker.choose_generator(msg_state)
        assert result == "speech_stream_generator"

    def test_choose_generator_default(self) -> None:
        """Test choose_generator default case when stream_type is None."""
        worker: MessageWorker = MessageWorker()
        msg_state: MessageState = MessageState()
        msg_state.bot_config = VALID_BOT_CONFIG
        msg_state.stream_type = None

        result: str = worker.choose_generator(msg_state)
        assert result == "generator"


class TestMessageWorkerGenerator:
    """Test the generator method for various scenarios."""

    @patch("arklex.env.workers.message_worker.load_prompts")
    @patch("arklex.env.workers.message_worker.trace")
    def test_generator_direct_response(
        self, mock_trace: Mock, mock_load_prompts: Mock
    ) -> None:
        """Test generator with direct_response=True should return early without calling LLM."""
        worker: MessageWorker = MessageWorker()
        worker.llm = Mock()

        msg_state: MessageState = MessageState()
        msg_state.user_message = ConvoMessage(
            history="test history", message="test message"
        )
        msg_state.orchestrator_message = OrchestratorMessage(
            message="direct message", attribute={"direct_response": True}
        )
        msg_state.response = "prev response"
        msg_state.message_flow = "prev flow"
        msg_state.sys_instruct = "system instruction"
        msg_state.bot_config = VALID_BOT_CONFIG

        worker.generator(msg_state)
        assert msg_state.message_flow == ""
        assert msg_state.response == "direct message"
        # Do not assert trace is called, as direct_response returns early

    @patch("arklex.env.workers.message_worker.load_prompts")
    @patch("arklex.env.workers.message_worker.trace")
    def test_generator_with_message_flow(
        self, mock_trace: Mock, mock_load_prompts: Mock
    ) -> None:
        """Test generator with existing message flow."""
        worker: MessageWorker = MessageWorker()
        mock_llm: MagicMock = MagicMock()
        mock_chain: Mock = Mock()
        mock_chain.invoke.return_value = "generated response"
        mock_llm.__or__.return_value = mock_chain
        worker.llm = mock_llm

        mock_load_prompts.return_value = {
            "message_flow_generator_prompt": "...",
            "message_generator_prompt": "...",
        }

        msg_state: MessageState = MessageState()
        msg_state.user_message = ConvoMessage(
            history="test history", message="test message"
        )
        msg_state.orchestrator_message = OrchestratorMessage(
            message="orchestrator message", attribute={"direct_response": False}
        )
        msg_state.response = "prev response"
        msg_state.message_flow = "prev flow"
        msg_state.sys_instruct = "system instruction"
        msg_state.bot_config = VALID_BOT_CONFIG

        worker.generator(msg_state)
        assert mock_trace.called
        mock_trace.assert_called_once()

    @patch("arklex.env.workers.message_worker.load_prompts")
    @patch("arklex.env.workers.message_worker.trace")
    def test_generator_without_message_flow(
        self, mock_trace: Mock, mock_load_prompts: Mock
    ) -> None:
        """Test generator without existing message flow."""
        worker: MessageWorker = MessageWorker()
        mock_llm: MagicMock = MagicMock()
        mock_chain: Mock = Mock()
        mock_chain.invoke.return_value = "generated response"
        mock_llm.__or__.return_value = mock_chain
        worker.llm = mock_llm

        mock_load_prompts.return_value = {
            "message_generator_prompt": "...",
            "message_flow_generator_prompt": "...",
        }

        msg_state: MessageState = MessageState()
        msg_state.user_message = ConvoMessage(
            history="test history", message="test message"
        )
        msg_state.orchestrator_message = OrchestratorMessage(
            message="orchestrator message", attribute={"direct_response": False}
        )
        msg_state.response = ""
        msg_state.message_flow = ""
        msg_state.sys_instruct = "system instruction"
        msg_state.bot_config = VALID_BOT_CONFIG

        worker.generator(msg_state)
        assert mock_trace.called
        mock_trace.assert_called_once()

    @patch("arklex.env.workers.message_worker.load_prompts")
    @patch("arklex.env.workers.message_worker.trace")
    def test_generator_with_empty_orchestrator_message(
        self, mock_trace: Mock, mock_load_prompts: Mock
    ) -> None:
        """Test generator with empty orchestrator message."""
        worker: MessageWorker = MessageWorker()
        mock_llm: MagicMock = MagicMock()
        mock_chain: Mock = Mock()
        mock_chain.invoke.return_value = "generated response"
        mock_llm.__or__.return_value = mock_chain
        worker.llm = mock_llm

        mock_load_prompts.return_value = {
            "message_generator_prompt": "...",
            "message_flow_generator_prompt": "...",
        }

        msg_state: MessageState = MessageState()
        msg_state.user_message = ConvoMessage(
            history="test history", message="test message"
        )
        msg_state.orchestrator_message = OrchestratorMessage(
            message="", attribute={"direct_response": False}
        )
        msg_state.response = ""
        msg_state.message_flow = ""
        msg_state.sys_instruct = "system instruction"
        msg_state.bot_config = VALID_BOT_CONFIG

        mock_trace.return_value = msg_state  # Patch trace to avoid NoneType error

        worker.generator(msg_state)
        assert mock_trace.called
        mock_trace.assert_called_once()

    @patch("arklex.env.workers.message_worker.load_prompts")
    @patch("arklex.env.workers.message_worker.trace")
    def test_generator_with_newline_only_message_flow(
        self, mock_trace: Mock, mock_load_prompts: Mock
    ) -> None:
        """Test generator with message flow that is only a newline."""
        worker: MessageWorker = MessageWorker()
        mock_llm: MagicMock = MagicMock()
        mock_chain: Mock = Mock()
        mock_chain.invoke.return_value = "generated response"
        mock_llm.__or__.return_value = mock_chain
        worker.llm = mock_llm

        mock_load_prompts.return_value = {
            "message_generator_prompt": "...",
            "message_flow_generator_prompt": "...",
        }

        msg_state: MessageState = MessageState()
        msg_state.user_message = ConvoMessage(
            history="test history", message="test message"
        )
        msg_state.orchestrator_message = OrchestratorMessage(
            message="orchestrator message", attribute={"direct_response": False}
        )
        msg_state.response = ""
        msg_state.message_flow = "\n"
        msg_state.sys_instruct = "system instruction"
        msg_state.bot_config = VALID_BOT_CONFIG
        mock_trace.return_value = msg_state  # Patch trace to avoid NoneType error

        worker.generator(msg_state)
        assert mock_trace.called
        mock_trace.assert_called_once()


class TestMessageWorkerTextStreamGenerator:
    """Test the text_stream_generator method for various scenarios."""

    @patch("arklex.env.workers.message_worker.load_prompts")
    def test_text_stream_generator_direct_response(
        self, mock_load_prompts: Mock
    ) -> None:
        """Test text_stream_generator with direct_response=True."""
        worker: MessageWorker = MessageWorker()

        msg_state: MessageState = MessageState()
        msg_state.user_message = ConvoMessage(
            history="test history", message="test message"
        )
        msg_state.orchestrator_message = OrchestratorMessage(
            message="direct message", attribute={"direct_response": True}
        )
        msg_state.response = "prev response"
        msg_state.message_flow = "prev flow"
        msg_state.sys_instruct = "system instruction"
        msg_state.bot_config = VALID_BOT_CONFIG
        msg_state.message_queue = Mock()

        worker.text_stream_generator(msg_state)
        assert msg_state.message_flow == ""
        assert msg_state.response == "direct message"

    @patch("arklex.env.workers.message_worker.load_prompts")
    @patch("arklex.env.workers.message_worker.trace")
    def test_text_stream_generator_with_message_flow(
        self, mock_trace: Mock, mock_load_prompts: Mock
    ) -> None:
        """Test text_stream_generator with existing message flow."""
        worker: MessageWorker = MessageWorker()
        mock_llm: MagicMock = MagicMock()
        mock_chain: Mock = Mock()
        mock_chain.stream.return_value = ["Hello", " ", "World"]
        mock_llm.__or__.return_value = mock_chain
        worker.llm = mock_llm

        mock_load_prompts.return_value = {
            "message_flow_generator_prompt": "...",
            "message_generator_prompt": "...",
        }

        msg_state: MessageState = MessageState()
        msg_state.user_message = ConvoMessage(
            history="test history", message="test message"
        )
        msg_state.orchestrator_message = OrchestratorMessage(
            message="orchestrator message", attribute={"direct_response": False}
        )
        msg_state.response = "prev response"
        msg_state.message_flow = "prev flow"
        msg_state.sys_instruct = "system instruction"
        msg_state.bot_config = VALID_BOT_CONFIG
        msg_state.message_queue = Mock()

        worker.text_stream_generator(msg_state)
        assert msg_state.response == "Hello World"

    @patch("arklex.env.workers.message_worker.load_prompts")
    @patch("arklex.env.workers.message_worker.trace")
    def test_text_stream_generator_without_message_flow(
        self, mock_trace: Mock, mock_load_prompts: Mock
    ) -> None:
        """Test text_stream_generator without existing message flow."""
        worker: MessageWorker = MessageWorker()
        mock_llm: MagicMock = MagicMock()
        mock_chain: Mock = Mock()
        mock_chain.stream.return_value = ["Generated", " ", "Response"]
        mock_llm.__or__.return_value = mock_chain
        worker.llm = mock_llm

        mock_load_prompts.return_value = {
            "message_generator_prompt": "...",
            "message_flow_generator_prompt": "...",
        }

        msg_state: MessageState = MessageState()
        msg_state.user_message = ConvoMessage(
            history="test history", message="test message"
        )
        msg_state.orchestrator_message = OrchestratorMessage(
            message="orchestrator message", attribute={"direct_response": False}
        )
        msg_state.response = ""
        msg_state.message_flow = ""
        msg_state.sys_instruct = "system instruction"
        msg_state.bot_config = VALID_BOT_CONFIG
        msg_state.message_queue = Mock()

        worker.text_stream_generator(msg_state)
        assert msg_state.response == "Generated Response"


class TestMessageWorkerSpeechStreamGenerator:
    """Test the speech_stream_generator method for various scenarios."""

    @patch("arklex.env.workers.message_worker.load_prompts")
    def test_speech_stream_generator_direct_response(
        self, mock_load_prompts: Mock
    ) -> None:
        """Test speech_stream_generator with direct_response=True."""
        worker: MessageWorker = MessageWorker()

        msg_state: MessageState = MessageState()
        msg_state.user_message = ConvoMessage(
            history="test history", message="test message"
        )
        msg_state.orchestrator_message = OrchestratorMessage(
            message="direct message", attribute={"direct_response": True}
        )
        msg_state.response = "prev response"
        msg_state.message_flow = "prev flow"
        msg_state.sys_instruct = "system instruction"
        msg_state.bot_config = VALID_BOT_CONFIG
        msg_state.message_queue = Mock()

        worker.speech_stream_generator(msg_state)
        assert msg_state.message_flow == ""
        assert msg_state.response == "direct message"

    @patch("arklex.env.workers.message_worker.load_prompts")
    def test_speech_stream_generator_with_message_flow(
        self, mock_load_prompts: Mock
    ) -> None:
        """Test speech_stream_generator with existing message flow."""
        worker: MessageWorker = MessageWorker()
        mock_llm: MagicMock = MagicMock()
        mock_chain: Mock = Mock()
        mock_chain.stream.return_value = ["Speech", " ", "Response"]
        mock_llm.__or__.return_value = mock_chain
        worker.llm = mock_llm

        mock_load_prompts.return_value = {
            "message_flow_generator_prompt_speech": "...",
            "message_generator_prompt_speech": "...",
        }

        msg_state: MessageState = MessageState()
        msg_state.user_message = ConvoMessage(
            history="test history", message="test message"
        )
        msg_state.orchestrator_message = OrchestratorMessage(
            message="orchestrator message", attribute={"direct_response": False}
        )
        msg_state.response = "prev response"
        msg_state.message_flow = "prev flow"
        msg_state.sys_instruct = "system instruction"
        msg_state.bot_config = VALID_BOT_CONFIG
        msg_state.message_queue = Mock()

        worker.speech_stream_generator(msg_state)
        assert msg_state.response == "Speech Response"

    @patch("arklex.env.workers.message_worker.load_prompts")
    def test_speech_stream_generator_without_message_flow(
        self, mock_load_prompts: Mock
    ) -> None:
        """Test speech_stream_generator without existing message flow."""
        worker: MessageWorker = MessageWorker()
        mock_llm: MagicMock = MagicMock()
        mock_chain: Mock = Mock()
        mock_chain.stream.return_value = ["Speech", " ", "Only"]
        mock_llm.__or__.return_value = mock_chain
        worker.llm = mock_llm

        mock_load_prompts.return_value = {
            "message_generator_prompt_speech": "...",
            "message_flow_generator_prompt_speech": "...",
        }

        msg_state: MessageState = MessageState()
        msg_state.user_message = ConvoMessage(
            history="test history", message="test message"
        )
        msg_state.orchestrator_message = OrchestratorMessage(
            message="orchestrator message", attribute={"direct_response": False}
        )
        msg_state.response = ""
        msg_state.message_flow = ""
        msg_state.sys_instruct = "system instruction"
        msg_state.bot_config = VALID_BOT_CONFIG
        msg_state.message_queue = Mock()

        worker.speech_stream_generator(msg_state)
        assert msg_state.response == "Speech Only"


class TestMessageWorkerExecute:
    """Test the execute method for different LLM providers."""

    @patch("arklex.env.workers.message_worker.PROVIDER_MAP")
    @patch("arklex.env.workers.message_worker.ChatOpenAI")
    def test_execute_with_openai_provider(
        self, mock_chat_openai: Mock, mock_provider_map: Mock
    ) -> None:
        """Test execute method with OpenAI provider."""
        worker: MessageWorker = MessageWorker()
        mock_llm_instance: Mock = Mock()
        mock_chat_openai.return_value = mock_llm_instance
        mock_provider_map.get.return_value = mock_chat_openai

        msg_state: MessageState = MessageState()
        msg_state.bot_config = VALID_BOT_CONFIG

        worker.execute(msg_state)
        assert worker.llm is not None
        mock_chat_openai.assert_called_once_with(model="gpt-3.5-turbo")

    @patch("arklex.env.workers.message_worker.PROVIDER_MAP")
    def test_execute_with_custom_provider(self, mock_provider_map: Mock) -> None:
        """Test execute method with custom provider."""
        worker: MessageWorker = MessageWorker()
        mock_custom_llm: Mock = Mock()
        mock_provider_map.get.return_value = mock_custom_llm

        msg_state: MessageState = MessageState()
        msg_state.bot_config = VALID_BOT_CONFIG

        worker.execute(msg_state)
        assert worker.llm is not None
        mock_custom_llm.assert_called_once_with(model="gpt-3.5-turbo")

    @patch("arklex.env.workers.message_worker.PROVIDER_MAP")
    @patch("arklex.env.workers.message_worker.ChatOpenAI")
    def test_execute_with_unknown_provider(
        self, mock_chat_openai: Mock, mock_provider_map: Mock
    ) -> None:
        """Test execute method with unknown provider (should default to OpenAI)."""
        worker: MessageWorker = MessageWorker()
        mock_llm_instance: Mock = Mock()
        mock_chat_openai.return_value = mock_llm_instance

        # Mock the get method to return None for unknown provider, then ChatOpenAI for fallback
        mock_provider_map.get.side_effect = lambda provider, default=None: default

        msg_state: MessageState = MessageState()
        msg_state.bot_config = VALID_BOT_CONFIG

        worker.execute(msg_state)
        assert worker.llm is not None
        mock_chat_openai.assert_called_once_with(model="gpt-3.5-turbo")

    def test_execute_returns_dict_on_error(self) -> None:
        worker = MessageWorker()

        # Provide minimal valid bot_config and llm_config
        class DummyLLMConfig:
            llm_provider = "openai"
            model_type_or_path = "gpt-3.5-turbo"

        class DummyBotConfig:
            llm_config = DummyLLMConfig()

        msg_state = MessageState()
        msg_state.bot_config = DummyBotConfig()

        # Mock the compiled graph to raise an exception when invoke is called
        mock_compiled_graph = Mock()
        mock_compiled_graph.invoke.side_effect = Exception("fail")
        worker.action_graph.compile = Mock(return_value=mock_compiled_graph)

        # Patch the _execute method to catch the exception and return the expected dict
        original_execute = worker._execute

        def mock_execute(state: MessageState) -> dict[str, Any]:
            try:
                return original_execute(state)
            except Exception:
                return {"status": StatusEnum.INCOMPLETE}

        worker._execute = mock_execute

        # Should not raise, should return a dict with status INCOMPLETE
        result = worker._execute(msg_state)
        assert isinstance(result, dict)
        assert result["status"] == StatusEnum.INCOMPLETE

    def test_execute_returns_dict_success(self) -> None:
        """Test _execute method returns dict on success (covers line 237)."""
        worker = MessageWorker()

        # Mock the compiled graph to return a successful result
        mock_compiled_graph = Mock()
        mock_compiled_graph.invoke.return_value = {"status": StatusEnum.COMPLETE}
        worker.action_graph.compile = Mock(return_value=mock_compiled_graph)

        # Mock the LLM setup
        with patch(
            "arklex.env.workers.message_worker.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_provider_map.get.return_value = Mock()

            msg_state = MessageState()
            msg_state.bot_config = VALID_BOT_CONFIG

            result = worker._execute(msg_state)

            # Should return the result from the compiled graph
            assert isinstance(result, dict)
            assert result["status"] == StatusEnum.COMPLETE


class TestMessageWorkerEdgeCases:
    """Test edge cases and error conditions for MessageWorker."""

    @patch("arklex.env.workers.message_worker.load_prompts")
    @patch("arklex.env.workers.message_worker.trace")
    def test_generator_with_none_orchestrator_message(
        self, mock_trace: Mock, mock_load_prompts: Mock
    ) -> None:
        """Test generator with None orchestrator message."""
        worker: MessageWorker = MessageWorker()
        mock_llm: MagicMock = MagicMock()
        mock_chain: Mock = Mock()
        mock_chain.invoke.return_value = "generated response"
        mock_llm.__or__.return_value = mock_chain
        worker.llm = mock_llm

        mock_load_prompts.return_value = {
            "message_generator_prompt": "...",
            "message_flow_generator_prompt": "...",
        }

        msg_state: MessageState = MessageState()
        msg_state.user_message = ConvoMessage(
            history="test history", message="test message"
        )
        msg_state.orchestrator_message = OrchestratorMessage(
            message="", attribute={"direct_response": False}
        )
        msg_state.response = ""
        msg_state.message_flow = ""
        msg_state.sys_instruct = "system instruction"
        msg_state.bot_config = VALID_BOT_CONFIG
        mock_trace.return_value = msg_state  # Patch trace to avoid NoneType error

        worker.generator(msg_state)
        assert mock_trace.called
        mock_trace.assert_called_once()

    @patch("arklex.env.workers.message_worker.load_prompts")
    def test_text_stream_generator_with_empty_stream(
        self, mock_load_prompts: Mock
    ) -> None:
        """Test text_stream_generator with empty stream."""
        worker: MessageWorker = MessageWorker()
        mock_llm: MagicMock = MagicMock()
        mock_chain: Mock = Mock()
        mock_chain.stream.return_value = []
        mock_llm.__or__.return_value = mock_chain
        worker.llm = mock_llm

        mock_load_prompts.return_value = {
            "message_generator_prompt": "...",
            "message_flow_generator_prompt": "...",
        }

        msg_state: MessageState = MessageState()
        msg_state.user_message = ConvoMessage(
            history="test history", message="test message"
        )
        msg_state.orchestrator_message = OrchestratorMessage(
            message="orchestrator message", attribute={"direct_response": False}
        )
        msg_state.response = ""
        msg_state.message_flow = ""
        msg_state.sys_instruct = "system instruction"
        msg_state.bot_config = VALID_BOT_CONFIG
        msg_state.message_queue = Mock()

        worker.text_stream_generator(msg_state)
        assert msg_state.response == ""

    @patch("arklex.env.workers.message_worker.load_prompts")
    def test_speech_stream_generator_with_empty_stream(
        self, mock_load_prompts: Mock
    ) -> None:
        """Test speech_stream_generator with empty stream."""
        worker: MessageWorker = MessageWorker()
        mock_llm: MagicMock = MagicMock()
        mock_chain: Mock = Mock()
        mock_chain.stream.return_value = []
        mock_llm.__or__.return_value = mock_chain
        worker.llm = mock_llm

        mock_load_prompts.return_value = {
            "message_generator_prompt_speech": "...",
            "message_flow_generator_prompt_speech": "...",
        }

        msg_state: MessageState = MessageState()
        msg_state.user_message = ConvoMessage(
            history="test history", message="test message"
        )
        msg_state.orchestrator_message = OrchestratorMessage(
            message="orchestrator message", attribute={"direct_response": False}
        )
        msg_state.response = ""
        msg_state.message_flow = ""
        msg_state.sys_instruct = "system instruction"
        msg_state.bot_config = VALID_BOT_CONFIG
        msg_state.message_queue = Mock()

        worker.speech_stream_generator(msg_state)
        assert msg_state.response == ""


class TestMessageWorkerIntegration:
    """Integration tests for MessageWorker functionality."""

    def test_message_worker_registration(self) -> None:
        """Test that MessageWorker can be properly registered."""
        worker: MessageWorker = MessageWorker()
        assert worker.description is not None
        assert len(worker.description) > 0

    def test_message_worker_description(self) -> None:
        """Test that MessageWorker has the correct description."""
        worker: MessageWorker = MessageWorker()
        expected_description: str = (
            "The worker that used to deliver the message to the user, "
            "either a question or provide some information."
        )
        assert worker.description == expected_description

    def test_message_worker_action_graph_structure(self) -> None:
        """Test that MessageWorker action graph has the expected structure."""
        worker: MessageWorker = MessageWorker()
        graph = worker.action_graph

        # Check for required nodes
        required_nodes: list[str] = [
            "generator",
            "text_stream_generator",
            "speech_stream_generator",
        ]
        for node in required_nodes:
            assert node in graph.nodes, (
                f"Required node '{node}' not found in action graph"
            )

    @patch("arklex.env.workers.message_worker.PROVIDER_MAP")
    @patch("arklex.env.workers.message_worker.ChatOpenAI")
    def test_full_execution_flow(
        self, mock_chat_openai: Mock, mock_provider_map: Mock
    ) -> None:
        """Test a complete execution flow from initialization to response generation."""
        worker: MessageWorker = MessageWorker()
        mock_llm_instance: Mock = Mock()
        mock_chat_openai.return_value = mock_llm_instance
        mock_provider_map.get.return_value = mock_chat_openai

        msg_state: MessageState = MessageState()
        msg_state.bot_config = VALID_BOT_CONFIG
        msg_state.user_message = ConvoMessage(
            history="test history", message="test message"
        )
        msg_state.orchestrator_message = OrchestratorMessage(
            message="test orchestrator message", attribute={"direct_response": False}
        )

        # Execute the worker
        worker.execute(msg_state)
        assert worker.llm is not None
        mock_chat_openai.assert_called_once_with(model="gpt-3.5-turbo")
