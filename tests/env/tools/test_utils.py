"""Tests for the tools utils module.

This module contains comprehensive test cases for tool utility functions,
including prompt template generation, tool generation, and execution tracing.
"""

from queue import Queue
from unittest.mock import Mock, patch

import pytest

from arklex.env.tools.utils import (
    ToolGenerator,
    execute_tool,
    get_prompt_template,
    trace,
)
from arklex.types import StreamType
from arklex.utils.exceptions import ToolError
from arklex.utils.graph_state import MessageState


class TestGetPromptTemplate:
    """Test cases for get_prompt_template function."""

    @patch("arklex.env.tools.utils.load_prompts")
    def test_get_prompt_template_speech_non_chinese(
        self, mock_load_prompts: Mock
    ) -> None:
        """Test get_prompt_template with speech stream type for non-Chinese language."""
        # Setup
        state = Mock(spec=MessageState)
        state.stream_type = StreamType.SPEECH
        state.bot_config = Mock()
        state.bot_config.language = "EN"

        mock_load_prompts.return_value = {
            "test_prompt": "Regular prompt",
            "test_prompt_speech": "Speech prompt",
        }

        # Execute
        result = get_prompt_template(state, "test_prompt")

        # Assert
        assert result is not None
        mock_load_prompts.assert_called_once_with(state.bot_config)

    @patch("arklex.env.tools.utils.load_prompts")
    def test_get_prompt_template_speech_chinese(self, mock_load_prompts: Mock) -> None:
        """Test get_prompt_template with speech stream type for Chinese language."""
        # Setup
        state = Mock(spec=MessageState)
        state.stream_type = StreamType.SPEECH
        state.bot_config = Mock()
        state.bot_config.language = "CN"

        mock_load_prompts.return_value = {
            "test_prompt": "Regular prompt",
            "test_prompt_speech": "Speech prompt",
        }

        # Execute
        result = get_prompt_template(state, "test_prompt")

        # Assert
        assert result is not None
        mock_load_prompts.assert_called_once_with(state.bot_config)

    @patch("arklex.env.tools.utils.load_prompts")
    def test_get_prompt_template_non_speech(self, mock_load_prompts: Mock) -> None:
        """Test get_prompt_template with non-speech stream type."""
        # Setup
        state = Mock(spec=MessageState)
        state.stream_type = StreamType.TEXT
        state.bot_config = Mock()
        state.bot_config.language = "EN"

        mock_load_prompts.return_value = {
            "test_prompt": "Regular prompt",
        }

        # Execute
        result = get_prompt_template(state, "test_prompt")

        # Assert
        assert result is not None
        mock_load_prompts.assert_called_once_with(state.bot_config)


class TestToolGenerator:
    """Test cases for ToolGenerator class."""

    @patch("arklex.env.tools.utils.PROVIDER_MAP")
    @patch("arklex.env.tools.utils.get_prompt_template")
    @patch("arklex.env.tools.utils.log_context")
    def test_generate(
        self, mock_log_context: Mock, mock_get_prompt: Mock, mock_provider_map: Mock
    ) -> None:
        """Test ToolGenerator.generate method."""
        # Setup
        state = Mock(spec=MessageState)
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.llm_provider = "openai"
        state.bot_config.llm_config.model_type_or_path = "gpt-3.5-turbo"
        state.user_message = Mock()
        state.user_message.history = "test history"
        state.sys_instruct = "test instruction"

        mock_llm = Mock()
        mock_provider_map.get.return_value = Mock(return_value=mock_llm)

        mock_prompt = Mock()
        mock_prompt.invoke.return_value.text = "test prompt"
        mock_get_prompt.return_value = mock_prompt

        mock_chain = Mock()
        mock_chain.invoke.return_value = "test response"
        mock_llm.__or__ = Mock(return_value=mock_chain)

        # Execute
        result = ToolGenerator.generate(state)

        # Assert
        assert result == state
        assert state.response == "test response"
        mock_log_context.info.assert_called()

    @patch("arklex.env.tools.utils.PROVIDER_MAP")
    @patch("arklex.env.tools.utils.get_prompt_template")
    @patch("arklex.env.tools.utils.log_context")
    @patch("arklex.env.tools.utils.trace")
    def test_context_generate_with_records(
        self,
        mock_trace: Mock,
        mock_log_context: Mock,
        mock_get_prompt: Mock,
        mock_provider_map: Mock,
    ) -> None:
        """Test ToolGenerator.context_generate method with relevant records."""
        # Setup
        state = Mock(spec=MessageState)
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.llm_provider = "openai"
        state.bot_config.llm_config.model_type_or_path = "gpt-3.5-turbo"
        state.user_message = Mock()
        state.user_message.history = "test history"
        state.sys_instruct = "test instruction"
        state.message_flow = "test flow"
        state.relevant_records = [
            Mock(
                info="test info",
                personalized_intent="test intent",
                output="test output",
                steps=[{"step1": "value1"}, "step2"],
            )
        ]

        mock_llm = Mock()
        mock_provider_map.get.return_value = Mock(return_value=mock_llm)

        mock_prompt = Mock()
        mock_prompt.invoke.return_value.text = "test prompt"
        mock_get_prompt.return_value = mock_prompt

        mock_chain = Mock()
        mock_chain.invoke.return_value = "test response"
        mock_llm.__or__ = Mock(return_value=mock_chain)

        mock_trace.return_value = state

        # Execute
        result = ToolGenerator.context_generate(state)

        # Assert
        assert result == state
        assert state.response == "test response"
        assert state.message_flow == ""
        mock_trace.assert_called_once_with(input="test response", state=state)

    @patch("arklex.env.tools.utils.PROVIDER_MAP")
    @patch("arklex.env.tools.utils.get_prompt_template")
    @patch("arklex.env.tools.utils.log_context")
    @patch("arklex.env.tools.utils.trace")
    def test_context_generate_without_records(
        self,
        mock_trace: Mock,
        mock_log_context: Mock,
        mock_get_prompt: Mock,
        mock_provider_map: Mock,
    ) -> None:
        """Test ToolGenerator.context_generate method without relevant records."""
        # Setup
        state = Mock(spec=MessageState)
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.llm_provider = "openai"
        state.bot_config.llm_config.model_type_or_path = "gpt-3.5-turbo"
        state.user_message = Mock()
        state.user_message.history = "test history"
        state.sys_instruct = "test instruction"
        state.message_flow = "test flow"
        state.relevant_records = []

        mock_llm = Mock()
        mock_provider_map.get.return_value = Mock(return_value=mock_llm)

        mock_prompt = Mock()
        mock_prompt.invoke.return_value.text = "test prompt"
        mock_get_prompt.return_value = mock_prompt

        mock_chain = Mock()
        mock_chain.invoke.return_value = "test response"
        mock_llm.__or__ = Mock(return_value=mock_chain)

        mock_trace.return_value = state

        # Execute
        result = ToolGenerator.context_generate(state)

        # Assert
        assert result == state
        assert state.response == "test response"
        assert state.message_flow == ""
        mock_trace.assert_called_once_with(input="test response", state=state)

    @patch("arklex.env.tools.utils.PROVIDER_MAP")
    @patch("arklex.env.tools.utils.get_prompt_template")
    @patch("arklex.env.tools.utils.log_context")
    @patch("arklex.env.tools.utils.trace")
    def test_stream_context_generate(
        self,
        mock_trace: Mock,
        mock_log_context: Mock,
        mock_get_prompt: Mock,
        mock_provider_map: Mock,
    ) -> None:
        """Test ToolGenerator.stream_context_generate method."""
        # Setup
        state = Mock(spec=MessageState)
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.llm_provider = "openai"
        state.bot_config.llm_config.model_type_or_path = "gpt-3.5-turbo"
        state.user_message = Mock()
        state.user_message.history = "test history"
        state.sys_instruct = "test instruction"
        state.message_flow = "test flow"
        state.relevant_records = []
        state.message_queue = Queue()

        mock_llm = Mock()
        mock_provider_map.get.return_value = Mock(return_value=mock_llm)

        mock_prompt = Mock()
        mock_prompt.invoke.return_value.text = "test prompt"
        mock_get_prompt.return_value = mock_prompt

        mock_chain = Mock()
        mock_chain.stream.return_value = ["chunk1", "chunk2", "chunk3"]
        mock_llm.__or__ = Mock(return_value=mock_chain)

        mock_trace.return_value = state

        # Execute
        result = ToolGenerator.stream_context_generate(state)

        # Assert
        assert result == state
        assert state.response == "chunk1chunk2chunk3"
        assert state.message_flow == ""
        assert state.message_queue.qsize() == 3
        mock_trace.assert_called_once_with(input="chunk1chunk2chunk3", state=state)

    @patch("arklex.env.tools.utils.PROVIDER_MAP")
    @patch("arklex.env.tools.utils.get_prompt_template")
    @patch("arklex.env.tools.utils.log_context")
    @patch("arklex.env.tools.utils.trace")
    def test_stream_generate(
        self,
        mock_trace: Mock,
        mock_log_context: Mock,
        mock_get_prompt: Mock,
        mock_provider_map: Mock,
    ) -> None:
        """Test ToolGenerator.stream_generate method."""
        # Setup
        state = Mock(spec=MessageState)
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.llm_provider = "openai"
        state.bot_config.llm_config.model_type_or_path = "gpt-3.5-turbo"
        state.user_message = Mock()
        state.user_message.history = "test history"
        state.sys_instruct = "test instruction"
        state.message_queue = Queue()

        mock_llm = Mock()
        mock_provider_map.get.return_value = Mock(return_value=mock_llm)

        mock_prompt = Mock()
        mock_prompt.invoke.return_value.text = "test prompt"
        mock_get_prompt.return_value = mock_prompt

        mock_chain = Mock()
        mock_chain.stream.return_value = ["chunk1", "chunk2"]
        mock_llm.__or__ = Mock(return_value=mock_chain)

        # Execute
        result = ToolGenerator.stream_generate(state)

        # Assert
        assert result == state
        assert state.response == "chunk1chunk2"
        assert state.message_queue.qsize() == 2

    @patch("arklex.env.tools.utils.PROVIDER_MAP")
    @patch("arklex.env.tools.utils.get_prompt_template")
    @patch("arklex.env.tools.utils.log_context")
    def test_context_generate_with_dict_steps(
        self, mock_log_context: Mock, mock_get_prompt: Mock, mock_provider_map: Mock
    ) -> None:
        """Test ToolGenerator.context_generate with dictionary steps in records."""
        # Setup
        state = Mock(spec=MessageState)
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.llm_provider = "openai"
        state.bot_config.llm_config.model_type_or_path = "gpt-3.5-turbo"
        state.user_message = Mock()
        state.user_message.history = "test history"
        state.sys_instruct = "test instruction"
        state.message_flow = "test flow"
        state.relevant_records = [
            Mock(
                info="test info",
                personalized_intent="test intent",
                output="test output",
                steps=[{"key1": "value1", "key2": "value2"}],
            )
        ]
        state.trajectory = [[Mock(steps=[])]]

        mock_llm = Mock()
        mock_provider_map.get.return_value = Mock(return_value=mock_llm)

        mock_prompt = Mock()
        mock_prompt.invoke.return_value.text = "test prompt"
        mock_get_prompt.return_value = mock_prompt

        mock_chain = Mock()
        mock_chain.invoke.return_value = "test response"
        mock_llm.__or__ = Mock(return_value=mock_chain)

        # Execute
        result = ToolGenerator.context_generate(state)

        # Assert
        assert result == state
        assert state.response == "test response"

    @patch("arklex.env.tools.utils.PROVIDER_MAP")
    @patch("arklex.env.tools.utils.get_prompt_template")
    @patch("arklex.env.tools.utils.log_context")
    @patch("arklex.env.tools.utils.trace")
    def test_stream_context_generate_with_records_missing_fields(
        self,
        mock_trace: Mock,
        mock_log_context: Mock,
        mock_get_prompt: Mock,
        mock_provider_map: Mock,
    ) -> None:
        """Test ToolGenerator.stream_context_generate method with records missing some fields."""
        # Setup
        state = Mock(spec=MessageState)
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.llm_provider = "openai"
        state.bot_config.llm_config.model_type_or_path = "gpt-3.5-turbo"
        state.user_message = Mock()
        state.user_message.history = "test history"
        state.sys_instruct = "test instruction"
        state.message_flow = "test flow"
        # Record with missing fields
        state.relevant_records = [
            Mock(
                info=None,  # Missing info
                personalized_intent=None,  # Missing intent
                output=None,  # Missing output
                steps=[],  # Empty steps
            )
        ]
        state.message_queue = Queue()

        mock_llm = Mock()
        mock_provider_map.get.return_value = Mock(return_value=mock_llm)

        mock_prompt = Mock()
        mock_prompt.invoke.return_value.text = "test prompt"
        mock_get_prompt.return_value = mock_prompt

        mock_chain = Mock()
        mock_chain.stream.return_value = ["chunk1", "chunk2"]
        mock_llm.__or__ = Mock(return_value=mock_chain)

        mock_trace.return_value = state

        # Execute
        result = ToolGenerator.stream_context_generate(state)

        # Assert
        assert result == state
        assert state.response == "chunk1chunk2"
        assert state.message_flow == ""
        assert state.message_queue.qsize() == 2
        mock_trace.assert_called_once_with(input="chunk1chunk2", state=state)

    @patch("arklex.env.tools.utils.PROVIDER_MAP")
    @patch("arklex.env.tools.utils.get_prompt_template")
    @patch("arklex.env.tools.utils.log_context")
    @patch("arklex.env.tools.utils.trace")
    def test_stream_context_generate_with_records_mixed_step_types(
        self,
        mock_trace: Mock,
        mock_log_context: Mock,
        mock_get_prompt: Mock,
        mock_provider_map: Mock,
    ) -> None:
        """Test ToolGenerator.stream_context_generate method with mixed step types in records."""
        # Setup
        state = Mock(spec=MessageState)
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.llm_provider = "openai"
        state.bot_config.llm_config.model_type_or_path = "gpt-3.5-turbo"
        state.user_message = Mock()
        state.user_message.history = "test history"
        state.sys_instruct = "test instruction"
        state.message_flow = "test flow"
        # Record with mixed step types
        state.relevant_records = [
            Mock(
                info="test info",
                personalized_intent="test intent",
                output="test output",
                steps=[
                    {"key1": "value1"},  # Dict step
                    "string_step",  # String step
                    123,  # Non-dict, non-string step
                    None,  # None step
                ],
            )
        ]
        state.message_queue = Queue()

        mock_llm = Mock()
        mock_provider_map.get.return_value = Mock(return_value=mock_llm)

        mock_prompt = Mock()
        mock_prompt.invoke.return_value.text = "test prompt"
        mock_get_prompt.return_value = mock_prompt

        mock_chain = Mock()
        mock_chain.stream.return_value = ["chunk1"]
        mock_llm.__or__ = Mock(return_value=mock_chain)

        mock_trace.return_value = state

        # Execute
        result = ToolGenerator.stream_context_generate(state)

        # Assert
        assert result == state
        assert state.response == "chunk1"
        assert state.message_flow == ""
        assert state.message_queue.qsize() == 1
        mock_trace.assert_called_once_with(input="chunk1", state=state)

    @patch("arklex.env.tools.utils.PROVIDER_MAP")
    @patch("arklex.env.tools.utils.get_prompt_template")
    @patch("arklex.env.tools.utils.log_context")
    @patch("arklex.env.tools.utils.trace")
    def test_stream_context_generate_with_multiple_records(
        self,
        mock_trace: Mock,
        mock_log_context: Mock,
        mock_get_prompt: Mock,
        mock_provider_map: Mock,
    ) -> None:
        """Test ToolGenerator.stream_context_generate method with multiple records."""
        # Setup
        state = Mock(spec=MessageState)
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.llm_provider = "openai"
        state.bot_config.llm_config.model_type_or_path = "gpt-3.5-turbo"
        state.user_message = Mock()
        state.user_message.history = "test history"
        state.sys_instruct = "test instruction"
        state.message_flow = "test flow"
        # Multiple records with different structures
        state.relevant_records = [
            Mock(
                info="info1",
                personalized_intent="intent1",
                output="output1",
                steps=[{"step1": "value1"}],
            ),
            Mock(
                info="info2",
                personalized_intent=None,
                output="output2",
                steps=["step2"],
            ),
            Mock(
                info=None,
                personalized_intent="intent3",
                output=None,
                steps=[],
            ),
        ]
        state.message_queue = Queue()

        mock_llm = Mock()
        mock_provider_map.get.return_value = Mock(return_value=mock_llm)

        mock_prompt = Mock()
        mock_prompt.invoke.return_value.text = "test prompt"
        mock_get_prompt.return_value = mock_prompt

        mock_chain = Mock()
        mock_chain.stream.return_value = ["chunk1", "chunk2", "chunk3"]
        mock_llm.__or__ = Mock(return_value=mock_chain)

        mock_trace.return_value = state

        # Execute
        result = ToolGenerator.stream_context_generate(state)

        # Assert
        assert result == state
        assert state.response == "chunk1chunk2chunk3"
        assert state.message_flow == ""
        assert state.message_queue.qsize() == 3
        mock_trace.assert_called_once_with(input="chunk1chunk2chunk3", state=state)

    @patch("arklex.env.tools.utils.PROVIDER_MAP")
    @patch("arklex.env.tools.utils.get_prompt_template")
    @patch("arklex.env.tools.utils.log_context")
    def test_stream_generate_with_empty_stream(
        self, mock_log_context: Mock, mock_get_prompt: Mock, mock_provider_map: Mock
    ) -> None:
        """Test ToolGenerator.stream_generate method with empty stream response."""
        # Setup
        state = Mock(spec=MessageState)
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.llm_provider = "openai"
        state.bot_config.llm_config.model_type_or_path = "gpt-3.5-turbo"
        state.user_message = Mock()
        state.user_message.history = "test history"
        state.sys_instruct = "test instruction"
        state.message_queue = Queue()

        mock_llm = Mock()
        mock_provider_map.get.return_value = Mock(return_value=mock_llm)

        mock_prompt = Mock()
        mock_prompt.invoke.return_value.text = "test prompt"
        mock_get_prompt.return_value = mock_prompt

        mock_chain = Mock()
        mock_chain.stream.return_value = []  # Empty stream
        mock_llm.__or__ = Mock(return_value=mock_chain)

        # Execute
        result = ToolGenerator.stream_generate(state)

        # Assert
        assert result == state
        assert state.response == ""
        assert state.message_queue.qsize() == 0

    @patch("arklex.env.tools.utils.PROVIDER_MAP")
    @patch("arklex.env.tools.utils.get_prompt_template")
    @patch("arklex.env.tools.utils.log_context")
    def test_stream_generate_with_single_chunk(
        self, mock_log_context: Mock, mock_get_prompt: Mock, mock_provider_map: Mock
    ) -> None:
        """Test ToolGenerator.stream_generate method with single chunk response."""
        # Setup
        state = Mock(spec=MessageState)
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.llm_provider = "openai"
        state.bot_config.llm_config.model_type_or_path = "gpt-3.5-turbo"
        state.user_message = Mock()
        state.user_message.history = "test history"
        state.sys_instruct = "test instruction"
        state.message_queue = Queue()

        mock_llm = Mock()
        mock_provider_map.get.return_value = Mock(return_value=mock_llm)

        mock_prompt = Mock()
        mock_prompt.invoke.return_value.text = "test prompt"
        mock_get_prompt.return_value = mock_prompt

        mock_chain = Mock()
        mock_chain.stream.return_value = ["single_chunk"]  # Single chunk
        mock_llm.__or__ = Mock(return_value=mock_chain)

        # Execute
        result = ToolGenerator.stream_generate(state)

        # Assert
        assert result == state
        assert state.response == "single_chunk"
        assert state.message_queue.qsize() == 1


class TestTrace:
    """Test cases for trace function."""

    @patch("arklex.env.tools.utils.inspect")
    def test_trace(self, mock_inspect: Mock) -> None:
        """Test trace function adds response metadata to state."""
        # Setup
        state = Mock(spec=MessageState)
        state.trajectory = [[Mock(steps=[])]]

        mock_frame = Mock()
        mock_frame.f_back = Mock()
        mock_frame.f_back.f_code.co_name = "test_function"
        mock_inspect.currentframe.return_value = mock_frame

        input_text = "test input"

        # Execute
        result = trace(input_text, state)

        # Assert
        assert result == state
        assert len(state.trajectory[-1][-1].steps) == 1
        assert state.trajectory[-1][-1].steps[0]["test_function"] == input_text

    @patch("arklex.env.tools.utils.inspect")
    def test_trace_no_current_frame(self, mock_inspect: Mock) -> None:
        """Test trace function when no current frame is available."""
        # Setup
        state = Mock(spec=MessageState)
        state.trajectory = [[Mock(steps=[])]]

        mock_inspect.currentframe.return_value = None

        input_text = "test input"

        # Execute
        result = trace(input_text, state)

        # Assert
        assert result == state
        assert len(state.trajectory[-1][-1].steps) == 1
        assert state.trajectory[-1][-1].steps[0]["unknown"] == input_text

    @patch("arklex.env.tools.utils.inspect")
    def test_trace_no_previous_frame(self, mock_inspect: Mock) -> None:
        """Test trace function when no previous frame is available."""
        # Setup
        state = Mock(spec=MessageState)
        state.trajectory = [[Mock(steps=[])]]

        mock_frame = Mock()
        mock_frame.f_back = None
        mock_inspect.currentframe.return_value = mock_frame

        input_text = "test input"

        # Execute
        result = trace(input_text, state)

        # Assert
        assert result == state
        assert len(state.trajectory[-1][-1].steps) == 1
        assert state.trajectory[-1][-1].steps[0]["unknown"] == input_text


class TestExecuteTool:
    """Test cases for execute_tool function."""

    @patch("arklex.env.tools.utils.log_context")
    def test_execute_tool_success(self, mock_log_context: Mock) -> None:
        """Test execute_tool function with successful execution."""
        # Setup
        tool_instance = Mock()
        tool_instance.tools = {"test_tool": Mock()}
        tool_instance.tools["test_tool"].execute.return_value = "test result"

        tool_name = "test_tool"
        kwargs = {"param1": "value1", "param2": "value2"}

        # Execute
        result = execute_tool(tool_instance, tool_name, **kwargs)

        # Assert
        assert result == "test result"
        tool_instance.tools["test_tool"].execute.assert_called_once_with(**kwargs)
        mock_log_context.info.assert_called()

    @patch("arklex.env.tools.utils.log_context")
    def test_execute_tool_failure(self, mock_log_context: Mock) -> None:
        """Test execute_tool function with failed execution."""
        # Setup
        tool_instance = Mock()
        tool_instance.tools = {"test_tool": Mock()}
        tool_instance.tools["test_tool"].execute.side_effect = Exception("Tool error")

        tool_name = "test_tool"
        kwargs = {"param1": "value1"}

        # Execute & Assert
        with pytest.raises(ToolError) as exc_info:
            execute_tool(tool_instance, tool_name, **kwargs)

        assert "Tool execution failed: test_tool" in str(exc_info.value)
        mock_log_context.error.assert_called()

    @patch("arklex.env.tools.utils.log_context")
    def test_execute_tool_missing_tool(self, mock_log_context: Mock) -> None:
        """Test execute_tool function with missing tool."""
        # Setup
        tool_instance = Mock()
        tool_instance.tools = {}

        tool_name = "missing_tool"
        kwargs = {"param1": "value1"}

        # Execute & Assert
        with pytest.raises(ToolError) as exc_info:
            execute_tool(tool_instance, tool_name, **kwargs)

        assert "Tool execution failed: missing_tool" in str(exc_info.value)
        mock_log_context.error.assert_called()
