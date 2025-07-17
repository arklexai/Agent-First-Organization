"""Test utilities for the Arklex framework."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from arklex.env.tools.utils import ToolGenerator, execute_tool, trace
from arklex.orchestrator.entities.msg_state_entities import MessageState


class TestGetPromptTemplate:
    """Test cases for get_prompt_template function."""

    @patch("arklex.env.tools.utils.load_prompts")
    def test_get_prompt_template_speech_non_chinese(
        self, mock_load_prompts: Mock
    ) -> None:
        """Test get_prompt_template for speech non-Chinese."""
        # Setup
        state = Mock(spec=MessageState)
        state.stream_type = "speech"
        state.bot_config = Mock()
        state.bot_config.language = "EN"

        mock_prompts = {
            "test_prompt": "Regular prompt",
            "test_prompt_speech": "Speech prompt",
        }
        mock_load_prompts.return_value = mock_prompts

        # Execute
        from arklex.env.tools.utils import get_prompt_template

        result = get_prompt_template(state, "test_prompt")

        # Assert
        assert result.template == "Speech prompt"
        mock_load_prompts.assert_called_once_with(state.bot_config)

    @patch("arklex.env.tools.utils.load_prompts")
    def test_get_prompt_template_speech_chinese(self, mock_load_prompts: Mock) -> None:
        """Test get_prompt_template for speech Chinese."""
        # Setup
        state = Mock(spec=MessageState)
        state.stream_type = "speech"
        state.bot_config = Mock()
        state.bot_config.language = "CN"

        mock_prompts = {
            "test_prompt": "Regular prompt",
            "test_prompt_speech": "Speech prompt",
        }
        mock_load_prompts.return_value = mock_prompts

        # Execute
        from arklex.env.tools.utils import get_prompt_template

        result = get_prompt_template(state, "test_prompt")

        # Assert
        assert result.template == "Regular prompt"
        mock_load_prompts.assert_called_once_with(state.bot_config)

    @patch("arklex.env.tools.utils.load_prompts")
    def test_get_prompt_template_non_speech(self, mock_load_prompts: Mock) -> None:
        """Test get_prompt_template for non-speech."""
        # Setup
        state = Mock(spec=MessageState)
        state.stream_type = "text"
        state.bot_config = Mock()

        mock_prompts = {
            "test_prompt": "Regular prompt",
            "test_prompt_speech": "Speech prompt",
        }
        mock_load_prompts.return_value = mock_prompts

        # Execute
        from arklex.env.tools.utils import get_prompt_template

        result = get_prompt_template(state, "test_prompt")

        # Assert
        assert result.template == "Regular prompt"
        mock_load_prompts.assert_called_once_with(state.bot_config)


class TestToolGenerator:
    """Test cases for ToolGenerator class."""

    @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
    @patch("arklex.env.tools.utils.get_prompt_template")
    @patch("arklex.env.tools.utils.log_context")
    @patch("arklex.utils.provider_utils.validate_and_get_model_class")
    def test_generate(
        self,
        mock_validate_model: Mock,
        mock_log_context: Mock,
        mock_get_prompt: Mock,
        mock_provider_map: Mock,
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
        mock_validate_model.return_value = Mock(return_value=mock_llm)

        mock_prompt = Mock()
        mock_prompt.invoke.return_value.text = "test prompt"
        mock_get_prompt.return_value = mock_prompt

        # Create a proper mock chain that returns the expected string
        mock_chain = Mock()
        mock_chain.invoke.return_value = "test response"
        # Mock the __or__ method to return the chain
        mock_llm.__or__ = Mock(return_value=mock_chain)

        # Execute
        result = ToolGenerator.generate(state)

        # Assert
        assert result == state
        assert state.response == '{"result": "dummy response"}'

    @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
    @patch("arklex.env.tools.utils.get_prompt_template")
    @patch("arklex.env.tools.utils.log_context")
    @patch("arklex.utils.provider_utils.validate_and_get_model_class")
    def test_context_generate_with_dict_steps(
        self,
        mock_validate_model: Mock,
        mock_log_context: Mock,
        mock_get_prompt: Mock,
        mock_provider_map: Mock,
    ) -> None:
        """Test ToolGenerator.context_generate method with dict steps."""
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
                steps=[{"step1": "value1", "step2": "value2"}],
            )
        ]
        # Add mock trajectory to prevent AttributeError in trace
        mock_trajectory_item = Mock()
        mock_trajectory_item.steps = []
        state.trajectory = [[mock_trajectory_item]]

        mock_llm = Mock()
        mock_validate_model.return_value = Mock(return_value=mock_llm)

        mock_prompt = Mock()
        mock_prompt.invoke.return_value.text = "test prompt"
        mock_get_prompt.return_value = mock_prompt

        # Create a proper mock chain that returns the expected string
        mock_chain = Mock()
        mock_chain.invoke.return_value = "test response"
        # Mock the __or__ method to return the chain
        mock_llm.__or__ = Mock(return_value=mock_chain)

        # Execute
        result = ToolGenerator.context_generate(state)

        # Assert
        assert result == state
        assert state.response == '{"result": "dummy response"}'
        assert state.message_flow == ""

    @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
    @patch("arklex.env.tools.utils.get_prompt_template")
    @patch("arklex.env.tools.utils.log_context")
    @patch("arklex.utils.provider_utils.validate_and_get_model_class")
    def test_context_generate_without_relevant_records(
        self,
        mock_validate_model: Mock,
        mock_log_context: Mock,
        mock_get_prompt: Mock,
        mock_provider_map: Mock,
    ) -> None:
        """Test ToolGenerator.context_generate method without relevant_records."""
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
        state.relevant_records = None  # This is the missing coverage case
        # Add mock trajectory to prevent AttributeError in trace
        mock_trajectory_item = Mock()
        mock_trajectory_item.steps = []
        state.trajectory = [[mock_trajectory_item]]

        mock_llm = Mock()
        mock_validate_model.return_value = Mock(return_value=mock_llm)

        mock_prompt = Mock()
        mock_prompt.invoke.return_value.text = "test prompt"
        mock_get_prompt.return_value = mock_prompt

        # Create a proper mock chain that returns the expected string
        mock_chain = Mock()
        mock_chain.invoke.return_value = "test response"
        # Mock the __or__ method to return the chain
        mock_llm.__or__ = Mock(return_value=mock_chain)

        # Execute
        result = ToolGenerator.context_generate(state)

        # Assert
        assert result == state
        assert state.response == '{"result": "dummy response"}'
        assert state.message_flow == ""

    @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
    @patch("arklex.env.tools.utils.get_prompt_template")
    @patch("arklex.env.tools.utils.log_context")
    @patch("arklex.utils.provider_utils.validate_and_get_model_class")
    def test_stream_context_generate(
        self,
        mock_provider_map: Mock,
        mock_get_prompt: Mock,
        mock_log_context: Mock,
        mock_validate_model: Mock,
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
        state.relevant_records = [
            Mock(
                info="test info",
                personalized_intent="test intent",
                output="test output",
                steps=[{"step1": "value1", "step2": "value2"}],
            )
        ]
        state.message_queue = Mock()
        # Add mock trajectory to prevent AttributeError in trace
        mock_trajectory_item = Mock()
        mock_trajectory_item.steps = []
        state.trajectory = [[mock_trajectory_item]]

        # Create a proper mock LLM that can handle the __or__ operation
        mock_llm = Mock()
        mock_validate_model.return_value = Mock(return_value=mock_llm)

        mock_prompt = Mock()
        mock_prompt.invoke.return_value.text = "test prompt"
        mock_get_prompt.return_value = mock_prompt

        # Patch RunnableSequence.stream to return our mock iterator
        with patch(
            "langchain_core.runnables.base.RunnableSequence.stream",
            return_value=iter(["chunk1", "chunk2", "chunk3"]),
        ):
            # Execute
            result = ToolGenerator.stream_context_generate(state)

        # Assert
        assert result == state
        assert state.response == "chunk1chunk2chunk3"
        assert state.message_flow == ""
        assert state.message_queue.put.call_count == 3

    @patch("arklex.env.tools.utils.StrOutputParser")
    @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
    @patch("arklex.env.tools.utils.get_prompt_template")
    @patch("arklex.env.tools.utils.log_context")
    @patch("arklex.utils.provider_utils.validate_and_get_model_class")
    def test_stream_context_generate_without_relevant_records(
        self,
        mock_validate_model: Mock,
        mock_log_context: Mock,
        mock_get_prompt: Mock,
        mock_provider_map: Mock,
        mock_str_output_parser: Mock,
    ) -> None:
        """Test ToolGenerator.stream_context_generate method without relevant_records."""
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
        state.relevant_records = None
        state.message_queue = Mock()
        mock_trajectory_item = Mock()
        mock_trajectory_item.steps = []
        state.trajectory = [[mock_trajectory_item]]

        # Create a proper mock LLM that can handle the __or__ operation
        mock_llm = Mock()
        mock_validate_model.return_value = Mock(return_value=mock_llm)
        mock_prompt = Mock()
        mock_prompt.invoke.return_value.text = "test prompt"
        mock_get_prompt.return_value = mock_prompt
        mock_str_output_parser.return_value = Mock()

        # Patch RunnableSequence.stream to return our mock iterator
        with patch(
            "langchain_core.runnables.base.RunnableSequence.stream",
            return_value=iter(["chunk1", "chunk2"]),
        ):
            result = ToolGenerator.stream_context_generate(state)
        assert result == state
        assert state.response == "chunk1chunk2"
        assert state.message_flow == ""
        assert state.message_queue.put.call_count == 2

    @patch("arklex.utils.provider_utils.validate_and_get_model_class")
    @patch("arklex.env.tools.utils.log_context")
    @patch("arklex.env.tools.utils.get_prompt_template")
    @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
    def test_stream_generate(
        self,
        mock_provider_map: Mock,
        mock_get_prompt: Mock,
        mock_log_context: Mock,
        mock_validate_model: Mock,
    ) -> None:
        """Test ToolGenerator.stream_generate method."""
        state = Mock(spec=MessageState)
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.llm_provider = "openai"
        state.bot_config.llm_config.model_type_or_path = "gpt-3.5-turbo"
        state.user_message = Mock()
        state.user_message.history = "test history"
        state.sys_instruct = "test instruction"
        state.message_queue = Mock()
        mock_llm = MagicMock()
        mock_validate_model.return_value = Mock(return_value=mock_llm)
        mock_prompt = Mock()
        mock_prompt.invoke.return_value.text = "test prompt"
        mock_get_prompt.return_value = mock_prompt
        # Patch RunnableSequence.stream to return our mock iterator
        with patch(
            "langchain_core.runnables.base.RunnableSequence.stream",
            return_value=iter(["chunk1", "chunk2", "chunk3"]),
        ):
            from arklex.env.tools import utils as utils_mod

            result = utils_mod.ToolGenerator.stream_generate(state)
            assert result == state
            assert state.response == "chunk1chunk2chunk3"
            assert state.message_queue.put.call_count == 3

    @patch("arklex.env.tools.utils.StrOutputParser")
    @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
    @patch("arklex.env.tools.utils.get_prompt_template")
    @patch("arklex.env.tools.utils.log_context")
    @patch("arklex.utils.provider_utils.validate_and_get_model_class")
    def test_stream_generate_empty_stream(
        self,
        mock_validate_model: Mock,
        mock_log_context: Mock,
        mock_get_prompt: Mock,
        mock_provider_map: Mock,
        mock_str_output_parser: Mock,
    ) -> None:
        """Test ToolGenerator.stream_generate method with empty stream."""
        state = Mock(spec=MessageState)
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.llm_provider = "openai"
        state.bot_config.llm_config.model_type_or_path = "gpt-3.5-turbo"
        state.user_message = Mock()
        state.user_message.history = "test history"
        state.sys_instruct = "test instruction"
        state.message_queue = Mock()
        # Create a proper mock LLM that can handle the __or__ operation
        mock_llm = Mock()
        mock_validate_model.return_value = Mock(return_value=mock_llm)
        mock_prompt = Mock()
        mock_prompt.invoke.return_value.text = "test prompt"
        mock_get_prompt.return_value = mock_prompt
        mock_str_output_parser.return_value = Mock()

        # Patch RunnableSequence.stream to return our mock iterator
        with patch(
            "langchain_core.runnables.base.RunnableSequence.stream",
            return_value=iter([]),
        ):
            result = ToolGenerator.stream_generate(state)
        assert result == state
        assert state.response == ""
        assert state.message_queue.put.call_count == 0


class TestTrace:
    """Test cases for trace function."""

    @patch("arklex.env.tools.utils.inspect")
    def test_trace(self, mock_inspect: Mock) -> None:
        """Test trace function."""
        # Setup
        state = Mock(spec=MessageState)
        state.trajectory = [[Mock()]]
        state.trajectory[-1][-1].steps = []

        mock_frame = Mock()
        mock_frame.f_back = Mock()
        mock_frame.f_back.f_code.co_name = "test_function"
        mock_inspect.currentframe.return_value = mock_frame

        # Execute
        result = trace("test input", state)

        # Assert
        assert result == state
        assert len(state.trajectory[-1][-1].steps) == 1
        assert state.trajectory[-1][-1].steps[0] == {"test_function": "test input"}

    @patch("arklex.env.tools.utils.inspect")
    def test_trace_no_current_frame(self, mock_inspect: Mock) -> None:
        """Test trace function when no current frame is available."""
        # Setup
        state = Mock(spec=MessageState)
        state.trajectory = [[Mock()]]
        state.trajectory[-1][-1].steps = []

        mock_inspect.currentframe.return_value = None

        # Execute
        result = trace("test input", state)

        # Assert
        assert result == state
        assert len(state.trajectory[-1][-1].steps) == 1
        assert state.trajectory[-1][-1].steps[0] == {"unknown": "test input"}

    @patch("arklex.env.tools.utils.inspect")
    def test_trace_no_previous_frame(self, mock_inspect: Mock) -> None:
        """Test trace function when no previous frame is available."""
        # Setup
        state = Mock(spec=MessageState)
        state.trajectory = [[Mock()]]
        state.trajectory[-1][-1].steps = []

        mock_frame = Mock()
        mock_frame.f_back = None
        mock_inspect.currentframe.return_value = mock_frame

        # Execute
        result = trace("test input", state)

        # Assert
        assert result == state
        assert len(state.trajectory[-1][-1].steps) == 1
        assert state.trajectory[-1][-1].steps[0] == {"unknown": "test input"}


class TestExecuteTool:
    """Test cases for execute_tool function."""

    @patch("arklex.env.tools.utils.log_context")
    def test_execute_tool_success(self, mock_log_context: Mock) -> None:
        """Test execute_tool function with successful execution."""
        # Setup
        mock_self = Mock()
        mock_self.tools = {"test_tool": Mock()}
        mock_self.tools["test_tool"].execute.return_value = "test result"

        # Execute
        result = execute_tool(mock_self, "test_tool")

        # Assert
        assert result == "test result"
        mock_self.tools["test_tool"].execute.assert_called_once()

    @patch("arklex.env.tools.utils.log_context")
    def test_execute_tool_failure(self, mock_log_context: Mock) -> None:
        """Test execute_tool function with execution failure."""
        # Setup
        mock_self = Mock()
        mock_self.tools = {"test_tool": Mock()}
        mock_self.tools["test_tool"].execute.side_effect = Exception("test error")

        # Execute and Assert
        from arklex.utils.exceptions import ToolError

        with pytest.raises(ToolError):
            execute_tool(mock_self, "test_tool")

    @patch("arklex.env.tools.utils.log_context")
    def test_execute_tool_missing_tool(self, mock_log_context: Mock) -> None:
        """Test execute_tool function with missing tool."""
        # Setup
        mock_self = Mock()
        mock_self.tools = {}

        # Execute and Assert
        from arklex.utils.exceptions import ToolError

        with pytest.raises(ToolError):
            execute_tool(mock_self, "missing_tool")
