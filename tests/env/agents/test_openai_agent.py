from unittest.mock import Mock, patch

import pytest

from arklex.env.agents.openai_agent import OpenAIAgent, end_conversation
from arklex.utils.graph_state import MessageState, StatusEnum


@pytest.fixture
def mock_state() -> MessageState:
    """Create a mock MessageState for testing."""
    state = Mock(spec=MessageState)
    state.status = StatusEnum.INCOMPLETE
    state.function_calling_trajectory = []
    state.message_flow = ""
    state.response = ""
    state.sys_instruct = "System instructions"

    # Mock orchestrator_message
    state.orchestrator_message = Mock()
    state.orchestrator_message.message = "User message"

    # Mock bot_config
    state.bot_config = Mock()
    state.bot_config.llm_config = Mock()
    state.bot_config.llm_config.llm_provider = "openai"
    state.bot_config.llm_config.model_type_or_path = "gpt-3.5-turbo"

    return state


@pytest.fixture
def mock_tools() -> dict:
    """Create mock tools for testing."""
    tool_object = Mock()
    tool_object.to_openai_tool_def_v2.return_value = {
        "type": "function",
        "function": {"name": "mock_tool", "description": "Mock tool", "parameters": {}},
    }
    tool_object.func = Mock(return_value="tool result")

    return {
        "mock_tool_id": {
            "execute": lambda: tool_object,
            "fixed_args": {"fixed_param": "value"},
        }
    }


@pytest.fixture
def mock_nodes() -> list:
    """Create mock nodes for testing."""
    node = Mock()
    node.resource_id = "mock_tool_id"
    return [node]


class TestOpenAIAgent:
    @patch("arklex.env.agents.openai_agent.load_prompts")
    @patch("arklex.env.agents.openai_agent.trace")
    @patch("arklex.env.agents.openai_agent.PROVIDER_MAP")
    def test_init(
        self,
        mock_provider_map: Mock,
        mock_trace: Mock,
        mock_load_prompts: Mock,
        mock_state: Mock,
        mock_tools: Mock,
        mock_nodes: Mock,
    ) -> None:
        """Test OpenAIAgent initialization."""
        mock_trace.return_value = mock_state
        mock_load_prompts.return_value = {
            "function_calling_agent_prompt": "Test prompt"
        }

        agent = OpenAIAgent(
            successors=mock_nodes,
            predecessors=[],
            tools=mock_tools,
            state=mock_state,
        )

        assert agent.action_graph is not None
        assert agent.llm is None
        assert "mock_tool_id" in agent.available_tools
        assert "end_conversation" in agent.tool_map
        assert len(agent.tool_defs) >= 1

    @patch("arklex.env.agents.openai_agent.load_prompts")
    @patch("arklex.env.agents.openai_agent.trace")
    def test_generate_incomplete_status_no_prompt(
        self,
        mock_trace,  # noqa: ANN001
        mock_load_prompts,  # noqa: ANN001
        mock_state,  # noqa: ANN001
    ) -> None:
        """Test generate method with incomplete status and no prompt."""
        mock_load_prompts.return_value = {
            "function_calling_agent_prompt": "Test prompt: {sys_instruct} {message}"
        }
        mock_trace.return_value = mock_state

        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        mock_ai_message = Mock()
        mock_ai_message.content = "Test response"
        mock_ai_message.tool_calls = None

        agent.llm = Mock()
        agent.llm.invoke.return_value = mock_ai_message
        agent.prompt = ""

        result = agent.generate(mock_state)

        assert result.response == "Test response"
        assert len(mock_state.function_calling_trajectory) > 0
        mock_trace.assert_called_once()

    @patch("arklex.env.agents.openai_agent.trace")
    def test_generate_with_existing_prompt(self, mock_trace, mock_state) -> None:  # noqa: ANN001
        """Test generate method with existing prompt."""
        mock_trace.return_value = mock_state

        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        mock_ai_message = Mock()
        mock_ai_message.content = "Test response"
        mock_ai_message.tool_calls = None

        agent.llm = Mock()
        agent.llm.invoke.return_value = mock_ai_message
        agent.prompt = "existing prompt"

        result = agent.generate(mock_state)

        assert result.response == "Test response"

    @patch("arklex.env.agents.openai_agent.trace")
    @patch("arklex.env.agents.openai_agent.json.dumps")
    def test_generate_with_tool_calls(
        self,
        mock_json_dumps,  # noqa: ANN001
        mock_trace,  # noqa: ANN001
        mock_state,  # noqa: ANN001
        mock_tools,  # noqa: ANN001
    ) -> None:
        """Test generate method with tool calls."""
        mock_json_dumps.return_value = '{"result": "success"}'
        mock_trace.return_value = mock_state

        tool_call = {"name": "mock_tool", "args": {"param": "value"}, "id": "call_123"}

        # Mock AI messages for tool call and final response
        mock_ai_message_with_tools = Mock()
        mock_ai_message_with_tools.content = "Calling tool"
        mock_ai_message_with_tools.tool_calls = [tool_call]

        mock_ai_message_final = Mock()
        mock_ai_message_final.content = "Final response"
        mock_ai_message_final.tool_calls = None

        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools=mock_tools,
            state=mock_state,
        )

        # Setup tool map
        agent.tool_map["mock_tool"] = Mock(return_value="tool result")
        agent.tool_args["mock_tool"] = {}

        agent.llm = Mock()
        agent.llm.invoke.side_effect = [
            mock_ai_message_with_tools,
            mock_ai_message_final,
        ]
        agent.prompt = "test prompt"

        result = agent.generate(mock_state)

        assert result.response == "Final response"
        assert agent.llm.invoke.call_count == 2

    @patch("arklex.env.agents.openai_agent.trace")
    def test_generate_with_unknown_tool_call(self, mock_trace, mock_state) -> None:  # noqa: ANN001
        """Test generate method with unknown tool call."""
        mock_trace.return_value = mock_state

        tool_call = {
            "name": "unknown_tool",
            "args": {"param": "value"},
            "id": "call_123",
        }

        mock_ai_message = Mock()
        mock_ai_message.content = "Response"
        mock_ai_message.tool_calls = [tool_call]

        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        agent.llm = Mock()
        agent.llm.invoke.return_value = mock_ai_message
        agent.prompt = "test prompt"

        result = agent.generate(mock_state)

        assert result.response == "Response"

    @patch("arklex.env.agents.openai_agent.trace")
    def test_generate_with_complete_status(self, mock_trace, mock_state) -> None:  # noqa: ANN001
        """Test generate method when status is COMPLETE."""
        mock_trace.return_value = mock_state
        mock_state.status = StatusEnum.COMPLETE

        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        mock_ai_message = Mock()
        mock_ai_message.content = "Response"
        mock_ai_message.tool_calls = None

        agent.llm = Mock()
        agent.llm.invoke.return_value = mock_ai_message

        result = agent.generate(mock_state)

        assert result == mock_state

    @patch("arklex.env.agents.openai_agent.load_prompts")
    @patch("arklex.env.agents.openai_agent.trace")
    def test_generate_orchestrator_message_none(
        self,
        mock_trace,  # noqa: ANN001
        mock_load_prompts,  # noqa: ANN001
        mock_state,  # noqa: ANN001
    ) -> None:
        """Test generate method when orchestrator message is None."""
        mock_load_prompts.return_value = {
            "function_calling_agent_prompt": "Test: {sys_instruct} {message}"
        }
        mock_trace.return_value = mock_state
        mock_state.orchestrator_message.message = None

        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        mock_ai_message = Mock()
        mock_ai_message.content = "response"
        mock_ai_message.tool_calls = None

        agent.llm = Mock()
        agent.llm.invoke.return_value = mock_ai_message
        agent.prompt = ""

        result = agent.generate(mock_state)

        assert result.response == "response"

    @patch("arklex.env.agents.openai_agent.load_prompts")
    @patch("arklex.env.agents.openai_agent.trace")
    def test_generate_prompt_already_in_trajectory(
        self,
        mock_trace,  # noqa: ANN001
        mock_load_prompts,  # noqa: ANN001
        mock_state,  # noqa: ANN001
    ) -> None:
        """Test generate method when prompt is already in trajectory."""
        mock_load_prompts.return_value = {
            "function_calling_agent_prompt": "Test: {sys_instruct} {message}"
        }
        mock_trace.return_value = mock_state

        # Add a message to trajectory that matches the prompt
        mock_state.function_calling_trajectory = [
            {"content": "Test: System instructions User message"}
        ]

        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        mock_ai_message = Mock()
        mock_ai_message.content = "response"
        mock_ai_message.tool_calls = None

        agent.llm = Mock()
        agent.llm.invoke.return_value = mock_ai_message
        agent.prompt = ""

        initial_trajectory_length = len(mock_state.function_calling_trajectory)
        agent.generate(mock_state)

        # Should not add duplicate prompt
        assert len(mock_state.function_calling_trajectory) == initial_trajectory_length

    def test_create_action_graph(self, mock_state) -> None:  # noqa: ANN001
        """Test _create_action_graph method."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        graph = agent._create_action_graph()
        assert graph is not None

    @patch("arklex.env.agents.openai_agent.PROVIDER_MAP")
    def test_execute(self, mock_provider_map, mock_state) -> None:  # noqa: ANN001
        """Test _execute method."""
        mock_llm = Mock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_provider_map.get.return_value.return_value = mock_llm

        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        with patch.object(agent.action_graph, "compile") as mock_compile:  # noqa: ANN001
            mock_graph = Mock()
            mock_graph.invoke.return_value = {"result": "success"}
            mock_compile.return_value = mock_graph

            result = agent._execute(mock_state, prompt="custom prompt")

            assert agent.prompt == "custom prompt"
            assert result == {"result": "success"}

    def test_load_tools(self, mock_state, mock_tools, mock_nodes) -> None:  # noqa: ANN001
        """Test _load_tools method."""
        agent = OpenAIAgent(
            successors=mock_nodes,
            predecessors=[],
            tools=mock_tools,
            state=mock_state,
        )

        assert "mock_tool_id" in agent.available_tools

    def test_load_tools_empty_nodes(self, mock_state) -> None:  # noqa: ANN001
        """Test _load_tools method with empty nodes."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        assert len(agent.available_tools) == 0

    def test_load_tools_node_not_in_tools(self, mock_state) -> None:  # noqa: ANN001
        """Test _load_tools method when node not in tools."""
        node = Mock()
        node.resource_id = "nonexistent_tool"

        agent = OpenAIAgent(
            successors=[node],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        assert "nonexistent_tool" not in agent.available_tools

    def test_configure_tools(self, mock_state, mock_tools) -> None:  # noqa: ANN001
        """Test _configure_tools method."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        # Manually add tool to test configure
        agent.available_tools["test_tool_id"] = mock_tools["mock_tool_id"]
        agent._configure_tools()

        assert "mock_tool" in agent.tool_map
        assert "mock_tool" in agent.tool_args
        assert "end_conversation" in agent.tool_map

    def test_configure_tools_empty(self, mock_state) -> None:  # noqa: ANN001
        """Test _configure_tools method with no tools."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        # Should still have end_conversation tool
        assert "end_conversation" in agent.tool_map
        assert len(agent.tool_defs) == 1


class TestEndConversation:
    @patch("arklex.env.agents.openai_agent.PROVIDER_MAP")
    def test_end_conversation_success(self, mock_provider_map, mock_state) -> None:  # noqa: ANN001
        """Test end_conversation function success case."""
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = "Thank you for using Arklex. Goodbye!"
        mock_provider_map.get.return_value.return_value = mock_llm

        result = end_conversation().func(mock_state)

        assert mock_state.status == StatusEnum.COMPLETE
        assert result == "Thank you for using Arklex. Goodbye!"

    @patch("arklex.env.agents.openai_agent.PROVIDER_MAP")
    def test_end_conversation_error(self, mock_provider_map, mock_state) -> None:  # noqa: ANN001
        """Test end_conversation function error case."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM Error")
        mock_provider_map.get.return_value.return_value = mock_llm

        result = end_conversation().func(mock_state)

        assert mock_state.status == StatusEnum.COMPLETE
        assert result == "I hope I was able to help you today. Goodbye!"


class TestEdgeCases:
    def test_description_attribute(self, mock_state) -> None:  # noqa: ANN001
        """Test that OpenAIAgent has correct description."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        assert agent.description == "General-purpose Arklex agent for chat or voice."

    @patch("arklex.env.agents.openai_agent.trace")
    def test_generate_empty_tool_args(self, mock_trace, mock_state) -> None:  # noqa: ANN001
        """Test generate method with tool call but empty tool_args."""
        mock_trace.return_value = mock_state

        tool_call = {"name": "test_tool", "args": {"param": "value"}, "id": "call_123"}

        mock_ai_message_with_tools = Mock()
        mock_ai_message_with_tools.content = "Calling tool"
        mock_ai_message_with_tools.tool_calls = [tool_call]

        mock_ai_message_final = Mock()
        mock_ai_message_final.content = "Final response"
        mock_ai_message_final.tool_calls = None

        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        agent.tool_map["test_tool"] = Mock(return_value="tool result")
        # Don't add to tool_args to test empty case

        agent.llm = Mock()
        agent.llm.invoke.side_effect = [
            mock_ai_message_with_tools,
            mock_ai_message_final,
        ]
        agent.prompt = "test prompt"

        result = agent.generate(mock_state)

        assert result.response == "Final response"

    @patch("arklex.env.agents.openai_agent.trace")
    @patch("arklex.env.agents.openai_agent.json.dumps")
    def test_generate_tool_args_merge(
        self,
        mock_json_dumps,  # noqa: ANN001
        mock_trace,  # noqa: ANN001
        mock_state,  # noqa: ANN001
    ) -> None:
        """Test generate method merges tool args correctly."""
        mock_json_dumps.return_value = '{"result": "success"}'
        mock_trace.return_value = mock_state

        tool_call = {"name": "test_tool", "args": {"param": "value"}, "id": "call_123"}

        mock_ai_message_with_tools = Mock()
        mock_ai_message_with_tools.content = "Calling tool"
        mock_ai_message_with_tools.tool_calls = [tool_call]

        mock_ai_message_final = Mock()
        mock_ai_message_final.content = "Final response"
        mock_ai_message_final.tool_calls = None

        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        mock_tool_func = Mock(return_value="tool result")
        agent.tool_map["test_tool"] = mock_tool_func
        agent.tool_args["test_tool"] = {"fixed_param": "fixed_value"}

        agent.llm = Mock()
        agent.llm.invoke.side_effect = [
            mock_ai_message_with_tools,
            mock_ai_message_final,
        ]
        agent.prompt = "test prompt"

        result = agent.generate(mock_state)

        # Verify tool was called with merged args
        mock_tool_func.assert_called_once_with(
            state=mock_state, param="value", fixed_param="fixed_value"
        )
        assert result.response == "Final response"
