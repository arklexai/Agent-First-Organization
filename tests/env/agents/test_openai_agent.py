from unittest.mock import Mock, patch

import pytest

from arklex.env.agents.openai_agent import OpenAIAgent
from arklex.orchestrator.entities.msg_state_entities import MessageState, StatusEnum

pytestmark = pytest.mark.usefixtures("patch_openai")


@pytest.fixture(autouse=True)
def patch_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch openai.ChatCompletion.create to always return a mock response
    with patch("openai.ChatCompletion.create") as mock_create:
        mock_create.return_value = {
            "choices": [{"message": {"content": "mocked response"}}]
        }
        yield


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
    tool_func = Mock(return_value="tool result")
    tool_func.__name__ = "mock_tool"
    tool_object.func = tool_func

    return {
        "mock_tool_id": {
            "tool_instance": tool_object,
            "execute": lambda: tool_object,
            "fixed_args": {"fixed_param": "value"},
        }
    }


@pytest.fixture
def mock_nodes() -> list:
    """Create mock nodes for testing."""
    node = Mock()
    node.resource_id = "mock_tool_id"
    node.type = "tool"
    node.attributes = {}
    node.additional_args = {}
    return [node]


class TestOpenAIAgent:
    @patch("arklex.env.agents.openai_agent.load_prompts")
    @patch("arklex.env.agents.openai_agent.trace")
    @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
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
            successors=[],
            predecessors=mock_nodes,
            tools=mock_tools,
            state=mock_state,
        )

        assert agent.action_graph is not None
        assert agent.llm is None
        assert "mock_tool_id" in agent.available_tools
        assert len(agent.tool_defs) >= 1

    @patch("arklex.env.agents.openai_agent.load_prompts")
    @patch("arklex.env.agents.openai_agent.trace")
    def test_generate_incomplete_status_no_prompt(
        self,
        mock_trace: Mock,
        mock_load_prompts: Mock,
        mock_state: Mock,
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
    def test_generate_with_existing_prompt(
        self, mock_trace: Mock, mock_state: Mock
    ) -> None:
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
        mock_json_dumps: Mock,
        mock_trace: Mock,
        mock_state: Mock,
        mock_tools: Mock,
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
    def test_generate_with_unknown_tool_call(
        self, mock_trace: Mock, mock_state: Mock
    ) -> None:
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
    def test_generate_with_complete_status(
        self, mock_trace: Mock, mock_state: Mock
    ) -> None:
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
        mock_trace: Mock,
        mock_load_prompts: Mock,
        mock_state: Mock,
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
        mock_trace: Mock,
        mock_load_prompts: Mock,
        mock_state: Mock,
    ) -> None:
        """Test generate method when prompt is already in trajectory."""
        mock_load_prompts.return_value = {
            "function_calling_agent_prompt": "Test: {sys_instruct} {message}"
        }
        mock_trace.return_value = mock_state

        # Add existing prompt to trajectory as a proper message dict
        initial_trajectory_length = len(mock_state.function_calling_trajectory)
        mock_state.function_calling_trajectory.append({"content": "existing prompt"})

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
        agent.prompt = "existing prompt"

        result = agent.generate(mock_state)

        assert result.response == "response"
        # The trajectory should not change since the prompt is already there
        assert (
            len(mock_state.function_calling_trajectory) == initial_trajectory_length + 1
        )

    def test_create_action_graph(self, mock_state: Mock) -> None:
        """Test _create_action_graph method."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        graph = agent._create_action_graph()
        assert graph is not None

    @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
    def test_execute(self, mock_provider_map: Mock, mock_state: Mock) -> None:
        """Test _execute method."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="test response")
        mock_provider_map.get.return_value = mock_llm

        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        # Mock the required dependencies
        agent.llm = Mock()
        agent.llm.bind_tools.return_value = agent.llm
        mock_compiled_graph = Mock()
        mock_compiled_graph.invoke.return_value = {"result": "success"}
        agent.action_graph = Mock()
        agent.action_graph.compile.return_value = mock_compiled_graph

        result = agent._execute(mock_state)

        assert isinstance(result, dict)

    def test_load_tools(
        self, mock_state: Mock, mock_tools: Mock, mock_nodes: Mock
    ) -> None:
        """Test _load_tools method."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=mock_nodes,
            tools=mock_tools,
            state=mock_state,
        )

        # Test that tools are loaded during initialization
        assert "mock_tool_id" in agent.available_tools

    def test_load_tools_empty_nodes(self, mock_state: Mock) -> None:
        """Test _load_tools method with empty nodes."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        # Test that no tools are loaded when nodes are empty
        assert len(agent.available_tools) == 0

    def test_load_tools_node_not_in_tools(self, mock_state: Mock) -> None:
        """Test _load_tools method when node not in tools."""
        node = Mock()
        node.resource_id = "nonexistent_tool"

        agent = OpenAIAgent(
            successors=[node],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        # Test that non-existent tool is not loaded
        assert "nonexistent_tool" not in agent.available_tools

    def test_configure_tools(self, mock_state: Mock, mock_tools: Mock) -> None:
        """Test _configure_tools method."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools=mock_tools,
            state=mock_state,
        )

        # Manually add tool to test configure
        mock_node = Mock()
        mock_node.attributes = {}
        mock_node.additional_args = {}
        agent.available_tools["test_tool_id"] = (mock_tools["mock_tool_id"], mock_node)
        agent._configure_tools()

        assert "test_tool_id" in agent.tool_map
        assert "test_tool_id" in agent.tool_args
        assert agent.tool_map["test_tool_id"].__name__ == "mock_tool"

    def test_configure_tools_empty(self, mock_state: Mock) -> None:
        """Test _configure_tools method with no tools."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        agent._configure_tools()

    @patch("arklex.env.agents.openai_agent.load_prompts")
    @patch("arklex.env.agents.openai_agent.trace")
    def test_generate_adds_prompt_if_not_in_trajectory(
        self,
        mock_trace,  # noqa: ANN001
        mock_load_prompts,  # noqa: ANN001
        mock_state,  # noqa: ANN001
    ) -> None:
        """Test that the prompt is appended if not already present in trajectory."""
        mock_load_prompts.return_value = {
            "function_calling_agent_prompt": "Test: {sys_instruct} {message}"
        }
        mock_trace.return_value = mock_state
        mock_state.function_calling_trajectory = []

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

        _result = agent.generate(mock_state)
        # Should have appended the prompt
        assert len(mock_state.function_calling_trajectory) == 1
        assert mock_state.function_calling_trajectory[0]["content"].startswith("Test:")

    @patch("arklex.env.agents.openai_agent.load_prompts")
    @patch("arklex.env.agents.openai_agent.trace")
    def test_generate_does_not_add_duplicate_prompt(
        self,
        mock_trace,  # noqa: ANN001
        mock_load_prompts,  # noqa: ANN001
        mock_state,  # noqa: ANN001
    ) -> None:
        """Test that the prompt is NOT appended if already present in trajectory."""
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
        mock_ai_message.content = "Test response"
        mock_ai_message.tool_calls = None
        agent.llm = Mock()
        agent.llm.invoke.return_value = mock_ai_message
        agent.prompt = ""

        initial_len = len(mock_state.function_calling_trajectory)
        _result = agent.generate(mock_state)
        # Should NOT have appended a duplicate
        assert len(mock_state.function_calling_trajectory) == initial_len
