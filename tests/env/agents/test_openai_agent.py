from unittest.mock import Mock, patch

import pytest

from arklex.env.agents.openai_agent import OpenAIAgent, end_conversation
from arklex.orchestrator.entities.msg_state_entities import MessageState, StatusEnum


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
        assert "end_conversation" in agent.tool_map
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
        assert "end_conversation" in agent.tool_map

    def test_configure_tools_empty(self, mock_state: Mock) -> None:
        """Test _configure_tools method with no tools."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        agent._configure_tools()

        assert "end_conversation" in agent.tool_map

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


class TestEndConversation:
    @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
    @patch("arklex.env.agents.openai_agent.json.dumps")
    def test_end_conversation_success(
        self, mock_json_dumps: Mock, mock_provider_map: Mock, mock_state: Mock
    ) -> None:
        """Test end_conversation function success case."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="test response")
        mock_provider_map.get.return_value = mock_llm

        # Call the function that creates the tool, then call the tool
        tool_creator = end_conversation
        tool_instance = tool_creator()
        result = tool_instance.func(mock_state)

        assert isinstance(result, str)

    @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
    @patch("arklex.env.agents.openai_agent.json.dumps")
    def test_end_conversation_error(
        self, mock_json_dumps: Mock, mock_provider_map: Mock, mock_state: Mock
    ) -> None:
        """Test end_conversation function error case."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("Test error")
        mock_provider_map.get.return_value = mock_llm

        # Call the function that creates the tool, then call the tool
        tool_creator = end_conversation
        tool_instance = tool_creator()
        result = tool_instance.func(mock_state)

        assert isinstance(result, str)
        assert "Goodbye" in result


class TestEdgeCases:
    def test_description_attribute(self, mock_state: Mock) -> None:
        """Test that OpenAIAgent has correct description."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        assert agent.description == "General-purpose Arklex agent for chat or voice."

    @patch("arklex.env.agents.openai_agent.trace")
    def test_generate_empty_tool_args(self, mock_trace: Mock, mock_state: Mock) -> None:
        """Test generate method with tool call but empty tool_args."""
        mock_trace.return_value = mock_state

        tool_call = {
            "name": "mock_tool",
            "args": {},
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
    @patch("arklex.env.agents.openai_agent.json.dumps")
    def test_generate_tool_args_merge(
        self,
        mock_json_dumps: Mock,
        mock_trace: Mock,
        mock_state: Mock,
    ) -> None:
        """Test generate method merges tool args correctly."""
        mock_json_dumps.return_value = '{"result": "success"}'
        mock_trace.return_value = mock_state

        tool_call = {
            "name": "mock_tool",
            "args": {"param1": "value1"},
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

        agent.tool_map["mock_tool"] = Mock(return_value="tool result")
        agent.tool_args["mock_tool"] = {"param2": "value2"}

        agent.llm = Mock()
        agent.llm.invoke.return_value = mock_ai_message
        agent.prompt = "test prompt"

        result = agent.generate(mock_state)

        assert result.response == "Response"

    def test_execute_tool_http_tool_with_slots(self, mock_state: Mock) -> None:
        """Test _execute_tool method with http_tool and slots."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        # Mock slot objects
        mock_slot1 = Mock()
        mock_slot1.name = "param1"
        mock_slot2 = Mock()
        mock_slot2.name = "param2"

        agent.tool_slots["http_tool_example"] = [mock_slot1, mock_slot2]
        agent.tool_map["http_tool_example"] = Mock(return_value="http result")

        tool_args = {"param1": "value1", "param3": "value3"}

        result = agent._execute_tool("http_tool_example", mock_state, tool_args)

        assert result == "http result"
        # Verify the tool was called with slots parameter
        agent.tool_map["http_tool_example"].assert_called_once()
        call_args = agent.tool_map["http_tool_example"].call_args
        assert "slots" in call_args.kwargs
        slots = call_args.kwargs["slots"]
        assert len(slots) == 3  # param1, param3, and param2 (missing slot)
        assert {"name": "param1", "value": "value1"} in slots
        assert {"name": "param3", "value": "value3"} in slots
        assert {"name": "param2", "value": None} in slots

    def test_execute_tool_http_tool_with_slots_excluding_slots_param(
        self, mock_state: Mock
    ) -> None:
        """Test _execute_tool method with http_tool when 'slots' is in tool_args."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        # Mock slot objects
        mock_slot1 = Mock()
        mock_slot1.name = "param1"

        agent.tool_slots["http_tool_example"] = [mock_slot1]
        agent.tool_map["http_tool_example"] = Mock(return_value="http result")

        tool_args = {"param1": "value1", "slots": "existing_slots"}

        result = agent._execute_tool("http_tool_example", mock_state, tool_args)

        assert result == "http result"
        # Verify the tool was called with slots parameter
        agent.tool_map["http_tool_example"].assert_called_once()
        call_args = agent.tool_map["http_tool_example"].call_args
        assert "slots" in call_args.kwargs
        slots = call_args.kwargs["slots"]
        assert len(slots) == 1  # Only param1, slots param excluded
        assert {"name": "param1", "value": "value1"} in slots

    def test_execute_tool_http_tool_with_empty_slots(self, mock_state: Mock) -> None:
        """Test _execute_tool method with http_tool and empty slots."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        agent.tool_slots["http_tool_example"] = []
        agent.tool_map["http_tool_example"] = Mock(return_value="http result")

        tool_args = {"param1": "value1"}

        result = agent._execute_tool("http_tool_example", mock_state, tool_args)

        assert result == "http result"
        # Verify the tool was called with empty slots
        agent.tool_map["http_tool_example"].assert_called_once()
        call_args = agent.tool_map["http_tool_example"].call_args
        assert "slots" in call_args.kwargs
        slots = call_args.kwargs["slots"]
        assert len(slots) == 1
        assert {"name": "param1", "value": "value1"} in slots

    def test_execute_tool_http_tool_with_missing_slots(self, mock_state: Mock) -> None:
        """Test _execute_tool method with http_tool and missing slots."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        # Mock slot objects
        mock_slot1 = Mock()
        mock_slot1.name = "required_param"
        mock_slot2 = Mock()
        mock_slot2.name = "optional_param"

        agent.tool_slots["http_tool_example"] = [mock_slot1, mock_slot2]
        agent.tool_map["http_tool_example"] = Mock(return_value="http result")

        tool_args = {"provided_param": "value"}

        result = agent._execute_tool("http_tool_example", mock_state, tool_args)

        assert result == "http result"
        # Verify the tool was called with all slots including missing ones
        agent.tool_map["http_tool_example"].assert_called_once()
        call_args = agent.tool_map["http_tool_example"].call_args
        assert "slots" in call_args.kwargs
        slots = call_args.kwargs["slots"]
        assert len(slots) == 3  # provided_param, required_param, optional_param
        assert {"name": "provided_param", "value": "value"} in slots
        assert {"name": "required_param", "value": None} in slots
        assert {"name": "optional_param", "value": None} in slots

    def test_execute_tool_non_http_tool(self, mock_state: Mock) -> None:
        """Test _execute_tool method with non-http tool."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        agent.tool_map["regular_tool"] = Mock(return_value="regular result")

        tool_args = {"param1": "value1", "param2": "value2"}

        result = agent._execute_tool("regular_tool", mock_state, tool_args)

        assert result == "regular result"
        # Verify the tool was called with state and tool_args
        agent.tool_map["regular_tool"].assert_called_once_with(
            state=mock_state, param1="value1", param2="value2"
        )

    def test_execute_tool_non_http_tool_empty_args(self, mock_state: Mock) -> None:
        """Test _execute_tool method with non-http tool and empty args."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        agent.tool_map["regular_tool"] = Mock(return_value="regular result")

        tool_args = {}

        result = agent._execute_tool("regular_tool", mock_state, tool_args)

        assert result == "regular result"
        # Verify the tool was called with only state
        agent.tool_map["regular_tool"].assert_called_once_with(state=mock_state)

    def test_load_tools_with_task_attribute(self, mock_state: Mock) -> None:
        """Test _load_tools method with task attribute in node."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        # Create mock node with task attribute
        mock_node = Mock()
        mock_node.resource_id = "test_tool"
        mock_node.type = "tool"
        mock_node.attributes = {"task": "test_task"}
        mock_node.additional_args = {}

        mock_tools = {
            "test_tool": {
                "execute": lambda: Mock(),
                "fixed_args": {},
            }
        }

        agent._load_tools(successors=[], predecessors=[mock_node], tools=mock_tools)

        # Verify tool_id is created with task
        assert "test_tool_test_task" in agent.available_tools

    def test_load_tools_with_special_characters(self, mock_state: Mock) -> None:
        """Test _load_tools method with special characters in resource_id."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        # Create mock node with special characters
        mock_node = Mock()
        mock_node.resource_id = "test tool/with spaces"
        mock_node.type = "tool"
        mock_node.attributes = {}
        mock_node.additional_args = {}

        mock_tools = {
            "test tool/with spaces": {
                "execute": lambda: Mock(),
                "fixed_args": {},
            }
        }

        agent._load_tools(successors=[], predecessors=[mock_node], tools=mock_tools)

        # Verify tool_id has special characters replaced
        assert "test_tool_with_spaces" in agent.available_tools

    def test_configure_tools_with_slots(self, mock_state: Mock) -> None:
        """Test _configure_tools method with slots configuration."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        # Mock tool object with slots
        mock_tool_object = Mock()
        mock_tool_object.func.__name__ = "test_tool"
        mock_tool_object.slots = ["slot1", "slot2"]
        mock_tool_object.to_openai_tool_def_v2.return_value = {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "Test tool",
                "parameters": {},
            },
        }

        # Mock node with slots
        mock_node = Mock()
        mock_node.attributes = {"slots": ["slot1", "slot2"]}
        mock_node.additional_args = {"extra_arg": "value"}

        agent.available_tools["test_tool"] = (
            {
                "execute": lambda: mock_tool_object,
                "fixed_args": {"fixed_param": "value"},
            },
            mock_node,
        )

        agent._configure_tools()

        assert "test_tool" in agent.tool_map
        assert "test_tool" in agent.tool_slots
        assert "test_tool" in agent.tool_args
        assert agent.tool_args["test_tool"]["fixed_param"] == "value"
        assert agent.tool_args["test_tool"]["extra_arg"] == "value"

    def test_configure_tools_without_additional_args(self, mock_state: Mock) -> None:
        """Test _configure_tools method without additional_args."""
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        # Mock tool object
        mock_tool_object = Mock()
        mock_tool_object.func.__name__ = "test_tool"
        mock_tool_object.slots = []
        mock_tool_object.to_openai_tool_def_v2.return_value = {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "Test tool",
                "parameters": {},
            },
        }

        # Mock node without additional_args
        mock_node = Mock()
        mock_node.attributes = {}
        mock_node.additional_args = None

        agent.available_tools["test_tool"] = (
            {
                "execute": lambda: mock_tool_object,
                "fixed_args": {"fixed_param": "value"},
            },
            mock_node,
        )

        agent._configure_tools()

        assert "test_tool" in agent.tool_map
        assert agent.tool_args["test_tool"]["fixed_param"] == "value"
        # Should not have additional_args
        assert "extra_arg" not in agent.tool_args["test_tool"]

    def test_generate_with_multiple_tool_calls(self, mock_state: Mock) -> None:
        """Test generate method with multiple tool calls."""
        from unittest.mock import patch

        with patch("arklex.env.agents.openai_agent.trace") as mock_trace:
            mock_trace.return_value = mock_state

            tool_call1 = {"name": "tool1", "args": {"param1": "value1"}, "id": "call_1"}
            tool_call2 = {"name": "tool2", "args": {"param2": "value2"}, "id": "call_2"}

            # Mock AI messages for multiple tool calls
            mock_ai_message_with_tools = Mock()
            mock_ai_message_with_tools.content = "Calling tools"
            mock_ai_message_with_tools.tool_calls = [tool_call1, tool_call2]

            mock_ai_message_final = Mock()
            mock_ai_message_final.content = "Final response"
            mock_ai_message_final.tool_calls = None

            agent = OpenAIAgent(
                successors=[],
                predecessors=[],
                tools={},
                state=mock_state,
            )

            # Setup tool maps
            agent.tool_map["tool1"] = Mock(return_value="tool1 result")
            agent.tool_map["tool2"] = Mock(return_value="tool2 result")
            agent.tool_args["tool1"] = {}
            agent.tool_args["tool2"] = {}

            agent.llm = Mock()
            # Create a list of responses that will be returned in sequence
            # Need 3 responses: initial call, after tool1, after tool2
            agent.llm.invoke.side_effect = [
                mock_ai_message_with_tools,
                mock_ai_message_final,  # After tool1 execution
                mock_ai_message_final,  # After tool2 execution
            ]
            agent.prompt = "test prompt"

            result = agent.generate(mock_state)

            assert result.response == "Final response"
            # Should have been called multiple times: initial + after each tool call
            assert agent.llm.invoke.call_count >= 2
            # Verify both tools were called
            agent.tool_map["tool1"].assert_called_once()
            agent.tool_map["tool2"].assert_called_once()

    def test_generate_with_tool_call_no_id(self, mock_state: Mock) -> None:
        """Test generate method with tool call that has no id."""
        from unittest.mock import patch

        with patch("arklex.env.agents.openai_agent.trace") as mock_trace:
            mock_trace.return_value = mock_state

            tool_call = {"name": "mock_tool", "args": {"param": "value"}}  # No id

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
            agent.tool_map["mock_tool"] = Mock(return_value="tool result")
            agent.tool_args["mock_tool"] = {}

            result = agent.generate(mock_state)

            assert result.response == "Response"

    def test_generate_with_tool_call_no_args(self, mock_state: Mock) -> None:
        """Test generate method with tool call that has no args."""
        from unittest.mock import patch

        with patch("arklex.env.agents.openai_agent.trace") as mock_trace:
            mock_trace.return_value = mock_state

            tool_call = {"name": "mock_tool", "id": "call_123"}  # No args

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
            agent.tool_map["mock_tool"] = Mock(return_value="tool result")
            agent.tool_args["mock_tool"] = {}

            result = agent.generate(mock_state)

            assert result.response == "Response"

    @pytest.mark.no_llm_mock
    def test_end_conversation_with_exception_during_llm_invoke(
        self, mock_state: Mock
    ) -> None:
        """Test end_conversation function when LLM invoke raises an exception."""
        from unittest.mock import patch

        with patch(
            "arklex.utils.provider_utils.validate_and_get_model_class"
        ) as mock_validate:
            mock_llm_class = Mock()
            mock_llm = Mock()
            # Patch at a lower level to override global fixture
            mock_llm.invoke.side_effect = Exception("LLM error")
            mock_llm_class.return_value = mock_llm
            mock_validate.return_value = mock_llm_class

            result = end_conversation().func(mock_state)
            print(f"DEBUG: result = {result}, type = {type(result)}")

            # Should return the fallback message when exception occurs
            assert "I hope I was able to help you today" in str(result)

    @pytest.mark.no_llm_mock
    def test_end_conversation_with_invalid_llm_response(self, mock_state: Mock) -> None:
        """Test end_conversation function when LLM returns invalid response."""
        from unittest.mock import patch

        with patch(
            "arklex.utils.provider_utils.validate_and_get_model_class"
        ) as mock_validate:
            mock_llm_class = Mock()
            mock_llm = Mock()
            # Mock LLM to return None (invalid response)
            mock_llm.invoke.return_value = None
            mock_llm_class.return_value = mock_llm
            mock_validate.return_value = mock_llm_class

            result = end_conversation().func(mock_state)

            # Should return a fallback message
            assert "I hope I was able to help you today" in str(result)

    @pytest.mark.no_llm_mock
    def test_end_conversation_with_empty_llm_response(self, mock_state: Mock) -> None:
        """Test end_conversation function when LLM returns empty response."""
        from unittest.mock import patch

        with patch(
            "arklex.utils.provider_utils.validate_and_get_model_class"
        ) as mock_validate:
            mock_llm_class = Mock()
            mock_llm = Mock()
            # Mock LLM to return empty string
            mock_llm.invoke.return_value = ""
            mock_llm_class.return_value = mock_llm
            mock_validate.return_value = mock_llm_class

            result = end_conversation().func(mock_state)

            # Should return a fallback message
            assert "I hope I was able to help you today" in str(result)

    @pytest.mark.no_llm_mock
    def test_end_conversation_with_different_model_config(
        self, mock_state: Mock
    ) -> None:
        """Test end_conversation function with different model configuration."""
        from unittest.mock import patch

        with patch(
            "arklex.env.agents.openai_agent.validate_and_get_model_class"
        ) as mock_validate:
            mock_llm_class = Mock()
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Thank you for using our service. Goodbye!"
            # Patch at a lower level to override global fixture
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            mock_validate.return_value = mock_llm_class

            # Change model config
            mock_state.bot_config.llm_config.model_type_or_path = "gpt-4"

            result = end_conversation().func(mock_state)
            print(f"DEBUG: end_conversation result = {result!r}, type={type(result)}")
            assert result == "Thank you for using our service. Goodbye!"
            assert mock_state.status == StatusEnum.COMPLETE
            mock_llm_class.assert_called_once_with(model="gpt-4")

    def test_agent_inheritance(self, mock_state: Mock) -> None:
        """Test that OpenAIAgent properly inherits from BaseAgent."""
        from arklex.env.agents.agent import BaseAgent

        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        assert isinstance(agent, BaseAgent)
        assert hasattr(agent, "action_graph")
        assert hasattr(agent, "llm")
        assert hasattr(agent, "available_tools")
        assert hasattr(agent, "tool_map")
        assert hasattr(agent, "tool_defs")
        assert hasattr(agent, "tool_args")
        assert hasattr(agent, "tool_slots")

    def test_agent_registration(self, mock_state: Mock) -> None:
        """Test that OpenAIAgent is properly registered as an agent."""
        # This test checks that the agent can be instantiated and has the expected attributes
        agent = OpenAIAgent(
            successors=[],
            predecessors=[],
            tools={},
            state=mock_state,
        )

        # The agent should have the expected attributes
        assert hasattr(agent, "action_graph")
        assert hasattr(agent, "llm")
        assert hasattr(agent, "available_tools")
        assert hasattr(agent, "tool_map")
        assert hasattr(agent, "tool_defs")
        assert hasattr(agent, "tool_args")
        assert hasattr(agent, "tool_slots")

    @pytest.mark.no_llm_mock
    def test_end_conversation_tool_registration(self, mock_state: Mock) -> None:
        """Test that end_conversation is properly registered as a tool."""
        from unittest.mock import patch

        with patch(
            "arklex.env.agents.openai_agent.validate_and_get_model_class"
        ) as mock_validate:
            mock_llm_class = Mock()
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Thank you for using our service. Goodbye!"
            # Patch at a lower level to override global fixture
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            mock_validate.return_value = mock_llm_class

            result = end_conversation().func(mock_state)
            print(f"DEBUG: end_conversation result = {result!r}, type={type(result)}")
            assert result == "Thank you for using our service. Goodbye!"
            assert mock_state.status == StatusEnum.COMPLETE

    def test_agent_with_complex_tool_configuration(self, mock_state: Mock) -> None:
        """Test agent initialization with complex tool configuration."""
        from unittest.mock import patch

        with patch("arklex.env.agents.openai_agent.load_prompts") as mock_load_prompts:
            mock_load_prompts.return_value = {
                "function_calling_agent_prompt": "Test prompt"
            }

            # Create complex tool configuration
            mock_tool_object = Mock()
            mock_tool_object.func.__name__ = "complex_tool"
            mock_tool_object.slots = ["slot1", "slot2", "slot3"]
            mock_tool_object.to_openai_tool_def_v2.return_value = {
                "type": "function",
                "function": {
                    "name": "complex_tool",
                    "description": "Complex tool with multiple slots",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "slot1": {"type": "string"},
                            "slot2": {"type": "string"},
                            "slot3": {"type": "string"},
                        },
                    },
                },
            }

            mock_node = Mock()
            mock_node.resource_id = "complex_tool"
            mock_node.type = "tool"
            mock_node.attributes = {"slots": ["slot1", "slot2", "slot3"]}
            mock_node.additional_args = {"extra_param": "extra_value"}

            mock_tools = {
                "complex_tool": {
                    "execute": lambda: mock_tool_object,
                    "fixed_args": {"fixed_param": "fixed_value"},
                }
            }

            agent = OpenAIAgent(
                successors=[],
                predecessors=[mock_node],
                tools=mock_tools,
                state=mock_state,
            )

            assert "complex_tool" in agent.available_tools
            assert "complex_tool" in agent.tool_map
            assert "complex_tool" in agent.tool_slots
            assert "complex_tool" in agent.tool_args
            assert agent.tool_args["complex_tool"]["fixed_param"] == "fixed_value"
            assert agent.tool_args["complex_tool"]["extra_param"] == "extra_value"
            assert len(agent.tool_defs) >= 2  # complex_tool + end_conversation
