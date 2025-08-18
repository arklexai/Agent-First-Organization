# from unittest.mock import Mock, patch

# import pytest

# from arklex.env.agents.openai_agent import OpenAIAgent
# from arklex.orchestrator.entities.orchestrator_state_entities import (
#     OrchestratorState,
#     StatusEnum,
# )
# from arklex.types.stream_types import StreamType

# pytestmark = pytest.mark.usefixtures("patch_openai")


# @pytest.fixture(autouse=True)
# def patch_openai(monkeypatch: pytest.MonkeyPatch) -> None:
#     # Patch openai.ChatCompletion.create to always return a mock response
#     with patch("openai.ChatCompletion.create") as mock_create:
#         mock_create.return_value = {
#             "choices": [{"message": {"content": "mocked response"}}]
#         }
#         yield


# @pytest.fixture
# def mock_state() -> OrchestratorState:
#     """Create a mock OrchestratorState for testing."""
#     state = Mock(spec=OrchestratorState)
#     state.status = StatusEnum.INCOMPLETE
#     state.function_calling_trajectory = []
#     state.message_flow = ""
#     state.response = ""
#     state.sys_instruct = "System instructions"
#     state.stream_type = StreamType.NON_STREAM
#     state.message_queue = Mock()

#     # Mock orchestrator_message
#     state.orchestrator_message = Mock()
#     state.orchestrator_message.message = "User message"

#     # Mock bot_config
#     state.bot_config = Mock()
#     state.bot_config.llm_config = Mock()
#     state.bot_config.llm_config.llm_provider = "openai"
#     state.bot_config.llm_config.model_type_or_path = "gpt-3.5-turbo"

#     return state


# @pytest.fixture
# def mock_tools() -> dict:
#     """Create mock tools for testing."""
#     tool_object = Mock()
#     tool_object.to_openai_tool_def_v2.return_value = {
#         "type": "function",
#         "function": {"name": "mock_tool", "description": "Mock tool", "parameters": {}},
#     }
#     tool_func = Mock(return_value="tool result")
#     tool_func.__name__ = "mock_tool"
#     tool_object.func = tool_func
#     tool_object.slots = []
#     tool_object.node_specific_data = {}

#     return {
#         "mock_tool_id": {
#             "tool_instance": tool_object,
#             "execute": lambda: tool_object,
#             "fixed_args": {"fixed_param": "value"},
#         }
#     }


# @pytest.fixture
# def mock_nodes() -> list:
#     """Create mock nodes for testing."""
#     node = Mock()
#     node.resource = {"id": "mock_tool_id"}
#     node.data = {"name": "mock_tool"}
#     return [node]


# class TestOpenAIAgent:
#     @patch("arklex.env.agents.openai_agent.load_prompts")
#     def test_init(
#         self,
#         mock_load_prompts: Mock,
#         mock_tools: Mock,
#         mock_nodes: Mock,
#     ) -> None:
#         """Test OpenAIAgent initialization."""
#         mock_load_prompts.return_value = {
#             "function_calling_agent_prompt": "Test prompt"
#         }

#         agent = OpenAIAgent(
#             successors=[],
#             predecessors=mock_nodes,
#             tools=mock_tools,
#         )

#         assert agent.llm is None
#         assert "mock_tool_id" in agent.available_tools
#         assert len(agent.tool_defs) >= 1

#     @patch("arklex.env.agents.openai_agent.load_prompts")
#     @patch("arklex.utils.provider_utils.validate_and_get_model_class")
#     def test_generate_response_no_tools(
#         self,
#         mock_validate: Mock,
#         mock_load_prompts: Mock,
#         mock_state: Mock,
#     ) -> None:
#         """Test generate_response method with no tool calls."""
#         mock_load_prompts.return_value = {
#             "function_calling_agent_prompt": "Test prompt: {sys_instruct}"
#         }

#         # Mock the LLM
#         mock_llm_class = Mock()
#         mock_llm = Mock()
#         mock_ai_message = Mock()
#         mock_ai_message.content = "Test response"
#         mock_ai_message.tool_calls = None
#         mock_llm.invoke.return_value = mock_ai_message
#         mock_llm.bind_tools.return_value = mock_llm
#         mock_llm_class.return_value = mock_llm
#         mock_validate.return_value = mock_llm_class

#         agent = OpenAIAgent(
#             successors=[],
#             predecessors=[],
#             tools={},
#         )

#         # Initialize agent data
#         agent.init_agent_data(mock_state, node_specific_data={"prompt": "test prompt"})

#         # Set up the LLM
#         agent.llm = mock_llm

#         result_state, result_output = agent.generate_response(mock_state)
#         assert result_output.response == "Test response"
#         assert len(result_state.function_calling_trajectory) > 0

#     @patch("arklex.env.agents.openai_agent.load_prompts")
#     @patch("arklex.utils.provider_utils.validate_and_get_model_class")
#     def test_generate_response_with_tool_calls(
#         self,
#         mock_validate: Mock,
#         mock_load_prompts: Mock,
#         mock_state: Mock,
#         mock_tools: Mock,
#     ) -> None:
#         """Test generate_response method with tool calls."""
#         mock_load_prompts.return_value = {
#             "function_calling_agent_prompt": "Test prompt: {sys_instruct}"
#         }

#         tool_call = {
#             "name": "mock_tool_id",
#             "args": {"param": "value"},
#             "id": "call_123",
#         }

#         # Mock AI messages for tool call and final response
#         mock_ai_message_with_tools = Mock()
#         mock_ai_message_with_tools.content = "Calling tool"
#         mock_ai_message_with_tools.tool_calls = [tool_call]

#         mock_ai_message_final = Mock()
#         mock_ai_message_final.content = "Final response"
#         mock_ai_message_final.tool_calls = None

#         # Mock the LLM
#         mock_llm_class = Mock()
#         mock_llm = Mock()
#         mock_llm.invoke.side_effect = [
#             mock_ai_message_with_tools,
#             mock_ai_message_final,
#         ]
#         mock_llm.bind_tools.return_value = mock_llm
#         mock_llm_class.return_value = mock_llm
#         mock_validate.return_value = mock_llm_class

#         agent = OpenAIAgent(
#             successors=[],
#             predecessors=[],
#             tools=mock_tools,
#         )

#         # Setup tool map manually since we're not using the normal loading
#         agent.tool_map["mock_tool_id"] = Mock(return_value="tool result")
#         agent.tool_args["mock_tool_id"] = {}

#         # Initialize agent data
#         agent.init_agent_data(mock_state, node_specific_data={"prompt": "test prompt"})

#         # Set up the LLM
#         agent.llm = mock_llm

#         result_state, result_output = agent.generate_response(mock_state)

#         assert result_output.response == "Final response"
#         assert mock_llm.invoke.call_count == 2

#     @patch("arklex.utils.provider_utils.validate_and_get_model_class")
#     def test_execute(self, mock_validate: Mock, mock_state: Mock) -> None:
#         """Test _execute method."""
#         mock_llm_class = Mock()
#         mock_llm = Mock()
#         mock_ai_message = Mock()
#         mock_ai_message.content = "test response"
#         mock_ai_message.tool_calls = None
#         mock_llm.invoke.return_value = mock_ai_message
#         mock_llm.bind_tools.return_value = mock_llm
#         mock_llm_class.return_value = mock_llm
#         mock_validate.return_value = mock_llm_class

#         agent = OpenAIAgent(
#             successors=[],
#             predecessors=[],
#             tools={},
#         )

#         # Initialize agent data
#         agent.init_agent_data(mock_state, node_specific_data={"prompt": "test prompt"})

#         result_state, result_output = agent._execute()

#         assert result_output.response == '{"result": "dummy response"}'

#     def test_load_tools(self, mock_tools: Mock, mock_nodes: Mock) -> None:
#         """Test _load_tools method."""
#         agent = OpenAIAgent(
#             successors=[],
#             predecessors=mock_nodes,
#             tools=mock_tools,
#         )

#         # Test that tools are loaded during initialization
#         assert "mock_tool_id" in agent.available_tools

#     def test_load_tools_empty_nodes(self) -> None:
#         """Test _load_tools method with empty nodes."""
#         agent = OpenAIAgent(
#             successors=[],
#             predecessors=[],
#             tools={},
#         )

#         # Test that no tools are loaded when nodes are empty
#         assert len(agent.available_tools) == 0

#     def test_load_tools_node_not_in_tools(self) -> None:
#         """Test _load_tools method when node not in tools."""
#         node = Mock()
#         node.resource = {"id": "nonexistent_tool"}

#         agent = OpenAIAgent(
#             successors=[node],
#             predecessors=[],
#             tools={},
#         )

#         # Test that non-existent tool is not loaded
#         assert "nonexistent_tool" not in agent.available_tools

#     def test_configure_tools(self, mock_tools: Mock) -> None:
#         """Test _configure_tools method."""
#         agent = OpenAIAgent(
#             successors=[],
#             predecessors=[],
#             tools=mock_tools,
#         )

#         # Manually add tool to test configure
#         agent.available_tools["test_tool_id"] = mock_tools["mock_tool_id"]
#         agent._configure_tools()

#         assert "test_tool_id" in agent.tool_map
#         assert "test_tool_id" in agent.tool_args
#         assert agent.tool_map["test_tool_id"].__name__ == "mock_tool"

#     def test_configure_tools_empty(self) -> None:
#         """Test _configure_tools method with no tools."""
#         agent = OpenAIAgent(
#             successors=[],
#             predecessors=[],
#             tools={},
#         )

#         agent._configure_tools()

#     @patch("arklex.env.agents.openai_agent.load_prompts")
#     @patch("arklex.env.agents.openai_agent.trace")
#     def test_generate_adds_prompt_if_not_in_trajectory(
#         self,
#         mock_trace,  # noqa: ANN001
#         mock_load_prompts,  # noqa: ANN001
#         mock_state,  # noqa: ANN001
#     ) -> None:
#         """Test that the prompt is appended if not already present in trajectory."""
#         mock_load_prompts.return_value = {
#             "function_calling_agent_prompt": "Test: {sys_instruct} {message}"
#         }
#         mock_trace.return_value = mock_state
#         mock_state.function_calling_trajectory = []

#         agent = OpenAIAgent(
#             successors=[],
#             predecessors=[],
#             tools={},
#             state=mock_state,
#         )

#         mock_ai_message = Mock()
#         mock_ai_message.content = "Test response"
#         mock_ai_message.tool_calls = None
#         agent.llm = Mock()
#         agent.llm.invoke.return_value = mock_ai_message
#         agent.prompt = ""

#         _result = agent.generate(mock_state)
#         # Should have appended the prompt
#         assert len(mock_state.function_calling_trajectory) == 1
#         assert mock_state.function_calling_trajectory[0]["content"].startswith("Test:")

#     @patch("arklex.env.agents.openai_agent.load_prompts")
#     @patch("arklex.env.agents.openai_agent.trace")
#     def test_generate_does_not_add_duplicate_prompt(
#         self,
#         mock_trace,  # noqa: ANN001
#         mock_load_prompts,  # noqa: ANN001
#         mock_state,  # noqa: ANN001
#     ) -> None:
#         """Test that the prompt is NOT appended if already present in trajectory."""
#         mock_load_prompts.return_value = {
#             "function_calling_agent_prompt": "Test: {sys_instruct} {message}"
#         }
#         mock_trace.return_value = mock_state
#         # Add a message to trajectory that matches the prompt
#         mock_state.function_calling_trajectory = [
#             {"content": "Test: System instructions User message"}
#         ]

#         agent = OpenAIAgent(
#             successors=[],
#             predecessors=[],
#             tools={},
#             state=mock_state,
#         )

#         mock_ai_message = Mock()
#         mock_ai_message.content = "Test response"
#         mock_ai_message.tool_calls = None
#         agent.llm = Mock()
#         agent.llm.invoke.return_value = mock_ai_message
#         agent.prompt = ""

#         initial_len = len(mock_state.function_calling_trajectory)
#         _result = agent.generate(mock_state)
#         # Should NOT have appended a duplicate
#         assert len(mock_state.function_calling_trajectory) == initial_len


# class TestEndConversation:
#     @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
#     @patch("arklex.env.agents.openai_agent.json.dumps")
#     def test_end_conversation_success(
#         self, mock_json_dumps: Mock, mock_provider_map: Mock, mock_state: Mock
#     ) -> None:
#         """Test end_conversation function success case."""
#         mock_llm = Mock()
#         mock_llm.invoke.return_value = Mock(content="test response")
#         mock_provider_map.get.return_value = mock_llm

#         # Call the function that creates the tool, then call the tool
#         tool_creator = end_conversation
#         tool_instance = tool_creator()
#         result = tool_instance.func(mock_state)

#         assert isinstance(result, str)

#     @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
#     def test_end_conversation_error(
#         self, mock_provider_map: Mock, mock_state: Mock
#     ) -> None:
#         """Test end_conversation function error case."""
#         from unittest.mock import patch

#         mock_llm = Mock()
#         mock_llm.invoke.side_effect = Exception("Test error")
#         mock_provider_map.get.return_value = mock_llm
#         with patch("openai.ChatCompletion.create", side_effect=Exception("Test error")):
#             tool_creator = end_conversation
#             tool_instance = tool_creator()
#             result = tool_instance.func(mock_state)
#             assert isinstance(result, str)
#             assert any(
#                 phrase in result
#                 for phrase in [
#                     "Thank you",
#                     "Goodbye",
#                     "I hope I was able to help you today. Goodbye!",
#                 ]
#             )

#     @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
#     def test_end_conversation_invalid_llm_response(
#         self, mock_provider_map: Mock, mock_state: Mock
#     ) -> None:
#         """Test end_conversation function when LLM returns invalid response."""
#         from unittest.mock import patch

#         mock_llm = Mock()
#         mock_llm.invoke.return_value = None
#         mock_provider_map.get.return_value = mock_llm
#         with patch("openai.ChatCompletion.create", return_value=None):
#             result = end_conversation().func(mock_state)
#             assert any(
#                 phrase in result
#                 for phrase in [
#                     "Thank you",
#                     "Goodbye",
#                     "I hope I was able to help you today. Goodbye!",
#                 ]
#             )

#     @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
#     def test_end_conversation_empty_llm_response(
#         self, mock_provider_map: Mock, mock_state: Mock
#     ) -> None:
#         """Test end_conversation function when LLM returns empty response."""
#         from unittest.mock import patch

#         mock_llm = Mock()
#         mock_llm.invoke.return_value = ""
#         mock_provider_map.get.return_value = mock_llm
#         with patch("openai.ChatCompletion.create", return_value=""):
#             result = end_conversation().func(mock_state)
#             assert any(
#                 phrase in result
#                 for phrase in [
#                     "Thank you",
#                     "Goodbye",
#                     "I hope I was able to help you today. Goodbye!",
#                 ]
#             )

#     @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
#     def test_end_conversation_different_model_config(
#         self, mock_provider_map: Mock, mock_state: Mock
#     ) -> None:
#         """Test end_conversation function with different model configuration."""
#         from unittest.mock import patch

#         mock_llm = Mock()
#         mock_response = Mock()
#         mock_response.content = "Thank you for taking the time to interact with me today. I appreciate your input and questions. If you have any further inquiries, don't hesitate to reach out. Goodbye and take care!"
#         mock_provider_map.get.return_value = mock_llm
#         mock_state.bot_config.llm_config.model_type_or_path = "gpt-4"
#         with patch(
#             "openai.ChatCompletion.create",
#             return_value={"choices": [{"message": {"content": mock_response.content}}]},
#         ):
#             result = end_conversation().func(mock_state)
#             assert any(
#                 phrase in result
#                 for phrase in [
#                     "Thank you",
#                     "Goodbye",
#                     "I hope I was able to help you today. Goodbye!",
#                 ]
#             )

#     def test_agent_inheritance(self, mock_state: Mock) -> None:
#         """Test that OpenAIAgent properly inherits from BaseAgent."""
#         from arklex.env.agents.agent import BaseAgent

#         agent = OpenAIAgent(
#             successors=[],
#             predecessors=[],
#             tools={},
#         )

#         assert isinstance(agent, BaseAgent)
#         assert hasattr(agent, "llm")
#         assert hasattr(agent, "available_tools")
#         assert hasattr(agent, "tool_map")
#         assert hasattr(agent, "tool_defs")
#         assert hasattr(agent, "tool_args")
#         assert hasattr(agent, "tool_slots")

#     def test_agent_registration(self) -> None:
#         """Test that OpenAIAgent is properly registered as an agent."""
#         # This test checks that the agent can be instantiated and has the expected attributes
#         agent = OpenAIAgent(
#             successors=[],
#             predecessors=[],
#             tools={},
#         )

#         # The agent should have the expected attributes
#         assert hasattr(agent, "llm")
#         assert hasattr(agent, "available_tools")
#         assert hasattr(agent, "tool_map")
#         assert hasattr(agent, "tool_defs")
#         assert hasattr(agent, "tool_args")
#         assert hasattr(agent, "tool_slots")

#     def test_end_conversation_tool_registration(self, mock_state: Mock) -> None:
#         """Test that end_conversation is properly registered as a tool."""
#         from unittest.mock import patch

#         with patch(
#             "arklex.env.agents.openai_agent.validate_and_get_model_class"
#         ) as mock_validate:
#             mock_llm_class = Mock()
#             mock_llm = Mock()
#             mock_response = Mock()
#             mock_response.content = "Thank you for using our service. Goodbye!"
#             # Patch at a lower level to override global fixture
#             mock_llm.invoke.return_value = mock_response
#             mock_llm_class.return_value = mock_llm
#             mock_validate.return_value = mock_llm_class

#             result = end_conversation().func(mock_state)
#             print(f"DEBUG: end_conversation result = {result!r}, type={type(result)}")
#             assert result == "Thank you for using our service. Goodbye!"
#             assert mock_state.status == StatusEnum.COMPLETE

#     def test_agent_with_complex_tool_configuration(self, mock_state: Mock) -> None:
#         """Test agent initialization with complex tool configuration."""
#         from unittest.mock import patch

#         with patch("arklex.env.agents.openai_agent.load_prompts") as mock_load_prompts:
#             mock_load_prompts.return_value = {
#                 "function_calling_agent_prompt": "Test prompt"
#             }

#             # Create complex tool configuration
#             mock_tool_object = Mock()
#             mock_tool_object.func.__name__ = "complex_tool"
#             mock_tool_object.slots = ["slot1", "slot2", "slot3"]
#             mock_tool_object.to_openai_tool_def_v2.return_value = {
#                 "type": "function",
#                 "function": {
#                     "name": "complex_tool",
#                     "description": "Complex tool with multiple slots",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {
#                             "slot1": {"type": "string"},
#                             "slot2": {"type": "string"},
#                             "slot3": {"type": "string"},
#                         },
#                     },
#                 },
#             }

#             mock_node = Mock()
#             mock_node.resource_id = "complex_tool"
#             mock_node.type = "tool"
#             mock_node.attributes = {"slots": ["slot1", "slot2", "slot3"]}
#             mock_node.additional_args = {"extra_param": "extra_value"}

#             mock_tools = {
#                 "complex_tool": {
#                     "tool_instance": mock_tool_object,
#                     "execute": lambda: mock_tool_object,
#                     "fixed_args": {"fixed_param": "fixed_value"},
#                 }
#             }

#         agent = OpenAIAgent(
#             successors=[],
#             predecessors=[],
#             tools={},
#         )

#         # Initialize with no prompt
#         agent.init_agent_data(mock_state, node_specific_data={"prompt": ""})

#         result = agent._prepare_prompt(mock_state)
#         assert "Test: System instructions" in result

#     def test_execute_tool_slot_value_source_fixed(self, mock_state: Mock) -> None:
#         agent = OpenAIAgent(successors=[], predecessors=[], tools={}, state=mock_state)
#         slot = {
#             "name": "fixed_param",
#             "type": "str",
#             "valueSource": "fixed",
#             "value": "fixed_value",
#         }
#         agent.tool_slots["http_tool_example"] = [slot]
#         agent.tool_map["http_tool_example"] = Mock(return_value="fixed result")
#         tool_args = {}
#         result = agent._execute_tool("http_tool_example", mock_state, tool_args)
#         assert result == "fixed result"
#         call_args = agent.tool_map["http_tool_example"].call_args
#         slots = call_args.kwargs["slots"]
#         assert slots[0]["value"] == "fixed_value"

#     def test_execute_tool_slot_value_source_default_missing(
#         self, mock_state: Mock
#     ) -> None:
#         agent = OpenAIAgent(successors=[], predecessors=[], tools={}, state=mock_state)
#         slot = {
#             "name": "default_param",
#             "type": "str",
#             "valueSource": "default",
#             "value": "default_value",
#         }
#         agent.tool_slots["http_tool_example"] = [slot]
#         agent.tool_map["http_tool_example"] = Mock(return_value="default result")
#         tool_args = {}
#         result = agent._execute_tool("http_tool_example", mock_state, tool_args)
#         assert result == "default result"
#         call_args = agent.tool_map["http_tool_example"].call_args
#         slots = call_args.kwargs["slots"]
#         assert slots[0]["value"] == "default_value"

#     def test_execute_tool_slot_value_source_prompt_missing(
#         self, mock_state: Mock
#     ) -> None:
#         agent = OpenAIAgent(successors=[], predecessors=[], tools={}, state=mock_state)
#         slot = {"name": "prompt_param", "type": "str", "valueSource": "prompt"}
#         agent.tool_slots["http_tool_example"] = [slot]
#         agent.tool_map["http_tool_example"] = Mock(return_value="prompt result")
#         tool_args = {}
#         result = agent._execute_tool("http_tool_example", mock_state, tool_args)
#         assert result == "prompt result"
#         call_args = agent.tool_map["http_tool_example"].call_args
#         slots = call_args.kwargs["slots"]
#         assert slots[0]["value"] == ""

#     def test_execute_tool_group_slot_repeatable_default(self, mock_state: Mock) -> None:
#         agent = OpenAIAgent(successors=[], predecessors=[], tools={}, state=mock_state)
#         group_slot = {
#             "name": "group1",
#             "type": "group",
#             "repeatable": True,
#             "schema": [{"name": "item", "type": "str", "valueSource": "prompt"}],
#             "valueSource": "default",
#             "value": {"item": "default_value"},
#         }
#         agent.tool_slots["http_tool_example"] = [group_slot]
#         agent.tool_map["http_tool_example"] = Mock(return_value="group result")
#         tool_args = {}
#         result = agent._execute_tool("http_tool_example", mock_state, tool_args)
#         assert result == "group result"
#         call_args = agent.tool_map["http_tool_example"].call_args
#         slots = call_args.kwargs["slots"]
#         assert slots[0]["value"][0]["item"] == "default_value"

#     def test_execute_tool_group_slot_repeatable_fixed(self, mock_state: Mock) -> None:
#         agent = OpenAIAgent(successors=[], predecessors=[], tools={}, state=mock_state)
#         group_slot = {
#             "name": "group2",
#             "type": "group",
#             "repeatable": True,
#             "schema": [
#                 {
#                     "name": "item",
#                     "type": "str",
#                     "valueSource": "fixed",
#                     "value": "fixed_item",
#                 }
#             ],
#             "valueSource": "fixed",
#             "value": [{"item": "fixed_item"}],
#         }
#         agent.tool_slots["http_tool_example"] = [group_slot]
#         agent.tool_map["http_tool_example"] = Mock(return_value="group fixed result")
#         tool_args = {}
#         result = agent._execute_tool("http_tool_example", mock_state, tool_args)
#         assert result == "group fixed result"
#         call_args = agent.tool_map["http_tool_example"].call_args
#         slots = call_args.kwargs["slots"]
#         assert slots[0]["value"][0]["item"] == "fixed_item"

#     def test_execute_tool_type_convert_error(self, mock_state: Mock) -> None:
#         agent = OpenAIAgent(successors=[], predecessors=[], tools={}, state=mock_state)
#         # Simulate a slot with type 'int' and a value that can't be converted
#         slot = {"name": "int_param", "type": "int", "valueSource": "prompt"}
#         agent.tool_slots["http_tool_example"] = [slot]
#         agent.tool_map["http_tool_example"] = Mock(return_value="type error result")
#         tool_args = {"int_param": "not_an_int"}
#         # Patch TYPE_CONVERTERS to raise an exception
#         import arklex.env.agents.openai_agent as openai_agent_mod

#         orig_converter = openai_agent_mod.TYPE_CONVERTERS.get("int")
#         openai_agent_mod.TYPE_CONVERTERS["int"] = lambda v: (_ for _ in ()).throw(
#             ValueError("fail")
#         )
#         result = agent._execute_tool("http_tool_example", mock_state, tool_args)
#         openai_agent_mod.TYPE_CONVERTERS["int"] = orig_converter  # restore
#         assert result == "type error result"
#         call_args = agent.tool_map["http_tool_example"].call_args
#         slots = call_args.kwargs["slots"]
#         assert slots[0]["value"] == "not_an_int"  # fallback to original value

#     def test_execute_tool_slot_with_model_dump(self, mock_state: Mock) -> None:
#         agent = OpenAIAgent(successors=[], predecessors=[], tools={}, state=mock_state)

#         class FakeSlot:
#             def model_dump(self) -> dict:
#                 return {"name": "model_param", "type": "str", "valueSource": "prompt"}

#         agent.tool_slots["http_tool_example"] = [FakeSlot()]
#         agent.tool_map["http_tool_example"] = Mock(return_value="model_dump result")
#         tool_args = {}
#         result = agent._execute_tool("http_tool_example", mock_state, tool_args)
#         assert result == "model_dump result"
#         call_args = agent.tool_map["http_tool_example"].call_args
#         slots = call_args.kwargs["slots"]
#         assert slots[0]["name"] == "model_param"
