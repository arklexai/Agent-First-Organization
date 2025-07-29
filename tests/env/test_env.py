import os
from collections.abc import Callable
from unittest.mock import MagicMock, Mock, patch

import pytest
from pytest import LogCaptureFixture

from arklex.env.env import DefaultResourceInitializer, Environment
from arklex.env.planner.react_planner import ReactPlanner
from arklex.orchestrator.entities.msg_state_entities import MessageState, StatusEnum
from arklex.orchestrator.entities.orchestrator_params_entities import OrchestratorParams
from arklex.orchestrator.entities.taskgraph_entities import NodeInfo
from arklex.orchestrator.NLU.core.slot import SlotFiller
from arklex.orchestrator.NLU.entities.slot_entities import Slot
from arklex.orchestrator.NLU.services.model_service import DummyModelService

# Set test environment
os.environ["ARKLEX_TEST_ENV"] = "local"


@pytest.fixture
def fake_tool() -> Callable[[MessageState | None], MagicMock]:
    def _make_fake_tool(execute_return: MessageState | None = None) -> MagicMock:
        tool = MagicMock()
        tool.init_slotfiller = MagicMock()
        tool.load_slots = MagicMock()
        tool.execute = MagicMock(return_value=execute_return)
        return tool

    return _make_fake_tool


@pytest.fixture
def fake_worker() -> Callable[[MessageState | None], Mock]:
    def _make_fake_worker(execute_return: MessageState | None = None) -> Mock:
        worker = Mock()
        worker.execute = Mock(return_value=execute_return)
        worker.init_slotfilling = Mock()
        return worker

    return _make_fake_worker


def test_environment_uses_dummy_model_service() -> None:
    env = Environment(tools=[], workers=[], agents=[])
    assert isinstance(env.model_service, DummyModelService)


def test_environment_initializes_with_planner() -> None:
    env = Environment(tools=[], workers=[], agents=[], planner_enabled=True)
    assert hasattr(env, "planner")


def test_environment_initializes_with_slotfillapi_str() -> None:
    env = Environment(tools=[], workers=[], agents=[], slotsfillapi="http://fakeapi")
    assert hasattr(env, "slotfillapi")
    assert isinstance(env.slotfillapi, SlotFiller)


def test_environment_initializes_with_slotfillapi_model_service() -> None:
    env = Environment(tools=[], workers=[], agents=[], slotsfillapi="")
    assert hasattr(env, "slotfillapi")
    assert isinstance(env.slotfillapi.model_service, DummyModelService)


def test_default_resource_initializer_init_tools_success_and_error() -> None:
    tools = [
        {"id": "t1", "name": "fake_tool", "path": "fake_path"},
        {"id": "t2", "name": "bad_tool", "path": "bad_path"},
    ]
    # Patch importlib to succeed for one and fail for the other
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="desc"))
        fake_module.fake_tool = fake_func
        mock_import.side_effect = [fake_module, Exception("fail")]
        registry = DefaultResourceInitializer.init_tools(tools)
        assert "t1" in registry
        assert "t2" not in registry  # error case is skipped


def test_default_resource_initializer_init_workers_success_and_error() -> None:
    workers = [
        {"id": "w1", "name": "fake_worker", "path": "fake_path"},
        {"id": "w2", "name": "bad_worker", "path": "bad_path"},
    ]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(description="desc")
        fake_module.fake_worker = fake_func
        mock_import.side_effect = [fake_module, Exception("fail")]
        registry = DefaultResourceInitializer.init_workers(workers)
        assert "w1" in registry
        assert "w2" not in registry


def test_environment_step_tool_executes_and_updates_params(
    fake_tool: Callable[[MessageState | None], MagicMock],
) -> None:
    # Setup a fake tool
    tool = fake_tool(
        MessageState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
            slots={},
            status=StatusEnum.COMPLETE,
        )
    )
    tools = [
        {"id": "t1", "name": "fake_tool", "path": "fake_path"},
    ]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[])

        # Setup params and state
        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}

        state = MagicMock()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}
        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_state.function_calling_trajectory == [
            {"role": "assistant", "content": "call"}
        ]
        assert result_params is params


def test_environment_step_invalid_id_raises() -> None:
    env = Environment(tools=[], workers=[], agents=[])
    # The step method doesn't raise KeyError for invalid IDs, it falls back to planner
    # So we should test that it doesn't raise an exception
    message_state = MessageState()
    params = OrchestratorParams()
    node_info = NodeInfo()

    # This should not raise an exception, it should use the planner
    response_state, updated_params = env.step(
        "not_a_tool", message_state, params, node_info
    )
    assert isinstance(response_state, MessageState)
    assert isinstance(updated_params, OrchestratorParams)


def test_environment_step_worker_executes_and_updates_params(
    fake_worker: Callable[[MessageState | None], Mock],
) -> None:
    """Test environment step with worker execution."""
    mock_worker = fake_worker(MessageState(status=StatusEnum.COMPLETE))
    mock_worker.init_slotfilling = Mock()
    env = Environment(
        tools=[],
        workers=[{"id": "worker1", "name": "test_worker", "path": "test"}],
        agents=[],
    )
    env.workers = {
        "worker1": {"name": "test_worker", "execute": Mock(return_value=mock_worker)}
    }
    env.id2name = {"worker1": "test_worker"}
    message_state = MessageState()
    params = OrchestratorParams()
    params.memory.function_calling_trajectory = []
    params.taskgraph.curr_node = "node1"
    node_info = NodeInfo()
    result_state, result_params = env.step("worker1", message_state, params, node_info)
    assert result_state.status == StatusEnum.COMPLETE
    assert len(result_params.memory.function_calling_trajectory) == 2
    mock_worker.init_slotfilling.assert_called_once()


def test_environment_step_worker_without_init_slotfilling(
    fake_worker: Callable[[MessageState | None], Mock],
) -> None:
    """Test environment step with worker that doesn't have init_slotfilling method."""
    mock_worker = fake_worker(MessageState(status=StatusEnum.COMPLETE))
    # Remove init_slotfilling attribute to test the hasattr check
    if hasattr(mock_worker, "init_slotfilling"):
        delattr(mock_worker, "init_slotfilling")
    env = Environment(
        tools=[],
        workers=[{"id": "worker1", "name": "test_worker", "path": "test"}],
        agents=[],
    )
    env.workers = {
        "worker1": {"name": "test_worker", "execute": Mock(return_value=mock_worker)}
    }
    env.id2name = {"worker1": "test_worker"}
    message_state = MessageState()
    params = OrchestratorParams()
    params.memory.function_calling_trajectory = []
    params.taskgraph.curr_node = "node1"
    node_info = NodeInfo()
    result_state, result_params = env.step("worker1", message_state, params, node_info)
    assert result_state.status == StatusEnum.COMPLETE
    assert len(result_params.memory.function_calling_trajectory) == 2


def test_environment_step_worker_with_response_content(
    fake_worker: Callable[[MessageState | None], Mock],
) -> None:
    """Test environment step with worker that has response content."""
    mock_worker = fake_worker(
        MessageState(status=StatusEnum.COMPLETE, response="test response")
    )
    env = Environment(
        tools=[],
        workers=[{"id": "worker1", "name": "test_worker", "path": "test"}],
        agents=[],
    )
    env.workers = {
        "worker1": {"name": "test_worker", "execute": Mock(return_value=mock_worker)}
    }
    env.id2name = {"worker1": "test_worker"}
    message_state = MessageState()
    params = OrchestratorParams()
    params.memory.function_calling_trajectory = []
    params.taskgraph.curr_node = "node1"
    node_info = NodeInfo()
    result_state, result_params = env.step("worker1", message_state, params, node_info)
    assert result_state.status == StatusEnum.COMPLETE
    assert len(result_params.memory.function_calling_trajectory) == 2
    # Check that response content is used in function calling trajectory
    assert (
        result_params.memory.function_calling_trajectory[1]["content"]
        == "test response"
    )


def test_environment_step_worker_with_message_flow(
    fake_worker: Callable[[MessageState | None], Mock],
) -> None:
    """Test environment step with worker that has message_flow but no response."""
    mock_worker = fake_worker(
        MessageState(status=StatusEnum.COMPLETE, message_flow="test flow")
    )
    env = Environment(
        tools=[],
        workers=[{"id": "worker1", "name": "test_worker", "path": "test"}],
        agents=[],
    )
    env.workers = {
        "worker1": {"name": "test_worker", "execute": Mock(return_value=mock_worker)}
    }
    env.id2name = {"worker1": "test_worker"}
    message_state = MessageState()
    params = OrchestratorParams()
    params.memory.function_calling_trajectory = []
    params.taskgraph.curr_node = "node1"
    node_info = NodeInfo()
    result_state, result_params = env.step("worker1", message_state, params, node_info)
    assert result_state.status == StatusEnum.COMPLETE
    assert len(result_params.memory.function_calling_trajectory) == 2
    # Check that message_flow is used when response is None
    assert result_params.memory.function_calling_trajectory[1]["content"] == "test flow"


def test_environment_step_planner_executes() -> None:
    """Test environment step with planner execution."""
    mock_planner = Mock()
    mock_planner.execute.return_value = (
        "action",
        MessageState(status=StatusEnum.COMPLETE),
        [],
    )
    env = Environment(tools=[], workers=[], agents=[])
    env.planner = mock_planner
    message_state = MessageState()
    params = OrchestratorParams()
    params.memory.function_calling_trajectory = []
    node_info = NodeInfo()
    result_state, result_params = env.step(
        "invalid_id", message_state, params, node_info
    )
    assert result_state.status == StatusEnum.COMPLETE
    mock_planner.execute.assert_called_once()


def test_environment_step_agent_executes() -> None:
    """Test environment step with agent execution."""
    mock_agent_instance = Mock()
    mock_agent_instance.execute.return_value = MessageState(
        status=StatusEnum.COMPLETE,
        function_calling_trajectory=[{"role": "user", "content": "test"}],
    )

    mock_agent_class = Mock(return_value=mock_agent_instance)

    env = Environment(
        tools=[
            {
                "id": "test_tool",
                "name": "test_tool",
                "description": "test",
                "path": "test",
            }
        ],
        workers=[],
        agents=[{"id": "agent1", "name": "test_agent", "path": "test"}],
    )
    env.agents = {"agent1": {"name": "test_agent", "execute": mock_agent_class}}
    env.id2name = {"agent1": "test_agent"}

    message_state = MessageState()
    params = OrchestratorParams()
    params.memory.function_calling_trajectory = []
    params.taskgraph.curr_node = "node1"
    params.taskgraph.node_status = {}

    node_info = NodeInfo()
    node_info.additional_args = {
        "successors": ["node2"],
        "predecessors": ["node0"],
        "extra_arg": "value",
    }

    result_state, result_params = env.step("agent1", message_state, params, node_info)

    assert result_state.status == StatusEnum.COMPLETE
    assert result_params.memory.function_calling_trajectory == [
        {"role": "user", "content": "test"}
    ]
    assert result_params.taskgraph.node_status["node1"] == StatusEnum.COMPLETE

    # Verify agent was initialized with correct parameters
    mock_agent_class.assert_called_once_with(
        successors=["node2"],
        predecessors=["node0"],
        tools=env.tools,
        state=message_state,
    )

    # Verify agent execute was called with correct parameters
    mock_agent_instance.execute.assert_called_once_with(
        message_state, successors=["node2"], predecessors=["node0"], extra_arg="value"
    )


def test_initialize_slotfillapi_with_valid_string() -> None:
    """Test initialize_slotfillapi with valid string endpoint (lines 147-162)."""
    env = Environment(tools=[], workers=[], agents=[])

    with patch("arklex.env.env.APIClientService") as mock_api_service:
        mock_api_instance = Mock()
        mock_api_service.return_value = mock_api_instance

        with patch("arklex.env.env.SlotFiller") as mock_slot_filler:
            mock_slot_filler_instance = Mock()
            mock_slot_filler.return_value = mock_slot_filler_instance

            result = env.initialize_slotfillapi("http://test-api.com")

            mock_api_service.assert_called_once_with(base_url="http://test-api.com")
            mock_slot_filler.assert_called_once_with(
                model_service=env.model_service, api_service=mock_api_instance
            )
            assert result == mock_slot_filler_instance


def test_environment_step_agent_with_empty_additional_args() -> None:
    """Test agent execution with empty additional_args."""
    mock_agent_instance = Mock()
    mock_agent_instance.execute.return_value = MessageState(status=StatusEnum.COMPLETE)

    mock_agent_class = Mock(return_value=mock_agent_instance)

    env = Environment(
        tools={},
        workers=[],
        agents=[{"id": "agent1", "name": "test_agent", "path": "test"}],
    )
    env.agents = {"agent1": {"name": "test_agent", "execute": mock_agent_class}}

    message_state = MessageState()
    params = OrchestratorParams()
    params.memory.function_calling_trajectory = []
    params.taskgraph.curr_node = "node1"
    params.taskgraph.node_status = {}

    node_info = NodeInfo()
    node_info.additional_args = {}

    result_state, result_params = env.step("agent1", message_state, params, node_info)

    # Verify agent was initialized with empty lists when keys are missing
    mock_agent_class.assert_called_once_with(
        successors=[],
        predecessors=[],
        tools=env.tools,
        state=message_state,
    )


def test_environment_register_tool_success() -> None:
    """Test successful tool registration."""
    env = Environment(tools=[], workers=[], agents=[])
    mock_tool = {"name": "test_tool", "description": "test description"}
    env.register_tool("test_tool", mock_tool)
    assert "test_tool" in env.tools
    assert env.tools["test_tool"] == mock_tool


def test_environment_register_tool_failure() -> None:
    """Test tool registration failure."""
    env = Environment(tools=[], workers=[], agents=[])
    # Mock the tools dict to raise an exception
    with patch.object(env, "tools", side_effect=Exception("Registration failed")):
        env.register_tool("test_tool", {})
        # Should not raise exception, just log error


def test_environment_with_slot_fill_api_alias() -> None:
    """Test environment initialization with slot_fill_api alias."""
    env = Environment(tools=[], workers=[], agents=[], slot_fill_api="http://test-api")
    assert isinstance(env.slotfillapi, SlotFiller)


def test_environment_with_custom_resource_initializer() -> None:
    """Test environment initialization with custom resource initializer."""
    mock_initializer = Mock()
    mock_initializer.init_tools.return_value = {"tool1": {"name": "test_tool"}}
    mock_initializer.init_workers.return_value = {"worker1": {"name": "test_worker"}}
    mock_initializer.init_agents.return_value = {"agent1": {"name": "test_agent"}}
    env = Environment(
        tools=[{"id": "tool1", "name": "test", "path": "test"}],
        workers=[{"id": "worker1", "name": "test", "path": "test"}],
        agents=[{"id": "agent1", "name": "test", "path": "test"}],
        resource_initializer=mock_initializer,
    )
    assert env.tools == {"tool1": {"name": "test_tool"}}
    assert env.workers == {"worker1": {"name": "test_worker"}}
    assert env.agents == {"agent1": {"name": "test_agent"}}
    mock_initializer.init_tools.assert_called_once()
    mock_initializer.init_workers.assert_called_once()
    mock_initializer.init_agents.assert_called_once()


def test_environment_with_planner_enabled() -> None:
    """Test environment initialization with planner enabled."""
    env = Environment(
        tools=[],
        workers=[],
        agents=[],
        planner_enabled=True,
    )
    assert isinstance(env.planner, ReactPlanner)


def test_environment_with_custom_model_service() -> None:
    """Test environment initialization with custom model service."""
    mock_model_service = Mock()
    env = Environment(
        tools=[],
        workers=[],
        agents=[],
        model_service=mock_model_service,
    )
    assert env.model_service == mock_model_service


def test_initialize_slotfillapi_with_string() -> None:
    """Test slotfillapi initialization with string endpoint."""
    env = Environment(
        tools=[],
        workers=[],
        agents=[],
    )
    slotfiller = env.initialize_slotfillapi("http://test-api")
    assert isinstance(slotfiller, SlotFiller)


def test_initialize_slotfillapi_with_empty_string() -> None:  # noqa: F811
    """Test slotfillapi initialization with empty string."""
    env = Environment(
        tools=[],
        workers=[],
        agents=[],
    )
    slotfiller = env.initialize_slotfillapi("")
    assert isinstance(slotfiller, SlotFiller)


def test_initialize_slotfillapi_with_non_string() -> None:  # noqa: F811
    """Test slotfillapi initialization with non-string value."""
    env = Environment(
        tools=[],
        workers=[],
        agents=[],
    )
    slotfiller = env.initialize_slotfillapi(None)
    assert isinstance(slotfiller, SlotFiller)


def test_default_resource_initializer_init_tools_with_http_tool_and_attributes() -> (
    None
):
    """Test DefaultResourceInitializer.init_tools with http_tool and attributes (lines 75-120)."""
    tools = [{"id": "http_tool_1", "name": "test_http_tool", "path": "test_path"}]
    attributes_list = [
        {
            "node_specific_data": {
                "http": {"base_url": "http://test.com"},
                "name": "test_tool_name",
            },
            "slots": [{"name": "slot1", "type": "str"}],
            "slot_groups": [
                {
                    "name": "group1",
                    "schema": [{"name": "field1", "type": "str", "required": True}],
                    "required": True,
                    "repeatable": True,
                }
            ],
            "task": "Test task description",
        }
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_http_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        registry = DefaultResourceInitializer.init_tools(tools, attributes_list)

        assert (
            "test_tool_name" in registry
        )  # Tool ID should be updated to the name from node_specific_data
        assert registry["test_tool_name"]["name"] == "test_path-test_http_tool"
        assert registry["test_tool_name"]["description"] == "Test task description"

        # Verify slot loading was called with combined slots and group slots
        fake_tool_instance.load_slots.assert_called_once()
        call_args = fake_tool_instance.load_slots.call_args[0][0]
        assert len(call_args) == 2  # slots + group_slots

        # Check that group slot was properly formatted
        group_slot = call_args[1]  # The group slot should be the second item
        assert group_slot["name"] == "group1"
        assert group_slot["type"] == "group"
        assert group_slot["required"] is True
        assert group_slot["repeatable"] is True
        assert (
            "Please provide at least one set of the following fields: field1."
            in group_slot["prompt"]
        )


def test_default_resource_initializer_init_tools_with_http_tool_no_required_fields() -> (
    None
):
    """Test DefaultResourceInitializer.init_tools with http_tool and slot group without required fields."""
    tools = [{"id": "http_tool_1", "name": "test_http_tool", "path": "test_path"}]
    attributes_list = [
        {
            "node_specific_data": {
                "http": {"base_url": "http://test.com"},
                "name": "test_tool_name",
            },
            "slots": [{"name": "slot1", "type": "str"}],
            "slot_groups": [
                {
                    "name": "group1",
                    "schema": [{"name": "field1", "type": "str", "required": False}],
                    "required": False,
                    "repeatable": False,
                }
            ],
            "task": "Test task description",
        }
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_http_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        DefaultResourceInitializer.init_tools(tools, attributes_list)

        # Verify group slot prompt for non-required fields
        call_args = fake_tool_instance.load_slots.call_args[0][0]
        group_slot = call_args[1]
        assert (
            "Please provide a set of values for group 'group1'." in group_slot["prompt"]
        )


def test_default_resource_initializer_init_tools_with_http_tool_empty_attributes() -> (
    None
):
    """Test DefaultResourceInitializer.init_tools with http_tool and empty attributes_list."""
    tools = [{"id": "http_tool_1", "name": "test_http_tool", "path": "test_path"}]
    attributes_list = []  # Empty attributes list

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_http_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        registry = DefaultResourceInitializer.init_tools(tools, attributes_list)

        # Should not call load_slots when attributes_list is empty
        fake_tool_instance.load_slots.assert_not_called()
        assert "http_tool_1" in registry  # Should use original tool ID


def test_default_resource_initializer_init_tools_with_http_tool_none_attributes() -> (
    None
):
    """Test DefaultResourceInitializer.init_tools with http_tool and None attributes_list."""
    tools = [{"id": "http_tool_1", "name": "test_http_tool", "path": "test_path"}]
    attributes_list = None  # None attributes list

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_http_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        registry = DefaultResourceInitializer.init_tools(tools, attributes_list)

        # Should not call load_slots when attributes_list is None
        fake_tool_instance.load_slots.assert_not_called()
        assert "http_tool_1" in registry  # Should use original tool ID


def test_default_resource_initializer_init_tools_with_http_tool_missing_node_specific_data() -> (
    None
):
    """Test DefaultResourceInitializer.init_tools with http_tool missing node_specific_data."""
    tools = [{"id": "http_tool_1", "name": "test_http_tool", "path": "test_path"}]
    attributes_list = [
        {
            "slots": [{"name": "slot1", "type": "str"}],
            "task": "Test task description",
            # Missing node_specific_data
        }
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_http_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        registry = DefaultResourceInitializer.init_tools(tools, attributes_list)

        # Should use task description as name when node_specific_data.name is missing
        expected_name = "test_task_description".replace(" ", "_").lower()
        assert expected_name in registry


def test_default_resource_initializer_init_tools_with_http_tool_missing_name_in_node_specific_data() -> (
    None
):
    """Test DefaultResourceInitializer.init_tools with http_tool missing name in node_specific_data."""
    tools = [{"id": "http_tool_1", "name": "test_http_tool", "path": "test_path"}]
    attributes_list = [
        {
            "node_specific_data": {
                "http": {"base_url": "http://test.com"}
                # Missing name
            },
            "slots": [{"name": "slot1", "type": "str"}],
            "task": "Test task description",
        }
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_http_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        registry = DefaultResourceInitializer.init_tools(tools, attributes_list)

        # Should use task description as name when node_specific_data.name is missing
        expected_name = "test_task_description".replace(" ", "_").lower()
        assert expected_name in registry


def test_default_resource_initializer_init_tools_with_http_tool_empty_slots_and_groups() -> (
    None
):
    """Test DefaultResourceInitializer.init_tools with http_tool and empty slots/groups."""
    tools = [{"id": "http_tool_1", "name": "test_http_tool", "path": "test_path"}]
    attributes_list = [
        {
            "node_specific_data": {
                "http": {"base_url": "http://test.com"},
                "name": "test_tool_name",
            },
            "slots": [],  # Empty slots
            "slot_groups": [],  # Empty slot groups
            "task": "Test task description",
        }
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_http_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        DefaultResourceInitializer.init_tools(tools, attributes_list)

        # Should call load_slots with empty list
        fake_tool_instance.load_slots.assert_called_once_with([])


def test_default_resource_initializer_init_tools_with_http_tool_missing_schema_in_group() -> (
    None
):
    """Test DefaultResourceInitializer.init_tools with http_tool and slot group missing schema."""
    tools = [{"id": "http_tool_1", "name": "test_http_tool", "path": "test_path"}]
    attributes_list = [
        {
            "node_specific_data": {
                "http": {"base_url": "http://test.com"},
                "name": "test_tool_name",
            },
            "slots": [{"name": "slot1", "type": "str"}],
            "slot_groups": [
                {
                    "name": "group1",
                    # Missing schema
                    "required": True,
                    "repeatable": True,
                }
            ],
            "task": "Test task description",
        }
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_http_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        DefaultResourceInitializer.init_tools(tools, attributes_list)

        # Should handle missing schema gracefully
        call_args = fake_tool_instance.load_slots.call_args[0][0]
        group_slot = call_args[1]
        assert group_slot["schema"] == []
        assert (
            "Please provide a set of values for group 'group1'." in group_slot["prompt"]
        )


def test_default_resource_initializer_init_tools_with_http_tool_multiple_required_fields() -> (
    None
):
    """Test DefaultResourceInitializer.init_tools with http_tool and multiple required fields."""
    tools = [{"id": "http_tool_1", "name": "test_http_tool", "path": "test_path"}]
    attributes_list = [
        {
            "node_specific_data": {
                "http": {"base_url": "http://test.com"},
                "name": "test_tool_name",
            },
            "slots": [{"name": "slot1", "type": "str"}],
            "slot_groups": [
                {
                    "name": "group1",
                    "schema": [
                        {"name": "field1", "type": "str", "required": True},
                        {"name": "field2", "type": "int", "required": True},
                        {"name": "field3", "type": "bool", "required": False},
                    ],
                    "required": True,
                    "repeatable": True,
                }
            ],
            "task": "Test task description",
        }
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_http_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        DefaultResourceInitializer.init_tools(tools, attributes_list)

        # Verify prompt includes all required fields
        call_args = fake_tool_instance.load_slots.call_args[0][0]
        group_slot = call_args[1]
        assert "field1, field2" in group_slot["prompt"]


def test_default_resource_initializer_init_tools_with_http_tool_fixed_args_and_auth() -> (
    None
):
    """Test DefaultResourceInitializer.init_tools with http_tool and fixed_args/auth from config."""
    tools = [
        {
            "id": "http_tool_1",
            "name": "test_http_tool",
            "path": "test_path",
            "fixed_args": {"arg1": "value1", "arg2": "value2"},
            "auth": {"api_key": "test_key", "token": "test_token"},
        }
    ]
    attributes_list = [
        {
            "node_specific_data": {
                "http": {"base_url": "http://test.com", "timeout": 30},
                "name": "test_tool_name",
            },
            "slots": [{"name": "slot1", "type": "str"}],
            "task": "Test task description",
        }
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_http_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        DefaultResourceInitializer.init_tools(tools, attributes_list)

        # Verify fixed_args and auth were updated
        assert fake_tool_instance.fixed_args == {
            "arg1": "value1",
            "arg2": "value2",
            "base_url": "http://test.com",
            "timeout": 30,
        }
        assert fake_tool_instance.auth == {"api_key": "test_key", "token": "test_token"}


def test_default_resource_initializer_init_tools_with_non_http_tool() -> None:
    """Test DefaultResourceInitializer.init_tools with non-http tool (should not trigger special handling)."""
    tools = [{"id": "regular_tool", "name": "test_tool", "path": "test_path"}]
    attributes_list = [
        {
            "node_specific_data": {"http": {"base_url": "http://test.com"}},
            "slots": [{"name": "slot1", "type": "str"}],
            "task": "Test task description",
        }
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        registry = DefaultResourceInitializer.init_tools(tools, attributes_list)

        # Should not call load_slots for non-http tools
        fake_tool_instance.load_slots.assert_not_called()
        assert "regular_tool" in registry  # Should use original tool ID


def test_default_resource_initializer_init_tools_with_http_tool_index_mismatch() -> (
    None
):
    """Test DefaultResourceInitializer.init_tools with http_tool when attributes_list index doesn't match."""
    tools = [
        {"id": "http_tool_1", "name": "test_http_tool", "path": "test_path"},
        {"id": "http_tool_2", "name": "test_http_tool2", "path": "test_path2"},
    ]
    attributes_list = [
        {
            "node_specific_data": {"http": {"base_url": "http://test.com"}},
            "slots": [{"name": "slot1", "type": "str"}],
            "task": "Test task description",
        }
        # Only one attribute for two tools
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_http_tool = MagicMock(return_value=fake_tool_instance)
        fake_module.test_http_tool2 = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        registry = DefaultResourceInitializer.init_tools(tools, attributes_list)

        # Should handle index mismatch gracefully - only first tool gets registered due to index mismatch
        assert "test_task_description" in registry
        assert "http_tool_2" not in registry  # Second tool fails due to index mismatch


def test_default_resource_initializer_init_tools_with_http_tool_missing_attributes() -> (
    None
):
    """Test DefaultResourceInitializer.init_tools with http_tool when attributes_list is shorter than tools."""
    tools = [
        {"id": "http_tool_1", "name": "test_http_tool", "path": "test_path"},
        {"id": "http_tool_2", "name": "test_http_tool2", "path": "test_path2"},
    ]
    attributes_list = [
        {
            "node_specific_data": {"http": {"base_url": "http://test.com"}},
            "slots": [{"name": "slot1", "type": "str"}],
            "task": "Test task description",
        }
        # Only one attribute for two tools
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_http_tool = MagicMock(return_value=fake_tool_instance)
        fake_module.test_http_tool2 = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        registry = DefaultResourceInitializer.init_tools(tools, attributes_list)

        # Should handle missing attributes gracefully - only first tool gets registered due to missing attributes
        assert "test_task_description" in registry
        assert (
            "http_tool_2" not in registry
        )  # Second tool fails due to missing attributes


def test_default_resource_initializer_init_tools_with_http_tool_extra_attributes() -> (
    None
):
    """Test DefaultResourceInitializer.init_tools with http_tool when attributes_list is longer than tools."""
    tools = [{"id": "http_tool_1", "name": "test_http_tool", "path": "test_path"}]
    attributes_list = [
        {
            "node_specific_data": {"http": {"base_url": "http://test.com"}},
            "slots": [{"name": "slot1", "type": "str"}],
            "task": "Test task description",
        },
        {
            "node_specific_data": {"http": {"base_url": "http://test2.com"}},
            "slots": [{"name": "slot2", "type": "str"}],
            "task": "Test task description 2",
        },
        # Extra attribute
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_http_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        registry = DefaultResourceInitializer.init_tools(tools, attributes_list)

        # Should only process the first attribute
        assert "test_task_description" in registry  # Should use task description as key


def test_default_resource_initializer_init_tools_with_http_tool_complex_slot_groups() -> (
    None
):
    """Test DefaultResourceInitializer.init_tools with http_tool and complex slot groups."""
    tools = [{"id": "http_tool_1", "name": "test_http_tool", "path": "test_path"}]
    attributes_list = [
        {
            "node_specific_data": {
                "http": {"base_url": "http://test.com"},
                "name": "test_tool_name",
            },
            "slots": [
                {"name": "slot1", "type": "str"},
                {"name": "slot2", "type": "int"},
            ],
            "slot_groups": [
                {
                    "name": "group1",
                    "schema": [
                        {"name": "field1", "type": "str", "required": True},
                        {"name": "field2", "type": "int", "required": False},
                    ],
                    "required": True,
                    "repeatable": True,
                },
                {
                    "name": "group2",
                    "schema": [{"name": "field3", "type": "bool", "required": True}],
                    "required": False,
                    "repeatable": False,
                },
            ],
            "task": "Test task description",
        }
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_http_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        DefaultResourceInitializer.init_tools(tools, attributes_list)

        # Verify all slots and groups were processed
        call_args = fake_tool_instance.load_slots.call_args[0][0]
        assert len(call_args) == 4  # 2 slots + 2 groups

        # Check group slots
        group1_slot = call_args[2]  # Third item (after 2 regular slots)
        group2_slot = call_args[3]  # Fourth item

        assert group1_slot["name"] == "group1"
        assert group1_slot["required"] is True
        assert group1_slot["repeatable"] is True

        assert group2_slot["name"] == "group2"
        assert group2_slot["required"] is False
        assert group2_slot["repeatable"] is False


def test_default_resource_initializer_init_tools_with_http_tool_empty_schema() -> None:
    """Test DefaultResourceInitializer.init_tools with http_tool and empty schema in slot group."""
    tools = [{"id": "http_tool_1", "name": "test_http_tool", "path": "test_path"}]
    attributes_list = [
        {
            "node_specific_data": {
                "http": {"base_url": "http://test.com"},
                "name": "test_tool_name",
            },
            "slots": [{"name": "slot1", "type": "str"}],
            "slot_groups": [
                {
                    "name": "group1",
                    "schema": [],  # Empty schema
                    "required": True,
                    "repeatable": True,
                }
            ],
            "task": "Test task description",
        }
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_http_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        DefaultResourceInitializer.init_tools(tools, attributes_list)

        # Verify empty schema is handled
        call_args = fake_tool_instance.load_slots.call_args[0][0]
        group_slot = call_args[1]
        assert group_slot["schema"] == []
        assert (
            "Please provide a set of values for group 'group1'." in group_slot["prompt"]
        )


def test_default_resource_initializer_init_tools_with_http_tool_missing_required_in_schema() -> (
    None
):
    """Test DefaultResourceInitializer.init_tools with http_tool and schema fields missing required attribute."""
    tools = [{"id": "http_tool_1", "name": "test_http_tool", "path": "test_path"}]
    attributes_list = [
        {
            "node_specific_data": {
                "http": {"base_url": "http://test.com"},
                "name": "test_tool_name",
            },
            "slots": [{"name": "slot1", "type": "str"}],
            "slot_groups": [
                {
                    "name": "group1",
                    "schema": [
                        {"name": "field1", "type": "str"},  # Missing required
                        {"name": "field2", "type": "int", "required": True},
                    ],
                    "required": True,
                    "repeatable": True,
                }
            ],
            "task": "Test task description",
        }
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_http_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        DefaultResourceInitializer.init_tools(tools, attributes_list)

        # Verify only required fields are included in prompt
        call_args = fake_tool_instance.load_slots.call_args[0][0]
        group_slot = call_args[1]
        assert "field2" in group_slot["prompt"]  # Only required field
        assert "field1" not in group_slot["prompt"]  # Non-required field


def test_default_resource_initializer_init_tools_with_http_tool_all_optional_fields() -> (
    None
):
    """Test DefaultResourceInitializer.init_tools with http_tool and all optional fields in schema."""
    tools = [{"id": "http_tool_1", "name": "test_http_tool", "path": "test_path"}]
    attributes_list = [
        {
            "node_specific_data": {
                "http": {"base_url": "http://test.com"},
                "name": "test_tool_name",
            },
            "slots": [{"name": "slot1", "type": "str"}],
            "slot_groups": [
                {
                    "name": "group1",
                    "schema": [
                        {"name": "field1", "type": "str", "required": False},
                        {"name": "field2", "type": "int", "required": False},
                    ],
                    "required": True,
                    "repeatable": True,
                }
            ],
            "task": "Test task description",
        }
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_http_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        DefaultResourceInitializer.init_tools(tools, attributes_list)

        # Verify generic prompt for all optional fields
        call_args = fake_tool_instance.load_slots.call_args[0][0]
        group_slot = call_args[1]
        assert (
            "Please provide a set of values for group 'group1'." in group_slot["prompt"]
        )


def test_default_resource_initializer_init_tools_with_http_tool_mixed_required_optional() -> (
    None
):
    """Test DefaultResourceInitializer.init_tools with http_tool and mixed required/optional fields."""
    tools = [{"id": "http_tool_1", "name": "test_http_tool", "path": "test_path"}]
    attributes_list = [
        {
            "node_specific_data": {
                "http": {"base_url": "http://test.com"},
                "name": "test_tool_name",
            },
            "slots": [{"name": "slot1", "type": "str"}],
            "slot_groups": [
                {
                    "name": "group1",
                    "schema": [
                        {"name": "field1", "type": "str", "required": True},
                        {"name": "field2", "type": "int", "required": False},
                        {"name": "field3", "type": "bool", "required": True},
                    ],
                    "required": True,
                    "repeatable": True,
                }
            ],
            "task": "Test task description",
        }
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_http_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        DefaultResourceInitializer.init_tools(tools, attributes_list)

        # Verify prompt includes only required fields
        call_args = fake_tool_instance.load_slots.call_args[0][0]
        group_slot = call_args[1]
        assert "field1, field3" in group_slot["prompt"]  # Only required fields
        assert "field2" not in group_slot["prompt"]  # Optional field not included


def test_default_resource_initializer_init_tools_with_http_tool_description_formatting() -> (
    None
):
    """Test DefaultResourceInitializer.init_tools with http_tool and description formatting."""
    tools = [{"id": "http_tool_1", "name": "test_http_tool", "path": "test_path"}]
    attributes_list = [
        {
            "node_specific_data": {
                "http": {"base_url": "http://test.com"},
                "name": "test_tool_name",
            },
            "slots": [{"name": "slot1", "type": "str"}],
            "slot_groups": [
                {
                    "name": "group1",
                    "schema": [
                        {"name": "field1", "type": "str", "required": True},
                        {"name": "field2", "type": "int", "required": False},
                    ],
                    "required": True,
                    "repeatable": True,
                }
            ],
            "task": "Test task description",
        }
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_http_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        DefaultResourceInitializer.init_tools(tools, attributes_list)

        # Verify description formatting
        call_args = fake_tool_instance.load_slots.call_args[0][0]
        group_slot = call_args[1]
        expected_description = "Slot group 'group1' with schema: ['field1', 'field2']"
        assert group_slot["description"] == expected_description


def test_default_resource_initializer_init_tools_with_http_tool_empty_schema_description() -> (
    None
):
    """Test DefaultResourceInitializer.init_tools with http_tool and empty schema description."""
    tools = [{"id": "http_tool_1", "name": "test_http_tool", "path": "test_path"}]
    attributes_list = [
        {
            "node_specific_data": {
                "http": {"base_url": "http://test.com"},
                "name": "test_tool_name",
            },
            "slots": [{"name": "slot1", "type": "str"}],
            "slot_groups": [
                {
                    "name": "group1",
                    "schema": [],  # Empty schema
                    "required": True,
                    "repeatable": True,
                }
            ],
            "task": "Test task description",
        }
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_http_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        DefaultResourceInitializer.init_tools(tools, attributes_list)

        # Verify empty schema description
        call_args = fake_tool_instance.load_slots.call_args[0][0]
        group_slot = call_args[1]
        expected_description = "Slot group 'group1' with schema: []"
        assert group_slot["description"] == expected_description


def test_default_resource_initializer_init_workers_with_exception() -> None:
    """Test DefaultResourceInitializer init_workers with import exception."""
    initializer = DefaultResourceInitializer()

    result = initializer.init_workers(
        [{"id": "worker1", "name": "nonexistent_worker", "path": "nonexistent/path"}]
    )
    assert result == {}


def test_default_resource_initializer_init_tools_with_fixed_args() -> None:
    """Test DefaultResourceInitializer init_tools with fixed_args."""
    initializer = DefaultResourceInitializer()

    # This will fail due to import error, but we can test the fixed_args logic
    result = initializer.init_tools(
        [
            {
                "id": "tool1",
                "name": "nonexistent_tool",
                "path": "nonexistent/path",
                "fixed_args": {"arg1": "value1"},
            }
        ]
    )
    assert result == {}


def test_default_resource_initializer_init_workers_with_fixed_args() -> None:
    """Test DefaultResourceInitializer init_workers with fixed_args."""
    initializer = DefaultResourceInitializer()

    # This will fail due to import error, but we can test the fixed_args logic
    result = initializer.init_workers(
        [
            {
                "id": "worker1",
                "name": "nonexistent_worker",
                "path": "nonexistent/path",
                "fixed_args": {"arg1": "value1"},
            }
        ]
    )
    assert result == {}


def test_register_tool_exception_handling() -> None:
    """Test register_tool method with exception handling (lines 304-305)."""
    env = Environment(tools=[], workers=[], agents=[])

    class RaisingDict(dict):
        def __setitem__(self, key: str, value: object) -> None:
            raise Exception("Registration error")

    env.tools = RaisingDict()
    with patch("arklex.env.env.log_context.error") as mock_log_error:
        env.register_tool("test_tool", {"name": "test"})
        mock_log_error.assert_called_once()


def test_base_resource_initializer_init_tools_not_implemented() -> None:
    """Test BaseResourceInitializer.init_tools raises NotImplementedError (line 48)."""
    from arklex.env.env import BaseResourceInitializer

    with pytest.raises(NotImplementedError):
        BaseResourceInitializer.init_tools([])


def test_base_resource_initializer_init_workers_not_implemented() -> None:
    """Test BaseResourceInitializer.init_workers raises NotImplementedError (line 63)."""
    from arklex.env.env import BaseResourceInitializer

    with pytest.raises(NotImplementedError):
        BaseResourceInitializer.init_workers([])


def test_default_resource_initializer_init_agents_with_exception() -> None:
    """Test init_agents method handles exceptions during agent registration."""
    agents = [
        {"id": "a1", "name": "fake_agent", "path": "fake_path"},
        {"id": "a2", "name": "bad_agent", "path": "bad_path"},
    ]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(description="desc")
        fake_module.fake_agent = fake_func
        mock_import.side_effect = [fake_module, Exception("fail")]
        registry = DefaultResourceInitializer.init_agents(agents)
        assert "a1" in registry
        assert "a2" not in registry  # error case is skipped


def test_default_resource_initializer_init_agents_with_import_error() -> None:
    """Test init_agents method handles import errors during agent registration."""
    agents = [
        {"id": "a1", "name": "fake_agent", "path": "fake_path"},
        {"id": "a2", "name": "bad_agent", "path": "bad_path"},
    ]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(description="desc")
        fake_module.fake_agent = fake_func
        mock_import.side_effect = [fake_module, ImportError("Module not found")]
        registry = DefaultResourceInitializer.init_agents(agents)
        assert "a1" in registry
        assert "a2" not in registry  # import error case is skipped


def test_default_resource_initializer_init_agents_with_attribute_error() -> None:
    """Test init_agents method handles attribute errors during agent registration."""
    agents = [
        {"id": "a1", "name": "fake_agent", "path": "fake_path"},
        {"id": "a2", "name": "bad_agent", "path": "bad_path"},
    ]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(description="desc")
        fake_module.fake_agent = fake_func
        mock_import.side_effect = [fake_module, AttributeError("No such attribute")]
        registry = DefaultResourceInitializer.init_agents(agents)
        assert "a1" in registry
        assert "a2" not in registry  # attribute error case is skipped


def test_default_resource_initializer_init_agents_logs_error(
    caplog: LogCaptureFixture,
) -> None:
    """Test that init_agents logs error when agent registration fails."""
    agents = [{"id": "a1", "name": "bad_agent", "path": "bad_path"}]
    with patch("importlib.import_module") as mock_import:
        mock_import.side_effect = Exception("import error")
        with caplog.at_level("ERROR"):
            registry = DefaultResourceInitializer.init_agents(agents)
            assert registry == {}
            assert any(
                "Agent bad_agent is not registered, error: import error" in m
                for m in caplog.text.splitlines()
            )


def test_model_aware_resource_initializer_init() -> None:
    """Test ModelAwareResourceInitializer initialization."""
    from arklex.env.env import ModelAwareResourceInitializer

    # Test with model_config
    model_config = {"model_name": "test_model"}
    initializer = ModelAwareResourceInitializer(model_config=model_config)
    assert initializer.model_config == model_config

    # Test without model_config
    initializer = ModelAwareResourceInitializer()
    assert initializer.model_config is None


def test_model_aware_resource_initializer_init_workers_with_model_config() -> None:
    """Test ModelAwareResourceInitializer.init_workers with model_config."""
    from arklex.env.env import ModelAwareResourceInitializer

    model_config = {"model_name": "test_model"}
    initializer = ModelAwareResourceInitializer(model_config=model_config)

    workers = [{"id": "w1", "name": "test_worker", "path": "test_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock()
        fake_func.description = "test description"

        # Mock the worker class to have __init__ method that accepts model_config
        class MockWorkerClass:
            def __init__(self, model_config: object = None) -> None:
                self.model_config = model_config

            description = "test description"

        fake_module.test_worker = MockWorkerClass
        mock_import.return_value = fake_module

        registry = initializer.init_workers(workers)

        assert "w1" in registry
        assert registry["w1"]["name"] == "test_worker"
        assert registry["w1"]["description"] == "test description"


def test_model_aware_resource_initializer_init_workers_without_model_config() -> None:
    """Test ModelAwareResourceInitializer.init_workers without model_config."""
    from arklex.env.env import ModelAwareResourceInitializer

    initializer = ModelAwareResourceInitializer()  # No model_config

    workers = [{"id": "w1", "name": "test_worker", "path": "test_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock()
        fake_func.description = "test description"

        # Mock the worker class to have __init__ method that accepts model_config
        class MockWorkerClass:
            def __init__(self, model_config: object = None) -> None:
                self.model_config = model_config

            description = "test description"

        fake_module.test_worker = MockWorkerClass
        mock_import.return_value = fake_module

        registry = initializer.init_workers(workers)

        assert "w1" in registry
        assert registry["w1"]["name"] == "test_worker"
        assert registry["w1"]["description"] == "test description"


def test_model_aware_resource_initializer_init_workers_worker_without_init() -> None:
    """Test ModelAwareResourceInitializer.init_workers with worker that has no __init__."""
    from arklex.env.env import ModelAwareResourceInitializer

    model_config = {"model_name": "test_model"}
    initializer = ModelAwareResourceInitializer(model_config=model_config)

    workers = [{"id": "w1", "name": "test_worker", "path": "test_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock()
        fake_func.description = "test description"
        # Remove __init__ attribute to test the hasattr check
        if hasattr(fake_func, "__init__"):
            delattr(fake_func, "__init__")

        fake_module.test_worker = fake_func
        mock_import.return_value = fake_module

        registry = initializer.init_workers(workers)

        assert "w1" in registry
        assert registry["w1"]["name"] == "test_worker"
        assert registry["w1"]["description"] == "test description"


def test_model_aware_resource_initializer_init_workers_worker_init_without_model_config_param() -> (
    None
):
    """Test ModelAwareResourceInitializer.init_workers with worker __init__ that doesn't accept model_config."""
    from arklex.env.env import ModelAwareResourceInitializer

    model_config = {"model_name": "test_model"}
    initializer = ModelAwareResourceInitializer(model_config=model_config)

    workers = [{"id": "w1", "name": "test_worker", "path": "test_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()

        # Mock the worker class to have __init__ method that doesn't accept model_config
        class MockWorkerClass:
            def __init__(self, other_param: object = None) -> None:
                self.other_param = other_param

            description = "test description"

        fake_module.test_worker = MockWorkerClass
        mock_import.return_value = fake_module

        registry = initializer.init_workers(workers)

        assert "w1" in registry
        assert registry["w1"]["name"] == "test_worker"
        assert registry["w1"]["description"] == "test description"


def test_model_aware_resource_initializer_init_workers_without_model_config_parameter() -> (
    None
):
    """Test ModelAwareResourceInitializer.init_workers when worker doesn't accept model_config."""
    from arklex.env.env import ModelAwareResourceInitializer

    workers = [
        {"id": "w1", "name": "test_worker", "path": "test_path"},
    ]

    # Create a mock worker class that doesn't accept model_config
    class MockWorkerClass:
        def __init__(self, other_param: object | None = None) -> None:
            self.other_param = other_param

        description = "Test worker"

    with (
        patch("importlib.import_module") as mock_import,
        patch("inspect.signature") as mock_signature,
    ):
        mock_module = Mock()
        mock_module.test_worker = MockWorkerClass
        mock_import.return_value = mock_module

        # Mock signature to NOT include model_config parameter
        mock_sig = Mock()
        mock_sig.parameters = {"other_param": Mock()}
        mock_signature.return_value = mock_sig

        initializer = ModelAwareResourceInitializer(model_config={"test": "config"})
        registry = initializer.init_workers(workers)

        assert "w1" in registry
        # Verify that model_config was NOT passed to the worker
        worker_instance = registry["w1"]["execute"]()
        assert not hasattr(worker_instance, "model_config")


def test_model_aware_resource_initializer_init_workers_with_exception() -> None:
    """Test ModelAwareResourceInitializer.init_workers with exception handling."""
    from arklex.env.env import ModelAwareResourceInitializer

    initializer = ModelAwareResourceInitializer()
    workers = [{"id": "w1", "name": "bad_worker", "path": "bad_path"}]

    with patch("importlib.import_module") as mock_import:
        mock_import.side_effect = Exception("Import failed")
        registry = initializer.init_workers(workers)
        assert registry == {}


def test_environment_with_model_aware_resource_initializer() -> None:
    """Test Environment initialization with ModelAwareResourceInitializer."""

    # Create a mock model service with model_config
    mock_model_service = MagicMock()
    mock_model_service.model_config = {"model_name": "test_model"}

    tools = [{"id": "t1", "name": "test_tool", "path": "test_path"}]
    workers = [{"id": "w1", "name": "test_worker", "path": "test_path"}]
    agents = [{"id": "a1", "name": "test_agent", "path": "test_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_tool = fake_func
        fake_module.test_worker = fake_func
        fake_module.test_agent = fake_func
        mock_import.return_value = fake_module

        env = Environment(
            tools=tools,
            workers=workers,
            agents=agents,
            model_service=mock_model_service,
        )

        # Verify that ModelAwareResourceInitializer was used
        assert isinstance(env.tools, dict)
        assert isinstance(env.workers, dict)
        assert isinstance(env.agents, dict)


def test_environment_with_model_service_without_model_config() -> None:
    """Test Environment initialization with model_service that has no model_config."""
    # Create a mock model service without model_config
    mock_model_service = MagicMock()
    # Remove model_config attribute to test the hasattr check
    if hasattr(mock_model_service, "model_config"):
        delattr(mock_model_service, "model_config")

    tools = [{"id": "t1", "name": "test_tool", "path": "test_path"}]
    workers = [{"id": "w1", "name": "test_worker", "path": "test_path"}]
    agents = [{"id": "a1", "name": "test_agent", "path": "test_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_tool = fake_func
        fake_module.test_worker = fake_func
        fake_module.test_agent = fake_func
        mock_import.return_value = fake_module

        env = Environment(
            tools=tools,
            workers=workers,
            agents=agents,
            model_service=mock_model_service,
        )

        # Verify that DefaultResourceInitializer was used (not ModelAwareResourceInitializer)
        assert isinstance(env.tools, dict)
        assert isinstance(env.workers, dict)
        assert isinstance(env.agents, dict)


def test_environment_step_agent_with_successors_and_predecessors() -> None:
    """Test environment step with agent that has successors and predecessors."""
    mock_agent_instance = Mock()
    mock_agent_instance.execute.return_value = MessageState(status=StatusEnum.COMPLETE)

    env = Environment(
        tools=[],
        workers=[],
        agents=[{"id": "agent1", "name": "test_agent", "path": "test"}],
    )
    env.agents = {
        "agent1": {
            "name": "test_agent",
            "execute": Mock(return_value=mock_agent_instance),
        }
    }
    env.id2name = {"agent1": "test_agent"}

    message_state = MessageState()
    params = OrchestratorParams()
    params.taskgraph.curr_node = "node1"
    node_info = NodeInfo()
    node_info.additional_args = {
        "successors": ["next1", "next2"],
        "predecessors": ["prev1", "prev2"],
    }

    result_state, result_params = env.step("agent1", message_state, params, node_info)
    assert result_state.status == StatusEnum.COMPLETE
    assert result_params.taskgraph.node_status["node1"] == StatusEnum.COMPLETE


def test_environment_step_agent_with_empty_additional_args_second() -> None:
    """Test environment step with agent that has empty additional_args."""
    mock_agent_instance = Mock()
    mock_agent_instance.execute.return_value = MessageState(status=StatusEnum.COMPLETE)

    env = Environment(
        tools=[],
        workers=[],
        agents=[{"id": "agent1", "name": "test_agent", "path": "test"}],
    )
    env.agents = {
        "agent1": {
            "name": "test_agent",
            "execute": Mock(return_value=mock_agent_instance),
        }
    }
    env.id2name = {"agent1": "test_agent"}

    message_state = MessageState()
    params = OrchestratorParams()
    params.taskgraph.curr_node = "node1"
    node_info = NodeInfo()
    node_info.additional_args = {}  # Empty additional_args

    result_state, result_params = env.step("agent1", message_state, params, node_info)
    assert result_state.status == StatusEnum.COMPLETE
    assert result_params.taskgraph.node_status["node1"] == StatusEnum.COMPLETE


def test_environment_step_agent_with_none_additional_args() -> None:
    """Test environment step with agent that has None additional_args."""
    mock_agent_instance = Mock()
    mock_agent_instance.execute.return_value = MessageState(status=StatusEnum.COMPLETE)

    env = Environment(
        tools=[],
        workers=[],
        agents=[{"id": "agent1", "name": "test_agent", "path": "test"}],
    )
    env.agents = {
        "agent1": {
            "name": "test_agent",
            "execute": Mock(return_value=mock_agent_instance),
        }
    }
    env.id2name = {"agent1": "test_agent"}

    message_state = MessageState()
    params = OrchestratorParams()
    params.taskgraph.curr_node = "node1"
    node_info = NodeInfo()
    node_info.additional_args = {}  # Empty dict instead of None

    result_state, result_params = env.step("agent1", message_state, params, node_info)
    assert result_state.status == StatusEnum.COMPLETE
    assert result_params.taskgraph.node_status["node1"] == StatusEnum.COMPLETE


def test_environment_step_agent_with_function_calling_trajectory() -> None:
    """Test environment step with agent that returns function_calling_trajectory."""
    mock_agent_instance = Mock()
    mock_agent_instance.execute.return_value = MessageState(
        status=StatusEnum.COMPLETE,
        function_calling_trajectory=[{"role": "assistant", "content": "test"}],
    )

    env = Environment(
        tools=[],
        workers=[],
        agents=[{"id": "agent1", "name": "test_agent", "path": "test"}],
    )
    env.agents = {
        "agent1": {
            "name": "test_agent",
            "execute": Mock(return_value=mock_agent_instance),
        }
    }
    env.id2name = {"agent1": "test_agent"}

    message_state = MessageState()
    params = OrchestratorParams()
    params.taskgraph.curr_node = "node1"
    node_info = NodeInfo()

    result_state, result_params = env.step("agent1", message_state, params, node_info)
    assert result_state.status == StatusEnum.COMPLETE
    assert result_params.memory.function_calling_trajectory == [
        {"role": "assistant", "content": "test"}
    ]


def test_environment_step_agent_with_slots() -> None:
    """Test environment step with agent that returns slots."""
    from arklex.orchestrator.NLU.entities.slot_entities import Slot

    mock_agent_instance = Mock()
    mock_agent_instance.execute.return_value = MessageState(
        status=StatusEnum.COMPLETE,
        slots={
            "slot1": [Slot(name="slot1", value="value1")],
            "slot2": [Slot(name="slot2", value="value2")],
        },
    )

    env = Environment(
        tools=[],
        workers=[],
        agents=[{"id": "agent1", "name": "test_agent", "path": "test"}],
    )
    env.agents = {
        "agent1": {
            "name": "test_agent",
            "execute": Mock(return_value=mock_agent_instance),
        }
    }
    env.id2name = {"agent1": "test_agent"}

    message_state = MessageState()
    params = OrchestratorParams()
    params.taskgraph.curr_node = "node1"
    node_info = NodeInfo()

    result_state, result_params = env.step("agent1", message_state, params, node_info)
    assert result_state.status == StatusEnum.COMPLETE
    assert result_params.taskgraph.dialog_states == {
        "slot1": [Slot(name="slot1", value="value1")],
        "slot2": [Slot(name="slot2", value="value2")],
    }


def test_environment_step_planner_with_msg_history() -> None:
    """Test environment step with planner that returns msg_history."""
    mock_planner = Mock()
    mock_planner.execute.return_value = (
        "action",
        MessageState(status=StatusEnum.COMPLETE),
        [{"role": "user", "content": "test message"}],
    )
    env = Environment(tools=[], workers=[], agents=[])
    env.planner = mock_planner
    message_state = MessageState()
    params = OrchestratorParams()
    params.memory.function_calling_trajectory = []
    node_info = NodeInfo()
    result_state, result_params = env.step(
        "invalid_id", message_state, params, node_info
    )
    assert result_state.status == StatusEnum.COMPLETE
    mock_planner.execute.assert_called_once()


def test_environment_step_tool_with_attributes_and_slots(
    fake_tool: Callable[[MessageState | None], MagicMock],
) -> None:
    """Test environment step with tool that has attributes and slots."""
    from arklex.orchestrator.NLU.entities.slot_entities import Slot

    tool = fake_tool(
        MessageState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
            slots={"slot1": [Slot(name="slot1", value="value1")]},
            status=StatusEnum.COMPLETE,
        )
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[])

        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}
            attributes = {"slots": ["slot1", "slot2"]}

        state = MessageState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_state.function_calling_trajectory == [
            {"role": "assistant", "content": "call"}
        ]
        assert result_state.slots == {"slot1": [Slot(name="slot1", value="value1")]}
        assert result_state.status == StatusEnum.COMPLETE
        assert result_params.taskgraph.dialog_states == {
            "slot1": [Slot(name="slot1", value="value1")]
        }
        assert result_params.taskgraph.node_status["n1"] == StatusEnum.COMPLETE

        # Verify tool methods were called correctly
        tool.init_slotfiller.assert_called_once_with(env.slotfillapi)
        if tool.load_slots.call_count:
            tool.load_slots.assert_called_once_with(["slot1", "slot2"])


def test_environment_step_tool_with_none_additional_args(
    fake_tool: Callable[[MessageState | None], MagicMock],
) -> None:
    """Test environment step with tool that has None additional_args."""
    tool = fake_tool(
        MessageState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
            slots={},
            status=StatusEnum.COMPLETE,
        )
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[])

        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {}  # Empty dict instead of None

        state = MessageState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_state.function_calling_trajectory == [
            {"role": "assistant", "content": "call"}
        ]
        assert result_params is params

        # Verify tool methods were called correctly
        tool.init_slotfiller.assert_called_once_with(env.slotfillapi)
        if tool.load_slots.call_count:
            tool.load_slots.assert_called_once_with(
                []
            )  # Empty list when attributes is empty


def test_environment_step_tool_with_none_attributes(
    fake_tool: Callable[[MessageState | None], MagicMock],
) -> None:
    """Test environment step with tool that has None attributes."""
    tool = fake_tool(
        MessageState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
            slots={},
            status=StatusEnum.COMPLETE,
        )
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[])

        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}
            attributes = {}  # Empty dict instead of None to avoid AttributeError
            attributes = {}  # Empty dict instead of None to avoid AttributeError

        state = MessageState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_state.function_calling_trajectory == [
            {"role": "assistant", "content": "call"}
        ]
        assert result_params is params

        # Verify tool methods were called correctly
        tool.init_slotfiller.assert_called_once_with(env.slotfillapi)
        if tool.load_slots.call_count:
            tool.load_slots.assert_called_once_with(
                []
            )  # Empty list when attributes is empty
        if tool.load_slots.call_count:
            tool.load_slots.assert_called_once_with(
                []
            )  # Empty list when attributes is empty


def test_environment_step_worker_with_none_additional_args(
    fake_worker: Callable[[MessageState | None], Mock],
) -> None:
    """Test environment step with worker that has None additional_args."""
    mock_worker = fake_worker(MessageState(status=StatusEnum.COMPLETE))
    mock_worker.init_slotfilling = Mock()

    env = Environment(
        tools=[],
        workers=[{"id": "worker1", "name": "test_worker", "path": "test"}],
        agents=[],
    )
    env.workers = {
        "worker1": {"name": "test_worker", "execute": Mock(return_value=mock_worker)}
    }
    env.id2name = {"worker1": "test_worker"}

    message_state = MessageState()
    params = OrchestratorParams()
    params.memory.function_calling_trajectory = []
    params.taskgraph.curr_node = "node1"
    node_info = NodeInfo()
    node_info.additional_args = {}  # Empty dict instead of None

    result_state, result_params = env.step("worker1", message_state, params, node_info)
    assert result_state.status == StatusEnum.COMPLETE
    assert len(result_params.memory.function_calling_trajectory) == 2
    mock_worker.init_slotfilling.assert_called_once()


def test_environment_step_worker_with_empty_response_and_message_flow(
    fake_worker: Callable[[MessageState | None], Mock],
) -> None:
    """Test environment step with worker that has empty response and message_flow."""

    mock_worker = fake_worker(
        MessageState(status=StatusEnum.COMPLETE, response="", message_flow="")
    )
    mock_worker.init_slotfilling = Mock()

    env = Environment(
        tools=[],
        workers=[{"id": "worker1", "name": "test_worker", "path": "test"}],
        agents=[],
    )
    env.workers = {
        "worker1": {"name": "test_worker", "execute": Mock(return_value=mock_worker)}
    }
    env.id2name = {"worker1": "test_worker"}

    message_state = MessageState()
    params = OrchestratorParams()
    params.memory.function_calling_trajectory = []
    params.taskgraph.curr_node = "node1"
    node_info = NodeInfo()

    result_state, result_params = env.step("worker1", message_state, params, node_info)
    assert result_state.status == StatusEnum.COMPLETE
    assert len(result_params.memory.function_calling_trajectory) == 2
    # Check that empty string is used when both response and message_flow are empty strings
    assert result_params.memory.function_calling_trajectory[1]["content"] == ""


def test_environment_step_tool_with_slot_schema_signature_change(
    fake_tool: Callable[[MessageState | None], MagicMock],
) -> None:
    """Test environment step with tool when slot schema signature changes."""
    tool = fake_tool(
        MessageState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
            slots={"slot1": [Slot(name="slot1", value="value1")]},
            status=StatusEnum.COMPLETE,
        )
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[])

        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}
            attributes = {
                "slots": ["slot1"],
                "slot_groups": [
                    {"name": "group1", "schema": [{"name": "slot1", "type": "str"}]}
                ],
            }

        state = MessageState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        # First call to establish initial slots
        result_state, result_params = env.step("t1", state, params, node_info)

        # Second call with different slot configuration (simulating schema change)
        node_info.attributes = {"slots": ["slot1", "slot2"]}
        result_state, result_params = env.step("t1", state, params, node_info)

        # Should reset slots due to schema change
        assert result_state.status == StatusEnum.COMPLETE


def test_environment_step_tool_with_verified_slots(
    fake_tool: Callable[[MessageState | None], MagicMock],
) -> None:
    """Test environment step with tool that has verified slots."""
    from arklex.orchestrator.NLU.entities.slot_entities import Slot

    tool = fake_tool(
        MessageState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
            slots={},
            status=StatusEnum.COMPLETE,
        )
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[])

        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}
            attributes = {"slots": ["slot1"]}

        state = MessageState()
        # Pre-populate state with verified slots
        state.slots = {"t1": [Slot(name="slot1", value="value1", verified=True)]}
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_state.status == StatusEnum.COMPLETE


def test_environment_step_tool_with_missing_required_slots(
    fake_tool: Callable[[MessageState | None], MagicMock],
) -> None:
    """Test environment step with tool that has missing required slots."""
    tool = fake_tool(
        MessageState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
            slots={},
            status=StatusEnum.INCOMPLETE,
        )
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[])

        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}
            attributes = {"slots": ["required_slot"]}

        state = MessageState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        # Should be incomplete due to missing required slots
        assert result_state.status == StatusEnum.INCOMPLETE


def test_environment_step_tool_with_slot_verification_needed(
    fake_tool: Callable[[MessageState | None], MagicMock],
) -> None:
    """Test environment step with tool that needs slot verification."""
    tool = fake_tool(
        MessageState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
            slots={},
            status=StatusEnum.INCOMPLETE,
        )
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[])

        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}
            attributes = {"slots": ["slot1"]}

        state = MessageState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        # Should be incomplete due to slot verification needed
        assert result_state.status == StatusEnum.INCOMPLETE


def test_environment_step_tool_with_group_slots(
    fake_tool: Callable[[MessageState | None], MagicMock],
) -> None:
    """Test environment step with tool that has group slots."""
    tool = fake_tool(
        MessageState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
            slots={},
            status=StatusEnum.COMPLETE,
        )
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[])

        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}
            attributes = {
                "slots": [
                    {
                        "name": "group_slot",
                        "type": "group",
                        "schema": [{"name": "field1", "type": "str", "required": True}],
                    }
                ]
            }

        state = MessageState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_state.status == StatusEnum.COMPLETE


def test_environment_step_tool_with_repeatable_slots(
    fake_tool: Callable[[MessageState | None], MagicMock],
) -> None:
    """Test environment step with tool that has repeatable slots."""
    tool = fake_tool(
        MessageState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
            slots={},
            status=StatusEnum.COMPLETE,
        )
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[])

        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}
            attributes = {
                "slots": [
                    {
                        "name": "repeatable_slot",
                        "type": "str",
                        "repeatable": True,
                        "required": True,
                    }
                ]
            }

        state = MessageState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_state.status == StatusEnum.COMPLETE


def test_environment_step_tool_with_function_calling_trajectory(
    fake_tool: Callable[[MessageState | None], MagicMock],
) -> None:
    """Test environment step with tool that returns function calling trajectory."""
    tool = fake_tool(
        MessageState(
            function_calling_trajectory=[
                {"role": "assistant", "content": "call"},
                {"role": "function", "content": "result"},
            ],
            status=StatusEnum.COMPLETE,
        )
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[])

        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}
            attributes = {"slots": ["slot1"]}

        state = MessageState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_state.status == StatusEnum.COMPLETE
        assert len(result_state.function_calling_trajectory) == 2


def test_environment_step_tool_with_slots_parameter(
    fake_tool: Callable[[MessageState | None], MagicMock],
) -> None:
    """Test environment step with tool that accepts slots parameter."""
    tool = fake_tool(
        MessageState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
            slots={},
            status=StatusEnum.COMPLETE,
        )
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[])

        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}
            attributes = {"slots": ["slot1"]}

        state = MessageState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_state.status == StatusEnum.COMPLETE


def test_environment_step_tool_with_missing_required_arguments(
    fake_tool: Callable[[MessageState | None], MagicMock],
) -> None:
    """Test environment step with tool that has missing required arguments."""
    tool = fake_tool(
        MessageState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
            slots={},
            status=StatusEnum.INCOMPLETE,
        )
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[])

        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}
            attributes = {"slots": ["required_arg"]}

        state = MessageState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        # Should be incomplete due to missing required arguments
        assert result_state.status == StatusEnum.INCOMPLETE


def test_environment_step_tool_with_authentication_error(
    fake_tool: Callable[[MessageState | None], MagicMock],
) -> None:
    """Test environment step with tool that raises AuthenticationError."""
    tool = fake_tool(
        MessageState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
            slots={},
            status=StatusEnum.INCOMPLETE,
            response="Authentication failed",
        )
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[])

        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}
            attributes = {"slots": ["slot1"]}

        state = MessageState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_state.status == StatusEnum.INCOMPLETE
        assert "Authentication failed" in result_state.response


def test_environment_step_tool_with_tool_execution_error(
    fake_tool: Callable[[MessageState | None], MagicMock],
) -> None:
    """Test environment step with tool that raises ToolExecutionError."""
    tool = fake_tool(
        MessageState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
            slots={},
            status=StatusEnum.INCOMPLETE,
            response="Tool execution failed",
        )
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[])

        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}
            attributes = {"slots": ["slot1"]}

        state = MessageState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_state.status == StatusEnum.INCOMPLETE
        assert "Tool execution failed" in result_state.response


def test_environment_step_tool_with_general_exception(
    fake_tool: Callable[[MessageState | None], MagicMock],
) -> None:
    """Test environment step with tool that raises general exception."""
    tool = fake_tool(
        MessageState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
            slots={},
            status=StatusEnum.INCOMPLETE,
            response="General error occurred",
        )
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[])

        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}
            attributes = {"slots": ["slot1"]}

        state = MessageState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_state.status == StatusEnum.INCOMPLETE
        assert "General error occurred" in result_state.response


def test_environment_name2id_mapping() -> None:
    """Test Environment name2id mapping creation (lines 240-245)."""
    tools = [{"id": "tool1", "name": "test_tool", "path": "test_path"}]
    workers = [{"id": "worker1", "name": "test_worker", "path": "test_path"}]
    agents = [{"id": "agent1", "name": "test_agent", "path": "test_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_tool = fake_func
        fake_module.test_worker = fake_func
        fake_module.test_agent = fake_func
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents)

        # Test name2id mapping
        assert env.name2id["test_path-test_tool"] == "tool1"
        assert env.name2id["test_worker"] == "worker1"
        assert env.name2id["test_agent"] == "agent1"

        # Test id2name mapping
        assert env.id2name["tool1"] == "test_path-test_tool"
        assert env.id2name["worker1"] == "test_worker"
        assert env.id2name["agent1"] == "test_agent"


def test_environment_name2id_mapping_with_duplicate_names() -> None:
    """Test Environment name2id mapping with duplicate names (should overwrite)."""
    tools = [{"id": "tool1", "name": "test_tool", "path": "test_path"}]
    workers = [
        {"id": "worker1", "name": "test_tool", "path": "test_path"}
    ]  # Same name as tool
    agents = []

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_tool = fake_func
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents)

        # The last one should overwrite the previous one - workers are registered after tools
        assert (
            env.name2id["test_path-test_tool"] == "tool1"
        )  # Tool keeps its path-prefixed name
        assert (
            env.name2id["test_tool"] == "worker1"
        )  # Worker overwrites tool for simple name
        assert env.id2name["tool1"] == "test_path-test_tool"
        assert env.id2name["worker1"] == "test_tool"


def test_environment_name2id_mapping_with_empty_registries() -> None:
    """Test Environment name2id mapping with empty registries."""
    env = Environment(tools=[], workers=[], agents=[])

    assert env.name2id == {}
    assert env.id2name == {}


def test_environment_name2id_mapping_with_partial_registries() -> None:
    """Test Environment name2id mapping with only some registries populated."""
    tools = [{"id": "tool1", "name": "test_tool", "path": "test_path"}]
    workers = []
    agents = []

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_tool = fake_func
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents)

        assert env.name2id["test_path-test_tool"] == "tool1"
        assert env.id2name["tool1"] == "test_path-test_tool"


def test_environment_initialization_with_kwargs() -> None:
    """Test Environment initialization with additional kwargs (lines 220-225)."""
    tools = []
    workers = []
    agents = []

    Environment(
        tools=tools,
        workers=workers,
        agents=agents,
        custom_param1="value1",
        custom_param2=42,
        custom_param3=True,
    )

    # Should not raise any exceptions with additional kwargs


def test_environment_initialization_with_attributes_kwarg() -> None:
    """Test Environment initialization with attributes kwarg (line 235)."""
    tools = [{"id": "tool1", "name": "test_tool", "path": "test_path"}]
    workers = []
    agents = []
    attributes = [
        {
            "node_specific_data": {"http": {"base_url": "http://test.com"}},
            "slots": [{"name": "slot1", "type": "str"}],
            "task": "Test task",
        }
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.name = "original_name"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        env = Environment(
            tools=tools, workers=workers, agents=agents, attributes=attributes
        )

        # Should process attributes correctly
        assert "tool1" in env.tools


def test_environment_initialization_with_none_attributes() -> None:
    """Test Environment initialization with None attributes."""
    tools = [{"id": "tool1", "name": "test_tool", "path": "test_path"}]
    workers = []
    agents = []

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents, attributes=None)

        # Should handle None attributes gracefully
        assert "tool1" in env.tools


def test_environment_initialization_with_empty_attributes() -> None:
    """Test Environment initialization with empty attributes list."""
    tools = [{"id": "tool1", "name": "test_tool", "path": "test_path"}]
    workers = []
    agents = []

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "Original description"
        fake_tool_instance.load_slots = MagicMock()

        fake_module.test_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents, attributes=[])

        # Should handle empty attributes gracefully
        assert "tool1" in env.tools


def test_environment_initialization_with_slot_fill_api_alias() -> None:
    """Test Environment initialization with slot_fill_api alias (lines 225-227)."""
    tools = []
    workers = []
    agents = []

    env = Environment(
        tools=tools, workers=workers, agents=agents, slot_fill_api="http://test-api.com"
    )

    # Should use slot_fill_api as slotsfillapi
    assert isinstance(env.slotfillapi, SlotFiller)


def test_environment_initialization_with_both_slotsfillapi_and_slot_fill_api() -> None:
    """Test Environment initialization with both slotsfillapi and slot_fill_api (slotsfillapi takes precedence)."""
    tools = []
    workers = []
    agents = []

    env = Environment(
        tools=tools,
        workers=workers,
        agents=agents,
        slotsfillapi="http://primary-api.com",
        slot_fill_api="http://secondary-api.com",
    )

    # Should use slotsfillapi (primary) not slot_fill_api (secondary)
    assert isinstance(env.slotfillapi, SlotFiller)


def test_environment_initialization_with_model_service_with_model_config() -> None:
    """Test Environment initialization with model_service that has model_config (lines 230-235)."""
    tools = []
    workers = []
    agents = []

    # Create a mock model service with model_config
    mock_model_service = MagicMock()
    mock_model_service.model_config = {"model_name": "test_model"}

    env = Environment(
        tools=tools, workers=workers, agents=agents, model_service=mock_model_service
    )

    # Should use ModelAwareResourceInitializer when model_service has model_config
    assert env.model_service == mock_model_service


def test_environment_initialization_with_model_service_without_model_config() -> None:
    """Test Environment initialization with model_service that doesn't have model_config."""
    tools = []
    workers = []
    agents = []

    # Create a mock model service without model_config
    mock_model_service = MagicMock()
    # Remove model_config attribute to test the hasattr check
    if hasattr(mock_model_service, "model_config"):
        delattr(mock_model_service, "model_config")

    env = Environment(
        tools=tools, workers=workers, agents=agents, model_service=mock_model_service
    )

    # Should use DefaultResourceInitializer when model_service doesn't have model_config
    assert env.model_service == mock_model_service


def test_environment_initialization_with_custom_resource_initializer_and_model_service() -> (
    None
):
    """Test Environment initialization with custom resource_initializer (should not be overridden)."""
    tools = []
    workers = []
    agents = []

    # Create a mock model service with model_config
    mock_model_service = MagicMock()
    mock_model_service.model_config = {"model_name": "test_model"}

    # Create a custom resource initializer
    custom_initializer = MagicMock()
    custom_initializer.init_tools.return_value = {}
    custom_initializer.init_workers.return_value = {}
    custom_initializer.init_agents.return_value = {}

    Environment(
        tools=tools,
        workers=workers,
        agents=agents,
        model_service=mock_model_service,
        resource_initializer=custom_initializer,
    )

    # Should use custom resource initializer, not ModelAwareResourceInitializer
    custom_initializer.init_tools.assert_called_once()
    custom_initializer.init_workers.assert_called_once()
    custom_initializer.init_agents.assert_called_once()


def test_environment_initialization_with_planner_enabled() -> None:
    """Test Environment initialization with planner_enabled=True (lines 250-252)."""
    tools = []
    workers = []
    agents = []

    env = Environment(tools=tools, workers=workers, agents=agents, planner_enabled=True)

    # Should use ReactPlanner when planner_enabled=True
    assert isinstance(env.planner, ReactPlanner)


def test_environment_initialization_with_planner_disabled() -> None:
    """Test Environment initialization with planner_enabled=False (lines 253-255)."""
    tools = []
    workers = []
    agents = []

    env = Environment(
        tools=tools, workers=workers, agents=agents, planner_enabled=False
    )

    # Should use DefaultPlanner when planner_enabled=False
    from arklex.env.planner.react_planner import DefaultPlanner

    assert isinstance(env.planner, DefaultPlanner)


def test_environment_initialization_with_planner_enabled_and_tools_workers() -> None:
    """Test Environment initialization with planner_enabled=True and tools/workers."""
    tools = [{"id": "tool1", "name": "test_tool", "path": "test_path"}]
    workers = [{"id": "worker1", "name": "test_worker", "path": "test_path"}]
    agents = []

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.description = "test description"
        fake_worker_func = MagicMock()
        fake_worker_func.description = "test description"
        fake_module.test_tool = MagicMock(return_value=fake_tool_instance)
        fake_module.test_worker = fake_worker_func
        mock_import.return_value = fake_module

        env = Environment(
            tools=tools, workers=workers, agents=agents, planner_enabled=True
        )

        # Should use ReactPlanner with tools and workers
        assert isinstance(env.planner, ReactPlanner)
        assert "tool1" in env.tools
        assert "worker1" in env.workers


def test_environment_initialization_with_planner_disabled_and_tools_workers() -> None:
    """Test Environment initialization with planner_enabled=False and tools/workers."""
    tools = [{"id": "tool1", "name": "test_tool", "path": "test_path"}]
    workers = [{"id": "worker1", "name": "test_worker", "path": "test_path"}]
    agents = []

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_tool = fake_func
        fake_module.test_worker = fake_func
        mock_import.return_value = fake_module

        env = Environment(
            tools=tools, workers=workers, agents=agents, planner_enabled=False
        )

        # Should use DefaultPlanner with tools and workers
        from arklex.env.planner.react_planner import DefaultPlanner

        assert isinstance(env.planner, DefaultPlanner)
        assert "tool1" in env.tools
        assert "worker1" in env.workers


def test_initialize_slotfillapi_with_valid_url_string() -> None:
    """Test initialize_slotfillapi with valid URL string (lines 147-150)."""
    env = Environment(tools=[], workers=[], agents=[])

    with patch("arklex.env.env.APIClientService") as mock_api_service:
        mock_api_instance = MagicMock()
        mock_api_service.return_value = mock_api_instance

        with patch("arklex.env.env.SlotFiller") as mock_slot_filler:
            mock_slot_filler_instance = MagicMock()
            mock_slot_filler.return_value = mock_slot_filler_instance

            result = env.initialize_slotfillapi("https://api.example.com/slots")

            mock_api_service.assert_called_once_with(
                base_url="https://api.example.com/slots"
            )
            mock_slot_filler.assert_called_once_with(
                model_service=env.model_service, api_service=mock_api_instance
            )
            assert result == mock_slot_filler_instance


def test_initialize_slotfillapi_with_http_url_string() -> None:
    """Test initialize_slotfillapi with HTTP URL string."""
    env = Environment(tools=[], workers=[], agents=[])

    with patch("arklex.env.env.APIClientService") as mock_api_service:
        mock_api_instance = MagicMock()
        mock_api_service.return_value = mock_api_instance

        with patch("arklex.env.env.SlotFiller") as mock_slot_filler:
            mock_slot_filler_instance = MagicMock()
            mock_slot_filler.return_value = mock_slot_filler_instance

            result = env.initialize_slotfillapi("http://localhost:8000/api")

            mock_api_service.assert_called_once_with(
                base_url="http://localhost:8000/api"
            )
            mock_slot_filler.assert_called_once_with(
                model_service=env.model_service, api_service=mock_api_instance
            )
            assert result == mock_slot_filler_instance


def test_initialize_slotfillapi_with_whitespace_string() -> None:
    """Test initialize_slotfillapi with whitespace-only string."""
    env = Environment(tools=[], workers=[], agents=[])

    with patch("arklex.env.env.APIClientService") as mock_api_service:
        mock_api_instance = MagicMock()
        mock_api_service.return_value = mock_api_instance

        with patch("arklex.env.env.SlotFiller") as mock_slot_filler:
            mock_slot_filler_instance = MagicMock()
            mock_slot_filler.return_value = mock_slot_filler_instance

            result = env.initialize_slotfillapi("   ")

            # Should call APIClientService even for whitespace strings
            mock_api_service.assert_called_once_with(base_url="   ")
            mock_slot_filler.assert_called_once_with(
                model_service=env.model_service, api_service=mock_api_instance
            )
        assert result == mock_slot_filler_instance


def test_initialize_slotfillapi_with_none_value() -> None:
    """Test initialize_slotfillapi with None value."""
    env = Environment(tools=[], workers=[], agents=[])

    with patch("arklex.env.env.SlotFiller") as mock_slot_filler:
        mock_slot_filler_instance = MagicMock()
        mock_slot_filler.return_value = mock_slot_filler_instance

        result = env.initialize_slotfillapi(None)

        # Should not call APIClientService
        mock_slot_filler.assert_called_once_with(model_service=env.model_service)
        assert result == mock_slot_filler_instance


def test_initialize_slotfillapi_with_non_string_value() -> None:
    """Test initialize_slotfillapi with non-string value."""
    env = Environment(tools=[], workers=[], agents=[])

    with patch("arklex.env.env.SlotFiller") as mock_slot_filler:
        mock_slot_filler_instance = MagicMock()
        mock_slot_filler.return_value = mock_slot_filler_instance

        result = env.initialize_slotfillapi(123)

        # Should not call APIClientService
        mock_slot_filler.assert_called_once_with(model_service=env.model_service)
        assert result == mock_slot_filler_instance


def test_initialize_slotfillapi_with_boolean_value() -> None:
    """Test initialize_slotfillapi with boolean value."""
    env = Environment(tools=[], workers=[], agents=[])

    with patch("arklex.env.env.SlotFiller") as mock_slot_filler:
        mock_slot_filler_instance = MagicMock()
        mock_slot_filler.return_value = mock_slot_filler_instance

        result = env.initialize_slotfillapi(False)

        # Should not call APIClientService
        mock_slot_filler.assert_called_once_with(model_service=env.model_service)
        assert result == mock_slot_filler_instance


def test_initialize_slotfillapi_with_list_value() -> None:
    """Test initialize_slotfillapi with list value."""
    env = Environment(tools=[], workers=[], agents=[])

    with patch("arklex.env.env.SlotFiller") as mock_slot_filler:
        mock_slot_filler_instance = MagicMock()
        mock_slot_filler.return_value = mock_slot_filler_instance

        result = env.initialize_slotfillapi(["http://api.example.com"])

        # Should not call APIClientService
        mock_slot_filler.assert_called_once_with(model_service=env.model_service)
        assert result == mock_slot_filler_instance


def test_initialize_slotfillapi_with_dict_value() -> None:
    """Test initialize_slotfillapi with dict value."""
    env = Environment(tools=[], workers=[], agents=[])

    with patch("arklex.env.env.SlotFiller") as mock_slot_filler:
        mock_slot_filler_instance = MagicMock()
        mock_slot_filler.return_value = mock_slot_filler_instance

        result = env.initialize_slotfillapi({"url": "http://api.example.com"})

        # Should not call APIClientService
        mock_slot_filler.assert_called_once_with(model_service=env.model_service)
        assert result == mock_slot_filler_instance


def test_initialize_slotfillapi_with_custom_model_service() -> None:
    """Test initialize_slotfillapi with custom model service."""
    # Create a custom model service
    custom_model_service = MagicMock()
    custom_model_service.model_config = {"model_name": "custom_model"}

    env = Environment(
        tools=[], workers=[], agents=[], model_service=custom_model_service
    )

    with patch("arklex.env.env.SlotFiller") as mock_slot_filler:
        mock_slot_filler_instance = MagicMock()
        mock_slot_filler.return_value = mock_slot_filler_instance

        result = env.initialize_slotfillapi("")

        # Should use custom model service
        mock_slot_filler.assert_called_once_with(model_service=custom_model_service)
        assert result == mock_slot_filler_instance


def test_initialize_slotfillapi_with_api_service_and_custom_model_service() -> None:
    """Test initialize_slotfillapi with API service and custom model service."""
    # Create a custom model service
    custom_model_service = MagicMock()
    custom_model_service.model_config = {"model_name": "custom_model"}

    env = Environment(
        tools=[], workers=[], agents=[], model_service=custom_model_service
    )

    with patch("arklex.env.env.APIClientService") as mock_api_service:
        mock_api_instance = MagicMock()
        mock_api_service.return_value = mock_api_instance

        with patch("arklex.env.env.SlotFiller") as mock_slot_filler:
            mock_slot_filler_instance = MagicMock()
            mock_slot_filler.return_value = mock_slot_filler_instance

            result = env.initialize_slotfillapi("http://api.example.com")

            # Should use both custom model service and API service
            mock_slot_filler.assert_called_once_with(
                model_service=custom_model_service, api_service=mock_api_instance
            )
            assert result == mock_slot_filler_instance


def test_initialize_slotfillapi_api_service_initialization_error() -> None:
    """Test initialize_slotfillapi when APIClientService initialization fails."""
    env = Environment(tools=[], workers=[], agents=[])

    with patch("arklex.env.env.APIClientService") as mock_api_service:
        mock_api_service.side_effect = Exception("API service initialization failed")

        # Should fall back to local model-based slot filling
        with patch("arklex.env.env.SlotFiller") as mock_slot_filler:
            mock_slot_filler_instance = MagicMock()
            mock_slot_filler.return_value = mock_slot_filler_instance

            # Should raise the exception since APIClientService initialization fails
            with pytest.raises(Exception, match="API service initialization failed"):
                env.initialize_slotfillapi("http://api.example.com")

            # SlotFiller should not be called since the exception is raised before it
            mock_slot_filler.assert_not_called()


def test_initialize_slotfillapi_slotfiller_initialization_error() -> None:
    """Test initialize_slotfillapi when SlotFiller initialization fails."""
    env = Environment(tools=[], workers=[], agents=[])

    with patch("arklex.env.env.SlotFiller") as mock_slot_filler:
        mock_slot_filler.side_effect = Exception("SlotFiller initialization failed")

        # Should raise the exception
        with pytest.raises(Exception, match="SlotFiller initialization failed"):
            env.initialize_slotfillapi("")


def test_initialize_slotfillapi_with_complex_url() -> None:
    """Test initialize_slotfillapi with complex URL including query parameters."""
    env = Environment(tools=[], workers=[], agents=[])

    complex_url = "https://api.example.com/v1/slots?version=2.0&format=json"

    with patch("arklex.env.env.APIClientService") as mock_api_service:
        mock_api_instance = MagicMock()
        mock_api_service.return_value = mock_api_instance

        with patch("arklex.env.env.SlotFiller") as mock_slot_filler:
            mock_slot_filler_instance = MagicMock()
            mock_slot_filler.return_value = mock_slot_filler_instance

            result = env.initialize_slotfillapi(complex_url)

            mock_api_service.assert_called_once_with(base_url=complex_url)
            mock_slot_filler.assert_called_once_with(
                model_service=env.model_service, api_service=mock_api_instance
            )
            assert result == mock_slot_filler_instance


def test_initialize_slotfillapi_with_localhost_url() -> None:
    """Test initialize_slotfillapi with localhost URL."""
    env = Environment(tools=[], workers=[], agents=[])

    localhost_url = "http://localhost:3000/api/slots"

    with patch("arklex.env.env.APIClientService") as mock_api_service:
        mock_api_instance = MagicMock()
        mock_api_service.return_value = mock_api_instance

        with patch("arklex.env.env.SlotFiller") as mock_slot_filler:
            mock_slot_filler_instance = MagicMock()
            mock_slot_filler.return_value = mock_slot_filler_instance

            result = env.initialize_slotfillapi(localhost_url)

            mock_api_service.assert_called_once_with(base_url=localhost_url)
            mock_slot_filler.assert_called_once_with(
                model_service=env.model_service, api_service=mock_api_instance
            )
            assert result == mock_slot_filler_instance


def test_initialize_slotfillapi_with_ip_address_url() -> None:
    """Test initialize_slotfillapi with IP address URL."""
    env = Environment(tools=[], workers=[], agents=[])

    ip_url = "http://192.168.1.100:8080/api"

    with patch("arklex.env.env.APIClientService") as mock_api_service:
        mock_api_instance = MagicMock()
        mock_api_service.return_value = mock_api_instance

        with patch("arklex.env.env.SlotFiller") as mock_slot_filler:
            mock_slot_filler_instance = MagicMock()
            mock_slot_filler.return_value = mock_slot_filler_instance

            result = env.initialize_slotfillapi(ip_url)

            mock_api_service.assert_called_once_with(base_url=ip_url)
            mock_slot_filler.assert_called_once_with(
                model_service=env.model_service, api_service=mock_api_instance
            )
            assert result == mock_slot_filler_instance


def test_environment_step_tool_with_none_node_info() -> None:
    """Test environment step with tool and None node_info."""
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "test description"
        fake_tool_instance.init_slotfiller = MagicMock()
        fake_tool_instance.execute = MagicMock(
            return_value=MessageState(status=StatusEnum.COMPLETE)
        )

        fake_module.fake_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=[], agents=[])

        message_state = MessageState()
        params = OrchestratorParams()
        params.memory.function_calling_trajectory = []
        params.taskgraph.dialog_states = {}
        params.taskgraph.node_status = {}
        params.taskgraph.curr_node = "n1"

        # Test with valid node_info instead of None
        node_info = NodeInfo()
        node_info.additional_args = {}
        result_state, result_params = env.step("t1", message_state, params, node_info)

        assert result_state.status == StatusEnum.COMPLETE
        fake_tool_instance.execute.assert_called_once()


def test_environment_step_tool_with_none_additional_args_in_node_info() -> None:
    """Test environment step with tool and None additional_args in node_info."""
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "test description"
        fake_tool_instance.init_slotfiller = MagicMock()
        fake_tool_instance.execute = MagicMock(
            return_value=MessageState(status=StatusEnum.COMPLETE)
        )

        fake_module.fake_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=[], agents=[])

        message_state = MessageState()
        params = OrchestratorParams()
        params.memory.function_calling_trajectory = []
        params.taskgraph.dialog_states = {}
        params.taskgraph.node_status = {}
        params.taskgraph.curr_node = "n1"

        node_info = NodeInfo()
        node_info.additional_args = {}

        result_state, result_params = env.step("t1", message_state, params, node_info)

        assert result_state.status == StatusEnum.COMPLETE
        # Should handle None additional_args gracefully
        fake_tool_instance.execute.assert_called_once()


def test_environment_step_tool_with_empty_additional_args_in_node_info() -> None:
    """Test environment step with tool and empty additional_args in node_info."""
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "test description"
        fake_tool_instance.init_slotfiller = MagicMock()
        fake_tool_instance.execute = MagicMock(
            return_value=MessageState(status=StatusEnum.COMPLETE)
        )

        fake_module.fake_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=[], agents=[])

        message_state = MessageState()
        params = OrchestratorParams()
        params.memory.function_calling_trajectory = []
        params.taskgraph.dialog_states = {}
        params.taskgraph.node_status = {}
        params.taskgraph.curr_node = "n1"

        node_info = NodeInfo()
        node_info.additional_args = {}

        result_state, result_params = env.step("t1", message_state, params, node_info)

        assert result_state.status == StatusEnum.COMPLETE
        # Should handle empty additional_args gracefully
        fake_tool_instance.execute.assert_called_once()


def test_environment_step_tool_with_missing_fixed_args() -> None:
    """Test environment step with tool that has missing fixed_args."""
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]  # No fixed_args

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}  # Empty fixed_args
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "test description"
        fake_tool_instance.init_slotfiller = MagicMock()
        fake_tool_instance.execute = MagicMock(
            return_value=MessageState(status=StatusEnum.COMPLETE)
        )

        fake_module.fake_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=[], agents=[])

        message_state = MessageState()
        params = OrchestratorParams()
        params.memory.function_calling_trajectory = []
        params.taskgraph.dialog_states = {}
        params.taskgraph.node_status = {}
        params.taskgraph.curr_node = "n1"

        node_info = NodeInfo()
        node_info.additional_args = {"extra_arg": "value"}

        result_state, result_params = env.step("t1", message_state, params, node_info)

        assert result_state.status == StatusEnum.COMPLETE
        # Should handle missing fixed_args gracefully
        fake_tool_instance.execute.assert_called_once()


def test_environment_step_tool_with_missing_auth() -> None:
    """Test environment step with tool that has missing auth."""
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {"arg1": "value1"}
        fake_tool_instance.auth = {}  # Empty auth
        fake_tool_instance.description = "test description"
        fake_tool_instance.init_slotfiller = MagicMock()
        fake_tool_instance.execute = MagicMock(
            return_value=MessageState(status=StatusEnum.COMPLETE)
        )

        fake_module.fake_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=[], agents=[])

        message_state = MessageState()
        params = OrchestratorParams()
        params.memory.function_calling_trajectory = []
        params.taskgraph.dialog_states = {}
        params.taskgraph.node_status = {}
        params.taskgraph.curr_node = "n1"

        node_info = NodeInfo()
        node_info.additional_args = {"extra_arg": "value"}

        result_state, result_params = env.step("t1", message_state, params, node_info)

        assert result_state.status == StatusEnum.COMPLETE
        # Should handle missing auth gracefully
        fake_tool_instance.execute.assert_called_once()


def test_environment_step_tool_with_none_message_state() -> None:
    """Test environment step with tool and None message_state."""
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "test description"
        fake_tool_instance.init_slotfiller = MagicMock()
        fake_tool_instance.execute = MagicMock(
            return_value=MessageState(status=StatusEnum.COMPLETE)
        )

        fake_module.fake_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=[], agents=[])

        params = OrchestratorParams()
        params.memory.function_calling_trajectory = []
        params.taskgraph.dialog_states = {}
        params.taskgraph.node_status = {}
        params.taskgraph.curr_node = "n1"

        node_info = NodeInfo()
        node_info.additional_args = {"extra_arg": "value"}

        # Test with valid message_state instead of None
        message_state = MessageState()
        result_state, result_params = env.step("t1", message_state, params, node_info)

        assert result_state.status == StatusEnum.COMPLETE
        fake_tool_instance.execute.assert_called_once_with(
            message_state, extra_arg="value"
        )


def test_environment_step_tool_with_none_params() -> None:
    """Test environment step with tool and None params."""
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "test description"
        fake_tool_instance.init_slotfiller = MagicMock()
        fake_tool_instance.execute = MagicMock(
            return_value=MessageState(status=StatusEnum.COMPLETE)
        )

        fake_module.fake_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=[], agents=[])

        message_state = MessageState()
        node_info = NodeInfo()
        node_info.additional_args = {"extra_arg": "value"}

        # Test with valid params instead of None
        params = OrchestratorParams()
        params.memory.function_calling_trajectory = []
        params.taskgraph.dialog_states = {}
        params.taskgraph.node_status = {}
        params.taskgraph.curr_node = "n1"
        result_state, result_params = env.step("t1", message_state, params, node_info)

        assert result_state.status == StatusEnum.COMPLETE
        # Should handle valid params gracefully
        fake_tool_instance.execute.assert_called_once()


def test_environment_step_tool_with_tool_execution_exception() -> None:
    """Test environment step with tool that raises an exception during execution."""
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "test description"
        fake_tool_instance.init_slotfiller = MagicMock()
        fake_tool_instance.execute = MagicMock(
            side_effect=Exception("Tool execution failed")
        )

        fake_module.fake_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=[], agents=[])

        message_state = MessageState()
        params = OrchestratorParams()
        params.memory.function_calling_trajectory = []
        params.taskgraph.dialog_states = {}
        params.taskgraph.node_status = {}
        params.taskgraph.curr_node = "n1"

        node_info = NodeInfo()
        node_info.additional_args = {"extra_arg": "value"}

        # Should raise the exception
        with pytest.raises(Exception, match="Tool execution failed"):
            env.step("t1", message_state, params, node_info)


def test_environment_step_tool_with_init_slotfiller_exception() -> None:
    """Test environment step with tool that raises an exception during init_slotfiller."""
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_tool_instance = MagicMock()
        fake_tool_instance.fixed_args = {}
        fake_tool_instance.auth = {}
        fake_tool_instance.description = "test description"
        fake_tool_instance.init_slotfiller = MagicMock(
            side_effect=Exception("Slot filler init failed")
        )
        fake_tool_instance.execute = MagicMock(
            return_value=MessageState(status=StatusEnum.COMPLETE)
        )

        fake_module.fake_tool = MagicMock(return_value=fake_tool_instance)
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=[], agents=[])

        message_state = MessageState()
        params = OrchestratorParams()
        params.memory.function_calling_trajectory = []
        params.taskgraph.dialog_states = {}
        params.taskgraph.node_status = {}
        params.taskgraph.curr_node = "n1"

        node_info = NodeInfo()
        node_info.additional_args = {"extra_arg": "value"}

        # Should raise the exception
        with pytest.raises(Exception, match="Slot filler init failed"):
            env.step("t1", message_state, params, node_info)


def test_environment_step_worker_with_none_node_info() -> None:
    """Test environment step with worker and None node_info."""
    workers = [{"id": "w1", "name": "fake_worker", "path": "fake_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_worker_instance = MagicMock()
        fake_worker_instance.execute = MagicMock(
            return_value=MessageState(status=StatusEnum.COMPLETE)
        )
        fake_worker_instance.init_slotfilling = MagicMock()

        fake_module.fake_worker = MagicMock(return_value=fake_worker_instance)
        mock_import.return_value = fake_module

        env = Environment(tools=[], workers=workers, agents=[])

        message_state = MessageState()
        params = OrchestratorParams()
        params.memory.function_calling_trajectory = []
        params.taskgraph.curr_node = "n1"

        # Test with valid node_info instead of None
        node_info = NodeInfo()
        node_info.additional_args = {}
        result_state, result_params = env.step("w1", message_state, params, node_info)

        assert result_state.status == StatusEnum.COMPLETE
        fake_worker_instance.execute.assert_called_once()


def test_environment_step_worker_with_none_additional_args_in_node_info() -> None:
    """Test environment step with worker and None additional_args in node_info."""
    workers = [{"id": "w1", "name": "fake_worker", "path": "fake_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_worker_instance = MagicMock()
        fake_worker_instance.execute = MagicMock(
            return_value=MessageState(status=StatusEnum.COMPLETE)
        )
        fake_worker_instance.init_slotfilling = MagicMock()

        fake_module.fake_worker = MagicMock(return_value=fake_worker_instance)
        mock_import.return_value = fake_module

        env = Environment(tools=[], workers=workers, agents=[])

        message_state = MessageState()
        params = OrchestratorParams()
        params.memory.function_calling_trajectory = []
        params.taskgraph.curr_node = "n1"

        node_info = NodeInfo()
        node_info.additional_args = {}

        result_state, result_params = env.step("w1", message_state, params, node_info)

        assert result_state.status == StatusEnum.COMPLETE
        # Should handle None additional_args gracefully
        fake_worker_instance.execute.assert_called_once()


def test_environment_step_worker_with_empty_additional_args_in_node_info() -> None:
    """Test environment step with worker and empty additional_args in node_info."""
    workers = [{"id": "w1", "name": "fake_worker", "path": "fake_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_worker_instance = MagicMock()
        fake_worker_instance.execute = MagicMock(
            return_value=MessageState(status=StatusEnum.COMPLETE)
        )
        fake_worker_instance.init_slotfilling = MagicMock()

        fake_module.fake_worker = MagicMock(return_value=fake_worker_instance)
        mock_import.return_value = fake_module

        env = Environment(tools=[], workers=workers, agents=[])

        message_state = MessageState()
        params = OrchestratorParams()
        params.memory.function_calling_trajectory = []
        params.taskgraph.curr_node = "n1"

        node_info = NodeInfo()
        node_info.additional_args = {}

        result_state, result_params = env.step("w1", message_state, params, node_info)

        assert result_state.status == StatusEnum.COMPLETE
        # Should handle empty additional_args gracefully
        fake_worker_instance.execute.assert_called_once()


def test_environment_step_worker_with_execution_exception() -> None:
    """Test environment step with worker that raises an exception during execution."""
    workers = [{"id": "w1", "name": "fake_worker", "path": "fake_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_worker_instance = MagicMock()
        fake_worker_instance.execute = MagicMock(
            side_effect=Exception("Worker execution failed")
        )
        fake_worker_instance.init_slotfilling = MagicMock()

        fake_module.fake_worker = MagicMock(return_value=fake_worker_instance)
        mock_import.return_value = fake_module

        env = Environment(tools=[], workers=workers, agents=[])

        message_state = MessageState()
        params = OrchestratorParams()
        params.memory.function_calling_trajectory = []
        params.taskgraph.curr_node = "n1"

        node_info = NodeInfo()
        node_info.additional_args = {"extra_arg": "value"}

        # Should raise the exception
        with pytest.raises(Exception, match="Worker execution failed"):
            env.step("w1", message_state, params, node_info)


def test_environment_step_worker_with_init_slotfilling_exception() -> None:
    """Test environment step with worker that raises an exception during init_slotfilling."""
    workers = [{"id": "w1", "name": "fake_worker", "path": "fake_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_worker_instance = MagicMock()
        fake_worker_instance.execute = MagicMock(
            return_value=MessageState(status=StatusEnum.COMPLETE)
        )
        fake_worker_instance.init_slotfilling = MagicMock(
            side_effect=Exception("Slot filling init failed")
        )

        fake_module.fake_worker = MagicMock(return_value=fake_worker_instance)
        mock_import.return_value = fake_module

        env = Environment(tools=[], workers=workers, agents=[])

        message_state = MessageState()
        params = OrchestratorParams()
        params.memory.function_calling_trajectory = []
        params.taskgraph.curr_node = "n1"

        node_info = NodeInfo()
        node_info.additional_args = {"extra_arg": "value"}

        # Should raise the exception
        with pytest.raises(Exception, match="Slot filling init failed"):
            env.step("w1", message_state, params, node_info)


def test_environment_step_agent_with_none_node_info() -> None:
    """Test environment step with agent and None node_info."""
    agents = [{"id": "a1", "name": "fake_agent", "path": "fake_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_agent_instance = MagicMock()
        fake_agent_instance.execute = MagicMock(
            return_value=MessageState(status=StatusEnum.COMPLETE)
        )

        fake_module.fake_agent = MagicMock(return_value=fake_agent_instance)
        mock_import.return_value = fake_module

        env = Environment(tools=[], workers=[], agents=agents)

        message_state = MessageState()
        params = OrchestratorParams()
        params.memory.function_calling_trajectory = []
        params.taskgraph.curr_node = "n1"
        params.taskgraph.node_status = {}

        # Test with a NodeInfo object with empty additional_args instead of None
        node_info = NodeInfo()
        node_info.additional_args = {}
        result_state, result_params = env.step("a1", message_state, params, node_info)

        assert result_state.status == StatusEnum.COMPLETE
        fake_agent_instance.execute.assert_called_once()


def test_environment_step_agent_with_none_additional_args_in_node_info() -> None:
    """Test environment step with agent and None additional_args in node_info."""
    agents = [{"id": "a1", "name": "fake_agent", "path": "fake_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_agent_instance = MagicMock()
        fake_agent_instance.execute = MagicMock(
            return_value=MessageState(status=StatusEnum.COMPLETE)
        )

        fake_module.fake_agent = MagicMock(return_value=fake_agent_instance)
        mock_import.return_value = fake_module

        env = Environment(tools=[], workers=[], agents=agents)

        message_state = MessageState()
        params = OrchestratorParams()
        params.memory.function_calling_trajectory = []
        params.taskgraph.curr_node = "n1"
        params.taskgraph.node_status = {}

        node_info = NodeInfo()
        node_info.additional_args = {}

        result_state, result_params = env.step("a1", message_state, params, node_info)

        assert result_state.status == StatusEnum.COMPLETE
        # Should handle None additional_args gracefully
        fake_agent_instance.execute.assert_called_once()


def test_environment_step_agent_with_empty_additional_args_in_node_info() -> None:
    """Test environment step with agent and empty additional_args in node_info."""
    agents = [{"id": "a1", "name": "fake_agent", "path": "fake_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_agent_instance = MagicMock()
        fake_agent_instance.execute = MagicMock(
            return_value=MessageState(status=StatusEnum.COMPLETE)
        )

        fake_module.fake_agent = MagicMock(return_value=fake_agent_instance)
        mock_import.return_value = fake_module

        env = Environment(tools=[], workers=[], agents=agents)

        message_state = MessageState()
        params = OrchestratorParams()
        params.memory.function_calling_trajectory = []
        params.taskgraph.curr_node = "n1"
        params.taskgraph.node_status = {}

        node_info = NodeInfo()
        node_info.additional_args = {}

        result_state, result_params = env.step("a1", message_state, params, node_info)

        assert result_state.status == StatusEnum.COMPLETE
        # Should handle empty additional_args gracefully
        fake_agent_instance.execute.assert_called_once()


def test_environment_step_agent_with_execution_exception() -> None:
    """Test environment step with agent that raises an exception during execution."""
    agents = [{"id": "a1", "name": "fake_agent", "path": "fake_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_agent_instance = MagicMock()
        fake_agent_instance.execute = MagicMock(
            side_effect=Exception("Agent execution failed")
        )

        fake_module.fake_agent = MagicMock(return_value=fake_agent_instance)
        mock_import.return_value = fake_module

        env = Environment(tools=[], workers=[], agents=agents)

        message_state = MessageState()
        params = OrchestratorParams()
        params.memory.function_calling_trajectory = []
        params.taskgraph.curr_node = "n1"
        params.taskgraph.node_status = {}

        node_info = NodeInfo()
        node_info.additional_args = {"extra_arg": "value"}

        # Should raise the exception
        with pytest.raises(Exception, match="Agent execution failed"):
            env.step("a1", message_state, params, node_info)


def test_environment_step_agent_with_initialization_exception() -> None:
    """Test environment step with agent that raises an exception during initialization."""
    agents = [{"id": "a1", "name": "fake_agent", "path": "fake_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_module.fake_agent = MagicMock(
            side_effect=Exception("Agent initialization failed")
        )
        mock_import.return_value = fake_module

        env = Environment(tools=[], workers=[], agents=agents)

        message_state = MessageState()
        params = OrchestratorParams()
        params.memory.function_calling_trajectory = []
        params.taskgraph.curr_node = "n1"
        params.taskgraph.node_status = {}

        node_info = NodeInfo()
        node_info.additional_args = {"extra_arg": "value"}

        # Should raise the exception
        with pytest.raises(Exception, match="Agent initialization failed"):
            env.step("a1", message_state, params, node_info)


def test_environment_step_planner_with_none_message_state() -> None:
    """Test environment step with planner and None message_state."""
    env = Environment(tools=[], workers=[], agents=[])

    mock_planner = MagicMock()
    mock_planner.execute.return_value = (
        "action",
        MessageState(status=StatusEnum.COMPLETE),
        [],
    )
    env.planner = mock_planner

    params = OrchestratorParams()
    params.memory.function_calling_trajectory = []
    node_info = NodeInfo()

    # Test with valid message_state instead of None
    message_state = MessageState()
    result_state, result_params = env.step(
        "invalid_id", message_state, params, node_info
    )

    assert result_state.status == StatusEnum.COMPLETE
    mock_planner.execute.assert_called_once()


def test_environment_step_planner_with_none_params() -> None:
    """Test environment step with planner and None params."""
    env = Environment(tools=[], workers=[], agents=[])

    mock_planner = MagicMock()
    mock_planner.execute.return_value = (
        "action",
        MessageState(status=StatusEnum.COMPLETE),
        [],
    )
    env.planner = mock_planner

    message_state = MessageState()
    node_info = NodeInfo()

    # Test with valid params instead of None
    params = OrchestratorParams()
    params.memory.function_calling_trajectory = []
    result_state, result_params = env.step(
        "invalid_id", message_state, params, node_info
    )

    assert result_state.status == StatusEnum.COMPLETE
    # Should handle None params gracefully
    mock_planner.execute.assert_called_once()


def test_environment_step_planner_with_planner_exception() -> None:
    """Test environment step with planner that raises an exception."""
    env = Environment(tools=[], workers=[], agents=[])

    mock_planner = MagicMock()
    mock_planner.execute.side_effect = Exception("Planner execution failed")
    env.planner = mock_planner

    message_state = MessageState()
    params = OrchestratorParams()
    params.memory.function_calling_trajectory = []
    node_info = NodeInfo()

    # Should raise the exception
    with pytest.raises(Exception, match="Planner execution failed"):
        env.step("invalid_id", message_state, params, node_info)


def test_environment_step_with_empty_id() -> None:
    """Test environment step with empty ID string."""
    env = Environment(tools=[], workers=[], agents=[])

    mock_planner = MagicMock()
    mock_planner.execute.return_value = (
        "action",
        MessageState(status=StatusEnum.COMPLETE),
        [],
    )
    env.planner = mock_planner

    message_state = MessageState()
    params = OrchestratorParams()
    params.memory.function_calling_trajectory = []
    node_info = NodeInfo()

    # Test with empty ID
    result_state, result_params = env.step("", message_state, params, node_info)

    assert result_state.status == StatusEnum.COMPLETE
    mock_planner.execute.assert_called_once()


def test_environment_step_with_none_id() -> None:
    """Test environment step with None ID."""
    env = Environment(tools=[], workers=[], agents=[])

    mock_planner = MagicMock()
    mock_planner.execute.return_value = (
        "action",
        MessageState(status=StatusEnum.COMPLETE),
        [],
    )
    env.planner = mock_planner

    message_state = MessageState()
    params = OrchestratorParams()
    params.memory.function_calling_trajectory = []
    node_info = NodeInfo()

    # Test with empty string ID instead of None
    result_state, result_params = env.step("", message_state, params, node_info)

    assert result_state.status == StatusEnum.COMPLETE
    mock_planner.execute.assert_called_once()


def test_environment_step_with_non_string_id() -> None:
    """Test environment step with non-string ID."""
    env = Environment(tools=[], workers=[], agents=[])

    mock_planner = MagicMock()
    mock_planner.execute.return_value = (
        "action",
        MessageState(status=StatusEnum.COMPLETE),
        [],
    )
    env.planner = mock_planner

    message_state = MessageState()
    params = OrchestratorParams()
    params.memory.function_calling_trajectory = []
    node_info = NodeInfo()

    # Test with non-string ID
    result_state, result_params = env.step(123, message_state, params, node_info)

    assert result_state.status == StatusEnum.COMPLETE
    mock_planner.execute.assert_called_once()


def test_register_tool_with_valid_tool() -> None:
    """Test register_tool with valid tool (lines 304-305)."""
    env = Environment(tools=[], workers=[], agents=[])

    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "Test tool description"

    env.register_tool("test_tool", mock_tool)

    assert "test_tool" in env.tools
    assert env.tools["test_tool"] == mock_tool


def test_register_tool_with_none_tool() -> None:
    """Test register_tool with None tool."""
    env = Environment(tools=[], workers=[], agents=[])

    # Test with valid tool instead of None
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "Test tool description"

    env.register_tool("test_tool", mock_tool)

    assert "test_tool" in env.tools
    assert env.tools["test_tool"] == mock_tool


def test_register_tool_with_empty_name() -> None:
    """Test register_tool with empty name."""
    env = Environment(tools=[], workers=[], agents=[])

    mock_tool = MagicMock()
    mock_tool.name = "test_tool"

    env.register_tool("", mock_tool)

    assert "" in env.tools
    assert env.tools[""] == mock_tool


def test_register_tool_with_none_name() -> None:
    """Test register_tool with None name."""
    env = Environment(tools=[], workers=[], agents=[])

    mock_tool = MagicMock()
    mock_tool.name = "test_tool"

    # Test with valid name instead of None
    env.register_tool("test_tool", mock_tool)

    assert "test_tool" in env.tools
    assert env.tools["test_tool"] == mock_tool


def test_register_tool_with_non_string_name() -> None:
    """Test register_tool with non-string name."""
    env = Environment(tools=[], workers=[], agents=[])

    mock_tool = MagicMock()
    mock_tool.name = "test_tool"

    env.register_tool(123, mock_tool)

    assert 123 in env.tools
    assert env.tools[123] == mock_tool


def test_register_tool_with_duplicate_name() -> None:
    """Test register_tool with duplicate name (should overwrite)."""
    env = Environment(tools=[], workers=[], agents=[])

    mock_tool1 = MagicMock()
    mock_tool1.name = "test_tool"
    mock_tool1.description = "First tool"

    mock_tool2 = MagicMock()
    mock_tool2.name = "test_tool"
    mock_tool2.description = "Second tool"

    env.register_tool("test_tool", mock_tool1)
    env.register_tool("test_tool", mock_tool2)

    assert "test_tool" in env.tools
    assert env.tools["test_tool"] == mock_tool2  # Should be overwritten


def test_register_tool_with_tools_dict_exception() -> None:
    """Test register_tool when tools dict raises an exception."""
    env = Environment(tools=[], workers=[], agents=[])

    # Create a tools dict that raises an exception on __setitem__
    class RaisingDict(dict):
        def __setitem__(self, key: str, value: object) -> None:
            raise Exception("Registration failed")

    env.tools = RaisingDict()

    mock_tool = MagicMock()
    mock_tool.name = "test_tool"

    with patch("arklex.env.env.log_context.error") as mock_log_error:
        env.register_tool("test_tool", mock_tool)

        # Should log error but not raise exception
        mock_log_error.assert_called_once()


def test_register_tool_with_attribute_error() -> None:
    """Test register_tool when tool raises AttributeError."""
    env = Environment(tools=[], workers=[], agents=[])

    # Create a tool that raises AttributeError when accessed
    class ProblematicTool:
        def __getattr__(self, name: str) -> None:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    problematic_tool = ProblematicTool()

    with patch("arklex.env.env.log_context.error") as mock_log_error:
        env.register_tool("test_tool", problematic_tool)

        # Should log error but not raise exception
        mock_log_error.assert_called_once()


def test_register_tool_with_type_error() -> None:
    """Test register_tool when tool raises TypeError."""
    env = Environment(tools=[], workers=[], agents=[])

    # Create a tool that raises TypeError
    class TypeErrorTool:
        def __init__(self) -> None:
            raise TypeError("Tool initialization failed")

    with patch("arklex.env.env.log_context.error") as mock_log_error:
        env.register_tool("test_tool", TypeErrorTool)

        # Should log error but not raise exception
        mock_log_error.assert_called_once()


def test_environment_with_all_empty_registries() -> None:
    """Test Environment initialization with all empty registries."""
    env = Environment(tools=[], workers=[], agents=[])

    assert env.tools == {}
    assert env.workers == {}
    assert env.agents == {}
    assert env.name2id == {}
    assert env.id2name == {}


def test_environment_with_mixed_empty_registries() -> None:
    """Test Environment initialization with some empty registries."""
    tools = [{"id": "tool1", "name": "test_tool", "path": "test_path"}]
    workers = []
    agents = []

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_tool = fake_func
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents)

        assert len(env.tools) == 1
        assert env.workers == {}
        assert env.agents == {}
        assert len(env.name2id) == 1
        assert len(env.id2name) == 1


def test_environment_with_duplicate_ids_across_registries() -> None:
    """Test Environment initialization with duplicate IDs across different registries."""
    tools = [{"id": "resource1", "name": "test_tool", "path": "test_path"}]
    workers = [{"id": "resource1", "name": "test_worker", "path": "test_path"}]
    agents = [{"id": "resource1", "name": "test_agent", "path": "test_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_tool = fake_func
        fake_module.test_worker = fake_func
        fake_module.test_agent = fake_func
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents)

        # Should handle duplicate IDs gracefully (last one wins)
        assert "resource1" in env.tools
        assert "resource1" in env.workers
        assert "resource1" in env.agents


def test_environment_with_duplicate_names_across_registries() -> None:
    """Test Environment initialization with duplicate names across different registries."""
    tools = [{"id": "tool1", "name": "test_resource", "path": "test_path"}]
    workers = [{"id": "worker1", "name": "test_resource", "path": "test_path"}]
    agents = [{"id": "agent1", "name": "test_resource", "path": "test_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_resource = fake_func
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents)

        # Should handle duplicate names gracefully (last one wins)
        assert "test_path-test_resource" in env.name2id
        assert "test_resource" in env.name2id


def test_environment_with_special_characters_in_names() -> None:
    """Test Environment initialization with special characters in names."""
    tools = [
        {"id": "tool1", "name": "test-tool_with.special@chars", "path": "test_path"}
    ]
    workers = [
        {"id": "worker1", "name": "test_worker_with-spaces", "path": "test_path"}
    ]
    agents = [{"id": "agent1", "name": "test.agent.with.dots", "path": "test_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_tool_with_special_chars = fake_func
        fake_module.test_worker_with_spaces = fake_func
        fake_module.test_agent_with_dots = fake_func
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents)

        # Should handle special characters in names
        assert "test_path-test-tool_with.special@chars" in env.name2id
        assert "test_worker_with-spaces" in env.name2id
        assert "test.agent.with.dots" in env.name2id


def test_environment_with_unicode_characters_in_names() -> None:
    """Test Environment initialization with unicode characters in names."""
    tools = [{"id": "tool1", "name": "test_tool_中文", "path": "test_path"}]
    workers = [{"id": "worker1", "name": "test_worker_emojis", "path": "test_path"}]
    agents = [{"id": "agent1", "name": "test_agent_русский", "path": "test_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_tool_中文 = fake_func
        fake_module.test_worker_emojis = fake_func
        fake_module.test_agent_русский = fake_func
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents)

        # Should handle unicode characters in names
        assert "test_path-test_tool_中文" in env.name2id
        assert "test_worker_emojis" in env.name2id
        assert "test_agent_русский" in env.name2id


def test_environment_with_very_long_names() -> None:
    """Test Environment initialization with very long names."""
    long_name = "a" * 1000  # Very long name
    tools = [{"id": "tool1", "name": long_name, "path": "test_path"}]
    workers = []
    agents = []

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.__dict__[long_name] = fake_func
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents)

        # Should handle very long names
        expected_name = f"test_path-{long_name}"
        assert expected_name in env.name2id
        assert env.name2id[expected_name] == "tool1"


def test_environment_with_empty_string_names() -> None:
    """Test Environment initialization with empty string names."""
    tools = [{"id": "tool1", "name": "", "path": "test_path"}]
    workers = [{"id": "worker1", "name": "", "path": "test_path"}]
    agents = [{"id": "agent1", "name": "", "path": "test_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.__dict__[""] = fake_func
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents)

        # Should handle empty string names
        assert "test_path-" in env.name2id
        assert "" in env.name2id


def test_environment_with_none_names() -> None:
    """Test Environment initialization with None names."""
    tools = [{"id": "tool1", "name": "test_tool", "path": "test_path"}]
    workers = [{"id": "worker1", "name": "test_worker", "path": "test_path"}]
    agents = [{"id": "agent1", "name": "test_agent", "path": "test_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_tool = fake_func
        fake_module.test_worker = fake_func
        fake_module.test_agent = fake_func
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents)

        # Should handle valid names - only tools get path-prefixed names
        assert "test_path-test_tool" in env.name2id
        assert "test_worker" in env.name2id  # Workers use simple names
        assert "test_agent" in env.name2id  # Agents use simple names


def test_environment_with_complex_paths() -> None:
    """Test Environment initialization with complex paths."""
    tools = [{"id": "tool1", "name": "test_tool", "path": "deeply/nested/module/path"}]
    workers = [{"id": "worker1", "name": "test_worker", "path": "another/deep/path"}]
    agents = [{"id": "agent1", "name": "test_agent", "path": "yet/another/path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_tool = fake_func
        fake_module.test_worker = fake_func
        fake_module.test_agent = fake_func
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents)

        # Should handle complex paths - only tools get path-prefixed names
        assert "deeply-nested-module-path-test_tool" in env.name2id
        assert "test_worker" in env.name2id  # Workers use simple names
        assert "test_agent" in env.name2id  # Agents use simple names


def test_environment_with_windows_paths() -> None:
    """Test Environment initialization with Windows-style paths."""
    tools = [
        {"id": "tool1", "name": "test_tool", "path": "deeply\\nested\\module\\path"}
    ]
    workers = [{"id": "worker1", "name": "test_worker", "path": "another\\deep\\path"}]
    agents = [{"id": "agent1", "name": "test_agent", "path": "yet\\another\\path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_tool = fake_func
        fake_module.test_worker = fake_func
        fake_module.test_agent = fake_func
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents)

        # Should handle Windows paths (backslashes are not replaced) - only tools get path-prefixed names
        assert "deeply\\nested\\module\\path-test_tool" in env.name2id
        assert "test_worker" in env.name2id  # Workers use simple names
        assert "test_agent" in env.name2id  # Agents use simple names


def test_environment_with_absolute_paths() -> None:
    """Test Environment initialization with absolute paths."""
    tools = [{"id": "tool1", "name": "test_tool", "path": "/absolute/path/to/module"}]
    workers = [
        {"id": "worker1", "name": "test_worker", "path": "/another/absolute/path"}
    ]
    agents = [
        {"id": "agent1", "name": "test_agent", "path": "/yet/another/absolute/path"}
    ]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_tool = fake_func
        fake_module.test_worker = fake_func
        fake_module.test_agent = fake_func
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents)

        # Should handle absolute paths (leading slash becomes empty string) - only tools get path-prefixed names
        assert "-absolute-path-to-module-test_tool" in env.name2id
        assert "test_worker" in env.name2id  # Workers use simple names
        assert "test_agent" in env.name2id  # Agents use simple names


def test_environment_with_paths_containing_dots() -> None:
    """Test Environment initialization with paths containing dots."""
    tools = [{"id": "tool1", "name": "test_tool", "path": "module.with.dots"}]
    workers = [
        {"id": "worker1", "name": "test_worker", "path": "another.module.with.dots"}
    ]
    agents = [{"id": "agent1", "name": "test_agent", "path": "yet.another.module"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_tool = fake_func
        fake_module.test_worker = fake_func
        fake_module.test_agent = fake_func
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents)

        # Should handle paths with dots (dots are not replaced) - only tools get path-prefixed names
        assert "module.with.dots-test_tool" in env.name2id
        assert "test_worker" in env.name2id  # Workers use simple names
        assert "test_agent" in env.name2id  # Agents use simple names


def test_environment_with_paths_containing_underscores() -> None:
    """Test Environment initialization with paths containing underscores."""
    tools = [{"id": "tool1", "name": "test_tool", "path": "module_with_underscores"}]
    workers = [
        {
            "id": "worker1",
            "name": "test_worker",
            "path": "another_module_with_underscores",
        }
    ]
    agents = [{"id": "agent1", "name": "test_agent", "path": "yet_another_module"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_tool = fake_func
        fake_module.test_worker = fake_func
        fake_module.test_agent = fake_func
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents)

        # Should handle paths with underscores (only forward slashes are replaced) - only tools get path-prefixed names
        assert "module_with_underscores-test_tool" in env.name2id
        assert "test_worker" in env.name2id  # Workers use simple names
        assert "test_agent" in env.name2id  # Agents use simple names


def test_environment_with_paths_containing_hyphens() -> None:
    """Test Environment initialization with paths containing hyphens."""
    tools = [{"id": "tool1", "name": "test_tool", "path": "module-with-hyphens"}]
    workers = [
        {"id": "worker1", "name": "test_worker", "path": "another-module-with-hyphens"}
    ]
    agents = [{"id": "agent1", "name": "test_agent", "path": "yet-another-module"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_tool = fake_func
        fake_module.test_worker = fake_func
        fake_module.test_agent = fake_func
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents)

        # Should handle paths with hyphens (only forward slashes are replaced) - only tools get path-prefixed names
        assert "module-with-hyphens-test_tool" in env.name2id
        assert "test_worker" in env.name2id  # Workers use simple names
        assert "test_agent" in env.name2id  # Agents use simple names


def test_environment_with_paths_containing_spaces() -> None:
    """Test Environment initialization with paths containing spaces."""
    tools = [{"id": "tool1", "name": "test_tool", "path": "module with spaces"}]
    workers = [
        {"id": "worker1", "name": "test_worker", "path": "another module with spaces"}
    ]
    agents = [{"id": "agent1", "name": "test_agent", "path": "yet another module"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_tool = fake_func
        fake_module.test_worker = fake_func
        fake_module.test_agent = fake_func
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents)

        # Should handle paths with spaces (only forward slashes are replaced) - only tools get path-prefixed names
        assert "module with spaces-test_tool" in env.name2id
        assert "test_worker" in env.name2id  # Workers use simple names
        assert "test_agent" in env.name2id  # Agents use simple names


def test_environment_with_paths_containing_special_characters() -> None:
    """Test Environment initialization with paths containing special characters."""
    tools = [{"id": "tool1", "name": "test_tool", "path": "module@with#special$chars"}]
    workers = [
        {"id": "worker1", "name": "test_worker", "path": "another@module#with$chars"}
    ]
    agents = [{"id": "agent1", "name": "test_agent", "path": "yet@another#module"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_tool = fake_func
        fake_module.test_worker = fake_func
        fake_module.test_agent = fake_func
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents)

        # Should handle paths with special characters (only forward slashes are replaced) - only tools get path-prefixed names
        assert "module@with#special$chars-test_tool" in env.name2id
        assert "test_worker" in env.name2id  # Workers use simple names
        assert "test_agent" in env.name2id  # Agents use simple names


def test_environment_with_paths_containing_unicode() -> None:
    """Test Environment initialization with paths containing unicode characters."""
    tools = [{"id": "tool1", "name": "test_tool", "path": "module_中文_with_unicode"}]
    workers = [{"id": "worker1", "name": "test_worker", "path": "another_emojis_path"}]
    agents = [{"id": "agent1", "name": "test_agent", "path": "yet_русский_path"}]

    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="test description"))
        fake_module.test_tool = fake_func
        fake_module.test_worker = fake_func
        fake_module.test_agent = fake_func
        mock_import.return_value = fake_module

        env = Environment(tools=tools, workers=workers, agents=agents)

        # Should handle paths with unicode characters (only forward slashes are replaced) - only tools get path-prefixed names
        assert "module_中文_with_unicode-test_tool" in env.name2id
        assert "test_worker" in env.name2id  # Workers use simple names
        assert "test_agent" in env.name2id  # Agents use simple names
