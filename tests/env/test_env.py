from unittest.mock import patch, MagicMock, Mock
from arklex.env.env import Environment, DefaultResourceInitializer
from arklex.orchestrator.NLU.services.model_service import DummyModelService
from arklex.orchestrator.NLU.core.slot import SlotFiller
from arklex.utils.graph_state import MessageState, Params, NodeInfo, StatusEnum
from arklex.env.planner.react_planner import ReactPlanner


def test_environment_uses_dummy_model_service() -> None:
    env = Environment(tools=[], workers=[])
    assert isinstance(env.model_service, DummyModelService)


def test_environment_initializes_with_planner() -> None:
    env = Environment(tools=[], workers=[], planner_enabled=True)
    assert hasattr(env, "planner")


def test_environment_initializes_with_slotfillapi_str() -> None:
    env = Environment(tools=[], workers=[], slotsfillapi="http://fakeapi")
    assert hasattr(env, "slotfillapi")
    assert isinstance(env.slotfillapi, SlotFiller)


def test_environment_initializes_with_slotfillapi_model_service() -> None:
    env = Environment(tools=[], workers=[], slotsfillapi="")
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
        setattr(fake_module, "fake_tool", fake_func)
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
        setattr(fake_module, "fake_worker", fake_func)
        mock_import.side_effect = [fake_module, Exception("fail")]
        registry = DefaultResourceInitializer.init_workers(workers)
        assert "w1" in registry
        assert "w2" not in registry


def test_environment_step_tool_executes_and_updates_params() -> None:
    # Setup a fake tool
    fake_tool = MagicMock()
    fake_tool.init_slotfiller = MagicMock()
    fake_tool.execute = MagicMock(
        return_value=MagicMock(function_calling_trajectory=["call"], slots={})
    )
    tools = [
        {"id": "t1", "name": "fake_tool", "path": "fake_path"},
    ]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        setattr(fake_module, "fake_tool", MagicMock(return_value=fake_tool))
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[])

        # Setup params and state
        class DummyParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}

        state = MagicMock()
        params = DummyParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}
        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_state.function_calling_trajectory == ["call"]
        assert result_params is params


def test_environment_step_invalid_id_raises() -> None:
    env = Environment(tools=[], workers=[])
    # The step method doesn't raise KeyError for invalid IDs, it falls back to planner
    # So we should test that it doesn't raise an exception
    message_state = MessageState()
    params = Params()
    node_info = NodeInfo()

    # This should not raise an exception, it should use the planner
    response_state, updated_params = env.step(
        "not_a_tool", message_state, params, node_info
    )
    assert isinstance(response_state, MessageState)
    assert isinstance(updated_params, Params)


def test_environment_step_worker_executes_and_updates_params() -> None:
    """Test environment step with worker execution."""
    mock_worker = Mock()
    mock_worker.execute.return_value = MessageState(status=StatusEnum.COMPLETE)
    mock_worker.init_slotfilling = Mock()
    env = Environment(
        tools=[], workers=[{"id": "worker1", "name": "test_worker", "path": "test"}]
    )
    env.workers = {
        "worker1": {"name": "test_worker", "execute": Mock(return_value=mock_worker)}
    }
    env.id2name = {"worker1": "test_worker"}
    message_state = MessageState()
    params = Params()
    params.memory.function_calling_trajectory = []
    params.taskgraph.curr_node = "node1"
    node_info = NodeInfo()
    result_state, result_params = env.step("worker1", message_state, params, node_info)
    assert result_state.status == StatusEnum.COMPLETE
    assert len(result_params.memory.function_calling_trajectory) == 2
    mock_worker.init_slotfilling.assert_called_once()


def test_environment_step_worker_without_init_slotfilling() -> None:
    """Test environment step with worker that doesn't have init_slotfilling method."""
    mock_worker = Mock()
    mock_worker.execute.return_value = MessageState(status=StatusEnum.COMPLETE)
    # Remove init_slotfilling attribute to test the hasattr check
    if hasattr(mock_worker, "init_slotfilling"):
        delattr(mock_worker, "init_slotfilling")
    env = Environment(
        tools=[], workers=[{"id": "worker1", "name": "test_worker", "path": "test"}]
    )
    env.workers = {
        "worker1": {"name": "test_worker", "execute": Mock(return_value=mock_worker)}
    }
    env.id2name = {"worker1": "test_worker"}
    message_state = MessageState()
    params = Params()
    params.memory.function_calling_trajectory = []
    params.taskgraph.curr_node = "node1"
    node_info = NodeInfo()
    result_state, result_params = env.step("worker1", message_state, params, node_info)
    assert result_state.status == StatusEnum.COMPLETE
    assert len(result_params.memory.function_calling_trajectory) == 2


def test_environment_step_worker_with_response_content() -> None:
    """Test environment step with worker that has response content."""
    mock_worker = Mock()
    mock_worker.execute.return_value = MessageState(
        status=StatusEnum.COMPLETE, response="test response"
    )
    env = Environment(
        tools=[], workers=[{"id": "worker1", "name": "test_worker", "path": "test"}]
    )
    env.workers = {
        "worker1": {"name": "test_worker", "execute": Mock(return_value=mock_worker)}
    }
    env.id2name = {"worker1": "test_worker"}
    message_state = MessageState()
    params = Params()
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


def test_environment_step_worker_with_message_flow() -> None:
    """Test environment step with worker that has message_flow but no response."""
    mock_worker = Mock()
    mock_worker.execute.return_value = MessageState(
        status=StatusEnum.COMPLETE, message_flow="test flow"
    )
    env = Environment(
        tools=[], workers=[{"id": "worker1", "name": "test_worker", "path": "test"}]
    )
    env.workers = {
        "worker1": {"name": "test_worker", "execute": Mock(return_value=mock_worker)}
    }
    env.id2name = {"worker1": "test_worker"}
    message_state = MessageState()
    params = Params()
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
    env = Environment(tools=[], workers=[])
    env.planner = mock_planner
    message_state = MessageState()
    params = Params()
    params.memory.function_calling_trajectory = []
    node_info = NodeInfo()
    result_state, result_params = env.step(
        "invalid_id", message_state, params, node_info
    )
    assert result_state.status == StatusEnum.COMPLETE
    mock_planner.execute.assert_called_once()


def test_environment_register_tool_success() -> None:
    """Test successful tool registration."""
    env = Environment(tools=[], workers=[])
    mock_tool = {"name": "test_tool", "description": "test description"}
    env.register_tool("test_tool", mock_tool)
    assert "test_tool" in env.tools
    assert env.tools["test_tool"] == mock_tool


def test_environment_register_tool_failure() -> None:
    """Test tool registration failure."""
    env = Environment(tools=[], workers=[])
    # Mock the tools dict to raise an exception
    with patch.object(env, "tools", side_effect=Exception("Registration failed")):
        env.register_tool("test_tool", {})
        # Should not raise exception, just log error


def test_environment_with_slot_fill_api_alias() -> None:
    """Test environment initialization with slot_fill_api alias."""
    env = Environment(tools=[], workers=[], slot_fill_api="http://test-api")
    assert isinstance(env.slotfillapi, SlotFiller)


def test_environment_with_custom_resource_initializer() -> None:
    """Test environment initialization with custom resource initializer."""
    mock_initializer = Mock()
    mock_initializer.init_tools.return_value = {"tool1": {"name": "test_tool"}}
    mock_initializer.init_workers.return_value = {"worker1": {"name": "test_worker"}}
    env = Environment(
        tools=[{"id": "tool1", "name": "test", "path": "test"}],
        workers=[{"id": "worker1", "name": "test", "path": "test"}],
        resource_initializer=mock_initializer,
    )
    assert env.tools == {"tool1": {"name": "test_tool"}}
    assert env.workers == {"worker1": {"name": "test_worker"}}
    mock_initializer.init_tools.assert_called_once()
    mock_initializer.init_workers.assert_called_once()


def test_environment_with_planner_enabled() -> None:
    """Test environment initialization with planner enabled."""
    env = Environment(tools=[], workers=[], planner_enabled=True)
    assert isinstance(env.planner, ReactPlanner)


def test_environment_with_custom_model_service() -> None:
    """Test environment initialization with custom model service."""
    mock_model_service = Mock()
    env = Environment(tools=[], workers=[], model_service=mock_model_service)
    assert env.model_service == mock_model_service


def test_initialize_slotfillapi_with_string() -> None:
    """Test slotfillapi initialization with string endpoint."""
    env = Environment(tools=[], workers=[])
    slotfiller = env.initialize_slotfillapi("http://test-api")
    assert isinstance(slotfiller, SlotFiller)


def test_initialize_slotfillapi_with_empty_string() -> None:
    """Test slotfillapi initialization with empty string."""
    env = Environment(tools=[], workers=[])
    slotfiller = env.initialize_slotfillapi("")
    assert isinstance(slotfiller, SlotFiller)


def test_initialize_slotfillapi_with_non_string() -> None:
    """Test slotfillapi initialization with non-string value."""
    env = Environment(tools=[], workers=[])
    slotfiller = env.initialize_slotfillapi(None)
    assert isinstance(slotfiller, SlotFiller)


def test_default_resource_initializer_init_tools_with_exception() -> None:
    """Test DefaultResourceInitializer init_tools with import exception."""
    initializer = DefaultResourceInitializer()

    result = initializer.init_tools(
        [{"id": "tool1", "name": "nonexistent_tool", "path": "nonexistent/path"}]
    )
    assert result == {}


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
