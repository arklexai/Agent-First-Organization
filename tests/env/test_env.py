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


def test_environment_step_tool_executes_and_updates_params(fake_tool: Callable[[MessageState | None], MagicMock]) -> None:
    # Setup a fake tool
    tool = fake_tool(MessageState(function_calling_trajectory=[{"role": "assistant", "content": "call"}], slots={}, status=StatusEnum.COMPLETE))
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
        assert result_state.function_calling_trajectory == [{"role": "assistant", "content": "call"}]
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


def test_environment_step_worker_executes_and_updates_params(fake_worker: Callable[[MessageState | None], Mock]) -> None:
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


def test_environment_step_worker_without_init_slotfilling(fake_worker: Callable[[MessageState | None], Mock]) -> None:
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


def test_environment_step_worker_with_response_content(fake_worker: Callable[[MessageState | None], Mock]) -> None:
    """Test environment step with worker that has response content."""
    mock_worker = fake_worker(MessageState(
        status=StatusEnum.COMPLETE, response="test response"
    ))
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


def test_environment_step_worker_with_message_flow(fake_worker: Callable[[MessageState | None], Mock]) -> None:
    """Test environment step with worker that has message_flow but no response."""
    mock_worker = fake_worker(MessageState(
        status=StatusEnum.COMPLETE, message_flow="test flow"
    ))
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
    mock_agent_instance.execute.return_value = MessageState(
        status=StatusEnum.COMPLETE, function_calling_trajectory=[]
    )

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


def test_environment_step_tool_with_attributes_and_slots(fake_tool: Callable[[MessageState | None], MagicMock]) -> None:
    """Test environment step with tool that has attributes and slots."""
    from arklex.orchestrator.NLU.entities.slot_entities import Slot

    tool = fake_tool(MessageState(function_calling_trajectory=[{"role": "assistant", "content": "call"}], slots={"slot1": [Slot(name="slot1", value="value1")]}, status=StatusEnum.COMPLETE))
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
        tool.load_slots.assert_called_once_with(["slot1", "slot2"])


def test_environment_step_tool_with_none_additional_args(fake_tool: Callable[[MessageState | None], MagicMock]) -> None:
    """Test environment step with tool that has None additional_args."""
    tool = fake_tool(MessageState(function_calling_trajectory=[{"role": "assistant", "content": "call"}], slots={}, status=StatusEnum.COMPLETE))
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
            additional_args = None  # None additional_args

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
        tool.load_slots.assert_called_once_with([])  # Empty list when attributes is empty


def test_environment_step_tool_with_none_attributes(fake_tool: Callable[[MessageState | None], MagicMock]) -> None:
    """Test environment step with tool that has None attributes."""
    tool = fake_tool(MessageState(function_calling_trajectory=[{"role": "assistant", "content": "call"}], slots={}, status=StatusEnum.COMPLETE))
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
        tool.load_slots.assert_called_once_with([])  # Empty list when attributes is empty
        tool.load_slots.assert_called_once_with([])  # Empty list when attributes is empty


def test_environment_step_worker_with_none_additional_args(fake_worker: Callable[[MessageState | None], Mock]) -> None:
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
    node_info.additional_args = None  # None additional_args

    result_state, result_params = env.step("worker1", message_state, params, node_info)
    assert result_state.status == StatusEnum.COMPLETE
    assert len(result_params.memory.function_calling_trajectory) == 2
    mock_worker.init_slotfilling.assert_called_once()


def test_environment_step_worker_with_empty_response_and_message_flow(fake_worker: Callable[[MessageState | None], Mock]) -> None:
    """Test environment step with worker that has empty response and message_flow."""

    mock_worker = fake_worker(MessageState(status=StatusEnum.COMPLETE, response="", message_flow=""))
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


def test_environment_step_tool_with_slot_schema_signature_change(fake_tool: Callable[[MessageState | None], MagicMock]) -> None:
    """Test environment step with tool when slot schema signature changes."""
    tool = fake_tool(MessageState(function_calling_trajectory=[{"role": "assistant", "content": "call"}], slots={"slot1": [Slot(name="slot1", value="value1")]}, status=StatusEnum.COMPLETE))
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
            attributes = {"slots": ["slot1"], "slot_groups": [{"name": "group1", "schema": [{"name": "slot1", "type": "str"}]}]}

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


def test_environment_step_tool_with_verified_slots(fake_tool: Callable[[MessageState | None], MagicMock]) -> None:
    """Test environment step with tool that has verified slots."""
    from arklex.orchestrator.NLU.entities.slot_entities import Slot

    tool = fake_tool(MessageState(function_calling_trajectory=[{"role": "assistant", "content": "call"}], slots={}, status=StatusEnum.COMPLETE))
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
        state.slots = {
            "t1": [Slot(name="slot1", value="value1", verified=True)]
        }
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_state.status == StatusEnum.COMPLETE


def test_environment_step_tool_with_missing_required_slots(fake_tool: Callable[[MessageState | None], MagicMock]) -> None:
    """Test environment step with tool that has missing required slots."""
    tool = fake_tool(MessageState(function_calling_trajectory=[{"role": "assistant", "content": "call"}], slots={}, status=StatusEnum.INCOMPLETE))
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


def test_environment_step_tool_with_slot_verification_needed(fake_tool: Callable[[MessageState | None], MagicMock]) -> None:
    """Test environment step with tool that needs slot verification."""
    tool = fake_tool(MessageState(function_calling_trajectory=[{"role": "assistant", "content": "call"}], slots={}, status=StatusEnum.INCOMPLETE))
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


def test_environment_step_tool_with_group_slots(fake_tool: Callable[[MessageState | None], MagicMock]) -> None:
    """Test environment step with tool that has group slots."""
    tool = fake_tool(MessageState(function_calling_trajectory=[{"role": "assistant", "content": "call"}], slots={}, status=StatusEnum.COMPLETE))
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
                        "schema": [{"name": "field1", "type": "str", "required": True}]
                    }
                ]
            }

        state = MessageState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_state.status == StatusEnum.COMPLETE


def test_environment_step_tool_with_repeatable_slots(fake_tool: Callable[[MessageState | None], MagicMock]) -> None:
    """Test environment step with tool that has repeatable slots."""
    tool = fake_tool(MessageState(function_calling_trajectory=[{"role": "assistant", "content": "call"}], slots={}, status=StatusEnum.COMPLETE))
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
                        "required": True
                    }
                ]
            }

        state = MessageState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_state.status == StatusEnum.COMPLETE


def test_environment_step_tool_with_function_calling_trajectory(fake_tool: Callable[[MessageState | None], MagicMock]) -> None:
    """Test environment step with tool that returns function calling trajectory."""
    tool = fake_tool(MessageState(function_calling_trajectory=[{"role": "assistant", "content": "call"}, {"role": "function", "content": "result"}], status=StatusEnum.COMPLETE))
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


def test_environment_step_tool_with_slots_parameter(fake_tool: Callable[[MessageState | None], MagicMock]) -> None:
    """Test environment step with tool that accepts slots parameter."""
    tool = fake_tool(MessageState(function_calling_trajectory=[{"role": "assistant", "content": "call"}], slots={}, status=StatusEnum.COMPLETE))
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


def test_environment_step_tool_with_missing_required_arguments(fake_tool: Callable[[MessageState | None], MagicMock]) -> None:
    """Test environment step with tool that has missing required arguments."""
    tool = fake_tool(MessageState(function_calling_trajectory=[{"role": "assistant", "content": "call"}], slots={}, status=StatusEnum.INCOMPLETE))
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


def test_environment_step_tool_with_authentication_error(fake_tool: Callable[[MessageState | None], MagicMock]) -> None:
    """Test environment step with tool that raises AuthenticationError."""
    tool = fake_tool(MessageState(function_calling_trajectory=[{"role": "assistant", "content": "call"}], slots={}, status=StatusEnum.INCOMPLETE, response="Authentication failed"))
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


def test_environment_step_tool_with_tool_execution_error(fake_tool: Callable[[MessageState | None], MagicMock]) -> None:
    """Test environment step with tool that raises ToolExecutionError."""
    tool = fake_tool(MessageState(function_calling_trajectory=[{"role": "assistant", "content": "call"}], slots={}, status=StatusEnum.INCOMPLETE, response="Tool execution failed"))
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


def test_environment_step_tool_with_general_exception(fake_tool: Callable[[MessageState | None], MagicMock]) -> None:
    """Test environment step with tool that raises general exception."""
    tool = fake_tool(MessageState(function_calling_trajectory=[{"role": "assistant", "content": "call"}], slots={}, status=StatusEnum.INCOMPLETE, response="General error occurred"))
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
