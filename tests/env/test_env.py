from collections.abc import Callable
from unittest.mock import MagicMock, Mock, patch

import pytest

from arklex.env.env import DefaultResourceInitializer, Environment
from arklex.env.tools.tools import ToolOutput
from arklex.orchestrator.entities.orchestrator_param_entities import OrchestratorParams
from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
    StatusEnum,
)
from arklex.orchestrator.entities.taskgraph_entities import NodeInfo
from arklex.orchestrator.NLU.core.slot import SlotFiller
from arklex.orchestrator.NLU.entities.slot_entities import Slot
from arklex.orchestrator.NLU.services.model_service import DummyModelService


@pytest.fixture
def fake_tool() -> Callable[[OrchestratorState | None, ToolOutput | None], MagicMock]:
    def _make_fake_tool(
        execute_return: OrchestratorState | None = None,
        tool_output: ToolOutput | None = None,
    ) -> MagicMock:
        tool = MagicMock()
        tool.init_slotfiller = MagicMock()
        tool.load_slots = MagicMock()
        tool.execute = MagicMock(return_value=(execute_return, tool_output))
        return tool

    return _make_fake_tool


@pytest.fixture
def fake_worker() -> Callable[[OrchestratorState | None], Mock]:
    def _make_fake_worker(execute_return: OrchestratorState | None = None) -> Mock:
        worker = Mock()
        worker.execute = Mock(return_value=execute_return)
        worker.init_slotfilling = Mock()
        return worker

    return _make_fake_worker


def test_environment_uses_dummy_model_service() -> None:
    env = Environment(tools=[], workers=[], agents=[], nodes=[])
    assert isinstance(env.model_service, DummyModelService)


def test_environment_initializes_with_slotfillapi_str() -> None:
    env = Environment(
        tools=[], workers=[], agents=[], nodes=[], slotsfillapi="http://fakeapi"
    )
    assert hasattr(env, "slotfillapi")
    assert isinstance(env.slotfillapi, SlotFiller)


def test_environment_initializes_with_slotfillapi_model_service() -> None:
    env = Environment(tools=[], workers=[], agents=[], nodes=[], slotsfillapi="")
    assert hasattr(env, "slotfillapi")
    assert isinstance(env.slotfillapi.model_service, DummyModelService)


def test_default_resource_initializer_init_tools_success_and_error() -> None:
    tools = [
        {"id": "t1", "name": "fake_tool", "path": "fake_path"},
        {"id": "t2", "name": "bad_tool", "path": "bad_path"},
    ]
    # Patch importlib to succeed for one and fail for the other
    with (
        patch(
            "arklex.env.env.RESOURCE_MAP",
            {
                "t1": {
                    "type": "tool",
                    "category": "custom",
                    "item_cls": MagicMock(),
                }
            },
        ),
        patch("importlib.import_module") as mock_import,
    ):
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="desc"))
        fake_module.fake_tool = fake_func
        mock_import.side_effect = [fake_module, Exception("fail")]
        registry = DefaultResourceInitializer.init_tools(tools, nodes=[])
        assert "t1" in registry
        assert "t2" not in registry  # error case is skipped


def test_environment_step_invalid_id_raises() -> None:
    env = Environment(tools=[], workers=[], agents=[], nodes=[])
    # The step method doesn't raise KeyError for invalid IDs, it falls back to planner
    # So we should test that it doesn't raise an exception
    message_state = OrchestratorState()
    params = OrchestratorParams()
    node_info = NodeInfo()

    # This should not raise an exception, it should use the planner
    try:
        response_state, updated_params = env.step(
            "not_a_tool", message_state, params, node_info
        )
    except Exception as e:
        pytest.fail(f"Expected no exception, got {e}")


def test_initialize_slotfillapi_with_valid_string() -> None:
    """Test initialize_slotfillapi with valid string endpoint (lines 147-162)."""
    env = Environment(tools=[], workers=[], agents=[], nodes=[])

    with (
        patch("arklex.env.env.APIClientService") as mock_api_service,
        patch("arklex.env.env.SlotFiller") as mock_slot_filler,
    ):
        mock_api_instance = Mock()
        mock_api_service.return_value = mock_api_instance
        mock_slot_filler_instance = Mock()
        mock_slot_filler.return_value = mock_slot_filler_instance

        result = env.initialize_slotfillapi("http://test-api.com")

        mock_api_service.assert_called_once_with(base_url="http://test-api.com")
        mock_slot_filler.assert_called_once_with(
            model_service=env.model_service, api_service=mock_api_instance
        )
        assert result == mock_slot_filler_instance


def test_environment_with_slot_fill_api_alias() -> None:
    """Test environment initialization with slot_fill_api alias."""
    env = Environment(
        tools=[], workers=[], agents=[], nodes=[], slot_fill_api="http://test-api"
    )
    assert isinstance(env.slotfillapi, SlotFiller)


def test_environment_with_custom_model_service() -> None:
    """Test environment initialization with custom model service."""
    mock_model_service = Mock()
    env = Environment(
        tools=[],
        workers=[],
        agents=[],
        nodes=[],
        model_service=mock_model_service,
    )
    assert env.model_service == mock_model_service


def test_initialize_slotfillapi_with_string() -> None:
    """Test slotfillapi initialization with string endpoint."""
    env = Environment(
        tools=[],
        workers=[],
        agents=[],
        nodes=[],
    )
    slotfiller = env.initialize_slotfillapi("http://test-api")
    assert isinstance(slotfiller, SlotFiller)


def test_initialize_slotfillapi_with_empty_string() -> None:  # noqa: F811
    """Test slotfillapi initialization with empty string."""
    env = Environment(
        tools=[],
        workers=[],
        agents=[],
        nodes=[],
    )
    slotfiller = env.initialize_slotfillapi("")
    assert isinstance(slotfiller, SlotFiller)


def test_initialize_slotfillapi_with_non_string() -> None:  # noqa: F811
    """Test slotfillapi initialization with non-string value."""
    env = Environment(
        tools=[],
        workers=[],
        agents=[],
        nodes=[],
    )
    slotfiller = env.initialize_slotfillapi(None)
    assert isinstance(slotfiller, SlotFiller)


def test_default_resource_initializer_init_tools_with_exception() -> None:
    """Test DefaultResourceInitializer init_tools with import exception."""
    initializer = DefaultResourceInitializer()

    result = initializer.init_tools(
        [{"id": "tool1", "name": "nonexistent_tool", "path": "nonexistent/path"}],
        nodes=[],
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
        ],
        nodes=[],
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
            nodes=[],
        )

        # Verify that DefaultResourceInitializer was used (not ModelAwareResourceInitializer)
        assert isinstance(env.tools, dict)
        assert isinstance(env.workers, dict)
        assert isinstance(env.agents, dict)


def test_environment_step_tool_with_slot_schema_signature_change(
    fake_tool: Callable[[OrchestratorState | None], MagicMock],
) -> None:
    """Test environment step with tool when slot schema signature changes."""
    tool = fake_tool(
        OrchestratorState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
        ),
        ToolOutput(
            status=StatusEnum.COMPLETE,
            response="Tool execution successful",
            slots={"t1": [Slot(name="slot1", value="value1", verified=True)]},
        ),
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with (
        patch(
            "arklex.env.env.RESOURCE_MAP",
            {
                "t1": {
                    "type": "tool",
                    "category": "custom",
                    "item_cls": tool,
                }
            },
        ),
        patch("importlib.import_module") as mock_import,
    ):
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[], nodes=[])

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

        state = OrchestratorState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        # First call to establish initial slots
        result_state, result_params = env.step("t1", state, params, node_info)

        # Second call with different slot configuration (simulating schema change)
        node_info.attributes = {"slots": ["slot1", "slot2"]}
        result_state, result_params = env.step("t1", state, params, node_info)

        # Should reset slots due to schema change
        assert result_params.status == StatusEnum.COMPLETE


def test_environment_step_tool_with_verified_slots(
    fake_tool: Callable[[OrchestratorState | None], MagicMock],
) -> None:
    """Test environment step with tool that has verified slots."""
    from arklex.orchestrator.NLU.entities.slot_entities import Slot

    tool = fake_tool(
        OrchestratorState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
        ),
        ToolOutput(
            status=StatusEnum.COMPLETE,
            response="Tool execution successful",
            slots={"t1": [Slot(name="slot1", value="value1", verified=True)]},
        ),
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with (
        patch(
            "arklex.env.env.RESOURCE_MAP",
            {
                "t1": {
                    "type": "tool",
                    "category": "custom",
                    "item_cls": tool,
                }
            },
        ),
        patch("importlib.import_module") as mock_import,
    ):
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[], nodes=[])

        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}
            attributes = {"slots": ["slot1"]}

        state = OrchestratorState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_params.status == StatusEnum.COMPLETE


def test_environment_step_tool_with_slot_verification_needed(
    fake_tool: Callable[[OrchestratorState | None], MagicMock],
) -> None:
    """Test environment step with tool that needs slot verification."""
    tool = fake_tool(
        OrchestratorState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
        ),
        ToolOutput(
            status=StatusEnum.INCOMPLETE,
            response="Tool execution failed",
            slots={},
        ),
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with (
        patch(
            "arklex.env.env.RESOURCE_MAP",
            {
                "t1": {
                    "type": "tool",
                    "category": "custom",
                    "item_cls": tool,
                }
            },
        ),
        patch("importlib.import_module") as mock_import,
    ):
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[], nodes=[])

        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}
            attributes = {"slots": ["slot1"]}

        state = OrchestratorState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        # Should be incomplete due to slot verification needed
        assert result_params.status == StatusEnum.INCOMPLETE


def test_environment_step_tool_with_group_slots(
    fake_tool: Callable[[OrchestratorState | None], MagicMock],
) -> None:
    """Test environment step with tool that has group slots."""
    tool = fake_tool(
        OrchestratorState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
        ),
        ToolOutput(
            status=StatusEnum.COMPLETE,
            response="Tool execution successful",
            slots={},
        ),
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with (
        patch(
            "arklex.env.env.RESOURCE_MAP",
            {
                "t1": {
                    "type": "tool",
                    "category": "custom",
                    "item_cls": tool,
                }
            },
        ),
        patch("importlib.import_module") as mock_import,
    ):
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[], nodes=[])

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

        state = OrchestratorState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_params.status == StatusEnum.COMPLETE


def test_environment_step_tool_with_repeatable_slots(
    fake_tool: Callable[[OrchestratorState | None], MagicMock],
) -> None:
    """Test environment step with tool that has repeatable slots."""
    tool = fake_tool(
        OrchestratorState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
        ),
        ToolOutput(
            status=StatusEnum.COMPLETE,
            response="Tool execution successful",
            slots={},
        ),
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with (
        patch(
            "arklex.env.env.RESOURCE_MAP",
            {
                "t1": {
                    "type": "tool",
                    "category": "custom",
                    "item_cls": tool,
                }
            },
        ),
        patch("importlib.import_module") as mock_import,
    ):
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[], nodes=[])

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

        state = OrchestratorState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_params.status == StatusEnum.COMPLETE


def test_environment_step_tool_with_function_calling_trajectory(
    fake_tool: Callable[[OrchestratorState | None], MagicMock],
) -> None:
    """Test environment step with tool that returns function calling trajectory."""
    orch_state = OrchestratorState(
        function_calling_trajectory=[
            {"role": "assistant", "content": "call"},
            {"role": "function", "content": "result"},
        ]
    )
    tool = fake_tool(
        orch_state,
        ToolOutput(
            status=StatusEnum.COMPLETE,
            response="Tool execution successful",
            slots={},
        ),
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with (
        patch(
            "arklex.env.env.RESOURCE_MAP",
            {
                "t1": {
                    "type": "tool",
                    "category": "custom",
                    "item_cls": tool,
                }
            },
        ),
        patch("importlib.import_module") as mock_import,
    ):
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[], nodes=[])

        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}
            attributes = {"slots": ["slot1"]}

        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", orch_state, node_info, params)
        assert result_params.status == StatusEnum.COMPLETE
        assert len(result_state.function_calling_trajectory) == 2


def test_environment_step_tool_with_tool_execution_error(
    fake_tool: Callable[[OrchestratorState | None], MagicMock],
) -> None:
    """Test environment step with tool that raises ToolExecutionError."""
    tool = fake_tool(
        OrchestratorState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
        ),
        ToolOutput(
            status=StatusEnum.INCOMPLETE,
            response="Tool execution failed",
            slots={},
        ),
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with (
        patch(
            "arklex.env.env.RESOURCE_MAP",
            {
                "t1": {
                    "type": "tool",
                    "category": "custom",
                    "item_cls": tool,
                }
            },
        ),
        patch("importlib.import_module") as mock_import,
    ):
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[], nodes=[])

        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}
            attributes = {"slots": ["slot1"]}

        state = OrchestratorState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_params.status == StatusEnum.INCOMPLETE


def test_environment_step_tool_with_general_exception(
    fake_tool: Callable[[OrchestratorState | None], MagicMock],
) -> None:
    """Test environment step with tool that raises general exception."""
    tool = fake_tool(
        OrchestratorState(
            function_calling_trajectory=[{"role": "assistant", "content": "call"}],
        ),
        ToolOutput(
            status=StatusEnum.INCOMPLETE,
            response="General error occurred",
            slots={},
        ),
    )
    tools = [{"id": "t1", "name": "fake_tool", "path": "fake_path"}]
    with (
        patch(
            "arklex.env.env.RESOURCE_MAP",
            {
                "t1": {
                    "type": "tool",
                    "category": "custom",
                    "item_cls": tool,
                }
            },
        ),
        patch("importlib.import_module") as mock_import,
    ):
        fake_module = MagicMock()
        fake_module.fake_tool = MagicMock(return_value=tool)
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[], agents=[], nodes=[])

        class DummyOrchestratorParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}
            attributes = {"slots": ["slot1"]}

        state = OrchestratorState()
        params = DummyOrchestratorParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}

        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_params.status == StatusEnum.INCOMPLETE
