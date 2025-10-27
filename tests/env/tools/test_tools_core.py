"""Comprehensive tests for arklex.env.tools.tools (Tool class and register_tool)."""

from typing import NoReturn
from unittest.mock import Mock

# Mocks for dependencies
from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
    StatusEnum,
)
from arklex.orchestrator.nlu.entities.slot_entities import Slot
from arklex.resources.tools import tools
from arklex.resources.tools.tools import Tool, register_tool


def dummy_func(a: object = None, b: object = None, **kwargs: object) -> str:
    return f"a={a}, b={b}"


def test_register_tool_decorator_creates_tool() -> None:
    """Test that register_tool decorates a function and returns a Tool."""
    tool_instance = register_tool(
        description="desc", slots=[{"name": "a", "type": "str", "description": "A"}]
    )(dummy_func)
    assert isinstance(tool_instance, Tool)
    assert tool_instance.description == "desc"
    assert any(slot.name == "a" for slot in tool_instance.slots)


def test_tool_init_slotfiller() -> None:
    """Test Tool.init_slotfiller sets the slotfiller attribute."""
    tool = Tool(dummy_func, "toolname", "desc", [])
    mock_sf = Mock()
    tool.init_slotfiller(mock_sf)
    assert tool.slotfiller is mock_sf


def test_tool__init_slots_populates_slots() -> None:
    """Test _init_slots populates slots from state.default_slots."""
    slot = Slot(
        name="a", type="str", description="A", required=True, value="foo", verified=True
    )
    state = Mock(spec=OrchestratorState)
    default_slots = {"default_slots": [slot]}
    state.function_calling_trajectory = []
    tool = Tool(
        dummy_func,
        "toolname",
        "desc",
        [{"name": "a", "type": "str", "description": "A", "required": True}],
    )
    tool._init_slots(state, default_slots)
    assert tool.slots[0].value == "foo"
    assert tool.slots[0].verified is True
    assert state.function_calling_trajectory


def test_tool_execute_successful() -> None:
    """Test Tool.execute with all slots filled and verified."""
    slot = Slot(
        name="a", type="str", description="A", required=True, value="bar", verified=True
    )
    state = Mock(spec=OrchestratorState)
    state.slots = {}
    state.function_calling_trajectory = []
    state.trajectory = [[Mock(input=None, output=None)]]
    state.bot_config = Mock()
    state.bot_config.llm_config = Mock()
    state.bot_config.llm_config.model_dump.return_value = {}
    state.message_flow = ""
    state.status = StatusEnum.INCOMPLETE
    tool = Tool(
        dummy_func,
        "toolname",
        "desc",
        [{"name": "a", "type": "str", "description": "A", "required": True}],
    )
    tool.slots = [slot]
    tool.slotfiller = Mock()
    tool.slotfiller.fill_slots.return_value = [slot]
    tool.isResponse = False
    result_state, result_output = tool.execute(state, {}, {})
    assert result_output.status == StatusEnum.COMPLETE
    assert result_output.slots["toolname"][0].value == "bar"


def test_tool_execute_incomplete_due_to_missing_slot() -> None:
    """Test Tool.execute when a required slot is missing."""
    slot = Slot(
        name="a",
        type="str",
        description="A",
        required=True,
        value=None,
        verified=False,
        prompt="Prompt for a",
    )
    state = Mock(spec=OrchestratorState)
    state.slots = {}
    state.function_calling_trajectory = []
    state.trajectory = [[Mock(input=None, output=None)]]
    state.bot_config = Mock()
    state.bot_config.llm_config = Mock()
    state.bot_config.llm_config.model_dump.return_value = {}
    state.message_flow = ""
    tool = Tool(
        dummy_func,
        "toolname",
        "desc",
        [{"name": "a", "type": "str", "description": "A", "required": True}],
    )
    tool.slots = [slot]
    tool.slotfiller = Mock()
    tool.slotfiller.fill_slots.return_value = [slot]
    result_state, result_output = tool.execute(state, {}, {})
    assert result_output.status == StatusEnum.INCOMPLETE
    assert (
        "Prompt for a" in result_state.message_flow or result_state.message_flow == ""
    )


def test_tool_execute_slot_verification_needed() -> None:
    """Test Tool.execute when slot verification is needed."""
    slot = Slot(
        name="a",
        type="str",
        description="A",
        required=True,
        value="foo",
        verified=False,
        prompt="Prompt for a",
    )
    state = Mock(spec=OrchestratorState)
    state.slots = {}
    state.function_calling_trajectory = []
    state.trajectory = [[Mock(input=None, output=None)]]
    state.bot_config = Mock()
    state.bot_config.llm_config = Mock()
    state.bot_config.llm_config.model_dump.return_value = {}
    state.message_flow = ""
    tool = Tool(
        dummy_func,
        "toolname",
        "desc",
        [{"name": "a", "type": "str", "description": "A", "required": True}],
    )
    tool.slots = [slot]
    tool.slotfiller = Mock()
    tool.slotfiller.fill_slots.return_value = [slot]
    tool.slotfiller.verify_slot.return_value = (True, "Need verification")
    result_state, result_output = tool.execute(state, {}, {})
    assert result_output.status == StatusEnum.INCOMPLETE


def test_tool_execute_tool_execution_error() -> None:
    """Test Tool.execute handles ToolExecutionError."""

    def error_func(**kwargs: object) -> NoReturn:
        raise tools.ToolExecutionError("toolname", "fail", extra_message="extra")

    slot = Slot(
        name="a", type="str", description="A", required=True, value="foo", verified=True
    )
    state = Mock(spec=OrchestratorState)
    state.slots = {}
    state.function_calling_trajectory = []
    state.trajectory = [[Mock(input=None, output=None)]]
    state.bot_config = Mock()
    state.bot_config.llm_config = Mock()
    state.bot_config.llm_config.model_dump.return_value = {}
    state.message_flow = ""
    tool = Tool(
        error_func,
        "toolname",
        "desc",
        [{"name": "a", "type": "str", "description": "A", "required": True}],
    )
    tool.slots = [slot]
    tool.slotfiller = Mock()
    tool.slotfiller.fill_slots.return_value = [slot]
    result_state, result_output = tool.execute(state, {}, {})
    assert result_output.status == StatusEnum.INCOMPLETE


def test_tool_execute_authentication_error() -> None:
    """Test Tool.execute handles AuthenticationError."""

    def error_func(**kwargs: object) -> NoReturn:
        raise tools.AuthenticationError("auth fail")

    slot = Slot(
        name="a", type="str", description="A", required=True, value="foo", verified=True
    )
    state = Mock(spec=OrchestratorState)
    state.slots = {}
    state.function_calling_trajectory = []
    state.trajectory = [[Mock(input=None, output=None)]]
    state.bot_config = Mock()
    state.bot_config.llm_config = Mock()
    state.bot_config.llm_config.model_dump.return_value = {}
    state.message_flow = ""
    tool = Tool(
        error_func,
        "toolname",
        "desc",
        [{"name": "a", "type": "str", "description": "A", "required": True}],
    )
    tool.slots = [slot]
    tool.slotfiller = Mock()
    tool.slotfiller.fill_slots.return_value = [slot]
    result_state, result_output = tool.execute(state, {}, {})
    assert result_output.status == StatusEnum.INCOMPLETE


def test_tool_str_and_repr() -> None:
    """Test __str__ and __repr__ methods."""
    tool = Tool(dummy_func, "toolname", "desc", [])
    assert str(tool) == "Tool"
    assert repr(tool) == "Tool"
