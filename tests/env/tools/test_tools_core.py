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


# Tests for _validate_and_clean_slot_values ----------------------------------------------------------
def test_validate_and_clean_slot_values_simple_object_array() -> None:
    """Test validation of a simple array of objects."""
    slot_schema_object_array = {
        "type": "object",
        "required": ["vehicles_array"],
        "properties": {
            "vehicles_array": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["make", "model"],
                    "properties": {
                        "make": {
                            "type": "string",
                            "description": "make of the vehicle",
                        },
                        "model": {
                            "type": "string",
                            "description": "model of the vehicle",
                        },
                    },
                },
                "description": "Array of vehicle objects",
            }
        },
    }

    slot = Slot(
        name="vehicles_array",
        type="array",
        description="Array of vehicles",
        required=True,
        value=[{"make": "Toyota", "model": "Camry"}],
        verified=True,
        slot_schema={
            "type": "function",
            "function": {"name": "tool", "parameters": slot_schema_object_array},
        },
    )

    tool = Tool(dummy_func, "toolname", "desc", [])
    tool.slots = [slot]
    tool._validate_and_clean_slot_values()

    assert tool.slots[0].value == [{"make": "Toyota", "model": "Camry"}]


def test_validate_and_clean_slot_values_removes_extra_fields() -> None:
    """Test that validation removes extra fields not in schema."""
    slot_schema = {
        "type": "object",
        "required": ["vehicles_array"],
        "properties": {
            "vehicles_array": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["make", "model"],
                    "properties": {
                        "make": {"type": "string"},
                        "model": {"type": "string"},
                    },
                },
            }
        },
    }

    slot = Slot(
        name="vehicles_array",
        type="array",
        description="Array of vehicles",
        required=True,
        value=[
            {
                "make": "Toyota",
                "model": "Camry",
                "extra_field": "should_be_removed",
            }
        ],
        verified=True,
        slot_schema={
            "type": "function",
            "function": {"name": "tool", "parameters": slot_schema},
        },
    )

    tool = Tool(dummy_func, "toolname", "desc", [])
    tool.slots = [slot]
    tool._validate_and_clean_slot_values()

    # Extra field should be removed
    assert tool.slots[0].value == [{"make": "Toyota", "model": "Camry"}]
    assert "extra_field" not in tool.slots[0].value[0]


def test_validate_and_clean_slot_values_nested_objects() -> None:
    """Test validation of nested objects within array."""
    slot_schema_nested = {
        "type": "object",
        "required": ["cars_array"],
        "properties": {
            "cars_array": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["make", "model", "car"],
                    "properties": {
                        "car": {
                            "type": "object",
                            "required": ["car_id", "car_insurance_type"],
                            "properties": {
                                "car_id": {"type": "string"},
                                "car_insurance_type": {"type": "integer"},
                            },
                        },
                        "make": {"type": "string"},
                        "model": {"type": "string"},
                    },
                },
            }
        },
    }

    slot = Slot(
        name="cars_array",
        type="array",
        description="Array of cars",
        required=True,
        value=[
            {
                "make": "Toyota",
                "model": "Camry",
                "car": {
                    "car_id": "toyota_camry_2025",
                    "car_insurance_type": 1,
                    "dummy_field": "should_be_removed",
                },
            }
        ],
        verified=True,
        slot_schema={
            "type": "function",
            "function": {"name": "tool", "parameters": slot_schema_nested},
        },
    )

    tool = Tool(dummy_func, "toolname", "desc", [])
    tool.slots = [slot]
    tool._validate_and_clean_slot_values()

    # Check that dummy_field was removed from nested object
    validated_value = tool.slots[0].value[0]
    assert validated_value["make"] == "Toyota"
    assert validated_value["model"] == "Camry"
    assert validated_value["car"]["car_id"] == "toyota_camry_2025"
    assert validated_value["car"]["car_insurance_type"] == 1
    assert "dummy_field" not in validated_value["car"]


def test_validate_and_clean_slot_values_optional_fields() -> None:
    """Test validation with optional fields (not all required)."""
    object_schema = {
        "type": "object",
        "required": ["cars_array"],
        "properties": {
            "cars_array": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["make", "model", "car"],
                    "properties": {
                        "car": {
                            "type": "object",
                            "required": ["car_id"],
                            "properties": {
                                "car_id": {"type": "string"},
                                "car_insurance_type": {
                                    "type": "integer"
                                },  # Not required
                            },
                        },
                        "make": {"type": "string"},
                        "model": {"type": "string"},
                    },
                },
            }
        },
    }

    # Test with optional field present
    slot = Slot(
        name="cars_array",
        type="array",
        description="Array of cars",
        required=True,
        value=[
            {
                "make": "Toyota",
                "model": "Camry",
                "car": {"car_id": "toyota_camry_2025", "car_insurance_type": 1},
            }
        ],
        verified=True,
        slot_schema={
            "type": "function",
            "function": {"name": "tool", "parameters": object_schema},
        },
    )

    tool = Tool(dummy_func, "toolname", "desc", [])
    tool.slots = [slot]
    tool._validate_and_clean_slot_values()

    assert tool.slots[0].value[0]["car"]["car_insurance_type"] == 1

    # Test with optional field missing
    slot2 = Slot(
        name="cars_array",
        type="array",
        description="Array of cars",
        required=True,
        value=[
            {
                "make": "Honda",
                "model": "Accord",
                "car": {"car_id": "honda_accord_2025"},
            }
        ],
        verified=True,
        slot_schema={
            "type": "function",
            "function": {"name": "tool", "parameters": object_schema},
        },
    )

    tool2 = Tool(dummy_func, "toolname", "desc", [])
    tool2.slots = [slot2]
    tool2._validate_and_clean_slot_values()

    # Optional field should be set to None
    assert tool2.slots[0].value[0]["car"]["car_insurance_type"] is None


def test_validate_and_clean_slot_values_type_conversion() -> None:
    """Test that validation performs type conversion."""
    object_schema = {
        "type": "object",
        "required": ["data"],
        "properties": {
            "data": {
                "type": "object",
                "required": ["count", "price", "active"],
                "properties": {
                    "count": {"type": "integer"},
                    "price": {"type": "number"},
                    "active": {"type": "boolean"},
                },
            }
        },
    }

    slot = Slot(
        name="data",
        type="object",
        description="Test data",
        required=True,
        value={"count": 42, "price": 99.99, "active": True},
        verified=True,
        slot_schema={
            "type": "function",
            "function": {"name": "tool", "parameters": object_schema},
        },
    )

    tool = Tool(dummy_func, "toolname", "desc", [])
    tool.slots = [slot]
    tool._validate_and_clean_slot_values()

    validated_value = tool.slots[0].value
    assert isinstance(validated_value["count"], int)
    assert isinstance(validated_value["price"], float)
    assert isinstance(validated_value["active"], bool)


def test_validate_and_clean_slot_values_missing_required_field() -> None:
    """Test that validation raises error for missing required fields."""
    from arklex.utils.logging.exceptions import ValidationError

    object_schema = {
        "type": "object",
        "required": ["vehicles_array"],
        "properties": {
            "vehicles_array": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["make", "model"],
                    "properties": {
                        "make": {"type": "string"},
                        "model": {"type": "string"},
                    },
                },
            }
        },
    }

    # Test case 1: Missing required 'model' field completely
    slot = Slot(
        name="vehicles_array",
        type="array",
        description="Array of vehicles",
        required=True,
        value=[{"make": "Toyota"}],
        verified=True,
        slot_schema={
            "type": "function",
            "function": {"name": "tool", "parameters": object_schema},
        },
    )

    tool = Tool(dummy_func, "toolname", "desc", [])
    tool.slots = [slot]

    try:
        tool._validate_and_clean_slot_values()
        raise AssertionError("Expected ValidationError to be raised for missing field")
    except ValidationError as e:
        assert "Failed to validate slot values" in str(e)

    # Test case 2: Required field is None (should throw error)
    slot2 = Slot(
        name="vehicles_array",
        type="array",
        description="Array of vehicles",
        required=True,
        value=[{"make": "Toyota", "model": None}],
        verified=True,
        slot_schema={
            "type": "function",
            "function": {"name": "tool", "parameters": object_schema},
        },
    )

    tool2 = Tool(dummy_func, "toolname", "desc", [])
    tool2.slots = [slot2]

    try:
        tool2._validate_and_clean_slot_values()
        raise AssertionError("Expected ValidationError to be raised for None value")
    except ValidationError as e:
        assert "Failed to validate slot values" in str(e)


def test_validate_and_clean_slot_values_empty_string_allowed() -> None:
    """Test that validation allows empty strings for required fields.

    According to the function docstring: 'allow empty values for required fields (e.g. ""), but not None'
    """
    object_schema = {
        "type": "object",
        "required": ["user_info"],
        "properties": {
            "user_info": {
                "type": "object",
                "required": ["name", "email", "phone"],
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"},
                },
            }
        },
    }

    # Test with empty strings for required fields (should pass)
    slot = Slot(
        name="user_info",
        type="object",
        description="User information",
        required=True,
        value={
            "name": "John Doe",
            "email": "",  # Empty string should be allowed
            "phone": "",  # Empty string should be allowed
        },
        verified=True,
        slot_schema={
            "type": "function",
            "function": {"name": "tool", "parameters": object_schema},
        },
    )

    tool = Tool(dummy_func, "toolname", "desc", [])
    tool.slots = [slot]

    # This should not raise an error - empty strings are allowed
    tool._validate_and_clean_slot_values()

    # Verify the values are preserved
    validated_value = tool.slots[0].value
    assert validated_value["name"] == "John Doe"
    assert validated_value["email"] == ""
    assert validated_value["phone"] == ""


def test_validate_and_clean_slot_values_multiple_slots() -> None:
    """Test validation with multiple slots."""
    object_schema1 = {
        "type": "object",
        "required": ["name"],
        "properties": {"name": {"type": "string"}},
    }

    object_schema2 = {
        "type": "object",
        "required": ["count"],
        "properties": {"count": {"type": "integer"}},
    }

    slot1 = Slot(
        name="name",
        type="string",
        description="Name",
        required=True,
        value="John",
        verified=True,
        slot_schema={
            "type": "function",
            "function": {"name": "tool", "parameters": object_schema1},
        },
    )

    slot2 = Slot(
        name="count",
        type="integer",
        description="Count",
        required=True,
        value=42,
        verified=True,
        slot_schema={
            "type": "function",
            "function": {"name": "tool", "parameters": object_schema2},
        },
    )

    tool = Tool(dummy_func, "toolname", "desc", [])
    tool.slots = [slot1, slot2]
    tool._validate_and_clean_slot_values()

    assert tool.slots[0].value == "John"
    assert tool.slots[1].value == 42


def test_validate_and_clean_slot_values_deeply_nested_objects() -> None:
    """Test validation with deeply nested object structures."""
    object_schema = {
        "type": "object",
        "required": ["data"],
        "properties": {
            "data": {
                "type": "object",
                "required": ["level1"],
                "properties": {
                    "level1": {
                        "type": "object",
                        "required": ["level2"],
                        "properties": {
                            "level2": {
                                "type": "object",
                                "required": ["value"],
                                "properties": {"value": {"type": "string"}},
                            }
                        },
                    }
                },
            }
        },
    }

    slot = Slot(
        name="data",
        type="object",
        description="Nested data",
        required=True,
        value={
            "level1": {"level2": {"value": "deep_value", "extra": "should_be_removed"}}
        },
        verified=True,
        slot_schema={
            "type": "function",
            "function": {"name": "tool", "parameters": object_schema},
        },
    )

    tool = Tool(dummy_func, "toolname", "desc", [])
    tool.slots = [slot]
    tool._validate_and_clean_slot_values()

    validated_value = tool.slots[0].value
    assert validated_value["level1"]["level2"]["value"] == "deep_value"
    assert "extra" not in validated_value["level1"]["level2"]


# Tests for _validate_and_clean_slot_values ------------------------------------------------------------
