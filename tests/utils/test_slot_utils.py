"""Tests for slot management and type handling module.

This module tests the slot management functionality including type mappings,
slot definitions, input/output formatting, and validation utilities.
"""

from typing import List
import pytest
from unittest.mock import MagicMock

from arklex.utils.slot import (
    TypeMapping,
    Slot,
    SlotInput,
    SlotInputList,
    Verification,
    structured_input_output,
    format_slotfiller_output,
    format_slot_output,
    validate_slot_values,
    convert_slot_values,
)


class TestTypeMapping:
    """Test cases for TypeMapping class."""

    def test_string_to_type_basic_types(self) -> None:
        """Test conversion of basic type strings."""
        assert TypeMapping.string_to_type("str") == str
        assert TypeMapping.string_to_type("int") == int
        assert TypeMapping.string_to_type("float") == float
        assert TypeMapping.string_to_type("bool") == bool

    def test_string_to_type_list_types(self) -> None:
        """Test conversion of list type strings."""
        assert TypeMapping.string_to_type("list[str]") == List[str]
        assert TypeMapping.string_to_type("list[int]") == List[int]
        assert TypeMapping.string_to_type("list[float]") == List[float]
        assert TypeMapping.string_to_type("list[bool]") == List[bool]

    def test_string_to_type_unsupported_type(self) -> None:
        """Test conversion of unsupported type string."""
        assert TypeMapping.string_to_type("unsupported") is None

    def test_string_to_type_empty_string(self) -> None:
        """Test conversion of empty string."""
        assert TypeMapping.string_to_type("") is None

    def test_string_to_type_none(self) -> None:
        """Test conversion of None."""
        assert TypeMapping.string_to_type(None) is None


class TestSlot:
    """Test cases for Slot class."""

    def test_slot_initialization_basic(self) -> None:
        """Test basic slot initialization."""
        slot = Slot(name="test_slot")
        assert slot.name == "test_slot"
        assert slot.type == "str"
        assert slot.value is None
        assert slot.enum == []
        assert slot.description == ""
        assert slot.prompt == ""
        assert slot.required is False
        assert slot.verified is False
        assert slot.items is None

    def test_slot_initialization_with_all_fields(self) -> None:
        """Test slot initialization with all fields."""
        slot = Slot(
            name="age",
            type="int",
            value=25,
            enum=[18, 25, 30, 35],
            description="User's age",
            prompt="What is your age?",
            required=True,
            verified=True,
            items={"min": 18, "max": 65},
        )
        assert slot.name == "age"
        assert slot.type == "int"
        assert slot.value == 25
        assert slot.enum == [18, 25, 30, 35]
        assert slot.description == "User's age"
        assert slot.prompt == "What is your age?"
        assert slot.required is True
        assert slot.verified is True
        assert slot.items == {"min": 18, "max": 65}

    def test_slot_with_different_types(self) -> None:
        """Test slots with different types."""
        str_slot = Slot(name="name", type="str", value="John")
        int_slot = Slot(name="age", type="int", value=25)
        float_slot = Slot(name="score", type="float", value=95.5)
        bool_slot = Slot(name="active", type="bool", value=True)
        list_slot = Slot(name="tags", type="list[str]", value=["tag1", "tag2"])

        assert str_slot.value == "John"
        assert int_slot.value == 25
        assert float_slot.value == 95.5
        assert bool_slot.value is True
        assert list_slot.value == ["tag1", "tag2"]

    def test_slot_copy(self) -> None:
        """Test slot copying functionality."""
        original = Slot(name="test", value="original")
        copied = original.copy()

        assert copied.name == original.name
        assert copied.value == original.value
        assert copied is not original

        # Modify copied slot
        copied.value = "modified"
        assert original.value == "original"  # Original unchanged
        assert copied.value == "modified"


class TestSlotInput:
    """Test cases for SlotInput class."""

    def test_slot_input_initialization(self) -> None:
        """Test slot input initialization."""
        slot_input = SlotInput(
            name="test",
            value="test_value",
            enum=["option1", "option2"],
            description="Test description",
        )
        assert slot_input.name == "test"
        assert slot_input.value == "test_value"
        assert slot_input.enum == ["option1", "option2"]
        assert slot_input.description == "Test description"

    def test_slot_input_with_none_values(self) -> None:
        """Test slot input with None values."""
        slot_input = SlotInput(name="test", value=None, enum=None, description="")
        assert slot_input.name == "test"
        assert slot_input.value is None
        assert slot_input.enum is None
        assert slot_input.description == ""


class TestSlotInputList:
    """Test cases for SlotInputList class."""

    def test_slot_input_list_initialization(self) -> None:
        """Test slot input list initialization."""
        slot_inputs = [
            SlotInput(name="slot1", value="value1", enum=None, description=""),
            SlotInput(name="slot2", value="value2", enum=None, description=""),
        ]
        slot_input_list = SlotInputList(slot_input_list=slot_inputs)
        assert len(slot_input_list.slot_input_list) == 2
        assert slot_input_list.slot_input_list[0].name == "slot1"
        assert slot_input_list.slot_input_list[1].name == "slot2"

    def test_slot_input_list_empty(self) -> None:
        """Test slot input list with empty list."""
        slot_input_list = SlotInputList(slot_input_list=[])
        assert len(slot_input_list.slot_input_list) == 0


class TestVerification:
    """Test cases for Verification class."""

    def test_verification_initialization(self) -> None:
        """Test verification initialization."""
        verification = Verification(
            thought="This value looks correct", verification_needed=False
        )
        assert verification.thought == "This value looks correct"
        assert verification.verification_needed is False

    def test_verification_with_verification_needed(self) -> None:
        """Test verification with verification needed."""
        verification = Verification(
            thought="This value needs verification", verification_needed=True
        )
        assert verification.thought == "This value needs verification"
        assert verification.verification_needed is True


class TestStructuredInputOutput:
    """Test cases for structured_input_output function."""

    def test_structured_input_output_basic(self) -> None:
        """Test basic structured input/output formatting."""
        slots = [
            Slot(name="name", type="str", description="User's name"),
            Slot(name="age", type="int", description="User's age"),
        ]
        input_format, output_type = structured_input_output(slots)

        assert isinstance(input_format, SlotInputList)
        assert len(input_format.slot_input_list) == 2
        assert input_format.slot_input_list[0].name == "name"
        assert input_format.slot_input_list[1].name == "age"
        assert hasattr(output_type, "__annotations__")

    def test_structured_input_output_empty_list(self) -> None:
        """Test structured input/output with empty slot list."""
        slots = []
        input_format, output_type = structured_input_output(slots)

        assert isinstance(input_format, SlotInputList)
        assert len(input_format.slot_input_list) == 0
        assert hasattr(output_type, "__annotations__")

    def test_structured_input_output_with_enums(self) -> None:
        """Test structured input/output with enum values."""
        slots = [
            Slot(
                name="category",
                type="str",
                enum=["A", "B", "C"],
                description="Category",
            ),
            Slot(name="priority", type="int", enum=[1, 2, 3], description="Priority"),
        ]
        input_format, output_type = structured_input_output(slots)

        assert len(input_format.slot_input_list) == 2
        assert input_format.slot_input_list[0].enum == ["A", "B", "C"]
        assert input_format.slot_input_list[1].enum == [1, 2, 3]


class TestFormatSlotfillerOutput:
    """Test cases for format_slotfiller_output function."""

    def test_format_slotfiller_output_valid_response(self) -> None:
        """Test formatting slotfiller output with valid response."""
        slots = [Slot(name="name", type="str"), Slot(name="age", type="int")]
        response = MagicMock()
        response.model_dump.return_value = {"name": "John", "age": 25}
        result = format_slotfiller_output(slots, response)

        assert len(result) == 2
        assert result[0].name == "name"
        assert result[0].value == "John"
        assert result[1].name == "age"
        assert result[1].value == 25

    def test_format_slotfiller_output_partial_response(self) -> None:
        """Test formatting slotfiller output with partial response."""
        slots = [Slot(name="name", type="str"), Slot(name="age", type="int")]
        response = MagicMock()
        response.model_dump.return_value = {"name": "John"}  # age missing
        result = format_slotfiller_output(slots, response)

        assert len(result) == 2
        assert result[0].name == "name"
        assert result[0].value == "John"
        assert result[1].name == "age"
        assert result[1].value is None

    def test_format_slotfiller_output_empty_response(self) -> None:
        """Test formatting slotfiller output with empty response."""
        slots = [Slot(name="name", type="str"), Slot(name="age", type="int")]
        response = MagicMock()
        response.model_dump.return_value = {}
        result = format_slotfiller_output(slots, response)

        assert len(result) == 2
        assert result[0].value is None
        assert result[1].value is None

    def test_format_slotfiller_output_none_response(self) -> None:
        """Test formatting slotfiller output with None response."""
        slots = [Slot(name="name", type="str"), Slot(name="age", type="int")]
        response = MagicMock()
        response.model_dump.return_value = {"name": None, "age": None}
        result = format_slotfiller_output(slots, response)

        assert len(result) == 2
        assert result[0].value is None
        assert result[1].value is None


class TestFormatSlotOutput:
    """Test cases for format_slot_output function."""

    def test_format_slot_output_valid_response(self) -> None:
        """Test formatting slot output with valid response."""
        slots = [Slot(name="name", type="str"), Slot(name="age", type="int")]
        response = {
            "slots": [{"name": "name", "value": "John"}, {"name": "age", "value": 25}]
        }
        result = format_slot_output(slots, response)

        assert len(result) == 2
        assert result[0].name == "name"
        assert result[0].value == "John"
        assert result[1].name == "age"
        assert result[1].value == 25

        # Also test with dict with slot names as keys
        response2 = {"name": "John", "age": 25}
        result2 = format_slot_output(slots, response2)
        assert len(result2) == 2
        assert result2[0].name == "name"
        assert result2[0].value == "John"
        assert result2[1].name == "age"
        assert result2[1].value == 25

    def test_format_slot_output_missing_slots_key(self) -> None:
        """Test formatting slot output with missing slots key."""
        slots = [Slot(name="name", type="str"), Slot(name="age", type="int")]
        response = {"other_key": "value"}
        result = format_slot_output(slots, response)

        assert len(result) == 2
        assert result[0].value is None
        assert result[1].value is None

    def test_format_slot_output_empty_slots_list(self) -> None:
        """Test formatting slot output with empty slots list."""
        slots = [Slot(name="name", type="str"), Slot(name="age", type="int")]
        response = {"slots": []}
        result = format_slot_output(slots, response)

        assert len(result) == 2
        assert result[0].value is None
        assert result[1].value is None


class TestValidateSlotValues:
    """Test cases for validate_slot_values function."""

    def test_validate_slot_values_all_valid(self) -> None:
        """Test validation with all valid slot values."""
        slots = [
            Slot(name="name", type="str", value="John", required=True),
            Slot(name="age", type="int", value=25, required=True),
        ]
        errors = validate_slot_values(slots)
        assert len(errors) == 0

    def test_validate_slot_values_missing_required(self) -> None:
        """Test validation with missing required values."""
        slots = [
            Slot(name="name", type="str", value=None, required=True),
            Slot(name="age", type="int", value=25, required=True),
        ]
        errors = validate_slot_values(slots)
        assert len(errors) == 1
        assert "name" in errors[0]

    def test_validate_slot_values_invalid_enum(self) -> None:
        """Test validation with invalid enum values."""
        slots = [
            Slot(
                name="category",
                type="str",
                value="invalid",
                enum=["A", "B", "C"],
                required=True,
            )
        ]
        errors = validate_slot_values(slots)
        assert len(errors) == 1
        assert "category" in errors[0]

    def test_validate_slot_values_valid_enum(self) -> None:
        """Test validation with valid enum values."""
        slots = [
            Slot(
                name="category",
                type="str",
                value="A",
                enum=["A", "B", "C"],
                required=True,
            )
        ]
        errors = validate_slot_values(slots)
        assert len(errors) == 0

    def test_validate_slot_values_empty_slots(self) -> None:
        """Test validation with empty slot list."""
        slots = []
        errors = validate_slot_values(slots)
        assert len(errors) == 0

    def test_validate_slot_values_optional_slots(self) -> None:
        """Test validation with optional slots."""
        slots = [
            Slot(name="name", type="str", value=None, required=False),
            Slot(name="age", type="int", value=None, required=False),
        ]
        errors = validate_slot_values(slots)
        assert len(errors) == 0


class TestConvertSlotValues:
    """Test cases for convert_slot_values function."""

    def test_convert_slot_values_basic_types(self) -> None:
        """Test conversion of basic types."""
        slots = [
            Slot(name="name", type="str", value="John"),
            Slot(name="age", type="integer", value="25"),
            Slot(name="score", type="float", value="95.5"),
            Slot(name="active", type="boolean", value="true"),
        ]
        result = convert_slot_values(slots)

        assert result[0].value == "John"  # str stays str
        assert result[1].value == 25  # str "25" -> int 25
        assert result[2].value == 95.5  # str "95.5" -> float 95.5
        assert result[3].value is True  # str "true" -> bool True

    def test_convert_slot_values_list_types(self) -> None:
        """Test conversion of list types (should remain unchanged)."""
        slots = [
            Slot(name="tags", type="list[str]", value=["tag1", "tag2"]),
            Slot(name="numbers", type="list[int]", value=["1", "2", "3"]),
        ]
        result = convert_slot_values(slots)

        assert result[0].value == ["tag1", "tag2"]
        assert result[1].value == ["1", "2", "3"]  # No conversion for lists

    def test_convert_slot_values_none_values(self) -> None:
        """Test conversion with None values."""
        slots = [
            Slot(name="name", type="str", value=None),
            Slot(name="age", type="integer", value=None),
        ]
        result = convert_slot_values(slots)

        assert result[0].value is None
        assert result[1].value is None

    def test_convert_slot_values_invalid_conversion(self) -> None:
        """Test conversion with invalid type conversions (should raise ValueError for int/float, but not for bool)."""
        slots = [
            Slot(name="age", type="integer", value="not_a_number"),
            Slot(name="score", type="float", value="not_a_float"),
            Slot(name="active", type="boolean", value="not_a_bool"),
        ]
        # Each should raise ValueError when convert_slot_values is called
        with pytest.raises(ValueError):
            convert_slot_values([slots[0]])
        with pytest.raises(ValueError):
            convert_slot_values([slots[1]])
        # For bool, should just set value to False
        result = convert_slot_values([slots[2]])
        assert result[0].value is False

    def test_convert_slot_values_empty_list(self) -> None:
        """Test conversion with empty slot list."""
        slots = []
        result = convert_slot_values(slots)
        assert len(result) == 0
