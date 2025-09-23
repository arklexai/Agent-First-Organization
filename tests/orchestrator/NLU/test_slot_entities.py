"""Tests for NLU slot entities module."""

import pytest
from arklex.orchestrator.NLU.entities.slot_entities import (
    Slot,
    SlotInput,
    SlotInputList,
    Verification,
    convert_value_for_type,
)


class TestSlot:
    """Test cases for Slot class."""

    def test_slot_creation_basic(self):
        """Test basic slot creation."""
        slot = Slot(
            name="test_slot", type="string", required=True, description="Test slot"
        )

        assert slot.name == "test_slot"
        assert slot.type == "string"
        assert slot.required is True
        assert slot.description == "Test slot"
        assert slot.value is None
        assert slot.enum == []
        assert slot.verified is False

    def test_slot_creation_with_all_fields(self):
        """Test slot creation with all fields."""
        slot = Slot(
            name="test_slot",
            type="string",
            required=False,
            description="Test slot",
            value="default",
            enum=["option1", "option2"],
            prompt="Please provide a value",
        )

        assert slot.name == "test_slot"
        assert slot.type == "string"
        assert slot.required is False
        assert slot.description == "Test slot"
        assert slot.value == "default"
        assert slot.enum == ["option1", "option2"]
        assert slot.prompt == "Please provide a value"

    def test_slot_copy(self):
        """Test slot copying."""
        original_slot = Slot(
            name="test_slot", type="string", required=True, description="Test slot"
        )

        copied_slot = original_slot.copy()

        assert copied_slot.name == original_slot.name
        assert copied_slot.type == original_slot.type
        assert copied_slot.required == original_slot.required
        assert copied_slot.description == original_slot.description
        assert copied_slot is not original_slot


class TestSlotInput:
    """Test cases for SlotInput class."""

    def test_slot_input_creation(self):
        """Test slot input creation."""
        slot_input = SlotInput(
            name="test_slot",
            value="test_value",
            enum=["option1", "option2"],
            description="Test slot description",
        )

        assert slot_input.name == "test_slot"
        assert slot_input.value == "test_value"
        assert slot_input.enum == ["option1", "option2"]
        assert slot_input.description == "Test slot description"

    def test_slot_input_with_none_values(self):
        """Test slot input with None values."""
        slot_input = SlotInput(
            name="test_slot", value=None, enum=None, description="Test slot description"
        )

        assert slot_input.name == "test_slot"
        assert slot_input.value is None
        assert slot_input.enum is None
        assert slot_input.description == "Test slot description"


class TestSlotInputList:
    """Test cases for SlotInputList class."""

    def test_slot_input_list_creation(self):
        """Test slot input list creation."""
        slot_inputs = [
            SlotInput(
                name="slot1", value="value1", enum=["option1"], description="Slot 1"
            ),
            SlotInput(
                name="slot2", value="value2", enum=["option2"], description="Slot 2"
            ),
        ]

        slot_input_list = SlotInputList(slot_input_list=slot_inputs)

        assert len(slot_input_list.slot_input_list) == 2
        assert slot_input_list.slot_input_list[0].name == "slot1"
        assert slot_input_list.slot_input_list[1].name == "slot2"

    def test_slot_input_list_empty(self):
        """Test empty slot input list."""
        slot_input_list = SlotInputList(slot_input_list=[])

        assert len(slot_input_list.slot_input_list) == 0


class TestVerification:
    """Test cases for Verification class."""

    def test_verification_creation(self):
        """Test verification creation."""
        verification = Verification(thought="Valid value", verification_needed=False)

        assert verification.thought == "Valid value"
        assert verification.verification_needed is False

    def test_verification_with_verification_needed(self):
        """Test verification with verification needed."""
        verification = Verification(
            thought="Value needs verification", verification_needed=True
        )

        assert verification.thought == "Value needs verification"
        assert verification.verification_needed is True


class TestConvertValueForType:
    """Test cases for convert_value_for_type function."""

    def test_convert_value_string_type(self):
        """Test converting value to string type."""
        result = convert_value_for_type("test", "string")
        assert result == "test"

        result = convert_value_for_type(123, "string")
        assert result == "123"

    def test_convert_value_integer_type(self):
        """Test converting value to integer type."""
        result = convert_value_for_type("123", "integer")
        assert result == 123

        result = convert_value_for_type(45.6, "integer")
        assert result == 45

    def test_convert_value_number_type(self):
        """Test converting value to number type."""
        result = convert_value_for_type("45.6", "number")
        assert result == 45.6

        result = convert_value_for_type(123, "number")
        assert result == 123.0

    def test_convert_value_boolean_type(self):
        """Test converting value to boolean type."""
        result = convert_value_for_type("true", "boolean")
        assert result is True

        result = convert_value_for_type("false", "boolean")
        assert result is False

        result = convert_value_for_type(True, "boolean")
        assert result is True

        result = convert_value_for_type(False, "boolean")
        assert result is False

    def test_convert_value_unknown_type(self):
        """Test converting value with unknown type."""
        result = convert_value_for_type("test", "unknown")
        assert result == "test"

    def test_convert_value_none(self):
        """Test converting None value."""
        result = convert_value_for_type(None, "string")
        assert result == "None"  # None gets converted to string "None"

    def test_convert_value_conversion_error(self):
        """Test converting value that causes conversion error."""
        # This should handle the error gracefully
        result = convert_value_for_type("not_a_number", "integer")
        assert result == "not_a_number"
