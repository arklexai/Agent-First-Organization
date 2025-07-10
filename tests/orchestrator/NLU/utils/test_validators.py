"""Tests for NLU validators module.

This module tests the validation utilities for NLU operations including
intent detection, slot filling, and slot verification responses.
"""

from arklex.orchestrator.NLU.entities.slot_entities import Slot
from arklex.orchestrator.NLU.utils import validators


class TestValidateIntentResponse:
    """Test cases for validate_intent_response function."""

    def test_validate_intent_response_direct_match(self) -> None:
        """Test direct mapping match."""
        response = "greet"
        idx2intents_mapping = {"greet": "greet", "goodbye": "goodbye"}
        result = validators.validate_intent_response(response, idx2intents_mapping)
        assert result == "greet"

    def test_validate_intent_response_numeric_match(self) -> None:
        """Test numeric response mapping."""
        response = "1"
        idx2intents_mapping = {"1": "greet", "2": "goodbye"}
        result = validators.validate_intent_response(response, idx2intents_mapping)
        assert result == "greet"

    def test_validate_intent_response_numeric_not_in_mapping(self) -> None:
        """Test numeric response not in mapping returns 'others'."""
        response = "3"
        idx2intents_mapping = {"1": "greet", "2": "goodbye"}
        result = validators.validate_intent_response(response, idx2intents_mapping)
        assert result == "others"

    def test_validate_intent_response_invalid_response(self) -> None:
        """Test invalid response returns 'others'."""
        response = "invalid_response"
        idx2intents_mapping = {"greet": "greet", "goodbye": "goodbye"}
        result = validators.validate_intent_response(response, idx2intents_mapping)
        assert result == "others"

    def test_validate_intent_response_empty_string(self) -> None:
        """Test empty string response returns 'others'."""
        response = ""
        idx2intents_mapping = {"greet": "greet", "goodbye": "goodbye"}
        result = validators.validate_intent_response(response, idx2intents_mapping)
        assert result == "others"

    def test_validate_intent_response_whitespace(self) -> None:
        """Test response with whitespace is properly stripped."""
        response = "  greet  "
        idx2intents_mapping = {"greet": "greet", "goodbye": "goodbye"}
        result = validators.validate_intent_response(response, idx2intents_mapping)
        assert result == "greet"

    def test_validate_intent_response_exception_handling(self) -> None:
        """Test exception handling returns 'others'."""
        response = "greet"
        idx2intents_mapping = None  # This will cause an exception
        result = validators.validate_intent_response(response, idx2intents_mapping)
        assert result == "others"

    def test_validate_intent_response_empty_mapping(self) -> None:
        """Test with empty mapping."""
        response = "greet"
        idx2intents_mapping = {}
        result = validators.validate_intent_response(response, idx2intents_mapping)
        assert result == "others"


class TestValidateSlotResponse:
    """Test cases for validate_slot_response function."""

    def test_validate_slot_response_valid_json(self) -> None:
        """Test valid JSON response with slot values."""
        response = '{"slots": [{"name": "user_name", "value": "John"}]}'
        slots = [Slot(name="user_name", type="str")]
        result = validators.validate_slot_response(response, slots)
        assert len(result) == 1
        assert result[0].name == "user_name"
        assert result[0].value == "John"

    def test_validate_slot_response_multiple_slots(self) -> None:
        """Test response with multiple slots."""
        response = '{"slots": [{"name": "user_name", "value": "John"}, {"name": "age", "value": "25"}]}'
        slots = [Slot(name="user_name", type="str"), Slot(name="age", type="int")]
        result = validators.validate_slot_response(response, slots)
        assert len(result) == 2
        assert result[0].value == "John"
        assert result[1].value == "25"

    def test_validate_slot_response_slot_not_in_response(self) -> None:
        """Test when a slot is not found in response."""
        response = '{"slots": [{"name": "user_name", "value": "John"}]}'
        slots = [Slot(name="user_name", type="str"), Slot(name="age", type="int")]
        result = validators.validate_slot_response(response, slots)
        assert len(result) == 2
        assert result[0].value == "John"
        assert result[1].value is None

    def test_validate_slot_response_invalid_json_structure(self) -> None:
        """Test invalid JSON structure returns original slots."""
        response = '{"invalid": "structure"}'
        slots = [Slot(name="user_name", type="str")]
        result = validators.validate_slot_response(response, slots)
        assert result == slots

    def test_validate_slot_response_missing_slots_key(self) -> None:
        """Test response missing 'slots' key."""
        response = '{"data": [{"name": "user_name", "value": "John"}]}'
        slots = [Slot(name="user_name", type="str")]
        result = validators.validate_slot_response(response, slots)
        assert result == slots

    def test_validate_slot_response_json_decode_error(self) -> None:
        """Test JSON decode error returns original slots."""
        response = '{"slots": [{"name": "user_name", "value": "John"}'  # Invalid JSON
        slots = [Slot(name="user_name", type="str")]
        result = validators.validate_slot_response(response, slots)
        assert result == slots

    def test_validate_slot_response_general_exception(self) -> None:
        """Test general exception handling."""
        slots = [Slot(name="user_name", type="str")]

        # Test with a malformed response that will cause a general exception
        # when trying to access data["slots"] after JSON parsing
        malformed_response = '{"slots": null}'  # slots is null, not a list

        result = validators.validate_slot_response(malformed_response, slots)
        assert result == slots

    def test_validate_slot_response_empty_slots_list(self) -> None:
        """Test with empty slots list in response."""
        response = '{"slots": []}'
        slots = [Slot(name="user_name", type="str")]
        result = validators.validate_slot_response(response, slots)
        assert len(result) == 1
        assert result[0].value is None

    def test_validate_slot_response_none_value(self) -> None:
        """Test slot with None value."""
        response = '{"slots": [{"name": "user_name", "value": null}]}'
        slots = [Slot(name="user_name", type="str")]
        result = validators.validate_slot_response(response, slots)
        assert result[0].value is None


class TestValidateVerificationResponse:
    """Test cases for validate_verification_response function."""

    def test_validate_verification_response_valid_json(self) -> None:
        """Test valid JSON response with verification data."""
        response = '{"verification_needed": true, "thought": "Please confirm"}'
        result = validators.validate_verification_response(response)
        assert result == (True, "Please confirm")

    def test_validate_verification_response_false_verification(self) -> None:
        """Test verification_needed set to false."""
        response = '{"verification_needed": false, "thought": "No need to verify"}'
        result = validators.validate_verification_response(response)
        assert result == (False, "No need to verify")

    def test_validate_verification_response_missing_fields(self) -> None:
        """Test response with missing fields uses defaults."""
        response = '{"other_field": "value"}'
        result = validators.validate_verification_response(response)
        assert result == (False, "No need to verify")

    def test_validate_verification_response_invalid_json_structure(self) -> None:
        """Test invalid JSON structure returns defaults."""
        response = "[1, 2, 3]"  # List instead of dict
        result = validators.validate_verification_response(response)
        assert result == (False, "No need to verify")

    def test_validate_verification_response_json_decode_error(self) -> None:
        """Test JSON decode error returns defaults."""
        response = (
            '{"verification_needed": true, "thought": "Please confirm"'  # Invalid JSON
        )
        result = validators.validate_verification_response(response)
        assert result == (False, "No need to verify")

    def test_validate_verification_response_general_exception(self) -> None:
        """Test general exception handling."""

        # Test with a malformed response that will cause a general exception
        # when trying to access data.get() after JSON parsing
        malformed_response = "null"  # data is null, not a dict

        result = validators.validate_verification_response(malformed_response)
        assert result == (False, "No need to verify")

    def test_validate_verification_response_empty_dict(self) -> None:
        """Test empty dictionary response."""
        response = "{}"
        result = validators.validate_verification_response(response)
        assert result == (False, "No need to verify")

    def test_validate_verification_response_only_verification_needed(self) -> None:
        """Test response with only verification_needed field."""
        response = '{"verification_needed": true}'
        result = validators.validate_verification_response(response)
        assert result == (True, "No need to verify")

    def test_validate_verification_response_only_thought(self) -> None:
        """Test response with only thought field."""
        response = '{"thought": "Custom thought"}'
        result = validators.validate_verification_response(response)
        assert result == (False, "Custom thought")

    def test_validate_verification_response_string_values(self) -> None:
        """Test response with string values instead of boolean."""
        response = '{"verification_needed": "true", "thought": "Please confirm"}'
        result = validators.validate_verification_response(response)
        assert result == ("true", "Please confirm")  # String "true" is truthy

    def test_validate_verification_response_invalid_structure(self) -> None:
        """Test invalid structure returns defaults."""
        # Not a dict
        result = validators.validate_verification_response("not a dict")
        assert result == (False, "No need to verify")
        # Dict but missing fields
        import json

        s = json.dumps({"foo": "bar"})
        result = validators.validate_verification_response(s)
        assert result == (False, "No need to verify")
