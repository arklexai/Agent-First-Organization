"""Tests for NLU formatters module.

This module tests the formatting utilities for NLU operations including
intent detection, slot filling, and slot verification prompt formatting.
"""

from unittest.mock import Mock

from arklex.orchestrator.NLU.entities.slot_entities import Slot
from arklex.orchestrator.NLU.utils.formatters import (
    format_intent_input,
    format_slot_input,
    format_verification_input,
)


class TestFormatIntentInput:
    """Test cases for format_intent_input function."""

    def test_format_intent_input_single_intent(self) -> None:
        """Test formatting with single intent definition."""
        intents = {
            "greeting": [
                {
                    "attribute": {
                        "definition": "User greets the system",
                        "sample_utterances": ["Hello", "Hi there", "Good morning"],
                    }
                }
            ]
        }
        chat_history = "User: Hello\nAssistant: Hi! How can I help you?"

        prompt, mapping = format_intent_input(intents, chat_history)

        assert "greeting" in prompt
        assert "User greets the system" in prompt
        assert "Hello" in prompt
        assert "Hi there" in prompt
        assert "Good morning" in prompt
        assert "1) greeting" in prompt
        assert mapping == {"1": "greeting"}

    def test_format_intent_input_multiple_intents(self) -> None:
        """Test formatting with multiple intent definitions."""
        intents = {
            "greeting": [
                {
                    "attribute": {
                        "definition": "User greets the system",
                        "sample_utterances": ["Hello", "Hi"],
                    }
                }
            ],
            "booking": [
                {
                    "attribute": {
                        "definition": "User wants to book something",
                        "sample_utterances": ["I want to book", "Can I reserve"],
                    }
                }
            ],
        }
        chat_history = "User: I want to book a table\nAssistant: Sure!"

        prompt, mapping = format_intent_input(intents, chat_history)

        assert "greeting" in prompt
        assert "booking" in prompt
        assert "User greets the system" in prompt
        assert "User wants to book something" in prompt
        assert "1) greeting" in prompt
        assert "2) booking" in prompt
        assert mapping == {"1": "greeting", "2": "booking"}

    def test_format_intent_input_multiple_definitions_per_intent(self) -> None:
        """Test formatting with multiple definitions for the same intent."""
        intents = {
            "booking": [
                {
                    "attribute": {
                        "definition": "Book a table",
                        "sample_utterances": ["Book table", "Reserve table"],
                    }
                },
                {
                    "attribute": {
                        "definition": "Book a room",
                        "sample_utterances": ["Book room", "Reserve room"],
                    }
                },
            ]
        }
        chat_history = "User: I need a room\nAssistant: I can help with that."

        prompt, mapping = format_intent_input(intents, chat_history)

        assert "booking__<0>" in prompt
        assert "booking__<1>" in prompt
        assert "Book a table" in prompt
        assert "Book a room" in prompt
        assert "1) booking__<0>" in prompt
        assert "2) booking__<1>" in prompt
        assert mapping == {"1": "booking__<0>", "2": "booking__<1>"}

    def test_format_intent_input_no_definitions(self) -> None:
        """Test formatting with intents that have no definitions."""
        intents = {
            "greeting": [{"attribute": {"definition": "", "sample_utterances": []}}]
        }
        chat_history = "User: Hello\nAssistant: Hi!"

        prompt, mapping = format_intent_input(intents, chat_history)

        assert "1) greeting" in prompt
        assert "greeting:" not in prompt  # No definition to show
        assert mapping == {"1": "greeting"}

    def test_format_intent_input_no_sample_utterances(self) -> None:
        """Test formatting with intents that have no sample utterances."""
        intents = {
            "greeting": [
                {
                    "attribute": {
                        "definition": "User greets the system",
                        "sample_utterances": [],
                    }
                }
            ]
        }
        chat_history = "User: Hello\nAssistant: Hi!"

        prompt, mapping = format_intent_input(intents, chat_history)

        assert "User greets the system" in prompt
        # The definition should still appear in the Intent Definitions section
        assert "1) greeting: User greets the system" in prompt
        assert mapping == {"1": "greeting"}

    def test_format_intent_input_mixed_content(self) -> None:
        """Test formatting with mixed content (some with definitions, some without)."""
        intents = {
            "greeting": [
                {
                    "attribute": {
                        "definition": "User greets the system",
                        "sample_utterances": ["Hello", "Hi"],
                    }
                }
            ],
            "unknown": [{"attribute": {"definition": "", "sample_utterances": []}}],
        }
        chat_history = "User: Hello\nAssistant: Hi!"

        prompt, mapping = format_intent_input(intents, chat_history)

        assert "User greets the system" in prompt
        assert "Hello" in prompt
        assert "Hi" in prompt
        assert "1) greeting" in prompt
        assert "2) unknown" in prompt
        assert mapping == {"1": "greeting", "2": "unknown"}

    def test_format_intent_input_empty_intents(self) -> None:
        """Test formatting with empty intents dictionary."""
        intents = {}
        chat_history = "User: Hello\nAssistant: Hi!"

        prompt, mapping = format_intent_input(intents, chat_history)

        assert "Intent Definitions:" in prompt
        assert "Sample Utterances:" in prompt
        assert "Available Intents:" in prompt
        assert mapping == {}

    def test_format_intent_input_complex_chat_history(self) -> None:
        """Test formatting with complex chat history."""
        intents = {
            "booking": [
                {
                    "attribute": {
                        "definition": "Book a service",
                        "sample_utterances": ["I want to book", "Can I reserve"],
                    }
                }
            ]
        }
        chat_history = """User: Hi, I need help
Assistant: Hello! I'm here to help you.
User: I want to book a table for tonight
Assistant: I can help you with that booking."""

        prompt, mapping = format_intent_input(intents, chat_history)

        assert "Book a service" in prompt
        assert "I want to book" in prompt
        assert "Can I reserve" in prompt
        assert chat_history in prompt
        assert mapping == {"1": "booking"}

    def test_format_intent_input_missing_attribute_keys(self) -> None:
        """Test formatting with missing attribute keys."""
        intents = {"greeting": [{"attribute": {}}]}
        chat_history = "User: Hello\nAssistant: Hi!"

        prompt, mapping = format_intent_input(intents, chat_history)

        assert "1) greeting" in prompt
        assert mapping == {"1": "greeting"}

    def test_format_intent_input_missing_attribute(self) -> None:
        """Test formatting with missing attribute."""
        intents = {"greeting": [{}]}
        chat_history = "User: Hello\nAssistant: Hi!"

        prompt, mapping = format_intent_input(intents, chat_history)

        assert "1) greeting" in prompt
        assert mapping == {"1": "greeting"}


class TestFormatSlotInput:
    """Test cases for format_slot_input function."""

    def test_format_slot_input_basic(self) -> None:
        """Test basic slot input formatting."""
        slots = [
            Slot(name="name", description="User's full name"),
            Slot(name="age", description="User's age"),
        ]
        context = "User: My name is John and I'm 25 years old."
        result = format_slot_input(slots, context)
        assert "Given the following context and slot requirements" in result
        assert "name: User's full name" in result
        assert "age: User's age" in result
        assert context in result
        assert "JSON format" in result
        assert "slots" in result

    def test_format_slot_input_with_required_slots(self) -> None:
        """Test slot input formatting with required slots."""
        slots = [
            Slot(name="name", description="User's full name", required=True),
            Slot(name="email", description="User's email address"),
        ]
        context = "User: My name is John."
        result = format_slot_input(slots, context)
        assert "name: User's full name (required)" in result
        assert "email: User's email address" in result
        assert context in result

    def test_format_slot_input_with_type_information(self) -> None:
        """Test slot input formatting with type information."""
        slots = [
            Slot(name="age", description="User's age", required=True, type="integer")
        ]
        context = "User: I am 25 years old."
        result = format_slot_input(slots, context)
        assert "age: User's age (required) (type: integer)" in result
        assert context in result

    def test_format_slot_input_empty_slots(self) -> None:
        """Test slot input formatting with empty slots list."""
        slots = []
        context = "User: Hello"
        result = format_slot_input(slots, context)
        assert "Given the following context and slot requirements" in result
        assert context in result
        assert "Required Slots:" in result

    def test_format_slot_input_different_type(self) -> None:
        """Test slot input formatting with different type parameter."""
        slots = [Slot(name="name", description="User's name")]
        context = "User: My name is John."
        result = format_slot_input(slots, context, type="document")
        assert "Given the following context and slot requirements" in result
        assert context in result


class TestFormatVerificationInput:
    """Test cases for format_verification_input function."""

    def test_format_verification_input_basic(self) -> None:
        """Test basic verification input formatting."""
        slot = {
            "name": "email",
            "description": "User's email address",
            "value": "john@example.com",
            "type": "string",
        }
        chat_history = "User: My email is john@example.com\nAssistant: Thank you!"

        result = format_verification_input(slot, chat_history)

        assert "Given the following slot and chat history" in result
        assert "Name: email" in result
        assert "Description: User's email address" in result
        assert "Value: john@example.com" in result
        assert "Type: string" in result
        assert chat_history in result
        assert "verification_needed" in result
        assert "thought" in result

    def test_format_verification_input_missing_value(self) -> None:
        """Test verification input formatting with missing value."""
        slot = {
            "name": "email",
            "description": "User's email address",
            "type": "string",
        }
        chat_history = "User: I don't have an email\nAssistant: No problem."

        result = format_verification_input(slot, chat_history)

        assert "Value: Not provided" in result
        assert "Type: string" in result

    def test_format_verification_input_missing_type(self) -> None:
        """Test verification input formatting with missing type."""
        slot = {
            "name": "email",
            "description": "User's email address",
            "value": "john@example.com",
        }
        chat_history = "User: My email is john@example.com\nAssistant: Thank you!"

        result = format_verification_input(slot, chat_history)

        assert "Value: john@example.com" in result
        assert "Type: Not specified" in result

    def test_format_verification_input_empty_value(self) -> None:
        """Test verification input formatting with empty value."""
        slot = {
            "name": "email",
            "description": "User's email address",
            "value": "",
            "type": "string",
        }
        chat_history = "User: I don't have an email\nAssistant: No problem."

        result = format_verification_input(slot, chat_history)

        assert "Value: " in result  # Empty value should be shown as empty

    def test_format_verification_input_none_value(self) -> None:
        """Test verification input formatting with None value."""
        slot = {
            "name": "email",
            "description": "User's email address",
            "value": None,
            "type": "string",
        }
        chat_history = "User: I don't have an email\nAssistant: No problem."

        result = format_verification_input(slot, chat_history)

        assert "Value: None" in result

    def test_format_verification_input_complex_chat_history(self) -> None:
        """Test verification input formatting with complex chat history."""
        slot = {
            "name": "phone",
            "description": "User's phone number",
            "value": "555-1234",
            "type": "string",
        }
        chat_history = """User: Hi, I need to update my contact information
Assistant: I can help you with that. What information would you like to update?
User: My phone number is 555-1234
Assistant: I've updated your phone number to 555-1234. Is there anything else?"""

        result = format_verification_input(slot, chat_history)

        assert "Name: phone" in result
        assert "Description: User's phone number" in result
        assert "Value: 555-1234" in result
        assert "Type: string" in result
        assert chat_history in result

    def test_format_verification_input_json_format_instruction(self) -> None:
        """Test that the JSON format instruction is included."""
        slot = {
            "name": "email",
            "description": "User's email address",
            "value": "john@example.com",
            "type": "string",
        }
        chat_history = "User: My email is john@example.com"

        result = format_verification_input(slot, chat_history)

        assert (
            "JSON format Only without any markdown formatting or code blocks" in result
        )
        assert '"verification_needed": true/false' in result
        assert '"thought": "reasoning for the decision"' in result


class TestPrivateFunctions:
    """Test cases for private functions (indirectly through public functions)."""

    def test_format_slot_description_through_slot_input(self) -> None:
        """Test _format_slot_description through format_slot_input."""
        slots = [
            Slot(name="age", description="User's age", required=True, type="integer")
        ]
        context = "User: I am 25 years old."
        result = format_slot_input(slots, context)
        assert "age: User's age (required) (type: integer)" in result

    def test_format_slot_description_optional_slot(self) -> None:
        """Test _format_slot_description with optional slot."""
        slots = [
            Slot(name="age", description="User's age", required=False, type="integer")
        ]
        context = "User: I am 25 years old."
        result = format_slot_input(slots, context)
        assert "age: User's age (type: integer)" in result
        assert "(required)" not in result

    def test_format_slot_description_no_type(self) -> None:
        """Test _format_slot_description without type."""
        slots = [Slot(name="age", description="User's age", required=True)]
        context = "User: I am 25 years old."
        result = format_slot_input(slots, context)
        assert "age: User's age (required) (type: str)" in result

    def test_format_slot_prompt_through_slot_input(self) -> None:
        """Test _format_slot_prompt through format_slot_input."""
        # Create a mock object that behaves like SlotInputList but is iterable
        mock_slots = Mock()
        mock_slots.__iter__ = Mock(
            return_value=iter(
                [
                    Mock(
                        name="name",
                        description="User's name",
                        required=False,
                        type=None,
                    )
                ]
            )
        )

        context = "User: My name is John."

        result = format_slot_input(mock_slots, context)

        # This should trigger _format_slot_prompt
        assert "Given the following context and slot requirements" in result
        assert "extract the values for each slot" in result
        assert "Required Slots:" in result
        assert "JSON format" in result
        assert '"slots": [' in result
        assert '"name": "slot_name"' in result
        assert '"value": "extracted_value"' in result
