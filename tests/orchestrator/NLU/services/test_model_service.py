"""Tests for the ModelService class.

This module contains tests for the ModelService class, focusing on
slot filling functionality and response processing.
"""

import pytest
from typing import Dict, Any, List
from arklex.orchestrator.NLU.services.model_service import ModelService


@pytest.fixture
def model_service():
    """Create a ModelService instance for testing."""
    model_config = {
        "model_type_or_path": "gpt-3.5-turbo",
        "llm_provider": "openai",
    }
    return ModelService(model_config)


@pytest.fixture
def sample_slots():
    """Create sample slot definitions for testing."""
    return [
        {
            "name": "location",
            "type": "string",
            "description": "The city or location where the event will take place",
            "required": True,
        },
        {
            "name": "date",
            "type": "string",
            "description": "The date of the event",
            "required": True,
        },
        {
            "name": "time",
            "type": "string",
            "description": "The time of the event",
            "required": False,
        },
    ]


@pytest.fixture
def sample_context():
    """Create sample context for testing."""
    return (
        "I want to schedule a meeting in New York on March 15th "
        "at 2:00 PM in the afternoon."
    )


def test_format_slot_input(model_service, sample_slots, sample_context):
    """Test the format_slot_input method."""
    user_prompt, system_prompt = model_service.format_slot_input(
        sample_slots, sample_context
    )

    # Check that both prompts are strings
    assert isinstance(user_prompt, str)
    assert isinstance(system_prompt, str)

    # Check that the prompts contain the necessary information
    assert "Context:" in user_prompt
    assert sample_context in user_prompt
    assert "Slot definitions:" in user_prompt
    assert "location" in user_prompt
    assert "date" in user_prompt
    assert "time" in user_prompt
    assert "required" in user_prompt
    assert "optional" in user_prompt

    # Check system prompt
    assert "slot filling assistant" in system_prompt.lower()
    assert "extract" in system_prompt.lower()
    assert "json format" in system_prompt.lower()


def test_process_slot_response(model_service, sample_slots):
    """Test the process_slot_response method."""
    # Sample response from the model
    response = '{"location": "New York", "date": "March 15th", "time": "2:00 PM"}'

    # Process the response
    updated_slots = model_service.process_slot_response(response, sample_slots)

    # Check that all slots were processed
    assert len(updated_slots) == len(sample_slots)

    # Check that values were correctly assigned
    for slot in updated_slots:
        if slot["name"] == "location":
            assert slot["value"] == "New York"
        elif slot["name"] == "date":
            assert slot["value"] == "March 15th"
        elif slot["name"] == "time":
            assert slot["value"] == "2:00 PM"


def test_process_slot_response_with_missing_values(model_service, sample_slots):
    """Test the process_slot_response method with missing values."""
    # Sample response with missing values
    response = '{"location": "New York", "date": "March 15th"}'

    # Process the response
    updated_slots = model_service.process_slot_response(response, sample_slots)

    # Check that all slots were processed
    assert len(updated_slots) == len(sample_slots)

    # Check that values were correctly assigned
    for slot in updated_slots:
        if slot["name"] == "location":
            assert slot["value"] == "New York"
        elif slot["name"] == "date":
            assert slot["value"] == "March 15th"
        elif slot["name"] == "time":
            assert slot["value"] is None


def test_process_slot_response_with_invalid_json(model_service, sample_slots):
    """Test the process_slot_response method with invalid JSON."""
    # Invalid JSON response
    response = "invalid json"

    # Check that it raises ValueError
    with pytest.raises(ValueError):
        model_service.process_slot_response(response, sample_slots)


def test_format_slot_input_with_enum_values(model_service):
    """Test the format_slot_input method with enum values."""
    slots = [
        {
            "name": "category",
            "type": "string",
            "description": "The category of the item",
            "required": True,
            "items": {"enum": ["electronics", "clothing", "books", "food"]},
        }
    ]
    context = "I want to buy some electronics."

    user_prompt, system_prompt = model_service.format_slot_input(slots, context)

    # Check that enum values are included in the prompt
    assert "Possible values:" in user_prompt
    assert "electronics" in user_prompt
    assert "clothing" in user_prompt
    assert "books" in user_prompt
    assert "food" in user_prompt
