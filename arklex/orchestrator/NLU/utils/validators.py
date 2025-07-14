"""Response validation utilities for NLU.

This module provides utilities for validating and processing responses
from NLU operations, ensuring data integrity and proper error handling.
It includes validators for intent detection, slot filling, and slot
verification responses.

Key features:
- Response structure validation
- Error handling and logging
- Default value handling
- JSON parsing and validation
"""

import json

from arklex.orchestrator.NLU.entities.slot_entities import Slot
from arklex.utils.logging_utils import LOG_MESSAGES, LogContext

log_context = LogContext(__name__)


def validate_intent_response(response: str, idx2intents_mapping: dict[str, str]) -> str:
    """Validate and process intent detection response.

    This function validates and processes the raw response from an intent
    detection operation, ensuring it maps to a valid intent name. It handles
    various response formats and provides appropriate fallbacks.

    Args:
        response: Raw response string from the model
        idx2intents_mapping: Dictionary mapping response indices to intent names

    Returns:
        The validated intent name, or "others" if validation fails

    Note:
        The function handles several cases:
        - Direct mapping match
        - Numeric response mapping
        - Invalid or malformed responses
    """
    try:
        # Clean and normalize response
        response = response.strip()

        # Check if response is in mapping
        if response in idx2intents_mapping:
            return idx2intents_mapping[response]

        # Try to extract number from response
        if response.isdigit():
            return idx2intents_mapping.get(response, "others")

        log_context.warning(
            LOG_MESSAGES["WARNING"]["VALIDATION_ERROR"].format(
                error=f"Invalid intent response: {response}"
            )
        )
        return "others"
    except Exception as e:
        log_context.error(
            LOG_MESSAGES["ERROR"]["VALIDATION_ERROR"].format(error=str(e)), exc_info=e
        )
        return "others"


def validate_slot_response(response: str, slots: list[Slot]) -> list[Slot]:
    """Validate and process slot filling response.

    This function validates and processes the raw response from a slot
    filling operation, ensuring it contains valid slot values and
    maintaining the original slot structure.

    Args:
        response: Raw JSON response string from the model
        slots: List of original slots to be filled

    Returns:
        List of filled slots with validated values

    Note:
        The function handles several cases:
        - Valid JSON response with slot values
        - Invalid JSON structure
        - Missing or malformed slot values
        - JSON parsing errors
    """
    try:
        # Parse JSON response
        data = json.loads(response)

        # Validate response structure
        if not isinstance(data, dict) or "slots" not in data:
            log_context.warning(
                LOG_MESSAGES["WARNING"]["VALIDATION_ERROR"].format(
                    error="Invalid slot response structure"
                )
            )
            return slots

        # Process filled slots
        filled_slots = []
        for slot in slots:
            filled_slot = slot.copy()

            # Find matching slot in response
            for filled in data["slots"]:
                if filled["name"] == slot.name:
                    filled_slot.value = filled.get("value")
                    break

            filled_slots.append(filled_slot)

        return filled_slots
    except json.JSONDecodeError as e:
        log_context.error(
            LOG_MESSAGES["ERROR"]["VALIDATION_ERROR"].format(
                error=f"Error parsing slot response JSON: {str(e)}"
            ),
            exc_info=e,
        )
        return slots
    except Exception as e:
        log_context.error(
            LOG_MESSAGES["ERROR"]["VALIDATION_ERROR"].format(error=str(e)), exc_info=e
        )
        return slots


def validate_verification_response(response: str) -> tuple[bool, str]:
    """Validate and process slot verification response.

    This function validates and processes the raw response from a slot
    verification operation, ensuring it contains valid verification
    decisions and reasoning.

    Args:
        response: Raw JSON response string from the model

    Returns:
        A tuple containing:
            - bool: Whether verification is needed
            - str: Reasoning for the verification decision

    Note:
        The function handles several cases:
        - Valid JSON response with verification data
        - Invalid JSON structure
        - Missing verification fields
        - JSON parsing errors
    """
    try:
        # Parse JSON response
        data = json.loads(response)

        # Validate response structure
        if not isinstance(data, dict):
            log_context.warning(
                LOG_MESSAGES["WARNING"]["VALIDATION_ERROR"].format(
                    error="Invalid verification response structure"
                )
            )
            return False, "No need to verify"

        # Extract verification decision and reasoning
        verification_needed = data.get("verification_needed", False)
        thought = data.get("thought", "No need to verify")

        return verification_needed, thought
    except json.JSONDecodeError as e:
        log_context.error(
            LOG_MESSAGES["ERROR"]["VALIDATION_ERROR"].format(
                error=f"Error parsing verification response JSON: {str(e)}"
            ),
            exc_info=e,
        )
        return False, "No need to verify"
    except Exception as e:
        log_context.error(
            LOG_MESSAGES["ERROR"]["VALIDATION_ERROR"].format(error=str(e)), exc_info=e
        )
        return False, "No need to verify"
