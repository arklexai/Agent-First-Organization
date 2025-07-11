"""Slot utility functions and type handling for the Arklex framework.

This module provides utility functions for managing slots in conversation systems,
including type mapping, value validation, and slot formatting operations.

Key Components:
    - TypeMapping: Maps string type names to Python types for type-safe conversion
    - Slot formatting functions for input/output operations
    - Value validation and type conversion utilities
    - Dynamic Pydantic model generation for slot filling

Functions:
    - structured_input_output(): Creates structured input/output formats for slot filling
    - format_slotfiller_output(): Formats slot filler response output
    - format_slot_output(): Formats slot output from various response types
    - validate_slot_values(): Validates slot values against type and enum constraints
    - convert_slot_values(): Converts slot values to their appropriate Python types

The module supports basic Python types (str, int, float, bool) and their list variants,
with enum-based value validation and dynamic type checking capabilities.
"""

from pydantic import create_model

from arklex.orchestrator.NLU.entities.slot_entities import (
    Slot,
    SlotInput,
    SlotInputList,
)
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class TypeMapping:
    """Mapping between string type names and Python types.

    This class provides a mapping between string representations of types and their
    corresponding Python types, along with a method to convert between them.
    It supports basic types (str, int, float, bool) and their list variants.

    The class maintains a static mapping of type names to their corresponding Python types,
    enabling type-safe conversion between string representations and actual types.

    Attributes:
        STRING_TO_TYPE (Dict[str, Type]): Dictionary mapping string type names to Python types.
            Supported types include:
            - Basic types: str, int, float, bool
            - List types: list[str], list[int], list[float], list[bool]
    """

    STRING_TO_TYPE: dict[str, type] = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list[str]": list[str],
        "list[int]": list[int],
        "list[float]": list[float],
        "list[bool]": list[bool],
    }

    @classmethod
    def string_to_type(cls, type_string: str) -> type:
        """Convert a string representation to its corresponding Python type.

        This method looks up the Python type corresponding to the given string
        representation in the STRING_TO_TYPE mapping. It provides a type-safe way
        to convert between string type names and actual Python types.

        Args:
            type_string (str): String representation of the type (e.g., 'str', 'int').

        Returns:
            Type: The corresponding Python type, or None if the type is not supported.
        """
        return cls.STRING_TO_TYPE.get(type_string)


def structured_input_output(slots: list[Slot]) -> tuple[SlotInputList, type]:
    """Format slots for slot filling input and output.

    This function converts a list of slots into a structured format suitable for
    slot filling operations. It creates both the input format (SlotInputList) and
    a dynamic output type definition for the slot filling results.

    The function performs two main tasks:
    1. Creates a list of SlotInput objects from the provided slots
    2. Generates a dynamic Pydantic model for the output format

    The output type is dynamically created based on the slot names and types,
    ensuring type safety in the slot filling process.

    Args:
        slots (List[Slot]): List of slots to format.

    Returns:
        Tuple[SlotInputList, Type]: A tuple containing:
            - SlotInputList: Formatted input structure with slot information
            - Type: Dynamic output type for slot filling results, with fields
                   matching the slot names and types

    Example:
        slots = [
            Slot(name="name", type="str"),
            Slot(name="age", type="int")
        ]
        input_format, output_type = structured_input_output(slots)
        # input_format contains SlotInputList with slot information
        # output_type is a dynamic Pydantic model with name and age fields
    """
    # Convert slots to SlotInput format
    input_slots = [
        SlotInput(
            name=slot.name,
            value=slot.value,
            enum=slot.enum,
            description=slot.description,
        )
        for slot in slots
    ]

    # Create dynamic output type with fields matching slot names and types
    output_format = create_model(
        "DynamicSlotOutputs",
        **{
            slot.name: (TypeMapping.string_to_type(slot.type) | None, None)
            for slot in slots
        },
    )
    return SlotInputList(slot_input_list=input_slots), output_format


def format_slotfiller_output(slots: list[Slot], response: object) -> list[Slot]:
    """Format the output of slot filler.

    Args:
        slots (List[Slot]): List of slots to format
        response (object): Response from slot filler

    Returns:
        List[Slot]: Formatted slots
    """
    log_context.info(f"filled_slots: {response}")
    filled_slots = response.model_dump()
    for slot in slots:
        slot.value = filled_slots.get(slot.name, None)
    return slots


def format_slot_output(slots: list[Slot], response: object) -> list[Slot]:
    """Format slot output from response.

    Args:
        slots: List of slots to format
        response: Response to format

    Returns:
        List of formatted slots
    """
    updated_slots = []
    # If response has a 'slots' key, use that list
    if isinstance(response, dict) and "slots" in response:
        slot_dict = {item["name"]: item["value"] for item in response["slots"]}
        for slot in slots:
            slot.value = slot_dict.get(slot.name, None)
            updated_slots.append(slot)
    # If response is a dict with slot names as keys
    elif isinstance(response, dict):
        for slot in slots:
            slot.value = response.get(slot.name, None)
            updated_slots.append(slot)
    return updated_slots


def validate_slot_values(slots: list[Slot]) -> list[str]:
    """Validate slot values.

    Args:
        slots: List of slots to validate

    Returns:
        List of validation errors
    """
    errors = []
    for slot in slots:
        if slot.required and not slot.value:
            errors.append(f"Required slot '{slot.name}' is missing")
        elif slot.value and slot.type == "integer":
            try:
                int(slot.value)
            except ValueError:
                errors.append(f"Slot '{slot.name}' must be an integer")
        elif slot.value and slot.type == "float":
            try:
                float(slot.value)
            except ValueError:
                errors.append(f"Slot '{slot.name}' must be a float")
        elif slot.value and slot.type == "boolean":
            if slot.value.lower() not in ["true", "false"]:
                errors.append(f"Slot '{slot.name}' must be a boolean")
        elif slot.value and slot.enum and slot.value not in slot.enum:
            errors.append(f"Slot '{slot.name}' must be one of {slot.enum}")
    return errors


def convert_slot_values(slots: list[Slot]) -> list[Slot]:
    """Convert slot values to appropriate types.

    Args:
        slots: List of slots to convert

    Returns:
        List of converted slots
    """
    for slot in slots:
        if slot.value:
            if slot.type == "integer":
                slot.value = int(slot.value)
            elif slot.type == "float":
                slot.value = float(slot.value)
            elif slot.type == "boolean":
                slot.value = slot.value.lower() == "true"
    return slots
