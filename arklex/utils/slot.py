"""Slot management and type handling for the Arklex framework.

This module provides functionality for managing slots in the conversation system,
including slot type definitions, value validation, and slot filling operations.
It includes classes for representing slots, handling type conversions, and managing
slot verification processes. The module supports various data types and provides
utilities for structured input/output handling in slot filling operations.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)


class TypeMapping:
    """Mapping between string type names and Python types.

    This class provides a mapping between string representations of types and their
    corresponding Python types, along with a method to convert between them.

    Attributes:
        STRING_TO_TYPE (Dict[str, Type]): Dictionary mapping string type names to Python types.
    """

    STRING_TO_TYPE: Dict[str, Type] = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list[str]": List[str],
        "list[int]": List[int],
        "list[float]": List[float],
        "list[bool]": List[bool],
    }

    @classmethod
    def string_to_type(cls, type_string: str) -> Type:
        """Convert a string representation to its corresponding Python type.

        Args:
            type_string (str): String representation of the type (e.g., 'str', 'int').

        Returns:
            Type: The corresponding Python type.
        """
        return cls.STRING_TO_TYPE.get(type_string)


class Slot(BaseModel):
    """Represents a slot in the conversation system.

    A slot is a named container for a value of a specific type, with optional
    validation rules and metadata.

    Attributes:
        name (str): The name of the slot.
        type (str): The type of the slot value (default: "str").
        value (Union[str, int, float, bool, List[str], None]): The current value of the slot.
        enum (Optional[List[Union[str, int, float, bool, None]]]): List of valid values.
        description (str): Description of the slot's purpose.
        prompt (str): Prompt to use when filling the slot.
        required (bool): Whether the slot must be filled.
        verified (bool): Whether the slot's value has been verified.
    """

    name: str
    type: str = Field(default="str")
    value: Union[str, int, float, bool, List[str], None] = Field(default=None)
    enum: Optional[List[Union[str, int, float, bool, None]]] = Field(default=[])
    description: str = Field(default="")
    prompt: str = Field(default="")
    required: bool = Field(default=False)
    verified: bool = Field(default=False)


class SlotInput(BaseModel):
    """Input structure for slot filling operations.

    Attributes:
        name (str): The name of the slot.
        value (Union[str, int, float, bool, List[str], None]): The current value.
        enum (Optional[List[Union[str, int, float, bool, None]]]): Valid values.
        description (str): Description of the slot's purpose.
    """

    name: str
    value: Union[str, int, float, bool, List[str], None]
    enum: Optional[List[Union[str, int, float, bool, None]]]
    description: str


class SlotInputList(BaseModel):
    """Container for a list of slot inputs.

    Attributes:
        slot_input_list (List[SlotInput]): List of slot inputs to process.
    """

    slot_input_list: List[SlotInput]


class Verification(BaseModel):
    """Verification result for a slot value.

    Attributes:
        thought (str): Reasoning behind the verification decision.
        verification_needed (bool): Whether additional verification is required.
    """

    thought: str
    verification_needed: bool


def structured_input_output(slots: List[Slot]) -> Tuple[SlotInputList, Type]:
    """Format slots for slot filling input and output.

    This function converts a list of slots into a structured format suitable for
    slot filling operations, including both input format and output type definition.

    Args:
        slots (List[Slot]): List of slots to format.

    Returns:
        Tuple[SlotInputList, Type]: A tuple containing:
            - SlotInputList: Formatted input structure
            - Type: Dynamic output type for slot filling results
    """
    input_slots = [
        SlotInput(
            name=slot.name,
            value=slot.value,
            enum=slot.enum,
            description=slot.description,
        )
        for slot in slots
    ]

    output_format = create_model(
        "DynamicSlotOutputs",
        **{
            slot.name: (Optional[TypeMapping.string_to_type(slot.type)], None)
            for slot in slots
        },
    )
    return SlotInputList(slot_input_list=input_slots), output_format


def format_slotfilling_output(slots: List[Slot], response: Any) -> List[Slot]:
    """Format the output of slot filling operations.

    This function updates the values of slots based on the response from a slot
    filling operation.

    Args:
        slots (List[Slot]): List of slots to update.
        response (Any): Response from the slot filling operation.

    Returns:
        List[Slot]: Updated list of slots with filled values.
    """
    logger.info(f"filled_slots: {response}")
    filled_slots = response.model_dump()
    for slot in slots:
        slot.value = filled_slots[slot.name]
    return slots
