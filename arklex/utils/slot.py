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
        """Convert a string representation to its corresponding Python type."""
        return cls.STRING_TO_TYPE.get(type_string)


class Slot(BaseModel):
    name: str
    type: str = Field(default="str")
    value: Union[str, int, float, bool, List[str], None] = Field(default=None)
    enum: Optional[List[Union[str, int, float, bool, None]]] = Field(default=[])
    description: str = Field(default="")
    prompt: str = Field(default="")
    required: bool = Field(default=False)
    verified: bool = Field(default=False)


class SlotInput(BaseModel):
    name: str
    value: Union[str, int, float, bool, List[str], None]
    enum: Optional[List[Union[str, int, float, bool, None]]]
    description: str


class SlotInputList(BaseModel):
    slot_input_list: List[SlotInput]


class Verification(BaseModel):
    thought: str
    verification_needed: bool


# format slots for slotfilling input and output
def structured_input_output(slots: List[Slot]) -> Tuple[SlotInputList, Type]:
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


# format slots after slotfilling
def format_slotfilling_output(slots: List[Slot], response: Any) -> List[Slot]:
    logger.info(f"filled_slots: {response}")
    filled_slots = response.model_dump()
    for slot in slots:
        slot.value = filled_slots[slot.name]
    return slots
