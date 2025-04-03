from pydantic import BaseModel
from typing import Union, List, Dict, Type, Optional
import logging

logger = logging.getLogger(__name__)


class TypeMapping:
    STRING_TO_TYPE: Dict[str, Type] = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list
    }

    @classmethod
    def string_to_type(cls, type_string: str) -> Type:
        """Convert a string representation to its corresponding Python type."""
        return cls.STRING_TO_TYPE.get(type_string)


class Slot(BaseModel):
    name: str
    type: str
    value: Union[str, int, float, bool, List[str], None]
    enum: Optional[list[Union[str, int, float, bool, None]]]
    description: str
    prompt: str
    required: bool
    verified: bool

class SlotInput(BaseModel):
    name: str
    type: type
    value: Union[str, int, float, bool, List[str], None]
    enum: Optional[list[Union[str, int, float, bool, None]]]
    description: str

class SlotInputList(BaseModel):
    slot_input_list: list[SlotInput]

class SlotOutput(BaseModel):
    name: str
    value: Union[str, int, float, bool, List[str], None]

class SlotOutputGemini(BaseModel):
    name: str
    value: Union[str, int, float, bool, List[str]] = None

class SlotOutputList(BaseModel):
    slot_output_list: list[SlotOutput]

class SlotOutputListGemini(BaseModel):
    slot_output_list: list[SlotOutputGemini]

class Verification(BaseModel):
    thought: str
    verification_needed: bool


def format_slots(slots: list[dict]) -> list[Slot]:
    format_slots = []
    for slot in slots:
        format_slots.append(Slot(
            name=slot["name"],
            type=slot["type"],
            value=slot.get("value", None),
            enum=slot.get("enum", []),
            description=slot["description"],
            prompt=slot["prompt"],
            required=slot.get("required", False),
            verified=slot.get("verified", False)
        ))
    return format_slots


# format slots for slotfilling input
def format_slotfilling_input(slots: list[Slot]) -> SlotInputList:
    input_slots = []
    for slot in slots:
        input_slots.append(SlotInput(
            name=slot.name,
            type=TypeMapping.string_to_type(slot.type),
            value=slot.value,
            enum=slot.enum,
            description=slot.description
        ))
    return SlotInputList(slot_input_list=input_slots)


# format slots after slotfilling
def format_slotfilling_output(slots: list[Slot], response: SlotOutputList) -> list[Slot]:
    filled_slots = response.slot_output_list
    logger.info(f"filled_slots: {filled_slots}")
    for slot in slots:
        for filled_slot in filled_slots:
            if slot.name == filled_slot.name:
                slot.value = filled_slot.value
                break
    return slots
