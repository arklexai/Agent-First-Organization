from pydantic import BaseModel
from typing import Union, List, Dict, Type
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

    TYPE_TO_STRING: Dict[Type, str] = {
        str: "str",
        int: "int",
        float: "float",
        bool: "bool",
        list: "list"
    }

    @classmethod
    def string_to_type(cls, type_string: str) -> Type:
        """Convert a string representation to its corresponding Python type."""
        return cls.STRING_TO_TYPE.get(type_string)
    
    @classmethod
    def type_to_string(cls, type: Type) -> str:
        """Convert a Python type to its string representation."""
        return cls.TYPE_TO_STRING.get(type)


class Slot(BaseModel):
    name: str
    type: str
    value: Union[str, int, float, bool, List[str], None]
    description: str
    prompt: str
    required: bool
    verified: bool

class SlotInput(BaseModel):
    name: str
    type: type
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


# format slots for slotfilling input
def format_slotfilling_input(slots: list[dict]) -> SlotInputList:
    input_slots = []
    for slot in slots:
        input_slots.append(SlotInput(
            name=slot["name"],
            type=TypeMapping.string_to_type(slot["type"]),
            description=slot["description"]
        ))
    return SlotInputList(slot_input_list=input_slots)


# format slots after slotfilling
def format_slotfilling_output(slots: list[dict], response: SlotOutputList) -> list[Slot]:
    filled_slots = response.slot_output_list
    logger.info(f"filled_slots: {filled_slots}")
    for slot in slots:
        filled = False
        for filled_slot in filled_slots:
            if slot["name"] == filled_slot.name:
                slot["value"] = filled_slot.value
                filled = True
                break
        if not filled:
            slot["value"] = None
    slots = [Slot(**slot) for slot in slots]
    return slots
