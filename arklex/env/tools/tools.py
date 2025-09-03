"""Tool management for the Arklex framework.

This module provides functionality for managing tools, including
initialization, execution, and slot filling integration.
"""

import asyncio
import inspect
import json
import traceback
import uuid
from collections.abc import Callable
from typing import Any

from agents import FunctionTool, RunContextWrapper
from pydantic import BaseModel, Field, create_model

from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
    StatusEnum,
)
from arklex.orchestrator.NLU.core.slot import SlotFiller
from arklex.orchestrator.NLU.entities.slot_entities import Slot
from arklex.utils.exceptions import AuthenticationError, ToolExecutionError
from arklex.utils.logging_utils import LogContext
from arklex.utils.utils import PYTHON_TO_JSON_SCHEMA, format_chat_history

log_context = LogContext(__name__)


class ToolOutput(BaseModel):
    status: StatusEnum
    message_flow: str | None = None
    response: str | None = None
    slots: dict[str, list[Slot]] | None = None


# Type conversion mapping for slot values
TYPE_CONVERTERS = {
    "int": int,
    "float": float,
    "bool": lambda v: v
    if isinstance(v, bool)
    else (v.lower() == "true" if isinstance(v, str) else bool(v)),
    "str": lambda v: v if isinstance(v, dict | list) else str(v),
}


def register_tool(
    description: str,
    slots: list[dict[str, Any]] | None = None,
) -> Callable:
    """Register a tool with the Arklex framework.

    This decorator registers a function as a tool with the specified description, slots,
    outputs, and response flag. It handles path normalization and tool initialization.

    Args:
        desc (str): Description of the tool's functionality.
        slots (List[Dict[str, Any]], optional): List of slot definitions. Defaults to None.

    Returns:
        Callable: A function that creates and returns a Tool instance.
    """
    if slots is None:
        slots = []

    def inner(func: Callable) -> Callable:
        name: str = f"{func.__name__}"
        return Tool(func, name, description, slots)

    return inner


class Tool:
    """Base class for tools in the Arklex framework.

    This class provides the core functionality for tool execution, slot management,
    and state handling. It supports slot filling, parameter validation, and error
    handling during tool execution.

    Attributes:
        func (Callable): The function implementing the tool's functionality.
        name (str): The name of the tool.
        description (str): Description of the tool's functionality.
        output (List[str]): List of output field names.
        slotfillapi (Optional[SlotFiller]): Slot filling API instance.
        info (Dict[str, Any]): Tool information including parameters and requirements.
        slots (List[Slot]): List of slot instances.
        llm_config (Dict[str, Any]): Language model configuration.
    """

    def __init__(
        self,
        func: Callable,
        name: str,
        description: str,
        slots: list[dict[str, Any]],
    ) -> None:
        """Initialize a new Tool instance.

        Args:
            func (Callable): The function implementing the tool's functionality.
            name (str): The name of the tool.
            description (str): Description of the tool's functionality.
            slots (List[Dict[str, Any]]): List of slot definitions.
            outputs (List[str]): List of output field names.
            isResponse (bool): Whether the tool is a response tool.
        """
        self.func: Callable = func
        self.name: str = name
        self.description: str = description
        self.slots: list[Slot] = []
        self.llm_config: dict[str, Any] = {}
        self.slotfiller: SlotFiller | None = None
        self.auth = {}
        self.node_specific_data: dict[str, Any] = {}
        self.fixed_args = {}
        self.properties: dict[str, dict[str, Any]] = {}

        # Load initial slots
        if slots:
            self.load_slots(slots)

    def copy(self) -> "Tool":
        """Create a copy of this tool instance.

        Returns:
            Tool: A new Tool instance with the same configuration but independent state.
        """
        return Tool(
            func=self.func,
            name=self.name,
            description=self.description,
            slots=[i.model_dump() for i in self.slots],
        )

    def _format_slots(self, slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format slots for OpenAI tool definition.

        Args:
            slots: List of slot definitions

        Returns:
            List of formatted slot definitions for OpenAI
        """
        formatted_slots = []
        for slot in slots:
            formatted_slot = {
                "name": slot["name"],
                "type": slot["type"],
                "description": slot.get("description", ""),
                "required": slot.get("required", False),
            }

            # Handle enum values
            if "enum" in slot:
                formatted_slot["enum"] = slot["enum"]

            # Handle items for array types
            if "items" in slot:
                formatted_slot["items"] = slot["items"]

            # Handle group schema
            if slot.get("type") == "group" and "schema" in slot:
                formatted_slot["slot_schema"] = slot["schema"]

            formatted_slots.append(formatted_slot)

        return formatted_slots

    def get_info(self, slots: list[dict[str, Any]]) -> dict[str, Any]:
        """Get tool information including parameters and requirements.

        This method processes the slot definitions to create a structured
        representation of the tool's parameters and requirements.

        Args:
            slots (List[Dict[str, Any]]): List of slot definitions.

        Returns:
            Dict[str, Any]: Tool information including parameters and requirements.
        """
        self.properties = {}
        for slot in slots:
            self.properties[slot["name"]] = {
                k: v
                for k, v in slot.items()
                if k in ["type", "description", "prompt", "items"]
            }
        required: list[str] = [
            slot["name"] for slot in slots if slot.get("required", False)
        ]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": required,
                },
            },
        }

    def init_slotfiller(self, slotfiller_api: SlotFiller) -> None:
        """Initialize the slot filler for this tool.

        Args:
            slotfiller_api: API endpoint for slot filling
        """
        self.slotfiller = slotfiller_api

    def init_default_slots(self, default_slots: list[Slot]) -> dict[str, Any]:
        """Initializes the default slots as provided and returns a dictionary of slots which have been populated."""
        populated_slots: dict[str, Any] = {}
        for default_slot in default_slots:
            populated_slots[default_slot.name] = default_slot.value
            for slot in self.slots:
                if slot.name == default_slot.name:
                    slot.value = default_slot.value
                    slot.verified = True
        return populated_slots

    def _init_slots(
        self, state: OrchestratorState, all_slots: dict[str, list[Slot]]
    ) -> None:
        """Initialize slots with default values from the message state.

        This method processes default slots from the message state and updates
        the tool's slots with their values.

        Args:
            state (MessageState): The current message state.
        """
        default_slots: list[Slot] = all_slots.get("default_slots", [])
        log_context.info(f"Default slots are: {default_slots}")
        if not default_slots:
            return
        response: dict[str, Any] = self.init_default_slots(default_slots)
        state.function_calling_trajectory.append(
            {
                "role": "tool",
                "tool_call_id": str(uuid.uuid4()),
                "name": "default_slots",
                "content": json.dumps(response),
            }
        )

        log_context.info(f"Slots after initialization are: {self.slots}")

    def load_slots(self, slots: list[dict[str, Any]]) -> None:
        """Load and merge slots with existing slots.

        This method handles the merging of new slots with the tool's existing slots.
        If a slot with the same name exists in both places, the new version takes precedence.
        New slots are added to the existing slots.

        Args:
            slots (List[Dict[str, Any]]): List of slot definitions to merge with existing slots.

        Example:
            Existing slots:
                [Slot(name="param1", type="str", required=True),
                 Slot(name="param2", type="int", required=False)]

            New slots:
                [{"name": "param1", "type": "str", "required": False},
                 {"name": "param3", "type="bool", "required": True}]

            Result:
                [Slot(name="param1", type="str", required=False),  # Updated
                 Slot(name="param2", type="int", required=False),  # Preserved
                 Slot(name="param3", type="bool", required=True)]  # Added
        """
        if not slots:
            return

        # Process slots to handle schema/slot_schema mapping for group slots
        processed_slots = []
        for slot in slots:
            if slot.get("type") == "group" and "schema" in slot:
                # Create a copy with slot_schema instead of schema
                processed_slot = slot.copy()
                processed_slot["slot_schema"] = processed_slot.pop("schema")
                processed_slots.append(processed_slot)
            else:
                processed_slots.append(slot)

        # Create a dictionary of existing slots for easy lookup
        existing_slots_dict = {slot.name: slot for slot in self.slots}

        # Process new slots
        for new_slot in processed_slots:
            slot_name = new_slot["name"]
            if slot_name in existing_slots_dict:
                existing_slot = existing_slots_dict[slot_name]
                for key, value in new_slot.items():
                    # Handle schema/slot_schema mapping for group slots
                    if key == "schema" and existing_slot.type == "group":
                        existing_slot.slot_schema = value
                    else:
                        setattr(existing_slot, key, value)
            else:
                if new_slot.get("type") == "group":
                    # Handle both "schema" and "slot_schema" keys for backward compatibility
                    schema = new_slot.get("slot_schema") or new_slot.get("schema", [])
                    self.slots.append(
                        Slot(
                            name=new_slot["name"],
                            type="group",
                            slot_schema=schema,
                            required=new_slot.get("required", False),
                            repeatable=new_slot.get("repeatable", True),
                            prompt=new_slot.get("prompt", ""),
                            description=new_slot.get("description", ""),
                            value=[],
                            valueSource=new_slot.get("valueSource", None),
                        )
                    )

                else:
                    self.slots.append(Slot.model_validate(new_slot))

        # Update tool info with merged slots
        self.info = self.get_info([slot.model_dump() for slot in self.slots])

    def _convert_value(self, value: Any, type_str: str) -> Any:  # noqa: ANN401
        if value is None:
            return value

        if type_str.startswith("list["):
            if isinstance(value, str):
                return [v.strip() for v in value.split(",") if v.strip()]
            return list(value)

        converter = TYPE_CONVERTERS.get(type_str)
        if converter:
            try:
                return converter(value)
            except Exception:
                return value
        return value

    def _fill_slots_recursive(
        self, slots: list[Slot], chat_history_str: str
    ) -> list[Slot]:
        """Fill slots recursively, handling both group and regular slots.

        Args:
            slots: List of slots to fill
            chat_history_str: Formatted chat history string

        Returns:
            List of filled slots
        """
        filled_slots = []
        for slot in slots:
            if slot.type == "group":
                filled_slot = self._fill_group_slot(slot, chat_history_str)
            else:
                filled_slot = self._fill_regular_slot(slot, chat_history_str)
            filled_slots.append(filled_slot)
        return filled_slots

    def _fill_group_slot(self, slot: Slot, chat_history_str: str) -> Slot:
        """Fill a group slot with its schema-based structure.

        Args:
            slot: The group slot to fill
            chat_history_str: Formatted chat history string

        Returns:
            Filled group slot
        """

        temp_group_slot = self._create_temp_group_slot(slot)

        # Use slotfiller to fill the group as a whole
        filled = self.slotfiller.fill_slots(
            [temp_group_slot], chat_history_str, self.llm_config
        )
        group_value = filled[0].value

        # Parse and validate group value
        group_value = self._parse_and_validate_group_value(slot, group_value)

        # Apply valueSource logic to each item in the group
        group_value = self._apply_valuesource_to_group_items(slot, group_value)

        slot.value = group_value
        return slot

    def _fill_regular_slot(self, slot: Slot, chat_history_str: str) -> Slot:
        """Fill a regular (non-group) slot.

        Args:
            slot: The regular slot to fill
            chat_history_str: Formatted chat history string

        Returns:
            Filled regular slot
        """
        if getattr(slot, "repeatable", False):
            return self._fill_repeatable_regular_slot(slot, chat_history_str)
        else:
            return self._fill_non_repeatable_regular_slot(slot, chat_history_str)

    def _fill_repeatable_regular_slot(self, slot: Slot, chat_history_str: str) -> Slot:
        """Fill a repeatable regular slot.

        Args:
            slot: The repeatable regular slot to fill
            chat_history_str: Formatted chat history string

        Returns:
            Filled repeatable regular slot
        """
        repeatable_prompt = self._build_repeatable_regular_slot_prompt(slot)
        temp_slot = self._create_temp_repeatable_slot(slot, repeatable_prompt)

        filled = self.slotfiller.fill_slots(
            [temp_slot], chat_history_str, self.llm_config
        )
        slot_value = filled[0].value

        # Parse and validate repeatable slot value
        slot_value = self._parse_and_validate_repeatable_value(slot, slot_value)

        slot.value = [self._convert_value(val, slot.type) for val in slot_value]
        return slot

    def _fill_non_repeatable_regular_slot(
        self, slot: Slot, chat_history_str: str
    ) -> Slot:
        """Fill a non-repeatable regular slot.

        Args:
            slot: The non-repeatable regular slot to fill
            chat_history_str: Formatted chat history string

        Returns:
            Filled non-repeatable regular slot
        """
        filled = self.slotfiller.fill_slots([slot], chat_history_str, self.llm_config)
        slot.value = self._convert_value(filled[0].value, slot.type)
        return slot

    def _build_group_prompt(self, slot: Slot) -> str:
        """Build a schema-driven prompt for a group slot.

        Args:
            slot: The group slot

        Returns:
            Formatted prompt string
        """
        example_fields = []
        schema_lines = []

        for field in (
            slot.slot_schema
            if hasattr(slot, "slot_schema")
            and isinstance(slot.slot_schema, list | tuple)
            else []
        ):
            field_type = field.get("type", "str")
            field_repeatable = field.get("repeatable", False)
            example_value = self._get_example_value_for_type(field_type)

            if field_repeatable:
                example_fields.append(
                    f'"{field["name"]}": [{example_value}, "another_{field["name"]}", "third_{field["name"]}"]'
                )
            else:
                example_fields.append(f'"{field["name"]}": {example_value}')

            desc_or_prompt = field.get("description") or field.get("prompt") or ""
            schema_lines.append(
                f"- {field['name']} ({field_type}){' [REQUIRED]' if field.get('required', False) else ''}{' [REPEATABLE]' if field_repeatable else ''}: {desc_or_prompt}"
            )

        example_obj = "{" + ", ".join(example_fields) + "}"
        schema_str = "\n".join(schema_lines)

        # Add comprehensive explanation about repeatable fields
        prompt = (
            f"Please provide a list of dictionaries (objects), e.g. [{{'key': 'value'}}], each matching this schema:\n"
            f"{schema_str}\n"
            f"Example:\n[{example_obj}]\n"
            f"IMPORTANT: Each object must have ALL the fields above, with the correct type. "
            f"Do not add extra fields. Return a list of dicts, each matching the schema exactly. "
            f"IMPORTANT: The field name is just a key, don't use it to find for the value. The value you provide must match the field's description and prompt, even if the user never says it directly.\n"
            f"\n"
            f"IMPORTANT - REPEATABLE FIELDS:\n"
            f"- Individual fields within schemas can also be repeatable\n"
            f"- If a field has repeatable=True, it becomes an ARRAY of values\n"
            f"- If a field has repeatable=False, it becomes a SINGLE value\n"
            f'- Example: if \'term\' field is repeatable=True, use: "term": ["Fall 2024", "Spring 2025"]\n'
            f'- Example: if \'term\' field is repeatable=False, use: "term": "Fall 2024"\n'
            f"- ALWAYS check the example structure to see which fields are arrays vs single values\n"
            f"- For repeatable fields, extract ALL values from the conversation and put them in an array\n"
            f"- Even if there's only one value, if the field is repeatable, it must be in an array\n"
            f"\n"
            f"REQUIRED STRUCTURE FOR '{slot.name}':\n"
            f"- Follow the exact schema shown above\n"
            f"- Do NOT add extra fields not in the schema\n"
            f"- Do NOT change field names (e.g., use 'term' not 'semester', 'terms')\n"
            f"- Follow the exact structure shown above\n"
            f"- Pay attention to ARRAYS vs SINGLE VALUES in the example structure\n"
            f'- If the example shows an array (e.g., "term": ["example"]), use an array in your response\n'
            f'- If the example shows a single value (e.g., "term": "example"), use a single value\n'
            f"- REPEATABLE FIELDS MUST BE ARRAYS - even if there's only one value\n"
            f"- Extract data from the conversation to populate the values\n"
            f"- Return ONLY valid JSON, no explanations"
        )

        return prompt

    def _create_temp_group_slot(self, slot: Slot) -> Slot:
        """Create a temporary group slot for filling.

        Args:
            slot: The original group slot
            group_prompt: The prompt to add to the description

        Returns:
            Temporary group slot
        """
        return Slot(
            name=slot.name,
            type="group",
            value=slot.value if slot.value else [],
            description=slot.description,
            required=slot.required,
            slot_schema=slot.slot_schema,
            repeatable=slot.repeatable,
        )

    def _create_temp_repeatable_slot(self, slot: Slot, repeatable_prompt: str) -> Slot:
        """Create a temporary repeatable slot for filling.

        Args:
            slot: The original repeatable slot
            repeatable_prompt: The prompt to add to the description

        Returns:
            Temporary repeatable slot
        """
        return Slot(
            name=slot.name,
            type=slot.type,
            value=slot.value if slot.value else [],
            description=slot.description + " " + repeatable_prompt,
            required=slot.required,
            repeatable=getattr(slot, "repeatable", False),
        )

    def _parse_and_validate_group_value(
        self, slot: Slot, group_value: object
    ) -> list[dict[str, object]]:
        """Parse and validate a group value, ensuring it's a list of dictionaries.

        Args:
            slot: The group slot
            group_value: The raw group value

        Returns:
            Validated list of dictionaries

        Raises:
            ValueError: If the group value cannot be parsed or validated
        """
        # If the value is a string, try to parse as JSON
        if isinstance(group_value, str):
            log_context.debug(
                f"Attempting to parse group_value as JSON for slot '{slot.name}': {group_value}"
            )
            try:
                group_value = json.loads(group_value)
            except Exception as e:
                log_context.error(
                    f"Failed to parse group_value as JSON for slot '{slot.name}': {group_value}. Error: {e}"
                )
                raise ValueError(
                    f"Slot group '{slot.name}' did not return a valid JSON list of objects: {group_value}"
                ) from e

        # Enforce that the value is a list of dicts
        if not (
            isinstance(group_value, list)
            and all(isinstance(item, dict) for item in group_value)
        ):
            # Handle case where group_value is None or not a list
            if group_value is None:
                log_context.warning(
                    f"Slot group '{slot.name}' returned None, converting to empty list"
                )
                group_value = []
            elif isinstance(group_value, dict):
                log_context.warning(
                    f"Slot group '{slot.name}' returned a single dict, converting to list"
                )
                group_value = [group_value]
            else:
                log_context.error(
                    f"Slot group '{slot.name}' returned invalid format: {type(group_value)} - {group_value}"
                )
                raise ValueError(
                    f"Slot group '{slot.name}' must be a list of dicts, got: {group_value}"
                )

        return group_value

    def _parse_and_validate_repeatable_value(
        self, slot: Slot, slot_value: object
    ) -> list[object]:
        """Parse and validate a repeatable slot value, ensuring it's a list.

        Args:
            slot: The repeatable slot
            slot_value: The raw slot value

        Returns:
            Validated list of values

        Raises:
            ValueError: If the repeatable value cannot be parsed or validated
        """
        # Handle repeatable flag for regular slots
        if isinstance(slot_value, str):
            # Only try to parse as JSON if it looks like JSON (starts with [ or {)
            if slot_value.strip().startswith(("[", "{")):
                try:
                    slot_value = json.loads(slot_value)
                except Exception as e:
                    log_context.error(
                        f"Failed to parse repeatable slot '{slot.name}' as JSON: {slot_value}. Error: {e}"
                    )
                    raise ValueError(
                        f"Repeatable slot '{slot.name}' did not return a valid JSON array: {slot_value}"
                    ) from e
            else:
                # Treat as a regular string value
                slot_value = [slot_value]
        if not isinstance(slot_value, list):
            if slot_value is None:
                log_context.warning(
                    f"Repeatable slot '{slot.name}' returned None, converting to empty list"
                )
                slot_value = []
            else:
                log_context.warning(
                    f"Repeatable slot '{slot.name}' returned single value, converting to list"
                )
                slot_value = [slot_value]

        return slot_value

    def _apply_valuesource_to_group_items(
        self, slot: Slot, group_value: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply valueSource logic to each item in a group.

        Args:
            slot: The group slot
            group_value: List of dictionaries representing group items

        Returns:
            Updated group value with valueSource logic applied
        """
        for item in group_value:
            for field in (
                slot.slot_schema
                if hasattr(slot, "slot_schema")
                and isinstance(slot.slot_schema, list | tuple)
                else []
            ):
                field_name = field["name"]
                field_repeatable = field.get("repeatable", False)
                val_source = field.get("valueSource", "Prompt User")
                field_type = field.get("type", "str")
                schema_value = field.get("value", "")
                if val_source == "fixed":
                    item[field_name] = self._apply_fixed_valuesource(
                        field_repeatable, field_type, schema_value
                    )
                elif val_source == "default":
                    item[field_name] = self._apply_default_valuesource(
                        item.get(field_name), field_repeatable, field_type, schema_value
                    )
                else:  # Prompt User or missing
                    item[field_name] = self._apply_prompt_user_valuesource(
                        item.get(field_name), field_repeatable, field_type
                    )

        return group_value

    def _apply_fixed_valuesource(
        self, field_repeatable: bool, field_type: str, schema_value: object
    ) -> object:
        """Apply fixed valueSource logic.

        Args:
            field_repeatable: Whether the field is repeatable
            field_type: The field type
            schema_value: The schema value

        Returns:
            Processed value
        """
        if field_repeatable:
            # For repeatable fields, ensure it's an array
            if isinstance(schema_value, list):
                return [self._convert_value(val, field_type) for val in schema_value]
            else:
                return [self._convert_value(schema_value, field_type)]
        else:
            return self._convert_value(schema_value, field_type)

    def _apply_default_valuesource(
        self,
        current_value: object,
        field_repeatable: bool,
        field_type: str,
        schema_value: object,
    ) -> object:
        """Apply default valueSource logic.

        Args:
            current_value: The current value in the item
            field_repeatable: Whether the field is repeatable
            field_type: The field type
            schema_value: The schema value

        Returns:
            Processed value
        """
        if field_repeatable:
            # For repeatable fields, ensure it's an array
            if current_value in [None, ""] or not isinstance(current_value, list):
                if isinstance(schema_value, list):
                    return [
                        self._convert_value(val, field_type) for val in schema_value
                    ]
                else:
                    return [self._convert_value(schema_value, field_type)]
            else:
                return [self._convert_value(val, field_type) for val in current_value]
        else:
            if current_value in [None, ""]:
                log_context.info("Current value is None/empty, using schema_value")
                return self._convert_value(schema_value, field_type)
            else:
                log_context.info("Current value exists, converting it")
                return self._convert_value(current_value, field_type)

    def _apply_prompt_user_valuesource(
        self, current_value: object, field_repeatable: bool, field_type: str
    ) -> object:
        """Apply prompt user valueSource logic.

        Args:
            current_value: The current value in the item
            field_repeatable: Whether the field is repeatable
            field_type: The field type

        Returns:
            Processed value
        """
        if field_repeatable:
            # For repeatable fields, ensure it's an array
            if current_value in [None, ""] or not isinstance(current_value, list):
                return []
            else:
                return [self._convert_value(val, field_type) for val in current_value]
        else:
            return self._convert_value(current_value or "", field_type)

    def _get_example_value_for_type(self, field_type: str) -> str:
        """Get an example value for a given field type.

        Args:
            field_type: The field type

        Returns:
            Example value string
        """
        return {
            "str": '"example string"',
            "int": "123",
            "float": "12.34",
            "bool": "true",
        }.get(field_type, '"example"')

    def _is_missing_required(self, slots: list[Slot]) -> bool:
        for slot in slots:
            if slot.type == "group":
                # For group, check if at least one item exists if required
                if slot.required and (
                    not slot.value
                    or not isinstance(slot.value, list)
                    or len(slot.value) == 0
                ):
                    return True
                # For each item, check required fields

                for item in slot.value or []:
                    for field in (
                        slot.slot_schema
                        if hasattr(slot, "slot_schema")
                        and isinstance(slot.slot_schema, list | tuple)
                        else []
                    ):
                        field_repeatable = field.get("repeatable", False)
                        if field.get("required", False):
                            if field_repeatable:
                                # For repeatable fields, check if array exists and has values
                                if (
                                    field["name"] not in item
                                    or not isinstance(item[field["name"]], list)
                                    or len(item[field["name"]]) == 0
                                ):
                                    return True
                                # Check each value in the array
                                for val in item[field["name"]]:
                                    if val in [None, ""]:
                                        return True
                            else:
                                # For non-repeatable fields, check single value
                                if item.get(field["name"]) in [None, ""]:
                                    return True
            else:
                # Handle regular slots (non-group)
                if getattr(slot, "repeatable", False):
                    # For repeatable regular slots, check if at least one item exists if required
                    if slot.required and (
                        not slot.value
                        or not isinstance(slot.value, list)
                        or len(slot.value) == 0
                    ):
                        return True
                    # Check each value in the list
                    if slot.value and isinstance(slot.value, list):
                        for val in slot.value:
                            if val in [None, ""]:
                                return True
                else:
                    # For non-repeatable regular slots
                    if slot.required and (not slot.value or not slot.verified):
                        return True
        return False

    def _missing_slots_recursive(self, slots: list[Slot]) -> list[str]:
        missing = []
        for slot in slots:
            if slot.type == "group":
                if slot.required and (
                    not slot.value
                    or not isinstance(slot.value, list)
                    or len(slot.value) == 0
                ):
                    missing.append(slot.prompt)
                for idx, item in enumerate(slot.value or []):
                    for field in (
                        slot.slot_schema
                        if hasattr(slot, "slot_schema")
                        and isinstance(slot.slot_schema, list | tuple)
                        else []
                    ):
                        if field.get("required", False) and (
                            item.get(field["name"]) in [None, ""]
                        ):
                            missing.append(
                                f"{field.get('prompt', field['name'])} (group '{slot.name}' item {idx + 1})"
                            )

            else:
                # Handle regular slots (non-group)
                if getattr(slot, "repeatable", False):
                    # For repeatable regular slots, check list structure
                    if slot.required and (
                        not slot.value
                        or not isinstance(slot.value, list)
                        or len(slot.value) == 0
                    ):
                        missing.append(slot.prompt)
                    elif slot.value and isinstance(slot.value, list):
                        for idx, val in enumerate(slot.value):
                            if val in [None, ""]:
                                missing.append(f"{slot.prompt} (item {idx + 1})")
                else:
                    # For non-repeatable regular slots
                    if slot.required and (not slot.value or not slot.verified):
                        missing.append(slot.prompt)
        return missing

    def execute(
        self,
        state: OrchestratorState,
        all_slots: dict[str, list[Slot]],
        auth: dict[str, Any],
    ) -> tuple[OrchestratorState, ToolOutput]:
        """Execute the tool with the current state and fixed arguments.

        This method is a wrapper around _execute that handles the execution flow
        and state management.

        Args:
            state (MessageState): The current message state.
            **fixed_args (FixedArgs): Additional fixed arguments for the tool.

        Returns:
            MessageState: The updated message state after tool execution.
        """
        self.llm_config = state.bot_config.llm_config.model_dump()
        state, tool_output = self._execute(state, all_slots, auth)
        return state, tool_output

    def to_openai_tool_def(self) -> dict:
        """Convert the tool to an OpenAI tool definition.

        Returns:
            dict: The OpenAI tool definition.
        """
        parameters = {
            "type": "object",
            "properties": {},
            "required": [
                slot.name
                for slot in self.slots
                if slot.required and not (slot.verified and slot.value)
            ],
        }
        for slot in self.slots:
            # If the default slots have been populated and verified, then don't show the slot in the tool definition
            if slot.verified and slot.value:
                continue
            if slot.type == "group":
                # For group, define as array of objects with schema
                group_properties = {}
                group_required = []
                for field in (
                    slot.slot_schema
                    if hasattr(slot, "slot_schema")
                    and isinstance(slot.slot_schema, list | tuple)
                    else []
                ):
                    field_source = field.get("valueSource", "")
                    if field_source == "fixed":
                        continue
                    field_repeatable = field.get("repeatable", False)
                    if field_repeatable:
                        # If field is repeatable, make it an array
                        group_properties[field["name"]] = {
                            "type": PYTHON_TO_JSON_SCHEMA.get(field["type"], "string"),
                            "items": {
                                "type": PYTHON_TO_JSON_SCHEMA.get(
                                    field["type"], "string"
                                ),
                            },
                            "description": field.get("description", ""),
                        }
                    else:
                        # If field is not repeatable, make it a single value
                        group_properties[field["name"]] = {
                            "type": PYTHON_TO_JSON_SCHEMA.get(field["type"], "string"),
                            "description": field.get("description", ""),
                        }
                    if field.get("required", False):
                        group_required.append(field["name"])
                parameters["properties"][slot.name] = {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": group_properties,
                        "required": group_required,
                    },
                    "description": slot.description,
                }
            elif slot.items:
                parameters["properties"][slot.name] = {
                    "type": "array",
                    "items": slot.items,
                }
            else:
                # Handle regular slots (non-group)
                if getattr(slot, "repeatable", False):
                    # For repeatable regular slots, define as array
                    parameters["properties"][slot.name] = {
                        "type": "array",
                        "items": {
                            "type": PYTHON_TO_JSON_SCHEMA[slot.type],
                        },
                        "description": slot.description,
                    }
                else:
                    # For non-repeatable regular slots, define as single value
                    parameters["properties"][slot.name] = {
                        "type": PYTHON_TO_JSON_SCHEMA[slot.type],
                        "description": slot.description,
                    }
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
        }

    def to_openai_tool_def_v2(self) -> dict:
        parameters = {
            "type": "object",
            "properties": {},
            "required": [
                slot.name for slot in self.slots if getattr(slot, "required", False)
            ],
        }
        for slot in self.slots:
            if getattr(slot, "valueSource", None) == "fixed":
                continue
            parameters["properties"][slot.name] = slot.to_openai_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def to_openai_agents_function_tool(self) -> "FunctionTool":
        """Convert this Arklex tool to an OpenAI Agents FunctionTool.

        This method creates a FunctionTool that can be used with the OpenAI Agents SDK.
        It handles parameter conversion, schema generation, and function wrapping.

        Args:
            **fixed_args: Fixed arguments to be passed to the tool function.

        Returns:
            FunctionTool: An OpenAI Agents FunctionTool instance.

        Raises:
            ImportError: If OpenAI Agents SDK is not available.
        """
        # Create a Pydantic model for the tool parameters
        fields = {}
        for slot in self.slots:
            # Skip slots with fixed valueSource
            if getattr(slot, "valueSource", None) == "fixed":
                continue

            # Convert slot type to Python type
            py_type = self._slot_type_to_python_type(slot.type)

            # Set default value based on required status
            default = None if getattr(slot, "required", False) else ...

            # Create field metadata
            metadata = {"description": getattr(slot, "description", "")}

            # Add enum values if available
            if hasattr(slot, "enum") and slot.enum:
                metadata["enum"] = slot.enum

            fields[slot.name] = (py_type, Field(default, **metadata))

        # Create the Pydantic model class
        model_cls = create_model(f"{self.name}_InputModel", **fields)

        # Create the async wrapper function
        async def on_invoke(ctx: RunContextWrapper[Any], raw_args: str) -> str:
            log_context.info(f"on_invoke tool {self.name}, input: {raw_args}")

            try:
                # Parse the input arguments
                user_args = model_cls.model_validate_json(raw_args).model_dump()
                # Update slots with the parsed values
                for slot in self.slots:
                    if slot.name in user_args:
                        slot.value = user_args[slot.name]
                    if slot.type == "group":
                        for schema_obj in slot.slot_schema:
                            if (
                                schema_obj.get("valueSource", "") == "fixed"
                                and schema_obj.get("name") in user_args[slot.name]
                            ):
                                for filled_ob in user_args[slot.name]:
                                    if schema_obj.get("type") == "bool":
                                        filled_ob[schema_obj.get("name")] = (
                                            schema_obj.get("value", "").lower()
                                            == "true"
                                        )
                                    else:
                                        filled_ob[schema_obj.get("name")] = (
                                            schema_obj.get("value")
                                        )

                # Merge with fixed arguments
                merged_args = {
                    "slots": self.slots,
                    "auth": self.auth,
                    "node_specific_data": self.node_specific_data,
                    **self.fixed_args,
                    **user_args,
                }

                # Call the original function - handle both sync and async functions
                if inspect.iscoroutinefunction(self.func):
                    result = await self.func(**merged_args)
                else:
                    result = await asyncio.to_thread(self.func, **merged_args)
                log_context.info(f"on_invoke result: {result}")
                return result
            except Exception as e:
                log_context.error(f"Error executing tool {self.name}: {e}")
                log_context.exception(e)
                return f"Error: {str(e)}"

        return FunctionTool(
            name=self.name,
            description=self.description,
            params_json_schema=model_cls.model_json_schema(),
            on_invoke_tool=on_invoke,
            strict_json_schema=True,
        )

    def _slot_type_to_python_type(self, type_str: str) -> type:
        """Convert slot type string to Python type.

        Args:
            type_str: The slot type string.

        Returns:
            The corresponding Python type.
        """
        mapping = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "dict": dict,
            "list": list,
        }
        return mapping.get(type_str, Any)

    def __str__(self) -> str:
        """Get a string representation of the tool.

        Returns:
            str: A string representation of the tool.
        """
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        """Get a detailed string representation of the tool.

        Returns:
            str: A detailed string representation of the tool.
        """
        return f"{self.__class__.__name__}"

    def _execute(
        self,
        state: OrchestratorState,
        all_slots: dict[str, list[Slot]],
        auth: dict[str, Any],
    ) -> tuple[OrchestratorState, ToolOutput]:
        """Execute the tool with the current state and fixed arguments.

        This method handles slot filling, parameter validation, and tool execution.
        It manages the execution flow, error handling, and state updates.

        Args:
            state (MessageState): The current message state.
            **fixed_args (FixedArgs): Additional fixed arguments for the tool.

        Returns:
            MessageState: The updated message state after tool execution.
        """
        slot_verification: bool = False
        reason: str = ""
        tool_output: ToolOutput = ToolOutput(status=StatusEnum.INCOMPLETE)

        self.slots = [Slot.model_validate(slot) for slot in self.slots]
        # init slot values saved in default slots
        self._init_slots(state, all_slots)
        # do slotfilling (now with valueSource logic)
        chat_history_str: str = format_chat_history(state.function_calling_trajectory)
        slots: list[Slot] = self._fill_slots_recursive(self.slots, chat_history_str)
        log_context.info(f"{slots=}")

        # Check if any required slots are missing or unverified (including groups)
        missing_required = self._is_missing_required(slots)
        if missing_required:
            response, is_verification = self._handle_missing_required_slots(
                slots, chat_history_str
            )
            if response:
                tool_output.status = StatusEnum.INCOMPLETE
                if is_verification:
                    slot_verification = True
                    reason = response

        # Re-check if any required slots are still missing after verification
        missing_required = self._is_missing_required(slots)

        # if all required slots are filled and verified, then execute the function
        if not missing_required:
            log_context.info("all required slots filled")
            # Get all slot values, including optional ones that have values
            kwargs: dict[str, Any] = {}
            for slot in slots:
                # Always include the slot value, even if None
                kwargs[slot.name] = slot.value if slot.value is not None else ""

            # Get the function signature to check parameters
            sig = inspect.signature(self.func)

            # Only include the slots list if the target function accepts it
            if "slots" in sig.parameters:
                kwargs["slots"] = [
                    slot.model_dump() if hasattr(slot, "model_dump") else slot
                    for slot in slots
                ]

            combined_kwargs: dict[str, Any] = {
                **kwargs,
                "auth": auth,
                "node_specific_data": self.node_specific_data,
                **self.llm_config,
            }
            try:
                required_args = [
                    name
                    for name, param in sig.parameters.items()
                    if param.default == inspect.Parameter.empty
                ]
                # Ensure all required arguments are present
                for arg in required_args:
                    if arg not in kwargs:
                        kwargs[arg] = ""
                response = self.func(**combined_kwargs)
                if hasattr(response, "message_flow"):
                    tool_output.message_flow = response.message_flow
                elif hasattr(response, "response"):
                    tool_output.response = response.response
                else:
                    tool_output.message_flow = str(response)
                tool_output.status = StatusEnum.COMPLETE
            except ToolExecutionError as tee:
                log_context.error(traceback.format_exc())
                tool_output.message_flow = tee.extra_message
            except AuthenticationError as ae:
                log_context.error(traceback.format_exc())
                tool_output.message_flow = str(ae)
            except Exception as e:
                log_context.error(traceback.format_exc())
                tool_output.message_flow = str(e)
            log_context.info(f"Tool {self.name} output: {tool_output}")
            call_id: str = str(uuid.uuid4())
            state.function_calling_trajectory.append(
                {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": json.dumps(kwargs),
                                "name": self.name,
                            },
                            "id": call_id,
                            "type": "function",
                        }
                    ],
                    "function_call": None,
                }
            )
            state.function_calling_trajectory.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": self.name,
                    "content": tool_output.message_flow
                    if tool_output.message_flow
                    else tool_output.response,
                }
            )
            # Trajectory for multi-agent
            # state.function_calling_trajectory.append({
            #     'type': 'function_call',
            #     'id': "fc_" + call_id,
            #     'call_id': "call_" + call_id,
            #     'name': self.name,
            #     'arguments': json.dumps(kwargs)
            # })
            # state.function_calling_trajectory.append({
            #     "type": "function_call_output",
            #     "call_id": "call_" + call_id,
            #     "output": response
            # })

        state.trajectory[-1][-1].input = slots
        state.trajectory[-1][-1].output = str(tool_output)

        if tool_output.status == StatusEnum.INCOMPLETE:
            # Tool execution failed
            if slot_verification:
                log_context.info("Tool execution INCOMPLETE due to slot verification")
                tool_output.message_flow = f"Context from {self.name} tool execution: {str(tool_output.message_flow)}\n Focus on the '{reason}' to generate the verification request in response please and make sure the request appear in the response."
            else:
                log_context.info(
                    "Tool execution INCOMPLETE due to tool execution failure"
                )
                # Make it clear that the LLM should ask the user for missing information
                missing_slots = self._missing_slots_recursive(slots)
                if missing_slots:
                    questions_text = " ".join(missing_slots)
                    tool_output.message_flow = (
                        state.message_flow
                        + f"IMPORTANT: The tool cannot proceed without required information. You MUST ask the user for: {questions_text}\n"
                        + "Do NOT provide any facts or information until you have collected this required information from the user.\n"
                    )
                else:
                    tool_output.message_flow = (
                        state.message_flow
                        + f"Context from {self.name} tool execution: {str(tool_output.message_flow)}\n"
                    )
        all_slots[self.name] = slots
        tool_output.slots = all_slots

        return state, tool_output

    def _build_repeatable_regular_slot_prompt(self, slot: Slot) -> str:
        """Build a prompt for repeatable regular slots.

        Args:
            slot: The repeatable regular slot

        Returns:
            str: The prompt for the repeatable regular slot
        """
        type_example = {
            "str": '"example string"',
            "int": "123",
            "float": "12.34",
            "bool": "true",
        }.get(slot.type, '"example"')

        return (
            f"IMPORTANT: This slot is repeatable and expects a list of {slot.type} values. "
            f"Please provide a JSON array of values, e.g. [{type_example}, {type_example}]. "
            f"Extract ALL matching values from the conversation into the array. "
            f"Do not return a single value - return an array even if there's only one value. "
            f"Return an empty array [] if no values are found."
        )

    def _ensure_repeatable_field_value(
        self, value: object, field_type: str
    ) -> list[object]:
        """
        Ensures that the value for a repeatable field is always a list of the correct type.
        If value is None or empty, returns [].
        If value is already a list, converts each element to the correct type.
        Otherwise, wraps the value in a list and converts it to the correct type.
        """
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [self._convert_value(v, field_type) for v in value]
        return [self._convert_value(value, field_type)]

    def _handle_missing_required_slots(
        self, slots: list[Slot], chat_history_str: str
    ) -> tuple[str, bool]:
        """Handle missing required slots and return appropriate response message.

        Args:
            slots: List of slots to check
            chat_history_str: Formatted chat history string

        Returns:
            Tuple of (response_message, is_verification) where is_verification indicates
            if this is a verification request (True) or missing slot request (False)
        """
        for slot in slots:
            if slot.type == "group":
                response = self._check_group_slot_missing_fields(slot)
                if response:
                    return response, False  # Group slots are missing, not verification
            else:
                # Handle regular slots (non-group)
                if getattr(slot, "repeatable", False):
                    # For repeatable regular slots, check list structure
                    if (
                        not slot.value
                        or not isinstance(slot.value, list)
                        or len(slot.value) == 0
                    ):
                        return slot.prompt, False  # Missing slot
                    # Check each value in the list
                    for idx, val in enumerate(slot.value):
                        if val in [None, ""]:
                            return (
                                f"Please provide a value for {slot.prompt} (item {idx + 1})",
                                False,
                            )  # Missing slot
                else:
                    # For non-repeatable regular slots
                    # if there is extracted slots values but haven't been verified
                    if slot.value and not slot.verified:
                        # check whether it verified or not
                        verification_needed: bool
                        thought: str
                        verification_needed, thought = self.slotfiller.verify_slot(
                            slot.model_dump(), chat_history_str, self.llm_config
                        )
                        if verification_needed:
                            return (
                                slot.prompt + "The reason is: " + thought,
                                True,
                            )  # Verification needed
                        else:
                            slot.verified = True
                            log_context.info(
                                f"Slot '{slot.name}' verified successfully"
                            )
                    # if there is no extracted slots values, then should prompt the user to fill the slot
                    if not slot.value and slot.required:
                        return slot.prompt, False  # Missing slot

        return "", False

    def _check_group_slot_missing_fields(self, slot: Slot) -> str:
        """Check for missing required fields in a group slot.

        Args:
            slot: The group slot to check

        Returns:
            Response message if missing fields, empty string otherwise
        """
        # For group, check each item in value list
        if not slot.value or not isinstance(slot.value, list):
            return slot.prompt

        for idx, item in enumerate(slot.value):
            missing_fields = []
            for field in (
                slot.slot_schema
                if hasattr(slot, "slot_schema")
                and isinstance(slot.slot_schema, list | tuple)
                else []
            ):
                field_repeatable = field.get("repeatable", False)
                if field.get("required", False):
                    if field_repeatable:
                        # For repeatable fields, check if array exists and has values
                        if (
                            field["name"] not in item
                            or not isinstance(item[field["name"]], list)
                            or len(item[field["name"]]) == 0
                        ):
                            missing_fields.append(f"{field['name']} (repeatable)")
                        else:
                            # Check each value in the array
                            for val_idx, val in enumerate(item[field["name"]]):
                                if val in [None, ""]:
                                    missing_fields.append(
                                        f"{field['name']} (value {val_idx + 1})"
                                    )
                    else:
                        # For non-repeatable fields, check single value
                        if item.get(field["name"]) in [None, ""]:
                            missing_fields.append(field["name"])
            if missing_fields:
                return f"Please provide the following fields for group '{slot.name}' item {idx + 1}: {', '.join(missing_fields)}."

        return ""
