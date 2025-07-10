"""Tool management for the Arklex framework.

This module provides functionality for managing tools, including
initialization, execution, and slot filling integration.
"""

import inspect
import json
import os
import traceback
import uuid
from collections.abc import Callable
from typing import Any, TypedDict

from arklex.orchestrator.NLU.core.slot import SlotFiller
from arklex.utils.exceptions import AuthenticationError, ToolExecutionError
from arklex.utils.graph_state import MessageState, StatusEnum
from arklex.utils.logging_utils import LogContext
from arklex.utils.slot import Slot
from arklex.utils.utils import format_chat_history

log_context = LogContext(__name__)

PYTHON_TO_JSON_SCHEMA = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}


def register_tool(
    desc: str,
    slots: list[dict[str, Any]] | None = None,
    outputs: list[str] | None = None,
    isResponse: bool = False,
) -> Callable:
    """Register a tool with the Arklex framework.

    This decorator registers a function as a tool with the specified description, slots,
    outputs, and response flag. It handles path normalization and tool initialization.

    Args:
        desc (str): Description of the tool's functionality.
        slots (List[Dict[str, Any]], optional): List of slot definitions. Defaults to None.
        outputs (List[str], optional): List of output field names. Defaults to None.
        isResponse (bool, optional): Whether the tool is a response tool. Defaults to False.

    Returns:
        Callable: A function that creates and returns a Tool instance.
    """
    if slots is None:
        slots = []
    if outputs is None:
        outputs = []

    current_file_dir: str = os.path.dirname(__file__)

    def inner(func: Callable) -> Callable:
        file_path: str = inspect.getfile(func)
        relative_path: str = os.path.relpath(file_path, current_file_dir)
        # reformat the relative path to replace / and \\ with -, and remove .py, because the function calling in openai only allow the function name match the patter the pattern '^[a-zA-Z0-9_-]+$'
        # different file paths format in Windows and linux systems
        relative_path = (
            relative_path.replace("/", "-").replace("\\", "-").replace(".py", "")
        )
        key: str = f"{relative_path}-{func.__name__}"

        def tool() -> "Tool":
            return Tool(func, key, desc, slots, outputs, isResponse)

        return tool

    return inner


class FixedArgs(TypedDict, total=False):
    """Type definition for fixed arguments passed to tool execution."""

    llm_provider: str
    model_type_or_path: str
    temperature: float
    shop_url: str
    api_version: str
    admin_token: str
    storefront_token: str
    limit: str
    navigate: str
    pageInfo: dict[str, Any]


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
        isResponse (bool): Whether the tool is a response tool.
        properties (Dict[str, Dict[str, Any]]): Tool properties.
        llm_config (Dict[str, Any]): Language model configuration.
    """

    def __init__(
        self,
        func: Callable,
        name: str,
        description: str,
        slots: list[dict[str, Any]],
        outputs: list[str],
        isResponse: bool,
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
        self.output: list[str] = outputs
        self.slotfiller: SlotFiller | None = None
        self.info: dict[str, Any] = self.get_info(slots)
        self.slots: list[Slot] = [Slot.model_validate(slot) for slot in slots]
        self.openai_slots: list[dict[str, Any]] = self._format_slots(slots)
        self.isResponse: bool = isResponse
        self.properties: dict[str, dict[str, Any]] = {}
        self.llm_config: dict[str, Any] = {}
        self.fixed_args = {}
        self.auth = {}

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

    def init_default_slots(self, default_slots: list[Slot]) -> None:
        """Initializes the default slots as provided and returns a dictionary of slots which have been populated."""
        populated_slots: dict[str:Any] = {}
        for default_slot in default_slots:
            populated_slots[default_slot.name] = default_slot.value
            for slot in self.slots:
                if slot.name == default_slot.name:
                    slot.value = default_slot.value
                    slot.verified = True
        return populated_slots

    def _init_slots(self, state: MessageState) -> None:
        """Initialize slots with default values from the message state.

        This method processes default slots from the message state and updates
        the tool's slots with their values.

        Args:
            state (MessageState): The current message state.
        """
        default_slots: list[Slot] = state.slots.get("default_slots", [])
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
                 {"name": "param3", "type": "bool", "required": True}]

            Result:
                [Slot(name="param1", type="str", required=False),  # Updated
                 Slot(name="param2", type="int", required=False),  # Preserved
                 Slot(name="param3", type="bool", required=True)]  # Added
        """
        if not slots:
            return

        # Create a dictionary of existing slots for easy lookup
        existing_slots_dict = {slot.name: slot for slot in self.slots}

        # Process new slots
        for new_slot in slots:
            slot_name = new_slot["name"]
            if slot_name in existing_slots_dict:
                existing_slot = existing_slots_dict[slot_name]
                for key, value in new_slot.items():
                    setattr(existing_slot, key, value)
            else:
                if new_slot.get("type") == "group":
                    self.slots.append(Slot(
                        name=new_slot["name"],
                        type="group",
                        schema=new_slot.get("schema", []),
                        required=new_slot.get("required", False),
                        repeatable=new_slot.get("repeatable", True),
                        prompt=new_slot.get("prompt", ""),
                        description=new_slot.get("description", ""),
                        value=[],
                        valueSource=new_slot.get("valueSource", None),
                    ))
                elif new_slot.get("type") == "nested_group":
                    self.slots.append(Slot(
                        name=new_slot["name"],
                        type="nested_group",
                        nested_schema=new_slot.get("nested_schema", []),
                        required=new_slot.get("required", False),
                        repeatable=new_slot.get("repeatable", True),
                        prompt=new_slot.get("prompt", ""),
                        description=new_slot.get("description", ""),
                        value=[],
                        valueSource=new_slot.get("valueSource", None),
                    ))
                else:
                    self.slots.append(Slot.model_validate(new_slot))

        # Update tool info with merged slots
        self.info = self.get_info([slot.model_dump() for slot in self.slots])

    def _convert_value(self, value, type_str):
        # Helper to convert string value to correct type
        if value is None:
            return value
        try:
            if type_str == "int":
                return int(value)
            elif type_str == "float":
                return float(value)
            elif type_str == "bool":
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.lower() == "true"
                return bool(value)
            elif type_str == "str":
                return str(value)
            elif type_str.startswith("list["):
                # Assume comma-separated string for lists
                if isinstance(value, str):
                    return [v.strip() for v in value.split(",") if v.strip()]
                return list(value)
            else:
                return value
        except Exception:
            return value

    def _fill_slots_recursive(self, slots, chat_history_str):
        filled_slots = []
        
        # First, separate regular slots from nested slot groups
        regular_slots = [slot for slot in slots if slot.type != "nested_group"]
        nested_slots = [slot for slot in slots if slot.type == "nested_group"]
        
        # Process regular slots first
        for slot in regular_slots:
            if slot.type == "group":
                # Build a schema-driven prompt for the slot group
                def build_group_prompt(slot):
                    example_fields = []
                    schema_lines = []
                    for field in (slot.schema or []):
                        field_type = field.get("type", "str")
                        example_value = {
                            "str": '"example string"',
                            "int": "123",
                            "float": "12.34",
                            "bool": "true"
                        }.get(field_type, '"example"')
                        example_fields.append(f'"{field["name"]}": {example_value}')
                        desc_or_prompt = field.get("description") or field.get("prompt", "")
                        schema_lines.append(
                            f'- {field["name"]} ({field_type}): {desc_or_prompt}'
                        )
                    example_obj = "{" + ", ".join(example_fields) + "}"
                    schema_str = "\n".join(schema_lines)
                    return (
                        f"Please provide a list of dictionaries (objects), e.g. [{{'key': 'value'}}], each matching this schema:\n"
                        f"{schema_str}\n"
                        f"Example:\n[{example_obj}]\n"
                        f"IMPORTANT: Each object must have ALL the fields above, with the correct type. "
                        f"Do not add extra fields. Do not return a dict of lists or a list of values. "
                        f"Return a list of dicts, each matching the schema exactly. "
                        f"Use the description for each field to determine where to find the value. The field name is the key in the dict."
                    )
                group_prompt = build_group_prompt(slot)
                # Create a temporary slot for the group, with the prompt
                temp_group_slot = Slot(
                    name=slot.name,
                    type="group",
                    value=slot.value if slot.value else [],
                    description=slot.description +" "+ group_prompt,
                    required=slot.required,
                    schema=slot.schema,
                    repeatable=slot.repeatable,
                )
                # Use slotfiller to fill the group as a whole
                filled = self.slotfiller.fill_slots([temp_group_slot], chat_history_str, self.llm_config)
                group_value = filled[0].value
                # If the value is a string, try to parse as JSON
                if isinstance(group_value, str):
                    log_context.debug(f"Attempting to parse group_value as JSON for slot '{slot.name}': {group_value}")
                    try:
                        group_value = json.loads(group_value)
                    except Exception as e:
                        log_context.error(f"Failed to parse group_value as JSON for slot '{slot.name}': {group_value}. Error: {e}")
                        raise ValueError(f"Slot group '{slot.name}' did not return a valid JSON list of objects: {group_value}")
                # Handle repeatable flag for slot groups
                if slot.repeatable:
                    # When repeatable is True, enforce that the value is a list of dicts
                    if not (isinstance(group_value, list) and all(isinstance(item, dict) for item in group_value)):
                        # Handle case where group_value is None or not a list
                        if group_value is None:
                            log_context.warning(f"Slot group '{slot.name}' returned None, converting to empty list")
                            group_value = []
                        elif isinstance(group_value, dict):
                            log_context.warning(f"Slot group '{slot.name}' returned a single dict, converting to list")
                            group_value = [group_value]
                        else:
                            log_context.error(f"Slot group '{slot.name}' returned invalid format: {type(group_value)} - {group_value}")
                            raise ValueError(f"Slot group '{slot.name}' must be a list of dicts when repeatable=True, got: {group_value}")
                else:
                    # When repeatable is False, enforce that the value is a single dict
                    if isinstance(group_value, list):
                        if len(group_value) == 0:
                            log_context.warning(f"Slot group '{slot.name}' returned empty list, converting to empty dict")
                            group_value = {}
                        elif len(group_value) == 1:
                            log_context.warning(f"Slot group '{slot.name}' returned list with one item, converting to single dict")
                            group_value = group_value[0]
                        else:
                            log_context.error(f"Slot group '{slot.name}' returned list with multiple items when repeatable=False")
                            raise ValueError(f"Slot group '{slot.name}' must be a single dict when repeatable=False, got list with {len(group_value)} items")
                    elif not isinstance(group_value, dict):
                        if group_value is None:
                            log_context.warning(f"Slot group '{slot.name}' returned None, converting to empty dict")
                            group_value = {}
                        else:
                            log_context.error(f"Slot group '{slot.name}' returned invalid format: {type(group_value)} - {group_value}")
                            raise ValueError(f"Slot group '{slot.name}' must be a single dict when repeatable=False, got: {group_value}")
                # For each dict, apply valueSource logic as defined in the schema
                if slot.repeatable:
                    # When repeatable is True, process as list
                    for item in group_value:
                        for field in (slot.schema or []):
                            field_name = field["name"]
                            val_source = field.get("valueSource", "Prompt User")
                            field_type = field.get("type", "str")
                            schema_value = field.get("value", "")
                            if val_source == "fixed":
                                item[field_name] = self._convert_value(schema_value, field_type)
                            elif val_source == "default":
                                if item.get(field_name) in [None, ""]:
                                    item[field_name] = self._convert_value(schema_value, field_type)
                                else:
                                    item[field_name] = self._convert_value(item[field_name], field_type)
                            else:  # Prompt User or missing
                                item[field_name] = self._convert_value(item.get(field_name, ""), field_type)
                    slot.value = group_value
                else:
                    # When repeatable is False, process as single dict
                    for field in (slot.schema or []):
                        field_name = field["name"]
                        val_source = field.get("valueSource", "Prompt User")
                        field_type = field.get("type", "str")
                        schema_value = field.get("value", "")
                        if val_source == "fixed":
                            group_value[field_name] = self._convert_value(schema_value, field_type)
                        elif val_source == "default":
                            if group_value.get(field_name) in [None, ""]:
                                group_value[field_name] = self._convert_value(schema_value, field_type)
                            else:
                                group_value[field_name] = self._convert_value(group_value[field_name], field_type)
                        else:  # Prompt User or missing
                            group_value[field_name] = self._convert_value(group_value.get(field_name, ""), field_type)
                    slot.value = group_value
                filled_slots.append(slot)
            else:
                filled = self.slotfiller.fill_slots([slot], chat_history_str, self.llm_config)
                slot.value = self._convert_value(filled[0].value, slot.type)
                filled_slots.append(slot)
        
        # Now process nested slot groups to build unified structure
        if nested_slots:
            for slot in nested_slots:
                # Check if we need to re-fill the nested slot group
                should_refill = False
                
                # Check if any referenced slots have changed
                for field in (slot.nested_schema or []):
                    group_ref = field.get("groupRef", "")
                    if group_ref and group_ref != "none":
                        # Find the referenced slot
                        for other_slot in self.slots:
                            if other_slot.name == group_ref:
                                # If the referenced slot has a value but the nested slot doesn't reflect it
                                if other_slot.value and (not slot.value or self._nested_slot_needs_update(slot, other_slot)):
                                    should_refill = True
                                    log_context.info(f"Nested slot '{slot.name}' needs update due to changes in referenced slot '{other_slot.name}'")
                                break
                
                # Fill or re-fill the nested slot group
                if not slot.value or should_refill:
                    log_context.info(f"Filling nested slot group '{slot.name}' via LLM...")
                    slot_value = self._fill_nested_slot_group(slot, chat_history_str)
                    log_context.info(f"LLM returned for '{slot.name}': {slot_value}")
                    slot.value = slot_value
                filled_slots.append(slot)
        
        return filled_slots

    def _build_unified_nested_structure(self, nested_slots, chat_history_str):
        """Build a unified nested structure from all nested slot groups.
        
        Args:
            nested_slots: List of nested slot groups
            chat_history_str: Chat history for context
            
        Returns:
            Dictionary mapping slot names to their values in the unified structure
        """
        return self._build_unified_nested_structure_v2(nested_slots, chat_history_str)

    def _build_complete_nested_structure(self, root_slot, slot_map, chat_history_str):
        """Build a complete nested structure by resolving all references.
        
        Args:
            root_slot: The root nested slot group
            slot_map: Mapping of slot names to slot objects
            chat_history_str: Chat history for context
            
        Returns:
            Dictionary mapping slot names to their values
        """
        # Build the structure starting from the root
        root_structure = self._fill_nested_slot_group(root_slot, chat_history_str)
        
        # Create the result mapping
        result = {}
        result[root_slot.name] = root_structure
        
        # Process other nested slots that might be referenced
        for slot_name, slot in slot_map.items():
            if slot_name != root_slot.name:
                # For now, we'll process them independently
                # In a more sophisticated approach, we'd build the unified structure
                slot_structure = self._fill_nested_slot_group(slot, chat_history_str)
                result[slot_name] = slot_structure
        
        return result

    def _build_unified_nested_structure_v2(self, nested_slots, chat_history_str):
        """Build a unified nested structure by asking the LLM to generate the complete structure.
        
        This method asks the LLM to generate the complete nested structure based on the user's input.
        
        Args:
            nested_slots: List of nested slot groups
            chat_history_str: Chat history for context
            
        Returns:
            Dictionary mapping slot names to their values in the unified structure
        """
        if not nested_slots:
            return {}
        
        # Find the root nested slot group (the one that should contain the main structure)
        # For now, let's assume the last nested slot group is the root (category)
        root_slot = nested_slots[-1]  # Assuming the last one is the root
        log_context.info(f"Building unified nested structure v2 starting with root slot: {root_slot.name}")
        
        # Let the LLM generate the complete structure
        complete_structure = self._fill_nested_slot_group(root_slot, chat_history_str)
        
        # Create the result mapping
        result = {}
        result[root_slot.name] = complete_structure
        
        return result

    def _build_complete_nested_structure_v2(self, root_slot, slot_map, chat_history_str):
        """Build a complete nested structure by resolving all references.
        
        Args:
            root_slot: The root nested slot group
            slot_map: Mapping of slot names to slot objects
            chat_history_str: Chat history for context
            
        Returns:
            Dictionary mapping slot names to their values
        """
        # Build the structure starting from the root
        root_structure = self._fill_nested_slot_group(root_slot, chat_history_str)
        
        # Create the result mapping
        result = {}
        result[root_slot.name] = root_structure
        
        # Process other nested slots that might be referenced
        for slot_name, slot in slot_map.items():
            if slot_name != root_slot.name:
                # For now, we'll process them independently
                # In a more sophisticated approach, we'd build the unified structure
                slot_structure = self._fill_nested_slot_group(slot, chat_history_str)
                result[slot_name] = slot_structure
        
        return result

    def _build_nested_structure_from_root(self, root_slot, all_nested_slots, chat_history_str):
        """Build nested structure starting from a root slot.
        
        Args:
            root_slot: The root nested slot group
            all_nested_slots: All nested slot groups
            chat_history_str: Chat history for context
            
        Returns:
            Dictionary mapping slot names to their values
        """
        # Create a mapping of slot names to their slot objects
        slot_map = {slot.name: slot for slot in all_nested_slots}
        
        # Build the structure starting from the root
        root_structure = self._fill_nested_slot_group(root_slot, chat_history_str)
        
        # Create the result mapping
        result = {}
        result[root_slot.name] = root_structure
        
        return result

    def _fill_nested_slot_group(self, slot, chat_history_str):
        """Recursively fill nested slot groups using LLM based on schema.
        
        Args:
            slot: The nested slot group to fill
            chat_history_str: Chat history for context
            
        Returns:
            The filled nested structure as a list of dictionaries
        """
        log_context.info(f"Filling nested slot group '{slot.name}' with schema: {slot.nested_schema}")
        if not slot.nested_schema:
            log_context.warning(f"Nested slot group '{slot.name}' has no nested_schema, treating as empty")
            return []
        
        # Build a comprehensive prompt that includes all slot groups and nested slot groups
        nested_prompt = self._build_comprehensive_nested_prompt(slot, chat_history_str)
        
        # Create a temporary slot for the nested group
        temp_nested_slot = Slot(
            name=slot.name,
            type="nested_group",
            value="",
            enum=[],
            description=f"Nested slot group '{slot.name}' with recursive schema" + nested_prompt,
            required=slot.required,
            verified=False,
            items=None,
            target=None,
            schema=None,
            nested_schema=slot.nested_schema,
            repeatable=slot.repeatable,
            valueSource=None
        )
        
        # Use slotfiller to get the complete nested structure
        filled = self.slotfiller.fill_slots([temp_nested_slot], chat_history_str, self.llm_config)
        complete_value = filled[0].value
        
        log_context.info(f"LLM response for nested slot group '{slot.name}': {complete_value}")
        
        # Parse JSON if it's a string
        if isinstance(complete_value, str):
            try:
                # Clean up markdown formatting
                if complete_value.startswith("```json"):
                    complete_value = complete_value.replace("```json", "").replace("```", "").strip()
                elif complete_value.startswith("```"):
                    complete_value = complete_value.replace("```", "").strip()
                
                complete_value = json.loads(complete_value)
                log_context.info(f"Successfully parsed JSON for slot '{slot.name}': {complete_value}")
            except Exception as e:
                log_context.error(f"Failed to parse complete_value as JSON for slot '{slot.name}': {complete_value}. Error: {e}")
                # Try to extract JSON from the response
                try:
                    import re
                    json_match = re.search(r'\{.*\}', complete_value, re.DOTALL)
                    if json_match:
                        complete_value = json.loads(json_match.group())
                        log_context.info(f"Extracted JSON from response for slot '{slot.name}': {complete_value}")
                    else:
                        raise ValueError(f"Could not extract JSON from response: {complete_value}")
                except Exception as e2:
                    log_context.error(f"Failed to extract JSON from response for slot '{slot.name}': {e2}")
                    raise ValueError(f"Nested slot group '{slot.name}' did not return a valid JSON structure: {complete_value}")
        
        # Handle repeatable flag for nested slot groups
        if slot.repeatable:
            # When repeatable is True, ensure it's a list
            if not isinstance(complete_value, list):
                if complete_value is None:
                    complete_value = []
                elif isinstance(complete_value, dict):
                    complete_value = [complete_value]
                else:
                    raise ValueError(f"Nested slot group '{slot.name}' must be a list when repeatable=True, got: {type(complete_value)}")
        else:
            # When repeatable is False, ensure it's a single dict
            if isinstance(complete_value, list):
                if len(complete_value) == 0:
                    log_context.warning(f"Nested slot group '{slot.name}' returned empty list, converting to empty dict")
                    complete_value = {}
                elif len(complete_value) == 1:
                    log_context.warning(f"Nested slot group '{slot.name}' returned list with one item, converting to single dict")
                    complete_value = complete_value[0]
                else:
                    log_context.error(f"Nested slot group '{slot.name}' returned list with multiple items when repeatable=False")
                    raise ValueError(f"Nested slot group '{slot.name}' must be a single dict when repeatable=False, got list with {len(complete_value)} items")
            elif not isinstance(complete_value, dict):
                if complete_value is None:
                    log_context.warning(f"Nested slot group '{slot.name}' returned None, converting to empty dict")
                    complete_value = {}
                else:
                    log_context.error(f"Nested slot group '{slot.name}' returned invalid format: {type(complete_value)} - {complete_value}")
                    raise ValueError(f"Nested slot group '{slot.name}' must be a single dict when repeatable=False, got: {complete_value}")
        
        return complete_value

    def _build_comprehensive_nested_prompt(self, slot, chat_history_str):
        """Build a comprehensive prompt that includes all slot groups and nested slot groups.
        
        Args:
            slot: The nested slot group to fill
            chat_history_str: Chat history for context
            
        Returns:
            A comprehensive prompt string
        """
        prompt_parts = []
        prompt_parts.append("You are a data extraction expert. Your task is to extract structured data from the conversation and format it exactly according to the schema provided.")
        prompt_parts.append("")
        
        # Get all slot groups and nested slot groups
        all_slot_groups = []
        all_nested_slot_groups = []
        
        # Collect all slot groups and nested slot groups from the current tool's slots
        for slot in self.slots:
            if slot.type == "group":
                all_slot_groups.append({
                    'name': slot.name,
                    'schema': slot.schema,
                    'repeatable': slot.repeatable,
                    'required': slot.required
                })
            elif slot.type == "nested_group":
                all_nested_slot_groups.append({
                    'name': slot.name,
                    'schema': slot.nested_schema,
                    'repeatable': slot.repeatable,
                    'required': slot.required
                })
        
        # Build comprehensive schema description
        prompt_parts.append("SCHEMA DEFINITION:")
        prompt_parts.append("=" * 50)
        
        # Describe slot groups
        if all_slot_groups:
            prompt_parts.append("SLOT GROUPS (regular groups):")
            for sg in all_slot_groups:
                prompt_parts.append(f"  Group: '{sg['name']}'")
                prompt_parts.append(f"    Repeatable: {sg.get('repeatable', False)}")
                prompt_parts.append(f"    Required: {sg.get('required', False)}")
                prompt_parts.append(f"    Schema:")
                for field in sg.get('schema', []):
                    field_type = field.get('type', 'str')
                    required = field.get('required', False)
                    description = field.get('description', '')
                    prompt_parts.append(f"      - {field['name']} ({field_type}){' [REQUIRED]' if required else ''}: {description}")
                prompt_parts.append("")
        
        # Describe nested slot groups
        if all_nested_slot_groups:
            prompt_parts.append("NESTED SLOT GROUPS:")
            for nsg in all_nested_slot_groups:
                prompt_parts.append(f"  Group: '{nsg['name']}'")
                prompt_parts.append(f"    Repeatable: {nsg.get('repeatable', False)}")
                prompt_parts.append(f"    Required: {nsg.get('required', False)}")
                prompt_parts.append(f"    Schema:")
                for field in nsg.get('schema', []):
                    field_type = field.get('type', 'str')
                    required = field.get('required', False)
                    field_repeatable = field.get('repeatable', False)
                    description = field.get('description', '')
                    group_ref = field.get('groupRef', '')
                    if group_ref and group_ref != 'none':
                        repeatable_text = ' [REPEATABLE]' if field_repeatable else ''
                        prompt_parts.append(f"      - {field['name']} (references group '{group_ref}'){' [REQUIRED]' if required else ''}{repeatable_text}: {description}")
                    else:
                        repeatable_text = ' [REPEATABLE]' if field_repeatable else ''
                        prompt_parts.append(f"      - {field['name']} ({field_type}){' [REQUIRED]' if required else ''}{repeatable_text}: {description}")
                prompt_parts.append("")
        
        # Explain how groupRef works
        prompt_parts.append("HOW groupRef WORKS:")
        prompt_parts.append("- When a field has groupRef, it means the field should contain data from the referenced group")
        prompt_parts.append("- Example: if field 'courses' has groupRef='courses', then the 'courses' field should contain the data structure from the 'courses' group")
        prompt_parts.append("- The field name in the nested structure should match the groupRef name")
        prompt_parts.append("")
        
        # Explain repeatable
        prompt_parts.append("HOW repeatable WORKS:")
        prompt_parts.append("- When repeatable=True: the group becomes a LIST of objects")
        prompt_parts.append("- When repeatable=False: the group becomes a SINGLE object")
        prompt_parts.append("- Example: if 'courses' is repeatable=True, then it should be: [{'course_name': 'Math101', 'modules': [...]}, {'course_name': 'CS101', 'modules': [...]}]")
        prompt_parts.append("- Example: if 'courses' is repeatable=False, then it should be: {'course_name': 'Math101', 'modules': [...]}")
        prompt_parts.append("")
        prompt_parts.append("IMPORTANT - REPEATABLE FIELDS:")
        prompt_parts.append("- Individual fields within schemas can also be repeatable")
        prompt_parts.append("- If a field has repeatable=True, it becomes an ARRAY of values")
        prompt_parts.append("- If a field has repeatable=False, it becomes a SINGLE value")
        prompt_parts.append("- Example: if 'term' field is repeatable=True, use: \"term\": [\"Fall 2024\", \"Spring 2025\"]")
        prompt_parts.append("- Example: if 'term' field is repeatable=False, use: \"term\": \"Fall 2024\"")
        prompt_parts.append("- ALWAYS check the example structure to see which fields are arrays vs single values")
        prompt_parts.append("- For repeatable fields, extract ALL values from the conversation and put them in an array")
        prompt_parts.append("- Even if there's only one value, if the field is repeatable, it must be in an array")
        prompt_parts.append("")
        
        # Build example structure for the specific slot
        prompt_parts.append(f"REQUIRED STRUCTURE FOR '{slot.name}':")
        prompt_parts.append("=" * 50)
        
        example_structure = self._build_comprehensive_example_structure(slot, all_slot_groups, all_nested_slot_groups)
        prompt_parts.append(json.dumps(example_structure, indent=2))
        
        prompt_parts.append("")
        prompt_parts.append("CRITICAL REQUIREMENTS:")
        prompt_parts.append("- Use EXACTLY the field names shown in the example")
        prompt_parts.append("- Do NOT add extra fields not in the schema")
        prompt_parts.append("- Do NOT change field names (e.g., use 'term' not 'semester', 'terms')")
        prompt_parts.append("- Follow the exact structure shown above")
        prompt_parts.append("- Pay attention to ARRAYS vs SINGLE VALUES in the example structure")
        prompt_parts.append("- If the example shows an array (e.g., \"term\": [\"example\"]), use an array in your response")
        prompt_parts.append("- If the example shows a single value (e.g., \"term\": \"example\"), use a single value")
        prompt_parts.append("- REPEATABLE FIELDS MUST BE ARRAYS - even if there's only one value")
        prompt_parts.append("- Extract data from the conversation to populate the values")
        prompt_parts.append("- Return ONLY valid JSON, no explanations")
        
        prompt_parts.append("")
        prompt_parts.append("CONVERSATION:")
        prompt_parts.append(chat_history_str)
        
        return "\n".join(prompt_parts)

    def _build_comprehensive_example_structure(self, target_slot, slot_groups, nested_slot_groups):
        """Build a comprehensive example structure for a specific nested slot group.
        
        Args:
            target_slot: The nested slot group to build example for (can be Slot object or dict)
            slot_groups: List of all slot groups
            nested_slot_groups: List of all nested slot groups
            
        Returns:
            An example structure for the target slot
        """
        example = {}
        
        # Handle both Slot objects and dicts
        if hasattr(target_slot, 'nested_schema'):
            # target_slot is a Slot object
            schema = target_slot.nested_schema
            repeatable = target_slot.repeatable
        else:
            # target_slot is a dict
            schema = target_slot.get('schema', [])
            repeatable = target_slot.get('repeatable', False)
        
        # Build example based on the target slot's schema
        for field in (schema or []):
            field_name = field["name"]
            field_type = field.get("type", "str")
            group_ref = field.get("groupRef", "")
            required = field.get("required", False)
            
            if group_ref and group_ref != "none":
                # This field references another group
                if group_ref in [sg['name'] for sg in slot_groups]:
                    # References a regular slot group
                    ref_group = next(sg for sg in slot_groups if sg['name'] == group_ref)
                    if ref_group.get('repeatable', False):
                        # Use the field name from the nested schema, not the groupRef name
                        example[field_name] = [
                            {
                                field['name']: self._example_value_for_type(field.get('type', 'str'), field['name'])
                                for field in ref_group.get('schema', [])
                            }
                        ]
                    else:
                        # Use the field name from the nested schema, not the groupRef name
                        example[field_name] = {
                            field['name']: self._example_value_for_type(field.get('type', 'str'), field['name'])
                            for field in ref_group.get('schema', [])
                        }
                elif group_ref in [nsg['name'] for nsg in nested_slot_groups]:
                    # References a nested slot group
                    ref_nested_group = next(nsg for nsg in nested_slot_groups if nsg['name'] == group_ref)
                    if ref_nested_group.get('repeatable', False):
                        # Use the field name from the nested schema, not the groupRef name
                        example[field_name] = [
                            self._build_comprehensive_example_structure(ref_nested_group, slot_groups, nested_slot_groups)
                        ]
                    else:
                        # Use the field name from the nested schema, not the groupRef name
                        example[field_name] = self._build_comprehensive_example_structure(ref_nested_group, slot_groups, nested_slot_groups)
                else:
                    # Unknown reference, create basic example
                    example[field_name] = [{"example_field": "example_value"}]
            else:
                # Regular field - check if it's repeatable
                field_repeatable = field.get("repeatable", False)
                if field_repeatable:
                    # If field is repeatable, make it an array of multiple values to make it obvious
                    example[field_name] = [
                        self._example_value_for_type(field_type, field_name),
                        f"another_{field_name}",
                        f"third_{field_name}"
                    ]
                else:
                    # If field is not repeatable, make it a single value
                    example[field_name] = self._example_value_for_type(field_type, field_name)
        
        # Handle repeatable flag
        if repeatable:
            return [example]
        else:
            return example

    def _nested_slot_needs_update(self, nested_slot, referenced_slot):
        """Check if a nested slot needs to be updated based on changes in referenced slot.
        
        Args:
            nested_slot: The nested slot group
            referenced_slot: The referenced slot group
            
        Returns:
            True if the nested slot needs update, False otherwise
        """
        if not nested_slot.value or not referenced_slot.value:
            return True
        
        # Check if the nested slot's value reflects the current referenced slot's value
        for field in (nested_slot.nested_schema or []):
            group_ref = field.get("groupRef", "")
            if group_ref == referenced_slot.name:
                # Find the field in the nested slot that references this group
                field_name = field["name"]
                
                # Check if the nested slot has the correct structure
                if isinstance(nested_slot.value, list):
                    for item in nested_slot.value:
                        if isinstance(item, dict):
                            # Check if the referenced field exists and has the right data
                            # Use field_name from the nested schema, not groupRef
                            if field_name in item:
                                # Compare the data - if they don't match, needs update
                                if item[field_name] != referenced_slot.value:
                                    log_context.info(f"Nested slot data mismatch: {item[field_name]} vs {referenced_slot.value}")
                                    return True
                            else:
                                # Field missing, needs update
                                return True
                elif isinstance(nested_slot.value, dict):
                    # Use field_name from the nested schema, not groupRef
                    if field_name in nested_slot.value:
                        if nested_slot.value[field_name] != referenced_slot.value:
                            return True
                    else:
                        return True
        
        return False

    def _build_recursive_nested_prompt(self, slot):
        """Build a recursive prompt that describes the exact nested structure based on schema.
        
        Args:
            slot: The nested slot group
            
        Returns:
            A formatted prompt string describing the recursive structure
        """
        prompt_parts = []
        prompt_parts.append(f"Extract the following nested information from the text:")
        
        # Build the recursive schema description
        schema_description = self._build_schema_description_recursive(slot.nested_schema, 0)
        prompt_parts.append(schema_description)
        
        prompt_parts.append(f"\nCRITICAL REQUIREMENTS:")
        prompt_parts.append("- Return ONLY valid JSON, no explanations or text")
        prompt_parts.append("- Use EXACTLY the field names specified in the schema")
        prompt_parts.append("- Do NOT add any extra fields not in the schema")
        prompt_parts.append("- Do NOT change field names (e.g., use 'term' not 'semester')")
        prompt_parts.append("- Follow the exact structure and field names shown above")
        prompt_parts.append("- Use the user's input to populate the values")
        prompt_parts.append("- Ensure all required fields are included")
        prompt_parts.append("- Handle nested lists and objects as specified")
        
        # Add example structure
        prompt_parts.append(f"\nEXAMPLE STRUCTURE:")
        example = self._build_example_structure_from_schema(slot.nested_schema)
        prompt_parts.append(json.dumps(example, indent=2))
        
        return "\n".join(prompt_parts)

    def _build_example_structure_from_schema(self, nested_schema):
        """Build an example structure from the nested schema.
        
        Args:
            nested_schema: The nested schema definition
            
        Returns:
            An example structure matching the schema
        """
        example = {}
        
        for field in (nested_schema or []):
            field_name = field["name"]
            field_type = field.get("type", "str")
            group_ref = field.get("groupRef", "")
            field_repeatable = field.get("repeatable", False)
            
            if group_ref and group_ref != "none":
                # This is a nested group reference
                if field_repeatable:
                    example[field_name] = [
                        {"example_field": "example_value"}
                    ]
                else:
                    example[field_name] = {"example_field": "example_value"}
            else:
                # Regular field
                if field_repeatable:
                    # Make it an array of multiple values to make it obvious
                    example[field_name] = [
                        self._example_value_for_type(field_type, field_name),
                        f"another_{field_name}",
                        f"third_{field_name}"
                    ]
                else:
                    example[field_name] = self._example_value_for_type(field_type, field_name)
        
        return example

    def _build_schema_description_recursive(self, schema, indent_level):
        """Recursively build a description of the schema structure.
        
        Args:
            schema: The schema to describe
            indent_level: Current indentation level
            
        Returns:
            A formatted string describing the schema structure
        """
        if not schema:
            return ""
        
        indent = "  " * indent_level
        lines = []
        
        for field in schema:
            field_name = field["name"]
            field_type = field.get("type", "str")
            group_ref = field.get("groupRef", "")
            required = field.get("required", False)
            description = field.get("description", "")
            
            # Build field description
            field_desc = f"{indent}- {field_name}"
            if field_type and field_type != "":
                field_desc += f" ({field_type})"
            if required:
                field_desc += " (required)"
            if description:
                field_desc += f" - {description}"
            
            lines.append(field_desc)
            
            # If this field references another group, recursively describe it
            if group_ref and group_ref != "none":
                # Find the referenced slot
                referenced_slot = None
                for other_slot in self.slots:
                    if other_slot.name == group_ref:
                        referenced_slot = other_slot
                        break
                
                if referenced_slot:
                    if referenced_slot.type == "group" and referenced_slot.schema:
                        lines.append(f"{indent}  (list of objects), each with:")
                        nested_desc = self._build_schema_description_recursive(referenced_slot.schema, indent_level + 2)
                        lines.append(nested_desc)
                    elif referenced_slot.type == "nested_group" and referenced_slot.nested_schema:
                        lines.append(f"{indent}  (nested structure), each with:")
                        nested_desc = self._build_schema_description_recursive(referenced_slot.nested_schema, indent_level + 2)
                        lines.append(nested_desc)
        
        return "\n".join(lines)

    def _process_nested_item_recursive(self, item, nested_schema, chat_history_str):
        """Process a single item in a nested structure, recursively filling referenced groups.
        
        Args:
            item: The item to process
            nested_schema: The schema for this level
            chat_history_str: Chat history for context
            
        Returns:
            The processed item
        """
        if not isinstance(item, dict):
            return item

        processed_item = {}

        for field in (nested_schema or []):
            field_name = field["name"]
            group_ref = field.get("groupRef", "")
            field_type = field.get("type", "str")
            repeatable = field.get("repeatable", True)

            if group_ref and group_ref != "none":
                # This is a nested group reference - recursively fill it
                # Check if the referenced group is a regular slot group or nested slot group
                referenced_slot = None
                for slot in self.slots:
                    if slot.name == group_ref:
                        referenced_slot = slot
                        break
                
                if referenced_slot and referenced_slot.type == "group":
                    # It's a regular slot group - use its value
                    if repeatable:
                        # For repeatable regular slot groups, create a list with the item data
                        if referenced_slot.value:
                            processed_item[field_name] = [referenced_slot.value]
                        else:
                            processed_item[field_name] = []
                    else:
                        # For non-repeatable regular slot groups, use the item data directly
                        processed_item[field_name] = referenced_slot.value if referenced_slot.value else {}
                elif referenced_slot and referenced_slot.type == "nested_group":
                    # It's a nested slot group - recursively fill it
                    nested_structure = self._fill_nested_slot_group(referenced_slot, chat_history_str)
                    processed_item[field_name] = nested_structure
                else:
                    # No referenced slot found - create empty structure
                    processed_item[field_name] = [] if repeatable else {}
            else:
                # Regular field
                processed_item[field_name] = self._convert_value(item.get(field_name, ""), field_type)

        return processed_item

    def _build_complete_structure_prompt(self, slot):
        """Build a comprehensive prompt that shows the exact structure we want, dynamically from the schema.
        
        Args:
            slot: The nested slot group
            
        Returns:
            A formatted prompt string with example structure
        """
        prompt_parts = []
        prompt_parts.append(f"Based on the user's request, generate the complete nested structure for '{slot.name}'.")
        
        # Dynamically build the required structure from the schema
        required_structure = self._build_example_from_schema(slot, set())
        
        prompt_parts.append(f"\nREQUIRED OUTPUT STRUCTURE:")
        prompt_parts.append("You must return a JSON object or array with this exact structure:")
        prompt_parts.append(json.dumps(required_structure, indent=2))
        
        prompt_parts.append(f"\nCRITICAL INSTRUCTIONS:")
        prompt_parts.append("- Return ONLY valid JSON, no explanations or text")
        prompt_parts.append("- Use the exact field names and nesting as shown above, based on the schema")
        prompt_parts.append("- Use the user's input to populate the values")
        prompt_parts.append("- Ensure all required fields are included")
        prompt_parts.append("- The structure must match exactly what is shown above")
        
        return "\n".join(prompt_parts)

    def _build_example_from_schema(self, slot, seen_groups):
        """Recursively build an example structure from the slot's schema, avoiding cycles."""
        # Prevent infinite recursion
        if slot.name in seen_groups:
            return [] if slot.repeatable else {}
        seen_groups = seen_groups | {slot.name}
        
        # If this is a nested_group, use nested_schema
        if slot.type == "nested_group" and slot.nested_schema:
            example = {} if not slot.repeatable else []
            if slot.repeatable:
                # Example: a list of one object
                example_obj = {}
                for field in slot.nested_schema:
                    field_name = field["name"]
                    group_ref = field.get("groupRef", "")
                    if group_ref and group_ref != "none":
                        # Find the referenced slot (group or nested_group)
                        ref_slot = next((s for s in self.slots if s.name == group_ref), None)
                        if ref_slot:
                            example_obj[field_name] = self._build_example_from_schema(ref_slot, seen_groups)
                        else:
                            example_obj[field_name] = []
                    else:
                        example_obj[field_name] = self._example_value_for_type(field.get("type", "str"), field_name)
                example.append(example_obj)
            else:
                for field in slot.nested_schema:
                    field_name = field["name"]
                    group_ref = field.get("groupRef", "")
                    if group_ref and group_ref != "none":
                        ref_slot = next((s for s in self.slots if s.name == group_ref), None)
                        if ref_slot:
                            example[field_name] = self._build_example_from_schema(ref_slot, seen_groups)
                        else:
                            example[field_name] = []
                    else:
                        example[field_name] = self._example_value_for_type(field.get("type", "str"), field_name)
            return example
        # If this is a group, use schema
        elif slot.type == "group" and slot.schema:
            example = [] if slot.repeatable else {}
            if slot.repeatable:
                example_obj = {}
                for field in slot.schema:
                    example_obj[field["name"]] = self._example_value_for_type(field.get("type", "str"), field["name"])
                example.append(example_obj)
            else:
                for field in slot.schema:
                    example[field["name"]] = self._example_value_for_type(field.get("type", "str"), field["name"])
            return example
        else:
            # Fallback: single value
            return self._example_value_for_type(slot.type, slot.name)

    def _example_value_for_type(self, type_str, field_name):
        if type_str == "str":
            return f"example_{field_name}"
        elif type_str == "int":
            return 1
        elif type_str == "float":
            return 1.0
        elif type_str == "bool":
            return True
        else:
            return f"example_{field_name}"

    def _build_expected_structure(self, nested_schema):
        """Build the expected structure based on the nested schema.
        
        Args:
            nested_schema: The nested schema definition
            
        Returns:
            Dictionary representing the expected structure
        """
        if not nested_schema:
            return {}
        
        # Analyze the nested schema to understand the structure
        category_name_field = None
        category_subcategories_field = None
        
        for field in nested_schema:
            if field["name"] == "name" and field.get("groupRef") == "none":
                category_name_field = field
            elif field.get("groupRef") == "subcategories":
                category_subcategories_field = field
        
        # Build the expected structure based on the schema analysis
        if category_name_field and category_subcategories_field:
            # This is a category structure with name and subcategories
            expected = {
                "data": [
                    {
                        "category": [
                            {
                                "subcategories": [
                                    {
                                        "name": "example_item",
                                        "price": 30.0
                                    }
                                ]
                            }
                        ],
                        "name": "example_category"
                    }
                ]
            }
        else:
            # Fallback structure
            expected = {
                "data": [
                    {
                        "category": [
                            {
                                "subcategories": [
                                    {
                                        "name": "example_item",
                                        "price": 30.0
                                    }
                                ]
                            }
                        ],
                        "name": "example_category"
                    }
                ]
            }
        
        return expected

    def _build_nested_group_prompt(self, slot):
        """Build a comprehensive prompt for nested slot groups.
        
        Args:
            slot: The nested slot group
            
        Returns:
            A formatted prompt string
        """
        prompt_parts = []
        prompt_parts.append(f"Please provide a nested structure for '{slot.name}' with the following schema:")
        
        def build_schema_description(schema, indent=0):
            """Recursively build schema description."""
            lines = []
            for field in schema:
                field_name = field["name"]
                field_type = field.get("type", "str")
                description = field.get("description", field.get("prompt", ""))
                group_ref = field.get("groupRef", "")
                
                indent_str = "  " * indent
                if group_ref and group_ref != "none":
                    lines.append(f"{indent_str}- {field_name}: nested group '{group_ref}'")
                    # Find the referenced group schema
                    ref_schema = self._find_group_schema(group_ref, slot.nested_schema)
                    if ref_schema:
                        lines.extend(build_schema_description(ref_schema, indent + 1))
                else:
                    lines.append(f"{indent_str}- {field_name} ({field_type}): {description}")
            
            return lines
        
        schema_lines = build_schema_description(slot.nested_schema)
        prompt_parts.extend(schema_lines)
        
        prompt_parts.append("\nIMPORTANT:")
        if slot.repeatable:
            prompt_parts.append("- Return a list of dictionaries matching the schema exactly")
        else:
            prompt_parts.append("- Return a single dictionary matching the schema exactly")
        prompt_parts.append("- Each dictionary should contain all required fields")
        prompt_parts.append("- For nested groups, provide the structure as specified by groupRef")
        prompt_parts.append("- Use the descriptions to determine appropriate values")
        
        return "\n".join(prompt_parts)

    def _find_group_schema(self, group_name, nested_schema):
        """Find the schema for a referenced group.
        
        Args:
            group_name: Name of the group to find
            nested_schema: The nested schema to search in
            
        Returns:
            The schema for the referenced group, or None if not found
        """
        # First search in the current nested schema
        for field in nested_schema:
            if field["name"] == group_name:
                return field.get("schema", [])
        
        # If not found in nested schema, search in regular slot groups
        for slot in self.slots:
            if slot.name == group_name and slot.type == "group":
                return slot.schema
        
        # If not found in regular slot groups, search in nested slot groups
        for slot in self.slots:
            if slot.name == group_name and slot.type == "nested_group":
                return slot.nested_schema
                
        return None

    def _build_nested_openai_schema(self, nested_schema):
        """Build OpenAI schema for nested slot groups.
        
        Args:
            nested_schema: The nested schema definition
            
        Returns:
            Dictionary representing the OpenAI schema
        """
        properties = {}
        
        for field in (nested_schema or []):
            field_name = field["name"]
            field_type = field.get("type", "str")
            group_ref = field.get("groupRef", "")
            field_repeatable = field.get("repeatable", False)
            
            if group_ref and group_ref != "none":
                # This is a nested group reference
                ref_schema = self._find_group_schema(group_ref, nested_schema)
                if ref_schema:
                    # Check if the referenced group is a regular slot group
                    referenced_slot = None
                    for slot in self.slots:
                        if slot.name == group_ref and slot.type == "group":
                            referenced_slot = slot
                            break
                    if referenced_slot and not referenced_slot.repeatable:
                        # If it's a non-repeatable regular slot group, use its schema directly
                        group_properties = {}
                        group_required = []
                        for schema_field in ref_schema:
                            group_properties[schema_field["name"]] = {
                                "type": PYTHON_TO_JSON_SCHEMA.get(schema_field.get("type", "str"), "string"),
                                "description": schema_field.get("description", ""),
                            }
                            if schema_field.get("required", False):
                                group_required.append(schema_field["name"])
                        if field_repeatable:
                            properties[field_name] = {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": group_properties,
                                    "required": group_required,
                                },
                                "description": field.get("description", ""),
                            }
                        else:
                            properties[field_name] = {
                                "type": "object",
                                "properties": group_properties,
                                "required": group_required,
                                "description": field.get("description", ""),
                            }
                    else:
                        # If it's a repeatable group or nested group, use array
                        nested_properties = self._build_nested_openai_schema(ref_schema)
                        properties[field_name] = {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": nested_properties,
                            },
                            "description": field.get("description", ""),
                        }
                else:
                    # Fallback to array of objects
                    properties[field_name] = {
                        "type": "array",
                        "items": {
                            "type": "object",
                        },
                        "description": field.get("description", ""),
                    }
            else:
                # Regular field
                if field_repeatable:
                    properties[field_name] = {
                        "type": "array",
                        "items": {
                            "type": PYTHON_TO_JSON_SCHEMA.get(field_type, "string"),
                        },
                        "description": field.get("description", ""),
                    }
                else:
                    properties[field_name] = {
                        "type": PYTHON_TO_JSON_SCHEMA.get(field_type, "string"),
                        "description": field.get("description", ""),
                    }
        
        return properties

    def _any_missing_required_recursive(self, slots):
        for slot in slots:
            if slot.type == "group":
                # For group, check if at least one item exists if required
                if slot.repeatable:
                    # When repeatable is True, check if at least one item exists if required
                    if slot.required and (not slot.value or not isinstance(slot.value, list) or len(slot.value) == 0):
                        return True
                    # For each item, check required fields
                    for item in (slot.value or []):
                        for field in (slot.schema or []):
                            if field.get("required", False) and (item.get(field["name"]) in [None, ""]):
                                return True
                else:
                    # When repeatable is False, check if the single dict exists if required
                    if slot.required and (not slot.value or not isinstance(slot.value, dict)):
                        return True
                    # Check required fields in the single dict
                    if slot.value and isinstance(slot.value, dict):
                        for field in (slot.schema or []):
                            if field.get("required", False) and (slot.value.get(field["name"]) in [None, ""]):
                                return True
            elif slot.type == "nested_group":
                # For nested group, check if at least one item exists if required
                if slot.repeatable:
                    # When repeatable is True, check if at least one item exists if required
                    if slot.required and (not slot.value or not isinstance(slot.value, list) or len(slot.value) == 0):
                        return True
                    # For each item, recursively check required fields in nested structure
                    for item in (slot.value or []):
                        if self._check_nested_required_fields(item, slot.nested_schema):
                            return True
                else:
                    # When repeatable is False, check if the single dict exists if required
                    if slot.required and (not slot.value or not isinstance(slot.value, dict)):
                        return True
                    # Recursively check required fields in the single dict
                    if slot.value and isinstance(slot.value, dict):
                        if self._check_nested_required_fields(slot.value, slot.nested_schema):
                            return True
            else:
                if slot.required and (not slot.value or not slot.verified):
                    return True
        return False

    def _check_nested_required_fields(self, item, nested_schema):
        """Recursively check required fields in a nested structure.
        
        Args:
            item: The item to check
            nested_schema: The schema for this level
            
        Returns:
            True if any required field is missing, False otherwise
        """
        if not isinstance(item, dict):
            return False
        
        for field in (nested_schema or []):
            field_name = field["name"]
            group_ref = field.get("groupRef", "")
            field_repeatable = field.get("repeatable", False)
            
            if group_ref and group_ref != "none":
                # This is a nested group reference - check if the referenced group field exists
                if field.get("required", False):
                    if field_name not in item or not item[field_name]:
                        return True
                    # Recursively check the nested group
                    if field_repeatable:
                        if not isinstance(item[field_name], list) or len(item[field_name]) == 0:
                            return True
                        for nested_item in item[field_name]:
                            ref_schema = self._find_group_schema(group_ref, nested_schema)
                            if ref_schema and self._check_nested_required_fields(nested_item, ref_schema):
                                return True
                    else:
                        if isinstance(item[field_name], list):
                            if len(item[field_name]) == 0:
                                return True
                            nested_value = item[field_name][0]
                        else:
                            nested_value = item[field_name]
                        ref_schema = self._find_group_schema(group_ref, nested_schema)
                        if ref_schema and self._check_nested_required_fields(nested_value, ref_schema):
                            return True
            else:
                # Regular field
                if field.get("required", False):
                    if field_repeatable:
                        if field_name not in item or not isinstance(item[field_name], list) or len(item[field_name]) == 0:
                            return True
                        for val in item[field_name]:
                            if val in [None, ""]:
                                return True
                    else:
                        if item.get(field_name) in [None, ""]:
                            return True
        
        return False

    def _missing_slots_recursive(self, slots):
        missing = []
        for slot in slots:
            if slot.type == "group":
                if slot.repeatable:
                    # When repeatable is True, check list structure
                    if slot.required and (not slot.value or not isinstance(slot.value, list) or len(slot.value) == 0):
                        missing.append(slot.prompt)
                    for idx, item in enumerate(slot.value or []):
                        for field in (slot.schema or []):
                            if field.get("required", False) and (item.get(field["name"]) in [None, ""]):
                                missing.append(f"{field.get('prompt', field['name'])} (group '{slot.name}' item {idx+1})")
                else:
                    # When repeatable is False, check single dict structure
                    if slot.required and (not slot.value or not isinstance(slot.value, dict)):
                        missing.append(slot.prompt)
                    elif slot.value and isinstance(slot.value, dict):
                        for field in (slot.schema or []):
                            if field.get("required", False) and (slot.value.get(field["name"]) in [None, ""]):
                                missing.append(f"{field.get('prompt', field['name'])} (group '{slot.name}')")
            elif slot.type == "nested_group":
                if slot.repeatable:
                    # When repeatable is True, check list structure
                    if slot.required and (not slot.value or not isinstance(slot.value, list) or len(slot.value) == 0):
                        missing.append(slot.prompt)
                    for idx, item in enumerate(slot.value or []):
                        nested_missing = self._get_nested_missing_fields(item, slot.nested_schema, slot.name, idx + 1)
                        missing.extend(nested_missing)
                else:
                    # When repeatable is False, check single dict structure
                    if slot.required and (not slot.value or not isinstance(slot.value, dict)):
                        missing.append(slot.prompt)
                    elif slot.value and isinstance(slot.value, dict):
                        nested_missing = self._get_nested_missing_fields(slot.value, slot.nested_schema, slot.name, 1)
                        missing.extend(nested_missing)
            else:
                if slot.required and (not slot.value or not slot.verified):
                    missing.append(slot.prompt)
        return missing

    def _get_nested_missing_fields(self, item, nested_schema, group_name, item_idx):
        """Recursively get missing fields in a nested structure.
        
        Args:
            item: The item to check
            nested_schema: The schema for this level
            group_name: Name of the parent group
            item_idx: Index of the item in the parent group
            
        Returns:
            List of missing field descriptions
        """
        missing = []
        
        if not isinstance(item, dict):
            return missing
        
        for field in (nested_schema or []):
            field_name = field["name"]
            group_ref = field.get("groupRef", "")
            field_repeatable = field.get("repeatable", False)
            
            if group_ref and group_ref != "none":
                # This is a nested group reference
                if field.get("required", False):
                    if field_name not in item or not item[field_name]:
                        missing.append(f"{field.get('prompt', field_name)} (nested group '{group_name}' item {item_idx})")
                    else:
                        if field_repeatable:
                            if not isinstance(item[field_name], list) or len(item[field_name]) == 0:
                                missing.append(f"{field.get('prompt', field_name)} (nested group '{group_name}' item {item_idx})")
                            else:
                                for nested_idx, nested_item in enumerate(item[field_name]):
                                    ref_schema = self._find_group_schema(group_ref, nested_schema)
                                    if ref_schema:
                                        nested_missing = self._get_nested_missing_fields(
                                            nested_item, ref_schema, f"{group_name}.{field_name}", nested_idx + 1
                                        )
                                        missing.extend(nested_missing)
                        else:
                            if isinstance(item[field_name], list):
                                if len(item[field_name]) == 0:
                                    missing.append(f"{field.get('prompt', field_name)} (nested group '{group_name}' item {item_idx})")
                                else:
                                    nested_value = item[field_name][0]
                            else:
                                nested_value = item[field_name]
                            ref_schema = self._find_group_schema(group_ref, nested_schema)
                            if ref_schema:
                                nested_missing = self._get_nested_missing_fields(
                                    nested_value, ref_schema, f"{group_name}.{field_name}", 1
                                )
                                missing.extend(nested_missing)
            else:
                # Regular field
                if field.get("required", False):
                    if field_repeatable:
                        if field_name not in item or not isinstance(item[field_name], list) or len(item[field_name]) == 0:
                            missing.append(f"{field.get('prompt', field_name)} (nested group '{group_name}' item {item_idx})")
                        else:
                            for val in item[field_name]:
                                if val in [None, ""]:
                                    missing.append(f"{field.get('prompt', field_name)} (nested group '{group_name}' item {item_idx})")
                    else:
                        if item.get(field_name) in [None, ""]:
                            missing.append(f"{field.get('prompt', field_name)} (nested group '{group_name}' item {item_idx})")
        
        return missing

    def _get_nested_missing_field_names(self, item, nested_schema):
        """Get missing field names in a nested structure.
        
        Args:
            item: The item to check
            nested_schema: The schema for this level
            
        Returns:
            List of missing field names
        """
        missing = []
        
        if not isinstance(item, dict):
            return missing
        
        for field in (nested_schema or []):
            field_name = field["name"]
            group_ref = field.get("groupRef", "")
            field_repeatable = field.get("repeatable", False)
            
            if group_ref and group_ref != "none":
                # This is a nested group reference
                if field.get("required", False):
                    if field_name not in item or not item[field_name]:
                        missing.append(field_name)
                    else:
                        if field_repeatable:
                            if not isinstance(item[field_name], list) or len(item[field_name]) == 0:
                                missing.append(field_name)
                            else:
                                for nested_item in item[field_name]:
                                    ref_schema = self._find_group_schema(group_ref, nested_schema)
                                    if ref_schema:
                                        nested_missing = self._get_nested_missing_field_names(nested_item, ref_schema)
                                        missing.extend(nested_missing)
                        else:
                            if isinstance(item[field_name], list):
                                if len(item[field_name]) == 0:
                                    missing.append(field_name)
                                else:
                                    nested_value = item[field_name][0]
                            else:
                                nested_value = item[field_name]
                            ref_schema = self._find_group_schema(group_ref, nested_schema)
                            if ref_schema:
                                nested_missing = self._get_nested_missing_field_names(nested_value, ref_schema)
                                missing.extend(nested_missing)
            else:
                # Regular field
                if field.get("required", False):
                    if field_repeatable:
                        if field_name not in item or not isinstance(item[field_name], list) or len(item[field_name]) == 0:
                            missing.append(field_name)
                        else:
                            for val in item[field_name]:
                                if val in [None, ""]:
                                    missing.append(field_name)
                    else:
                        if item.get(field_name) in [None, ""]:
                            missing.append(field_name)
        
        return missing

    def _normalize_nested_structure(self, item, nested_schema):
        """Normalize a nested structure to match the expected schema.
        
        Args:
            item: The item to normalize
            nested_schema: The expected schema
            
        Returns:
            The normalized item
        """
        if not isinstance(item, dict):
            return item
        
        # For nested structures, we should preserve the original field names
        # since the validation logic expects the groupRef names to be present
        # Don't normalize field names for nested structures
        return item

    def execute(self, state: MessageState, **fixed_args: FixedArgs) -> MessageState:
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
        state = self._execute(state, **fixed_args)
        return state

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
                # For group, define based on repeatable flag
                group_properties = {}
                group_required = []
                for field in (slot.schema or []):
                    group_properties[field["name"]] = {
                        "type": PYTHON_TO_JSON_SCHEMA.get(field["type"], "string"),
                        "description": field.get("description", ""),
                    }
                    if field.get("required", False):
                        group_required.append(field["name"])
                
                if slot.repeatable:
                    # When repeatable is True, define as array of objects
                    parameters["properties"][slot.name] = {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": group_properties,
                            "required": group_required,
                        },
                        "description": slot.description,
                    }
                else:
                    # When repeatable is False, define as single object
                    parameters["properties"][slot.name] = {
                        "type": "object",
                        "properties": group_properties,
                        "required": group_required,
                        "description": slot.description,
                    }
            elif slot.type == "nested_group":
                # For nested group, define based on repeatable flag
                nested_properties = self._build_nested_openai_schema(slot.nested_schema)
                
                if slot.repeatable:
                    # When repeatable is True, define as array of objects
                    parameters["properties"][slot.name] = {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": nested_properties,
                        },
                        "description": slot.description,
                    }
                else:
                    # When repeatable is False, define as single object
                    parameters["properties"][slot.name] = {
                        "type": "object",
                        "properties": nested_properties,
                        "description": slot.description,
                    }
            elif slot.items:
                parameters["properties"][slot.name] = {
                    "type": "array",
                    "items": slot.items,
                }
            else:
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
            "required": [slot.name for slot in self.openai_slots if slot.required],
        }
        for slot in self.openai_slots:
            if hasattr(slot, "items") and slot.items:
                parameters["properties"][slot.name] = {
                    "type": "array",
                    "items": slot.items,
                }
            else:
                parameters["properties"][slot.name] = {
                    "type": PYTHON_TO_JSON_SCHEMA[slot.type],
                    "description": slot.description,
                }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

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

    def _format_slots(self, slots: list) -> list[Slot]:
        format_slots = []
        for slot in slots:
            if slot.get("type") == "group":
                format_slots.append(
                    Slot(
                        name=slot["name"],
                        type="group",
                        value=[],
                        description=slot.get("description", ""),
                        prompt=slot.get("prompt", ""),
                        required=slot.get("required", False),
                        schema=slot.get("schema", []),
                        repeatable=slot.get("repeatable", True),
                    )
                )
            elif slot.get("type") == "nested_group":
                format_slots.append(
                    Slot(
                        name=slot["name"],
                        type="nested_group",
                        value=[],
                        description=slot.get("description", ""),
                        prompt=slot.get("prompt", ""),
                        required=slot.get("required", False),
                        nested_schema=slot.get("nested_schema", []),
                        repeatable=slot.get("repeatable", True),
                    )
                )
            else:
                format_slots.append(
                    Slot(
                        name=slot["name"],
                        type=slot["type"],
                        value="",
                        description=slot.get("description", ""),
                        prompt=slot.get("prompt", ""),
                        required=slot.get("required", False),
                        items=slot.get("items", None),
                    )
                )
        return format_slots

    def _execute(self, state: MessageState, **fixed_args: FixedArgs) -> MessageState:
        """Execute the tool with the current state and fixed arguments.

        This method handles slot filling, parameter validation, and tool execution.
        It manages the execution flow, error handling, and state updates.

        Args:
            state (MessageState): The current message state.
            **fixed_args (FixedArgs): Additional fixed arguments for the tool.

        Returns:
            MessageState: The updated message state after tool execution.
        """
        response = ""  # Initialize as empty string
        slot_verification: bool = False
        reason: str = ""
        response: str = ""  # Initialize response variable

        # Check if we need to reset slots for a new node
        # If this tool has been called before, check if the current slots are different
        # from the previously stored slots (indicating a different node)
        def slot_schema_signature(slots):
            import json
            return [
                (
                    slot.name,
                    slot.type,
                    json.dumps(slot.schema, sort_keys=True) if hasattr(slot, 'schema') and slot.schema else None
                )
                for slot in slots
            ]

        if state.slots.get(self.name):
            previous_slots = state.slots[self.name]
            if slot_schema_signature(self.slots) != slot_schema_signature(previous_slots):
                log_context.info(
                    f"Slot configuration or schema changed, resetting slots"
                )
                # Reset slots to the current node's configuration
                state.slots[self.name] = [Slot.model_validate(slot.model_dump()) for slot in self.slots]
                self.slots = state.slots[self.name]
            else:
                # Load previous slots if they're from the same node and schema
                self.slots = state.slots[self.name]
        else:
            state.slots[self.name] = [Slot.model_validate(slot.model_dump()) for slot in self.slots]
            self.slots = state.slots[self.name]

        # init slot values saved in default slots
        self._init_slots(state)
        # do slotfilling (now with valueSource logic)
        chat_history_str: str = format_chat_history(state.function_calling_trajectory)
        slots: list[Slot] = self._fill_slots_recursive(self.slots, chat_history_str)
        log_context.info(f"{slots=}")

        # Check if any required slots are missing or unverified (including groups)
        missing_required = self._any_missing_required_recursive(slots)
        if missing_required:
            for slot in slots:
                if slot.type == "group":
                    # For group, check each item in value list
                    if slot.repeatable:
                        # When repeatable is True, check list structure
                        if not slot.value or not isinstance(slot.value, list):
                            response = slot.prompt
                            break
                        for idx, item in enumerate(slot.value):
                            missing_fields = [
                                field["name"]
                                for field in (slot.schema or [])
                                if field.get("required", False) and (item.get(field["name"]) in [None, ""])
                            ]
                            if missing_fields:
                                response = f"Please provide the following fields for group '{slot.name}' item {idx+1}: {', '.join(missing_fields)}."
                                break
                    else:
                        # When repeatable is False, check single dict structure
                        if not slot.value or not isinstance(slot.value, dict):
                            response = slot.prompt
                            break
                        missing_fields = [
                            field["name"]
                            for field in (slot.schema or [])
                            if field.get("required", False) and (slot.value.get(field["name"]) in [None, ""])
                        ]
                        if missing_fields:
                            response = f"Please provide the following fields for group '{slot.name}': {', '.join(missing_fields)}."
                            break
                elif slot.type == "nested_group":
                    # For nested group, check each item in value list
                    if slot.repeatable:
                        # When repeatable is True, check list structure
                        if not slot.value or not isinstance(slot.value, list):
                            response = slot.prompt
                            break
                        for idx, item in enumerate(slot.value):
                            missing_fields = self._get_nested_missing_field_names(item, slot.nested_schema)
                            if missing_fields:
                                response = f"Please provide the following fields for nested group '{slot.name}' item {idx+1}: {', '.join(missing_fields)}."
                                break
                    else:
                        # When repeatable is False, check single dict structure
                        if not slot.value or not isinstance(slot.value, dict):
                            response = slot.prompt
                            break
                        missing_fields = self._get_nested_missing_field_names(slot.value, slot.nested_schema)
                        if missing_fields:
                            response = f"Please provide the following fields for nested group '{slot.name}': {', '.join(missing_fields)}."
                            break
                else:
                    # if there is extracted slots values but haven't been verified
                    if slot.value and not slot.verified:
                        # check whether it verified or not
                        verification_needed: bool
                        thought: str
                        verification_needed, thought = self.slotfiller.verify_slot(
                            slot.model_dump(), chat_history_str, self.llm_config
                        )
                        if verification_needed:
                            response: str = slot.prompt + "The reason is: " + thought
                            slot_verification = True
                            reason = thought
                            break
                        else:
                            slot.verified = True
                            log_context.info(f"Slot '{slot.name}' verified successfully")
                    # if there is no extracted slots values, then should prompt the user to fill the slot
                    if not slot.value and slot.required:
                        response = slot.prompt
                        break

            state.status = StatusEnum.INCOMPLETE

        # Re-check if any required slots are still missing after verification
        missing_required = self._any_missing_required_recursive(slots)

        # if all required slots are filled and verified, then execute the function
        tool_success: bool = False
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
                **fixed_args,
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
                tool_success = True
            except ToolExecutionError as tee:
                log_context.error(traceback.format_exc())
                response = tee.extra_message
            except AuthenticationError as ae:
                log_context.error(traceback.format_exc())
                response = str(ae)
            except Exception as e:
                log_context.error(traceback.format_exc())
                response = str(e)
            log_context.info(f"Tool {self.name} response: {response}")
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
                    "content": str(response),
                }
            )
            state.status = (
                StatusEnum.COMPLETE if tool_success else StatusEnum.INCOMPLETE
            )

        state.trajectory[-1][-1].input = slots
        state.trajectory[-1][-1].output = str(response)

        if tool_success:
            # Tool execution success
            if self.isResponse:
                log_context.info(
                    "Tool exeuction COMPLETE, and the output is stored in response"
                )
                state.response = str(response)
            else:
                log_context.info(
                    "Tool execution COMPLETE, and the output is stored in message flow"
                )
                state.message_flow = (
                    state.message_flow
                    + f"Context from {self.name} tool execution: {str(response)}\n"
                )
        else:
            # Tool execution failed
            if slot_verification:
                log_context.info("Tool execution INCOMPLETE due to slot verification")
                state.message_flow = f"Context from {self.name} tool execution: {str(response)}\n Focus on the '{reason}' to generate the verification request in response please and make sure the request appear in the response."
            else:
                log_context.info(
                    "Tool execution INCOMPLETE due to tool execution failure"
                )
                # Make it clear that the LLM should ask the user for missing information
                missing_slots = self._missing_slots_recursive(slots)
                if missing_slots:
                    questions_text = " ".join(missing_slots)
                    state.message_flow = (
                        state.message_flow
                        + f"IMPORTANT: The tool cannot proceed without required information. You MUST ask the user for: {questions_text}\n"
                        + "Do NOT provide any facts or information until you have collected this required information from the user.\n"
                    )
                else:
                    state.message_flow = (
                        state.message_flow
                        + f"Context from {self.name} tool execution: {str(response)}\n"
                    )
        state.slots[self.name] = slots
        return state


def build_group_prompt(slot):
    # Build a schema-driven prompt for the slot group
    example_fields = []
    for field in (slot.schema or []):
        # Use the field's type, prompt, and description
        example_value = {
            "str": "\"example string\"",
            "int": "123",
            "float": "12.34",
            "bool": "true"
        }.get(field.get("type", "str"), "\"example\"")
        example_fields.append(f'"{field["name"]}": {example_value}')
    example_obj = "{" + ", ".join(example_fields) + "}"
    schema_lines = []
    for field in (slot.schema or []):
        schema_lines.append(
            f'- {field["name"]} ({field.get("type", "str")}): {field.get("description", field.get("prompt", ""))}'
        )
    schema_str = "\n".join(schema_lines)
    
    if slot.repeatable:
        return (
            f"Please provide a list of dictionaries (objects), e.g. [{{'key': 'value'}}], each matching this schema:\n"
            f"{schema_str}\n"
            f"Example:\n[{example_obj}]\n"
            f"IMPORTANT: Each object must have ALL the fields above, with the correct type. "
            f"Do not add extra fields. Return a list of dicts, each matching the schema exactly. "
            f"IMPORTANT: The field name is just a key, don't use it to find for the value. The value you provide must match the field's description and prompt, even if the user never says it directly.\n"
        )
    else:
        return (
            f"Please provide a single dictionary (object) matching this schema:\n"
            f"{schema_str}\n"
            f"Example:\n{example_obj}\n"
            f"IMPORTANT: The object must have ALL the fields above, with the correct type. "
            f"Do not add extra fields. Return a single dict matching the schema exactly. "
            f"IMPORTANT: The field name is just a key, don't use it to find for the value. The value you provide must match the field's description and prompt, even if the user never says it directly.\n"
        )
