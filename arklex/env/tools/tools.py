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
        for slot in slots:
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
                        desc_or_prompt = field.get("description") or field.get("prompt") or ""
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
                # Enforce that the value is a list of dicts
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
                        raise ValueError(f"Slot group '{slot.name}' must be a list of dicts, got: {group_value}")
                # For each dict, apply valueSource logic as defined in the schema
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
                filled_slots.append(slot)
            else:
                filled = self.slotfiller.fill_slots([slot], chat_history_str, self.llm_config)
                slot.value = self._convert_value(filled[0].value, slot.type)
                filled_slots.append(slot)
        return filled_slots

    def _any_missing_required_recursive(self, slots):
        for slot in slots:
            if slot.type == "group":
                # For group, check if at least one item exists if required
                if slot.required and (not slot.value or not isinstance(slot.value, list) or len(slot.value) == 0):
                    return True
                # For each item, check required fields
                for item in (slot.value or []):
                    for field in (slot.schema or []):
                        if field.get("required", False) and (item.get(field["name"]) in [None, ""]):
                            return True
            else:
                if slot.required and (not slot.value or not slot.verified):
                    return True
        return False

    def _missing_slots_recursive(self, slots):
        missing = []
        for slot in slots:
            if slot.type == "group":
                if slot.required and (not slot.value or not isinstance(slot.value, list) or len(slot.value) == 0):
                    missing.append(slot.prompt)
                for idx, item in enumerate(slot.value or []):
                    for field in (slot.schema or []):
                        if field.get("required", False) and (item.get(field["name"]) in [None, ""]):
                            missing.append(f"{field.get('prompt', field['name'])} (group '{slot.name}' item {idx+1})")
            else:
                if slot.required and (not slot.value or not slot.verified):
                    missing.append(slot.prompt)
        return missing

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
                # For group, define as array of objects with schema
                group_properties = {}
                group_required = []
                for field in (slot.schema or []):
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
    return (
        f"Please provide a list of dictionaries (objects), e.g. [{{'key': 'value'}}], each matching this schema:\n"
        f"{schema_str}\n"
        f"Example:\n[{example_obj}]\n"
        f"IMPORTANT: Each object must have ALL the fields above, with the correct type. "
        f"Do not add extra fields. Return a list of dicts, each matching the schema exactly. "
        f"IMPORTANT: The field name is just a key, don't use it to find for the value. The value you provide must match the field's description and prompt, even if the user never says it directly.\n"
    )
