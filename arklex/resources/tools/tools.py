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
from arklex.orchestrator.nlu.core.slot import SlotFiller
from arklex.orchestrator.nlu.entities.slot_entities import (
    Slot,
)
from arklex.utils.logging.exceptions import AuthenticationError, ToolExecutionError
from arklex.utils.logging.logging_utils import LogContext
from arklex.utils.utils import format_chat_history

log_context = LogContext(__name__)

# Shared type mapping for JSON schema to Python types
JSON_SCHEMA_TO_PYTHON_TYPE = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


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
        self.runtime_args = {}

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
        for slot in slots:
            if "schema" in slot:
                self.slots.append(
                    Slot(
                        name=slot["name"],
                        type=slot.get("type", "str"),
                        slot_schema=slot["schema"],
                        required=slot.get("required", False),
                        repeatable=slot.get("repeatable", False),
                        prompt=slot.get("prompt", ""),
                        description=slot.get("description", ""),
                        value=slot.get("value", None),
                        valueSource=slot.get("valueSource", None),
                    )
                )
            else:
                self.slots.append(Slot.model_validate(slot))

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
        """Fill slots recursively.

        Args:
            slots: List of slots to fill
            chat_history_str: Formatted chat history string

        Returns:
            List of filled slots
        """
        filled_slots = []
        if slots:
            filled = self.slotfiller.fill_slots(
                slots, chat_history_str, self.llm_config
            )  # filled is a list of slots
            for i, slot in enumerate(slots):
                # propagate filled value and provenance
                slot.value = self._convert_value(filled[i].value, slot.type)
                try:
                    # carry over valueSource from filler result if present
                    if hasattr(filled[i], "valueSource"):
                        slot.valueSource = filled[i].valueSource
                    # mark verified if the filler marked it, or if value comes from fixed/default
                    if (
                        getattr(filled[i], "verified", False)
                        or getattr(slot, "valueSource", None) in ("fixed", "default")
                        and slot.value not in (None, "", [])
                    ):
                        slot.verified = True
                except Exception:
                    pass
                filled_slots.append(slot)
        return filled_slots

    def _is_missing_required(self, slots: list[Slot]) -> bool:
        for slot in slots:
            # Check if required slot is missing or unverified
            if slot.required and (not slot.value or not slot.verified):
                return True
        return False

    def _missing_slots_recursive(self, slots: list[Slot]) -> list[str]:
        missing = []

        for slot in slots:
            # Check if required slot is missing or unverified
            if slot.required:
                if (
                    getattr(slot, "valueSource", None) in ("fixed", "default")
                    and slot.value
                ):
                    continue
                if (not slot.value) or (not slot.verified):
                    # Prefer nested prompts when available
                    nested_prompts = self._collect_nested_required_prompts(slot)
                    if nested_prompts:
                        missing.extend(nested_prompts)
                    else:
                        missing.append(slot.prompt)
        # Filter out empty strings
        return [m for m in missing if m]

    def execute(
        self,
        state: OrchestratorState,
        all_slots: dict[str, list[Slot]],
        auth: dict[str, Any],
    ) -> tuple[OrchestratorState, ToolOutput]:
        """Execute the tool with the current state and fixed arguments.

        This method is a wrapper around _execute that handles the execution flow
        and state management.
        """
        self.llm_config = state.bot_config.llm_config.model_dump()
        state, tool_output = self._execute(state, all_slots, auth)
        return state, tool_output

    def to_openai_agents_function_tool(self) -> "FunctionTool":
        """Convert this Arklex tool to an OpenAI Agents FunctionTool.

        This method creates a FunctionTool that can be used with the OpenAI Agents SDK.
        It handles parameter conversion, schema generation, and function wrapping.

        Returns:
            FunctionTool: An OpenAI Agents FunctionTool instance.

        Raises:
            ImportError: If OpenAI Agents SDK is not available.
        """
        # Build Pydantic model fields from slots
        fields = self._build_pydantic_fields()

        # Create the Pydantic model class or use custom schema if available
        model_cls = self._create_model_class(fields)

        # Create the async wrapper function
        async def on_invoke(ctx: RunContextWrapper[Any], raw_args: str) -> str:
            log_context.info(f"on_invoke tool {self.name}, input: {raw_args}")

            try:
                # Parse input arguments
                user_args = json.loads(raw_args)
                log_context.info(f"Parsed user_args: {user_args}")

                # Apply fixed values from schemas
                self._apply_schema_fixed_values()

                # Update slots with parsed values (but don't override fixed values)
                self._update_slots_with_args(user_args)

                # Merge with fixed arguments
                merged_args = {
                    "slots": self.slots,
                    "auth": self.auth,
                    "node_specific_data": self.node_specific_data,
                    **self.fixed_args,
                    **self.runtime_args,
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
            # mark this as False to allow the optional json fields, which is not recommended by openai (https://github.com/openai/openai-agents-python/blob/9078e29c0c4134d1b850dcaf936a4ef8975d6fcb/src/agents/function_schema.py#L39)
            # If we keep it as True, the optional fields will still appear in the required fields list, and we need to use description to prompt the agent to fill the optional fields. (https://github.com/openai/openai-agents-python/issues/43#issuecomment-2722829809)
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
        log_context.info(f"slots: {slots}")
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
                kwargs["slots"] = slots

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
            call_id: str = str(uuid.uuid4())
            log_context.info(f"call_id: {call_id}")
            # update the slots to dict so the kwargs can be serialized
            kwargs["slots"] = [
                slot.model_dump() if hasattr(slot, "model_dump") else slot
                for slot in slots
            ]
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

        state.trajectory[-1][-1].input = slots
        state.trajectory[-1][-1].output = str(tool_output)

        if tool_output.status == StatusEnum.INCOMPLETE:
            # Tool execution failed
            if slot_verification:
                log_context.info("Tool execution INCOMPLETE due to slot verification")
                tool_output.message_flow = f"Context from {self.name} tool execution: {str(tool_output.message_flow)}\n Focus on the '{reason}' to generate the verification request in response please and make sure the request appear in the response."
            else:
                # Make it clear that the LLM should ask the user for missing information
                log_context.info(
                    "Tool execution INCOMPLETE due to tool execution failure"
                )
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

        def _collect_missing_prompts(slot: Slot) -> list[str]:
            return self._collect_nested_required_prompts(slot)

        for slot in slots:
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
                    log_context.info(f"Slot '{slot.name}' verified successfully")
            # if there is no extracted slots values, then should prompt the user to fill the slot
            if not slot.value and slot.required:
                # Try to surface nested required prompts from schema; fall back to top-level prompt
                nested_prompts = _collect_missing_prompts(slot)
                if nested_prompts:
                    return " ".join(nested_prompts), False
                return slot.prompt, False  # Missing slot

        return "", False

    def _build_pydantic_fields(self) -> dict[str, tuple[type, Field]]:
        """Build Pydantic model fields from slots.

        Returns:
            Dictionary mapping field names to (type, Field) tuples.
        """
        fields = {}
        for slot in self.slots:
            # Convert slot type to Python type
            py_type = self._slot_type_to_python_type(slot.type)

            # Use slot_schema if available to get the correct type
            if hasattr(slot, "slot_schema") and slot.slot_schema:
                py_type = self._extract_type_from_slot_schema(slot)

            # Set default value based on valueSource and required status
            value_source = getattr(slot, "valueSource", "prompt")
            if value_source == "fixed":
                default = getattr(slot, "value", "")
            elif not getattr(
                slot, "required", False
            ):  # set default to None if slot is not required
                default = None
                py_type = py_type | None
            else:
                default = ...

            # Create field metadata
            metadata = {"description": getattr(slot, "description", "")}

            # Add enum values if available
            if hasattr(slot, "enum") and slot.enum:
                metadata["enum"] = slot.enum

            fields[slot.name] = (py_type, Field(default, **metadata))

        return fields

    def _extract_type_from_slot_schema(self, slot: Slot) -> type:
        """Extract Python type from slot_schema.

        Args:
            slot: Slot object with slot_schema

        Returns:
            Python type for the slot
        """

        slot_schema = slot.slot_schema

        # Handle function-style schema
        if isinstance(slot_schema, dict) and "function" in slot_schema:
            params = slot_schema.get("function", {}).get("parameters", {})
            props = params.get("properties", {})
            prop_schema = props.get(slot.name, params)
        else:
            prop_schema = slot_schema

        # Extract type from schema
        schema_type = prop_schema.get("type", "string")

        if schema_type == "array":
            items_schema = prop_schema.get("items", {})
            items_type = items_schema.get("type", "string")

            # Map to Python types using shared mapping
            base_type = JSON_SCHEMA_TO_PYTHON_TYPE.get(items_type, str)
            return list[base_type]
        elif schema_type == "object":
            return dict
        else:
            # Primitive types - use shared mapping
            return JSON_SCHEMA_TO_PYTHON_TYPE.get(schema_type, str)

    def _create_model_class(self, fields: dict[str, tuple[type, Field]]) -> type:
        """Create Pydantic model class, using custom schema if available.

        Args:
            fields: Dictionary of field definitions.

        Returns:
            Pydantic model class.
        """
        # Use slot_schema directly if available (this is what the LLM needs to see)
        # Check if all slots have slot_schema
        all_slots_have_schema = all(
            hasattr(slot, "slot_schema") and slot.slot_schema for slot in self.slots
        )

        if all_slots_have_schema:
            # Combine all slot schemas into a single parameters schema
            import copy

            # Start with a base schema structure
            combined_schema = {"type": "object", "properties": {}, "required": []}

            # Extract properties from each slot's schema
            for slot in self.slots:
                slot_schema = slot.slot_schema

                # Extract the parameters part
                if isinstance(slot_schema, dict) and "function" in slot_schema:
                    params = slot_schema.get("function", {}).get("parameters", {})
                else:
                    params = slot_schema

                # Get the specific property for this slot
                props = params.get("properties", {})
                if slot.name in props:
                    # Deep copy the property to avoid modifying the original
                    import copy

                    slot_prop = copy.deepcopy(props[slot.name])

                    # Pre-populate fixed/default values in the schema
                    self._prepopulate_fixed_values_in_schema(slot_prop)

                    combined_schema["properties"][slot.name] = slot_prop
                    if slot.required:
                        combined_schema["required"].append(slot.name)

            # Create a simple Pydantic model that returns our custom schema
            model_cls = create_model(f"{self.name}_InputModel", **{})

            def custom_schema() -> dict[str, Any]:
                return combined_schema

            model_cls.model_json_schema = custom_schema
            return model_cls
        else:
            # Create the Pydantic model class from fields
            return create_model(f"{self.name}_InputModel", **fields)

    def _prepopulate_fixed_values_in_schema(self, schema_part: dict) -> None:
        """Pre-populate fixed and default values in schema to prevent LLM from asking for them.

        Args:
            schema_part: Part of the schema to process (can be nested)
        """
        if not isinstance(schema_part, dict):
            return

        # Handle array items
        if schema_part.get("type") == "array" and "items" in schema_part:
            self._prepopulate_fixed_values_in_schema(schema_part["items"])

        # Handle object properties
        elif schema_part.get("type") == "object" and "properties" in schema_part:
            properties = schema_part["properties"]
            for field_name, field_def in properties.items():
                if isinstance(field_def, dict):
                    value_source = field_def.get("valueSource")
                    if value_source == "fixed" and "value" in field_def:
                        # Pre-populate the field with the fixed value
                        field_def["default"] = field_def["value"]
                        # Remove from required since they can't be changed
                        required_fields = schema_part.get("required", [])
                        if field_name in required_fields:
                            required_fields.remove(field_name)
                    elif value_source == "default" and "value" in field_def:
                        # For default values, don't set as default in schema
                        # The default will be applied during execution if user doesn't provide a value
                        # Keep in required array so LLM will ask for it
                        pass
                    else:
                        # Recursively process nested structures
                        self._prepopulate_fixed_values_in_schema(field_def)

    def _collect_nested_required_prompts(self, slot: Slot) -> list[str]:
        """Collect prompts/descriptions for required nested fields within slot.slot_schema.

        Handles both object slots and arrays of objects. Falls back to field path when prompt/description missing.
        """

        def _collect_from_field(
            field_name: str, field_def: dict, path: str = ""
        ) -> list[str]:
            prompts: list[str] = []
            current_path = f"{path}.{field_name}" if path else field_name
            text = (
                field_def.get("prompt") or field_def.get("description") or current_path
            )
            prompts.append(text)
            if field_def.get("type") == "object":
                nested_props = field_def.get("properties", {}) or {}
                nested_required = field_def.get("required", []) or []
                for nested_name in nested_required:
                    nested_def = nested_props.get(nested_name, {})
                    prompts.extend(
                        _collect_from_field(nested_name, nested_def, current_path)
                    )
            elif field_def.get("type") == "array":
                items = field_def.get("items", {}) or {}
                if items.get("type") == "object":
                    nested_props = items.get("properties", {}) or {}
                    nested_required = items.get("required", []) or []
                    for nested_name in nested_required:
                        nested_def = nested_props.get(nested_name, {})
                        prompts.extend(
                            _collect_from_field(
                                nested_name, nested_def, current_path + "[]"
                            )
                        )
            return prompts

        try:
            schema_obj = getattr(slot, "slot_schema", None)
            if not isinstance(schema_obj, dict):
                return []
            function_block = schema_obj.get("function", {})
            parameters = function_block.get("parameters", {})
            properties = parameters.get("properties", {}) or {}
            slot_def = properties.get(slot.name, {})
            if not slot_def:
                return []
            # Object slot
            if slot_def.get("type") == "object":
                req = slot_def.get("required", []) or []
                props = slot_def.get("properties", {}) or {}
                prompts: list[str] = []
                for fname in req:
                    fdef = props.get(fname, {})
                    prompts.extend(_collect_from_field(fname, fdef, slot.name))
                return prompts
            # Array of objects slot
            if slot_def.get("type") == "array":
                items = slot_def.get("items", {}) or {}
                if items.get("type") == "object":
                    req = items.get("required", []) or []
                    props = items.get("properties", {}) or {}
                    prompts: list[str] = []
                    for fname in req:
                        fdef = props.get(fname, {})
                        prompts.extend(
                            _collect_from_field(fname, fdef, slot.name + "[]")
                        )
                    return prompts
        except Exception:
            return []
        return []

    def _update_slots_with_args(self, user_args: dict[str, Any]) -> None:
        """Update slots with parsed argument values.

        Args:
            user_args: Dictionary of parsed arguments.
        """
        for slot in self.slots:
            if slot.name in user_args:
                # Don't override fixed values - they should take precedence
                value_source = getattr(slot, "valueSource", "prompt")
                if value_source == "fixed":
                    log_context.info(
                        f"Skipping user arg for fixed slot '{slot.name}' (keeping fixed value)"
                    )
                    continue
                log_context.info(
                    f"Updating slot '{slot.name}' with value: {user_args[slot.name]}"
                )
                slot.value = user_args[slot.name]
            else:
                log_context.info(f"Slot '{slot.name}' not found in user_args")

    def _apply_schema_fixed_values(self) -> None:
        """Apply fixed values from slot schemas using the new format processing."""
        try:
            # Build slot values using the same logic as the main execution path
            for slot in self.slots:
                value_source = getattr(slot, "valueSource", "prompt")

                # For fixed values, always use the fixed value regardless of current value
                if value_source == "fixed":
                    fixed_value = getattr(slot, "value", None)
                    if fixed_value is not None:
                        slot.value = fixed_value
                        log_context.info(
                            f"Applied fixed value '{fixed_value}' to slot '{slot.name}'"
                        )

                # For default values, only use if current value is empty/None
                elif value_source == "default":
                    default_value = getattr(slot, "value", None)
                    if default_value is not None and (
                        not slot.value or slot.value == ""
                    ):
                        slot.value = default_value
                        log_context.info(
                            f"Applied default value '{default_value}' to slot '{slot.name}'"
                        )

            # Apply fixed/default values to slots with schema
            for slot in self.slots:
                if hasattr(slot, "slot_schema") and slot.slot_schema:
                    try:
                        from arklex.orchestrator.nlu.entities.slot_entities import (
                            apply_values_recursively,
                        )

                        apply_values_recursively(
                            slot.value, slot.slot_schema, slot.name
                        )
                    except Exception as e:
                        log_context.warning(
                            f"Failed to apply fixed values from schema for slot {slot.name}: {e}"
                        )

        except Exception as e:
            log_context.warning(f"Failed to apply schema fixed values: {e}")

    def _build_slot_values(
        self, schema: list[dict], tool_args: dict[str, Any]
    ) -> list[dict]:
        """Build slot values from schema using type conversion and valueSource logic.

        Args:
            schema: List of slot schema dictionaries.
            tool_args: Dictionary of tool arguments.

        Returns:
            List of processed slot dictionaries.
        """
        result = []
        for slot in schema:
            name = slot["name"]
            slot_type = slot["type"]
            value_source = slot.get("valueSource", "prompt")

            # Determine slot value based on valueSource
            if value_source == "fixed":
                slot_value = slot.get("value", "")
            elif value_source == "default":
                slot_value = tool_args.get(name, slot.get("value", ""))
            else:  # prompt or anything else
                slot_value = tool_args.get(name, "")

            # Apply type conversion
            slot_value = self._convert_value(slot_value, slot_type)

            # Create result slot dictionary
            slot_dict = slot.copy()
            slot_dict["value"] = slot_value
            result.append(slot_dict)

        return result
