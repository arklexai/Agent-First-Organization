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
from arklex.orchestrator.NLU.entities.slot_entities import (
    Slot,
)
from arklex.utils.exceptions import AuthenticationError, ToolExecutionError
from arklex.utils.logging_utils import LogContext
from arklex.utils.utils import format_chat_history

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
        if not slots:
            return

        # Process slots to handle schema/slot_schema mapping for all slot types
        processed_slots = []
        for slot in slots:
            if "schema" in slot:
                # Create a copy with slot_schema instead of schema for all slot types
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
                    # Handle schema/slot_schema mapping for all slot types
                    if key == "schema":
                        existing_slot.slot_schema = value
                    else:
                        setattr(existing_slot, key, value)
            else:
                if new_slot.get("slot_schema"):
                    # Handle slots with slot_schema for all types
                    self.slots.append(Slot(
                        name=new_slot["name"],
                        type=new_slot.get("type", "str"),
                        slot_schema=new_slot["slot_schema"],
                        required=new_slot.get("required", False),
                        repeatable=new_slot.get("repeatable", False),
                        prompt=new_slot.get("prompt", ""),
                        description=new_slot.get("description", ""),
                        value=new_slot.get("value", None),
                        valueSource=new_slot.get("valueSource", None),
                    ))
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
        """Fill slots recursively.

        Args:
            slots: List of slots to fill
            chat_history_str: Formatted chat history string

        Returns:
            List of filled slots
        """
        filled_slots = []
        if slots:
            filled = self.slotfiller.fill_slots(slots, chat_history_str, self.llm_config) # filled is a list of slots
            for i, slot in enumerate(slots):
                slot.value = self._convert_value(filled[i].value, slot.type) # we need to handle the case where all indexes are filled
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
            elif slot.items:
                parameters["properties"][slot.name] = {
                    "type": "array",
                    "items": slot.items,
                }
            else:
                # Use the slot's to_openai_schema method which handles all the complexity
                parameters["properties"][slot.name] = slot.to_openai_schema()
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
        }

    def to_openai_tool_def_v2(self) -> dict:
        # If any slot provides a full OpenAI function schema, use it directly
        for slot in self.slots:
            # Check for slot_schema first (new structure), then fall back to schema (legacy)
            schema_obj = getattr(slot, "slot_schema", None) or getattr(slot, "schema", None)
            if isinstance(schema_obj, dict) and ("function" in schema_obj or schema_obj.get("type") == "function"):
                function_block = schema_obj.get("function", {})
                # Ensure minimal structure
                if isinstance(function_block, dict) and function_block.get("parameters"):
                    return {
                        "type": "function",
                        "function": function_block,
                    }
        # Fallback: build schema from slots
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
                user_args = self._parse_input_args(raw_args, model_cls)
                
                # Update slots with parsed values
                self._update_slots_with_args(user_args)
                
                # Apply fixed values from schemas
                self._apply_schema_fixed_values()

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
            call_id: str = str(uuid.uuid4())
            log_context.info(f"call_id: {call_id}")
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
                    log_context.info(
                        f"Slot '{slot.name}' verified successfully"
                    )
            # if there is no extracted slots values, then should prompt the user to fill the slot
            if not slot.value and slot.required:
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

            # Set default value based on valueSource and required status
            value_source = getattr(slot, "valueSource", "prompt")
            if value_source == "fixed":
                default = getattr(slot, "value", "")
            elif getattr(slot, "required", False):
                default = None
            else:
                default = ...

            # Create field metadata
            metadata = {"description": getattr(slot, "description", "")}

            # Add enum values if available
            if hasattr(slot, "enum") and slot.enum:
                metadata["enum"] = slot.enum

            fields[slot.name] = (py_type, Field(default, **metadata))
            
        return fields

    def _create_model_class(self, fields: dict[str, tuple[type, Field]]) -> type:
        """Create Pydantic model class, using custom schema if available.
        
        Args:
            fields: Dictionary of field definitions.
            
        Returns:
            Pydantic model class.
        """
        # Use slot_schema directly if available (this is what the LLM needs to see)
        if len(self.slots) == 1 and hasattr(self.slots[0], "slot_schema") and self.slots[0].slot_schema:
            # The slot_schema contains the complete nested structure with correct field names
            import copy
            schema_copy = copy.deepcopy(self.slots[0].slot_schema)
            
            # Extract just the parameters part - OpenAI expects parameters, not the full function wrapper
            if 'function' in schema_copy and 'parameters' in schema_copy['function']:
                parameters_schema = schema_copy['function']['parameters']
            else:
                parameters_schema = schema_copy
            
            # Create a simple Pydantic model that returns our custom schema
            model_cls = create_model(f"{self.name}_InputModel", **{})
            def custom_schema() -> dict[str, Any]:
                return parameters_schema
            model_cls.model_json_schema = custom_schema
            return model_cls
        else:
            # Create the Pydantic model class from fields
            return create_model(f"{self.name}_InputModel", **fields)

    def _parse_input_args(self, raw_args: str, model_cls: type) -> dict[str, Any]:
        """Parse input arguments from JSON string.
        
        Args:
            raw_args: Raw JSON string arguments.
            model_cls: Pydantic model class for parsing.
            
        Returns:
            Parsed arguments dictionary.
        """
        # If we're using custom schema from slot_schema, parse JSON directly
        if len(self.slots) == 1 and hasattr(self.slots[0], "slot_schema") and self.slots[0].slot_schema:
            import json
            return json.loads(raw_args)
        else:
            return model_cls.model_validate_json(raw_args).model_dump()

    def _update_slots_with_args(self, user_args: dict[str, Any]) -> None:
        """Update slots with parsed argument values.
        
        Args:
            user_args: Dictionary of parsed arguments.
        """
        for slot in self.slots:
            if slot.name in user_args:
                slot.value = user_args[slot.name]

    def _apply_schema_fixed_values(self) -> None:
        """Apply fixed values from slot schemas using the new format processing."""
        try:
            # Build slot values using the same logic as the main execution path
            all_slots = [slot.model_dump() if hasattr(slot, "model_dump") else slot for slot in self.slots]
            processed_slots = self._build_slot_values(all_slots, {slot.name: slot.value for slot in self.slots})
            
            # Apply fixed/default values to slots with schema
            for slot_data in processed_slots:
                if slot_data.get("slot_schema"):
                    try:
                        from arklex.orchestrator.NLU.entities.slot_entities import (
                            apply_values_recursively,
                        )
                        apply_values_recursively(slot_data["value"], slot_data["slot_schema"], slot_data.get("name"))
                    except Exception as e:
                        log_context.warning(f"Failed to apply fixed values from schema for slot {slot_data.get('name')}: {e}")

            # Update self.slots with processed values
            for i, processed_slot in enumerate(processed_slots):
                if i < len(self.slots):
                    self.slots[i].value = processed_slot["value"]
                    
        except Exception as e:
            log_context.warning(f"Failed to apply schema fixed values: {e}")

    def _build_slot_values(self, schema: list[dict], tool_args: dict[str, Any]) -> list[dict]:
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