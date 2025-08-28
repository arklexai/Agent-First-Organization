import json
import re
from typing import Any

from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from arklex.env.agents.agent import BaseAgent, register_agent
from arklex.env.prompts import load_prompts
from arklex.env.tools.tools import TYPE_CONVERTERS
from arklex.orchestrator.NLU.entities.slot_entities import (
    convert_value_for_type,
    extract_fields_from_properties,
    find_fixed_default_fields_recursive,
    apply_fields_to_item_recursive,
    apply_values_recursively,
)
from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
)
from arklex.types.resource_types import ToolItem
from arklex.types.stream_types import EventType, StreamType
from arklex.utils.logging_utils import LogContext
from arklex.utils.provider_utils import validate_and_get_model_class

log_context = LogContext(__name__)


class OpenAIAgentData(BaseModel):
    """Data for the OpenAIAgent."""

    prompt: str


class OpenAIAgentOutput(BaseModel):
    """Output for the OpenAIAgent."""

    response: str


@register_agent
class OpenAIAgent(BaseAgent):
    description: str = "General-purpose Arklex agent for chat or voice."

    def __init__(
        self,
        successors: list,
        predecessors: list,
        tools: dict,
    ) -> None:
        super().__init__()
        self.prompt: str = ""
        self.llm: BaseChatModel | None = None
        self.available_tools: dict[str, tuple[dict[str, Any], Any]] = {}
        self.tool_map = {}
        self.tool_defs = []
        self.tool_args: dict[str, Any] = {}
        self.tool_slots: dict[str, Any] = {}
        self.tool_name_mapping: dict[str, str] = {}  # sanitized_name -> original_name

        self._load_tools(successors=successors, predecessors=predecessors, tools=tools)
        self._configure_tools()

        log_context.info(f"OpenAIAgent initialized with {len(self.tool_defs)} tools.")

    def _load_tools(self, successors: list, predecessors: list, tools: dict) -> None:
        """
        Load tools for the agent.
        This method is called during the initialization of the agent.
        """
        self.tools = tools.copy()
        self.http_tools = []
        all_nodes = successors + predecessors
        
        for node in all_nodes:
            
            if (
                node.resource.get("id") not in tools
                and node.resource.get("id") != ToolItem.HTTP_TOOL
            ):
                log_context.warning(
                    f"Tool {node.resource.get('id')} not found for openai agent"
                )
                continue

            if node.resource.get("id") == ToolItem.HTTP_TOOL:
                http_tool_id = node.data.get("name", "")
                self.available_tools[http_tool_id] = tools[http_tool_id]
                self.http_tools.append(http_tool_id)

            else:
                tool_id = node.resource.get("id")
                self.available_tools[tool_id] = tools[tool_id]

    def _configure_tools(self) -> None:
        """
        Configure tools for the agent.
        This method is called during the initialization of the agent.
        """
        for tool_id, tool in self.available_tools.items():
            tool_object = tool["tool_instance"]
            tool_def = tool_object.to_openai_tool_def_v2()
            
            # Sanitize tool name for OpenAI (only allow alphanumeric, underscore, hyphen)
            sanitized_tool_id = tool_id.replace("/", "_").replace(" ", "_").replace("-", "_")
            # Remove any other invalid characters
            sanitized_tool_id = re.sub(r'[^a-zA-Z0-9_-]', '_', sanitized_tool_id)
            
            tool_def["function"]["name"] = sanitized_tool_id
            self.tool_defs.append(tool_def)
            self.tool_slots[sanitized_tool_id] = tool_object.slots.copy()
            self.tool_map[sanitized_tool_id] = tool_object.func
            self.tool_name_mapping[sanitized_tool_id] = tool_id  # Store mapping
            combined_args: dict[str, Any] = {
                "node_specific_data": tool_object.node_specific_data,
            }
            self.tool_args[sanitized_tool_id] = combined_args

    def init_agent_data(
        self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
    ) -> None:
        """Initialize the agent data.

        Args:
            orch_state (OrchestratorState): The current orchestrator state.
            node_specific_data (dict[str, Any]): Additional keyword arguments for the execution.
        """
        self.orch_state = orch_state
        self.openai_agent_data: OpenAIAgentData = OpenAIAgentData(
            **node_specific_data,
        )

    def _prepare_prompt(self, state: OrchestratorState, is_speech: bool = False) -> str:
        """Prepare the input prompt for generation."""
        if self.prompt:
            return self.prompt

        prompts: dict[str, str] = load_prompts(state.bot_config)

        # Choose prompt based on speech flag
        if is_speech:
            prompt_key = "function_calling_agent_prompt_speech"
        else:
            prompt_key = "function_calling_agent_prompt"

        prompt: PromptTemplate = PromptTemplate.from_template(prompts[prompt_key])
        input_prompt = prompt.invoke(
            {
                "sys_instruct": state.sys_instruct,
            }
        )
        return input_prompt.text

    def _add_prompt_to_trajectory(
        self, state: OrchestratorState, input_prompt: str
    ) -> None:
        """Add the input prompt to the function calling trajectory if not already present."""
        if not any(
            message.get("content") == input_prompt
            for message in state.function_calling_trajectory
        ):
            state.function_calling_trajectory.append(
                SystemMessage(content=input_prompt).model_dump()
            )

    def _process_tool_calls(
        self, state: OrchestratorState, ai_message: AIMessage
    ) -> None:
        """Process tool calls and update the function calling trajectory."""
        if not ai_message.tool_calls:
            return

        log_context.info("Processing tool calls.")
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call.get("name")
            if tool_name in self.tool_map:
                # Ensure tool_call has proper structure
                tool_call_id = tool_call.get("id", f"call_{tool_name}")
                tool_call_args = tool_call.get("args", {})

                # Create properly structured tool call
                structured_tool_call = {
                    "name": tool_name,
                    "args": tool_call_args,
                    "id": tool_call_id,
                }

                state.function_calling_trajectory.append(
                    AIMessage(
                        content=f"Calling tool: {tool_name}",
                        tool_calls=[structured_tool_call],
                    ).model_dump()
                )

                # Prepare arguments for tool execution
                tool_args = {
                    **tool_call_args,
                    **self.tool_args.get(tool_name, {}),
                }

                # Call tool with unified interface
                tool_response = self._execute_tool(tool_name, state, tool_args)

                state.function_calling_trajectory.append(
                    ToolMessage(
                        name=tool_name,
                        content=json.dumps(tool_response),
                        tool_call_id=tool_call_id,
                    ).model_dump()
                )
            else:
                log_context.warning(f"Tool {tool_name} not found in tool map.")

    def _stream_response(
        self, state: OrchestratorState, final_chain: BaseChatModel
    ) -> str:
        """Stream the response and put chunks in the message queue."""
        answer = ""
        for chunk in final_chain.stream(state.function_calling_trajectory):
            if hasattr(chunk, "content") and chunk.content:
                answer += chunk.content
                state.message_queue.put(
                    {"event": EventType.CHUNK.value, "message_chunk": chunk.content}
                )
        return answer

    def _execute_tool(
        self, tool_name: str, state: OrchestratorState, tool_args: dict[str, Any]
    ) -> Any:  # noqa: ANN401
        """Execute a tool with unified interface.

        This method handles the different calling patterns for different types of tools.
        For http_tool, it prepares the slots parameter. For other tools, it passes state directly.

        Args:
            tool_name: Name of the tool to execute
            state: Current message state
            tool_args: Arguments for the tool

        Returns:
            Tool execution result
        """

        def build_slot_values(
            schema: list[dict[str, Any]], tool_args: dict[str, Any]
        ) -> list[dict[str, Any]]:
            def type_convert(value: object, slot_type: str) -> object:
                if value is None:
                    return value
                try:
                    converter = TYPE_CONVERTERS.get(slot_type)
                    if converter:
                        return converter(value)
                    return value
                except Exception:
                    return value

            def flatten_group_items(group_items: list[Any]) -> list[dict[str, Any]]:
                result: list[dict[str, Any]] = []
                for item in group_items:
                    if isinstance(item, list):
                        flat = {slot["name"]: slot["value"] for slot in item}
                        result.append(flat)
                    else:
                        result.append(item)
                return result

            def iter_group_fields(slot_def: dict[str, Any]) -> list[dict[str, Any]]:
                # Expect new OpenAI-style slot_schema only
                slot_schema = slot_def.get("slot_schema")
                if isinstance(slot_schema, (list, tuple)):
                    return list(slot_schema)
                if not isinstance(slot_schema, dict) or "function" not in slot_schema:
                    return []
                try:
                    function_block = slot_schema.get("function", {})
                    parameters = function_block.get("parameters", {})
                    properties = parameters.get("properties", {})
                    group_prop = properties.get(slot_def.get("name"))
                    if not group_prop:
                        return []
                    items = group_prop.get("items", {}) if group_prop.get("type") == "array" else group_prop
                    if items.get("type") != "object":
                        return []
                    inner_props = items.get("properties", {})
                    required_fields = set(items.get("required", []))
                    fields: list[dict[str, Any]] = []
                    for field_name, field_def in inner_props.items():
                        json_type = field_def.get("type", "string")
                        if json_type == "array":
                            item_type = (field_def.get("items", {}) or {}).get("type", "string")
                            repeatable = True
                        else:
                            item_type = json_type
                            repeatable = False
                        internal_type = {
                            "string": "str",
                            "integer": "int",
                            "number": "float",
                            "boolean": "bool",
                        }.get(item_type, "str")
                        field_entry: dict[str, Any] = {
                            "name": field_name,
                            "type": internal_type,
                            "description": field_def.get("description", ""),
                            "prompt": field_def.get("prompt", ""),
                            "required": field_name in required_fields,
                            "repeatable": repeatable,
                            "valueSource": field_def.get("valueSource"),
                        }
                        if "value" in field_def:
                            field_entry["value"] = field_def.get("value")
                        fields.append(field_entry)
                    return fields
                except Exception:
                    return []

            def reapply_group_fixed_default(fields: list[dict[str, Any]], obj: dict[str, Any]) -> dict[str, Any]:
                # Ensure fixed overrides; default applies only if missing
                for f in fields:
                    name = f.get("name")
                    vs = f.get("valueSource")
                    if vs == "fixed" and "value" in f:
                        fixed_value = f.get("value")
                        converted_value = TYPE_CONVERTERS.get(f.get("type", "str"), lambda x: x)(fixed_value)
                        obj[name] = converted_value
                    elif vs == "default" and "value" in f and (obj.get(name) in (None, "")):
                        default_value = f.get("value")
                        converted_value = TYPE_CONVERTERS.get(f.get("type", "str"), lambda x: x)(default_value)
                        obj[name] = converted_value
                return obj

            result = []
            for slot in schema:
                name = slot["name"]
                slot_type = slot["type"]
                value_source = slot.get("valueSource", "prompt")
                slot_value = None

                if slot_type == "group":
                    fields = iter_group_fields(slot)
                    if slot.get("repeatable", False):
                        group_values = tool_args.get(name, [])
                        if (
                            not group_values
                            and value_source == "default"
                            or not group_values
                            and value_source == "fixed"
                        ):
                            group_values = [slot.get("value", "")]
                        # TODO: temporary fix for slot group values (should be list of dicts instead of dict)
                        if isinstance(group_values, dict):
                            group_values = [group_values]
                        slot_value = [
                            build_slot_values(fields, item)
                            for item in group_values
                        ]
                        slot_value = flatten_group_items(slot_value)
                        # Reapply fixed/default at the group field level to override any user-provided values
                        slot_value = [reapply_group_fixed_default(fields, item) for item in slot_value]
                    else:
                        group_value = tool_args.get(name, {})
                        if (
                            not group_value
                            and value_source == "default"
                            or not group_value
                            and value_source == "fixed"
                        ):
                            group_value = slot.get("value", "")
                        slot_list = build_slot_values(fields, group_value)
                        # Convert list of slot dicts to single object for non-repeatable groups
                        slot_value = {
                            slot_dict["name"]: slot_dict["value"]
                            for slot_dict in slot_list
                        }
                        slot_value = reapply_group_fixed_default(fields, slot_value)
                else:
                    if value_source == "fixed":
                        slot_value = slot.get("value", "")
                    elif value_source == "default":
                        slot_value = tool_args.get(name, slot.get("value", ""))
                    else:  # prompt or anything else
                        slot_value = tool_args.get(name, "")
                    slot_value = type_convert(slot_value, slot_type)

                slot_dict = slot.copy()
                slot_dict["value"] = slot_value
                result.append(slot_dict)
            return result

        # Check if this is an HTTP tool using the mapping
        original_tool_name = self.tool_name_mapping.get(tool_name)
        is_http_tool = original_tool_name in self.http_tools if original_tool_name else False
        
        if is_http_tool:
            # This is an HTTP tool
            all_slots = self.tool_slots.get(tool_name, [])
            slots = build_slot_values(
                [
                    slot.model_dump() if hasattr(slot, "model_dump") else slot
                    for slot in all_slots
                ],
                tool_args,
            )
            
            # Apply fixed/default values to slots before calling HTTP tool
            for slot in slots:
                if slot.get("slot_schema"):
                    self._apply_fixed_default_values(slot)
            
            # Call http_tool with slots parameter, excluding slots from tool_args
            filtered_args = {k: v for k, v in tool_args.items() if k != "slots"}
            return self.tool_map[tool_name](slots=slots, **filtered_args)
        else:
            # Call other tools with state parameter
            return self.tool_map[tool_name](state=state, **tool_args)

    def generate_response(
        self, state: OrchestratorState, stream: bool = False, is_speech: bool = False
    ) -> tuple[OrchestratorState, OpenAIAgentOutput]:
        """Unified response generation method with optional streaming."""
        input_prompt = self._prepare_prompt(state, is_speech)
        self._add_prompt_to_trajectory(state, input_prompt)

        final_chain = self.llm
        ai_message: AIMessage = final_chain.invoke(state.function_calling_trajectory)

        log_context.info(f"Generated answer: {ai_message}")

        # Process tool calls first
        self._process_tool_calls(state, ai_message)

        # Generate final response
        if ai_message.tool_calls:
            # After tool execution, get final response
            if stream:
                answer = self._stream_response(state, final_chain)
            else:
                ai_message = final_chain.invoke(state.function_calling_trajectory)
                answer = ai_message.content
        else:
            # No tool calls
            if stream:
                answer = self._stream_response(state, final_chain)
            else:
                answer = ai_message.content

        state.message_flow = ""
        agent_output = OpenAIAgentOutput(response=answer)

        # if not stream:
        #     state = trace(input=answer, state=state)

        return state, agent_output

    def _execute(self) -> tuple[OrchestratorState, OpenAIAgentOutput]:
        model_class = validate_and_get_model_class(
            self.orch_state.bot_config.llm_config
        )

        self.llm = model_class(
            model=self.orch_state.bot_config.llm_config.model_type_or_path
        )
        self.llm = self.llm.bind_tools(self.tool_defs)
        self.prompt: str = self.openai_agent_data.prompt
        if self.orch_state.stream_type == StreamType.TEXT:
            return self.generate_response(self.orch_state, stream=True, is_speech=False)
        elif self.orch_state.stream_type == StreamType.SPEECH:
            return self.generate_response(self.orch_state, stream=True, is_speech=True)
        else:
            return self.generate_response(self.orch_state, stream=False)

    def _apply_fixed_default_values(self, slot: dict) -> None:
        """Apply fixed and default values from slot_schema to slot values recursively.
        
        Args:
            slot: Slot dictionary with slot_schema and value
        """
        slot_schema = slot.get("slot_schema", {})
        slot_value = slot.get("value")
        
        if not slot_schema or not slot_value:
            return
            
        # Apply fixed/default values recursively to the slot value
        apply_values_recursively(slot_value, slot_schema, slot.get("name"))
