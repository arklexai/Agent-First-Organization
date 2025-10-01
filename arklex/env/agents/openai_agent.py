import json
import re
import traceback
from typing import Any

from jinja2 import Template
from pydantic import BaseModel
from agents import Agent, Runner, ItemHelpers

from arklex.env.agents.agent import BaseAgent, register_agent, AgentOutput
from arklex.env.agents.entities import PromptVariable
from arklex.env.prompts import load_prompts
from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
    StatusEnum,
)
from arklex.orchestrator.entities.taskgraph_entities import StatusEnum as TaskGraphStatusEnum
from arklex.orchestrator.NLU.entities.slot_entities import (
    apply_values_recursively,
)
from arklex.types.resource_types import ToolItem
from arklex.types.stream_types import EventType, StreamType
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class OpenAIAgentData(BaseModel):
    """Data for the OpenAIAgent."""

    name: str
    prompt: str
    prompt_variables: list[PromptVariable] = []


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
        self.available_tools: dict[str, tuple[dict[str, Any], Any]] = {}
        self.tool_map = {}
        self.agents_sdk_tools = []
        self.tool_args: dict[str, Any] = {}
        self.tool_slots: dict[str, Any] = {}
        self.tool_name_mapping: dict[str, str] = {}  # sanitized_name -> original_name
        self.handoffs = []  # Store handoffs for later use
        self._handoff_detected = False  # Updated per turn based on SDK result

        self._load_tools(successors=successors, predecessors=predecessors, tools=tools)
        self._configure_tools()

        log_context.info(
            f"OpenAIAgent initialized with {len(self.agents_sdk_tools)} tools."
        )

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
        Configure tools for the Agents SDK.
        This method is called during the initialization of the agent.
        """
        self.agents_sdk_tools = []
        for tool_id, tool in self.available_tools.items():
            tool_object = tool["tool_instance"]
            try:
                sdk_tool = tool_object.to_openai_agents_function_tool()
                self.agents_sdk_tools.append(sdk_tool)
                # Keep slots metadata for potential HTTP special handling
                sanitized_tool_id = (
                    tool_id.replace("/", "_").replace(" ", "_").replace("-", "_")
                )
                sanitized_tool_id = re.sub(r"[^a-zA-Z0-9_-]", "_", sanitized_tool_id)
                self.tool_slots[sanitized_tool_id] = tool_object.slots.copy()
                self.tool_name_mapping[sanitized_tool_id] = tool_id
            except Exception as e:
                log_context.warning(
                    f"Failed to convert tool '{tool_id}' to Agents SDK tool: {e}"
                )
                continue
        log_context.info(f"Configured {len(self.agents_sdk_tools)} Agents SDK tools.")

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
        
        # Retrieve handoffs from node-specific data if available
        if "handoffs" in node_specific_data:
            self.handoffs = node_specific_data["handoffs"]
            log_context.info(f"Retrieved handoffs from node data: {self.handoffs}")
        else:
            log_context.warning("No handoffs found in node-specific data")

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

        template = Template(prompts[prompt_key])
        input_prompt = template.render({"sys_instruct": state.sys_instruct})
        return input_prompt

    def _add_prompt_to_trajectory(
        self, state: OrchestratorState, input_prompt: str
    ) -> None:
        """Add the input prompt to the function calling trajectory if not already present."""
        if not any(
            message.get("content") == input_prompt
            for message in state.function_calling_trajectory
        ):
            log_context.info("Adding input prompt to the function calling trajectory.")
            state.function_calling_trajectory.append(
                {"role": "system", "content": input_prompt}
            )

    # Tool calls are handled natively by the Agents SDK via provided Tool wrappers.

    async def _run_agent_and_stream(self, state: OrchestratorState) -> str:
        """Run the Agents SDK text agent and forward outputs incrementally when possible."""
        result = await Runner.run(self.text_agent, state.function_calling_trajectory)

        # Emit incremental text from newly produced items (best-effort streaming feel)
        try:
            incremental_text = ItemHelpers.text_message_outputs(result.new_items)
        except Exception:
            incremental_text = ""

        if incremental_text and state.message_queue is not None:
            state.message_queue.put(
                {"event": EventType.CHUNK.value, "message_chunk": incremental_text}
            )

        final_text = getattr(result, "final_output", "") or incremental_text
        return final_text

    def _ensure_tool_outputs_in_trajectory(self, state: OrchestratorState) -> None:
        """Ensure all tool outputs from previous calls are available in the trajectory.
        
        This method scans the trajectory for any tool calls that don't have corresponding
        tool outputs and adds them, ensuring the agent has full context for decision making.
        """
        if not state.function_calling_trajectory:
            return

        log_context.info("=== ENSURING TOOL OUTPUTS IN TRAJECTORY ===")
        log_context.info(f"Current trajectory length: {len(state.function_calling_trajectory)}")
        
        # Track tool calls and their outputs
        tool_calls = []
        tool_outputs = {}
        
        # First pass: collect all tool calls and outputs
        for i, message in enumerate(state.function_calling_trajectory):
            log_context.info(f"Message {i}: {message.get('role')} - {message.get('name', 'no name')}")
            if message.get("role") == "assistant" and "tool_calls" in message:
                for tool_call in message.get("tool_calls", []):
                    tool_calls.append(tool_call)
                    log_context.info(f"Found tool call: {tool_call.get('function', {}).get('name')}")
            elif message.get("role") == "tool":
                tool_call_id = message.get("tool_call_id")
                tool_name = message.get("name")
                if tool_call_id:
                    tool_outputs[tool_call_id] = message.get("content", "")
                    log_context.info(f"Found tool output: {tool_name} (ID: {tool_call_id})")
        
        log_context.info(f"Found {len(tool_calls)} tool calls and {len(tool_outputs)} tool outputs")
        
        # Second pass: add missing tool outputs
        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id")
            tool_name = tool_call.get("function", {}).get("name")
            
            if tool_call_id and tool_call_id not in tool_outputs:
                # This tool call doesn't have an output yet - add a placeholder
                # The Agents SDK will handle the actual execution
                log_context.info(f"Tool {tool_name} called but output not yet available")
                
                # Add a system message to indicate tool execution is in progress
                state.function_calling_trajectory.append({
                    "role": "system",
                    "content": f"Tool {tool_name} is being executed. Results will be available shortly."
                })
        
        log_context.info("=== END ENSURING TOOL OUTPUTS ===")

    def set_handoffs(self, handoffs: list) -> None:
        """Set handoffs for the text agent."""
        self.handoffs = handoffs
        log_context.info(f"Set handoffs for agent: {handoffs}")
        
        # If text agent already exists, update it
        if hasattr(self, "text_agent"):
            self.text_agent.handoffs = handoffs
            log_context.info(f"Updated existing text agent with handoffs: {handoffs}")

    def _append_tool_outputs_to_trajectory(self, state: OrchestratorState, result: Any) -> None:  # noqa: ANN401
        """Append tool outputs from the Agents SDK result to the trajectory.
        
        This ensures that all tool outputs are available for future context.
        """
        try:
            log_context.info("=== APPENDING TOOL OUTPUTS TO TRAJECTORY ===")
            # Try to extract tool outputs from the result
            if hasattr(result, "new_items"):
                log_context.info(f"Processing {len(result.new_items)} new items from Agents SDK")
                for i, item in enumerate(result.new_items):
                    log_context.info(f"Processing item {i}: {type(item)}")
                    # Look for tool output items - check both item_type and class name
                    is_tool_output = (
                        (hasattr(item, "item_type") and "tool" in str(item.item_type).lower()) or
                        "ToolCallOutputItem" in str(type(item))
                    )
                    if is_tool_output:
                        try:
                            # Log all available attributes for debugging
                            log_context.info(f"Tool output item {i} attributes: {[attr for attr in dir(item) if not attr.startswith('_')]}")
                            
                            # Extract tool output data
                            tool_name = getattr(item, "name", "unknown_tool")
                            tool_output = getattr(item, "output", "")
                            
                            # Try alternative attribute names if the above don't work
                            if not tool_name or tool_name == "unknown_tool":
                                tool_name = getattr(item, "tool_name", getattr(item, "function_name", "unknown_tool"))
                            
                            if not tool_output:
                                tool_output = getattr(item, "content", getattr(item, "result", ""))
                            
                            log_context.info(f"Found tool output: {tool_name} -> {str(tool_output)[:100]}...")
                            
                            if tool_output:
                                # Add to trajectory as system message to remain API-compliant
                                tool_message = {
                                    "role": "system",
                                    "content": f"[TOOL_OUTPUT name={tool_name}] {str(tool_output)}"
                                }
                                state.function_calling_trajectory.append(tool_message)
                                log_context.info(f"Successfully added tool output for {tool_name} to trajectory (as system)")
                            else:
                                log_context.warning(f"Tool {tool_name} has empty output")
                        except Exception as e:
                            log_context.warning(f"Could not extract tool output from item {i}: {e}")
                            continue
                    else:
                        log_context.info(f"Item {i} is not a tool output: {getattr(item, 'item_type', 'unknown')}")
            else:
                log_context.warning("Result has no new_items attribute")
            log_context.info("=== END APPENDING TOOL OUTPUTS ===")
        except Exception as e:
            log_context.warning(f"Could not append tool outputs to trajectory: {e}")

    def _inject_context_from_tool_outputs(self, state: OrchestratorState) -> None:
        """Derive essential IDs from prior tool outputs and inject a concise system hint.

        Example: Extract restaurantID from GetRestaurantFromPhone result and nudge the model
        to use it when calling VerifySupplierBusinessLink.
        """
        try:
            log_context.info("=== CONTEXT INJECTION DEBUG ===")
            log_context.info(f"Full trajectory length: {len(state.function_calling_trajectory)}")
            
            if not state.function_calling_trajectory:
                log_context.info("No trajectory available for context injection")
                return
                
            # Log all tool outputs in trajectory (we now store as system messages with [TOOL_OUTPUT])
            tool_outputs = []
            for i, msg in enumerate(state.function_calling_trajectory):
                if msg.get("role") == "system" and str(msg.get("content", "")).startswith("[TOOL_OUTPUT"):
                    tool_outputs.append({
                        "index": i,
                        "name": "",
                        "content": msg.get("content", "")[:200] + "..." if len(msg.get("content", "")) > 200 else msg.get("content", "")
                    })
            
            log_context.info(f"Found {len(tool_outputs)} tool outputs in trajectory:")
            for tool_out in tool_outputs:
                log_context.info(f"  [{tool_out['index']}] {tool_out['name']}: {tool_out['content']}")
            
            # Find latest GetRestaurantFromPhone tool output
            latest_business_json = None
            for msg in reversed(state.function_calling_trajectory):
                if msg.get("role") == "system" and str(msg.get("content", "")).startswith("[TOOL_OUTPUT") and (
                    "GetRestaurantFromPhone" in str(msg.get("content", ""))
                ):
                    # Strip the prefix and parse JSON portion
                    raw_content = str(msg.get("content", ""))
                    prefix_end = raw_content.find("] ")
                    content = raw_content[prefix_end+2:] if prefix_end != -1 else raw_content
                    log_context.info(f"Found GetRestaurantFromPhone output: {content[:500]}...")
                    try:
                        latest_business_json = json.loads(content)
                        log_context.info(f"Successfully parsed JSON from GetRestaurantFromPhone")
                    except Exception as e:
                        log_context.warning(f"Failed to parse JSON from GetRestaurantFromPhone: {e}")
                        latest_business_json = None
                    break

            if not latest_business_json:
                log_context.info("No valid GetRestaurantFromPhone output found for context injection")
                return

            # Extract restaurant id if present at data.businesses.docs[0].id
            restaurant_id = None
            try:
                docs = (
                    latest_business_json.get("data", {})
                    .get("businesses", {})
                    .get("docs", [])
                )
                log_context.info(f"Extracted docs from business JSON: {len(docs) if isinstance(docs, list) else 'not a list'}")
                if isinstance(docs, list) and len(docs) > 0:
                    rid = docs[0].get("id")
                    log_context.info(f"Found restaurant ID in docs[0]: {rid}")
                    if isinstance(rid, str) and len(rid) >= 6:
                        restaurant_id = rid
                        log_context.info(f"Valid restaurant ID extracted: {restaurant_id}")
                    else:
                        log_context.warning(f"Invalid restaurant ID format: {rid}")
                else:
                    log_context.warning("No docs found in business JSON or docs is not a list")
            except Exception as e:
                log_context.warning(f"Error extracting restaurant ID: {e}")
                restaurant_id = None

            if not restaurant_id:
                log_context.info("No valid restaurant ID found, skipping context injection")
                return

            # Avoid duplicating the same hint
            hint_text = (
                f"Context hint: restaurantID={restaurant_id}. When calling VerifySupplierBusinessLink, "
                f"use this restaurantID (the ID string), not the restaurant name."
            )
            
            # Check if hint already exists
            for msg in reversed(state.function_calling_trajectory):
                if msg.get("role") == "system" and hint_text in msg.get("content", ""):
                    log_context.info("Context hint already exists, skipping injection")
                    return

            log_context.info(f"Injecting context hint: {hint_text}")
            state.function_calling_trajectory.append({
                "role": "system",
                "content": hint_text,
            })
            log_context.info("=== END CONTEXT INJECTION DEBUG ===")
        except Exception as e:
            log_context.warning(f"Failed to inject context from tool outputs: {e}")

    # Tool execution is managed by the Agents SDK; no manual execution path needed here.

    async def generate_response(
        self, state: OrchestratorState, stream: bool = False, is_speech: bool = False
    ) -> tuple[OrchestratorState, OpenAIAgentOutput]:
        """Unified response generation method with optional streaming using Agents SDK."""
        generation_type = (
            "speech streaming"
            if is_speech and stream
            else "streaming"
            if stream
            else "standard"
        )
        log_context.info(f"\nGenerating {generation_type} response using the Agents SDK.")
        input_prompt = self._prepare_prompt(state, is_speech)
        self._add_prompt_to_trajectory(state, input_prompt)

        # Ensure all tool outputs are in the trajectory for context
        self._ensure_tool_outputs_in_trajectory(state)
        # Inject concise, structured context derived from prior tool outputs (e.g., IDs)
        self._inject_context_from_tool_outputs(state)

        log_context.info(f"\nagent messages: {state.function_calling_trajectory}")
        
        # Log detailed trajectory analysis
        log_context.info("=== TRAJECTORY ANALYSIS ===")
        for i, msg in enumerate(state.function_calling_trajectory):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            name = msg.get("name", "")
            log_context.info(f"  [{i}] {role}" + (f" ({name})" if name else "") + f": {content[:100]}{'...' if len(content) > 100 else ''}")
        log_context.info("=== END TRAJECTORY ANALYSIS ===")

        # Ensure text agent is created
        if not hasattr(self, "text_agent"):
            # Use stored handoffs if available
            handoffs = getattr(self, 'handoffs', [])
            self.text_agent = Agent(
                name="OpenAIAgent",
                instructions=self.prompt,
                tools=self.agents_sdk_tools,
                handoffs=handoffs,
                model=self.orch_state.bot_config.llm_config.model_type_or_path,
            )
            log_context.info(f"Created new text agent with {len(self.agents_sdk_tools)} tools")
            log_context.info(f"Available tools: {[tool.name for tool in self.agents_sdk_tools]}")
            if handoffs:
                log_context.info(f"Agent handoffs configured: {handoffs}")
            else:
                log_context.warning("No handoffs configured for this agent")

            if stream:
                answer = await self._run_agent_and_stream(state)
            else:
                result = await Runner.run(self.text_agent, state.function_calling_trajectory)
                # Emit best-effort incremental content even in non-stream mode for responsiveness
                try:
                    incremental_text = ItemHelpers.text_message_outputs(result.new_items)
                except Exception:
                    incremental_text = ""
                if incremental_text and state.message_queue is not None:
                    state.message_queue.put(
                        {"event": EventType.CHUNK.value, "message_chunk": incremental_text}
                    )
                answer = getattr(result, "final_output", "") or incremental_text

        # Log the result from Agents SDK
        log_context.info("=== AGENTS SDK RESULT DEBUG ===")
        log_context.info(f"Result type: {type(result)}")
        handoff_detected_local = False
        if hasattr(result, "new_items"):
            log_context.info(f"New items count: {len(result.new_items)}")
            for i, item in enumerate(result.new_items):
                log_context.info(f"  Item {i}: {type(item)} - {getattr(item, 'item_type', 'unknown_type')}")
                if hasattr(item, "name"):
                    log_context.info(f"    Name: {item.name}")
                if hasattr(item, "output"):
                    log_context.info(f"    Output: {str(item.output)[:200]}{'...' if len(str(item.output)) > 200 else ''}")
                # Detect handoff by class name or item name
                item_cls_name = type(item).__name__.lower()
                if "handoff" in item_cls_name or (hasattr(item, "name") and isinstance(item.name, str) and "handoff" in item.name.lower()):
                    handoff_detected_local = True
                    log_context.info("    *** HANDOFF EVENT DETECTED ***")
        if hasattr(result, "final_output"):
            log_context.info(f"Final output: {result.final_output}")
        log_context.info("=== END AGENTS SDK RESULT DEBUG ===")

        # Persist handoff detection for this turn
        self._handoff_detected = handoff_detected_local

        # Append any new tool outputs to the trajectory for future context
        try:
            self._append_tool_outputs_to_trajectory(state, result)
        except Exception:
            pass

        state.message_flow = ""
        agent_output = OpenAIAgentOutput(response=answer)
        return state, agent_output

    # Override execute to mark node COMPLETE on handoff so orchestrator advances
    def execute(
        self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
    ) -> tuple[OrchestratorState, Any]:  # noqa: ANN401
        try:
            self.init_agent_data(orch_state, node_specific_data)
            response_state, output = self._execute()
            if response_state.trajectory and response_state.trajectory[-1]:
                response_state.trajectory[-1][-1].output = output.response
            from arklex.orchestrator.entities.orchestrator_state_entities import StatusEnum as _StatusEnum
            from arklex.env.agents.agent import AgentOutput as _AgentOutput
            status = _StatusEnum.COMPLETE if getattr(self, "_handoff_detected", False) else _StatusEnum.INCOMPLETE
            agent_output: _AgentOutput = _AgentOutput(
                response=output.response,
                status=status,
            )
        except Exception:
            log_context.error(traceback.format_exc())
            from arklex.orchestrator.entities.orchestrator_state_entities import StatusEnum as _StatusEnum
            from arklex.env.agents.agent import AgentOutput as _AgentOutput
            agent_output: _AgentOutput = _AgentOutput(
                response="",
                status=_StatusEnum.INCOMPLETE,
            )
        return orch_state, agent_output

    def _execute(self) -> tuple[OrchestratorState, OpenAIAgentOutput]:
        if (
            self.openai_agent_data.prompt_variables
            and len(self.openai_agent_data.prompt_variables) > 0
        ):
            template = Template(self.openai_agent_data.prompt)
            prompt_variables_dict = {
                pv.name: pv.value for pv in self.openai_agent_data.prompt_variables
            }
            self.prompt = template.render(prompt_variables_dict)
        else:
            self.prompt = self.openai_agent_data.prompt
        
        # Use asyncio.run() to create a new event loop for async execution
        import asyncio
        
        async def _async_execute():
            if self.orch_state.stream_type == StreamType.TEXT:
                return await self.generate_response(self.orch_state, stream=True, is_speech=False)
            elif self.orch_state.stream_type == StreamType.SPEECH:
                return await self.generate_response(self.orch_state, stream=True, is_speech=True)
            else:
                return await self.generate_response(self.orch_state, stream=False)
        
        return asyncio.run(_async_execute())


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