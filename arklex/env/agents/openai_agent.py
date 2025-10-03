import contextlib
import re
import traceback
from typing import Any

from agents import (
    Agent,
    HandoffOutputItem,
    ItemHelpers,
    Runner,
    RunResult,
    ToolCallOutputItem,
)
from jinja2 import Template
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel

from arklex.env.agents.agent import BaseAgent, register_agent
from arklex.env.agents.entities import PromptVariable
from arklex.env.prompts import load_prompts
from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
)
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
                sanitized_tool_id = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_id)
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
        """Add the input prompt to both trajectories if not already present."""
        # Legacy trajectory
        try:
            if state.function_calling_trajectory is None:
                state.function_calling_trajectory = []
            if not any(
                message.get("content") == input_prompt
                for message in state.function_calling_trajectory
            ):
                state.function_calling_trajectory.append(
                    {"role": "system", "content": input_prompt}
                )
        except Exception:
            pass
        # SDK trajectory
        try:
            if state.openai_sdk_trajectory is None:
                state.openai_sdk_trajectory = []
            if not any(
                message.get("content") == input_prompt
                for message in state.openai_sdk_trajectory
            ):
                state.openai_sdk_trajectory.append(
                    {"role": "system", "content": input_prompt}
                )
        except Exception:
            pass

    # Tool calls are handled natively by the Agents SDK via provided Tool wrappers.
    async def _run_agent_and_stream(self, state: OrchestratorState) -> str:
        """Run the Agents SDK text agent with streaming and forward deltas."""
        sdk_traj = state.openai_sdk_trajectory or state.function_calling_trajectory or []
        streamed = Runner.run_streamed(self.text_agent, sdk_traj)

        buffer: list[str] = []
        try:
            async for event in streamed.stream_events():
                # Match the reference pattern exactly
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    delta = event.data.delta
                    if delta:
                        buffer.append(delta)
                        if state.message_queue is not None:
                            state.message_queue.put(
                                {"event": EventType.CHUNK.value, "message_chunk": delta}
                            )
        except Exception:
            # Fall back to best-effort partial output
            pass

        return "".join(buffer)

    def set_handoffs(self, handoffs: list) -> None:
        """Set handoffs for the text agent."""
        self.handoffs = handoffs
        log_context.info(f"Set handoffs for agent: {handoffs}")
        
        # If text agent already exists, update it
        if hasattr(self, "text_agent"):
            self.text_agent.handoffs = handoffs
            log_context.info(f"Updated existing text agent with handoffs: {handoffs}")

    def _append_tool_outputs_to_trajectory(self, state: OrchestratorState, result: RunResult) -> None:
        """Append tool outputs from the Agents SDK result to the trajectory.
        
        This ensures that all tool outputs are available for future context.
        """
        try:
            # Process new items from the result
            for item in result.new_items:
                # Handle ToolCallOutputItem specifically
                if isinstance(item, ToolCallOutputItem):
                    tool_name = getattr(item, "name", "unknown_tool")
                    tool_output = getattr(item, "output", "")
                    
                    if tool_output:
                        # Add to both trajectories for consistency
                        tool_message = {
                            "role": "system",
                            "content": f"[TOOL_OUTPUT name={tool_name}] {str(tool_output)}"
                        }
                        
                        # Legacy trajectory
                        if state.function_calling_trajectory is not None:
                            state.function_calling_trajectory.append(tool_message)
                        
                        # SDK trajectory
                        if state.openai_sdk_trajectory is not None:
                            state.openai_sdk_trajectory.append(tool_message)
        except Exception:
            pass

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

        try:
            cleaned_messages: list[dict[str, Any]] = []
            sdk_traj = state.openai_sdk_trajectory or []
            for msg in sdk_traj:
                role = msg.get("role")
                content = str(msg.get("content", ""))
                # Keep non-system, keep tool outputs, keep only the current agent's prompt
                if role != "system" or content.startswith("[TOOL_OUTPUT") or content == input_prompt:
                    cleaned_messages.append(msg)
            if len(cleaned_messages) != len(sdk_traj):
                log_context.info("Pruned prior system prompts from other agents to keep the conversation natural.")
                state.openai_sdk_trajectory = cleaned_messages
        except Exception:
            pass

        log_context.info("Prepared trajectory for Agents SDK run")

        # Ensure text agent is created
        if not hasattr(self, "text_agent"):
            handoffs = getattr(self, "handoffs", [])
            self.text_agent = Agent(
                name="OpenAIAgent",
                instructions=self.prompt,
                tools=self.agents_sdk_tools,
                handoffs=handoffs,
                model=self.orch_state.bot_config.llm_config.model_type_or_path,
            )
            log_context.info(
                f"Created new text agent with {len(self.agents_sdk_tools)} tools"
            )
            if handoffs:
                log_context.info(f"Agent handoffs configured: {handoffs}")

        # Run based on streaming mode
        result = None
        if stream:
            answer = await self._run_agent_and_stream(state)
        else:
            sdk_traj = state.openai_sdk_trajectory or state.function_calling_trajectory or []
            result = await Runner.run(self.text_agent, sdk_traj)
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

        # After non-stream run, log and detect handoffs from result if available
        handoff_detected_local = False
        if result is not None:
            log_context.info(f"Agents SDK result type: {type(result)}")
            if hasattr(result, "new_items"):
                log_context.info(f"New items: {len(result.new_items)}")
                for item in result.new_items:
                    # Use simple type checking like the reference code
                    if isinstance(item, HandoffOutputItem):
                        handoff_detected_local = True
                        log_context.info(f"Handoff detected: {item.source_agent.name} to {item.target_agent.name}")
            if hasattr(result, "final_output") and result.final_output:
                log_context.info("Agents SDK produced final output")

        # Persist handoff detection for this turn
        self._handoff_detected = handoff_detected_local

        # If a handoff occurred, suppress this agent's surface output so the next agent speaks
        if self._handoff_detected:
            answer = ""

        # Append any new tool outputs to the trajectory for future context
        with contextlib.suppress(Exception):
            self._append_tool_outputs_to_trajectory(state, result)

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
            from arklex.env.agents.agent import AgentOutput as _AgentOutput
            from arklex.orchestrator.entities.orchestrator_state_entities import (
                StatusEnum as _StatusEnum,
            )
            status = _StatusEnum.COMPLETE if getattr(self, "_handoff_detected", False) else _StatusEnum.INCOMPLETE
            agent_output: _AgentOutput = _AgentOutput(
                response=output.response,
                status=status,
            )
        except Exception:
            log_context.error(traceback.format_exc())
            from arklex.env.agents.agent import AgentOutput as _AgentOutput
            from arklex.orchestrator.entities.orchestrator_state_entities import (
                StatusEnum as _StatusEnum,
            )
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
        
        async def _async_execute() -> tuple[OrchestratorState, OpenAIAgentOutput]:
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