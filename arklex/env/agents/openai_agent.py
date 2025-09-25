import json
import re
from typing import Any

from jinja2 import Template
from pydantic import BaseModel
from agents import Agent, Runner, ItemHelpers

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
        """Run the Agents SDK text agent and forward outputs. Returns final text."""
        result = await Runner.run(self.text_agent, state.function_calling_trajectory)
        # Emit final output; the SDK accumulates new items internally
        final_text = result.final_output if hasattr(result, "final_output") else ""
        if final_text:
            state.message_queue.put(
                {"event": EventType.CHUNK.value, "message_chunk": final_text}
            )
        return final_text

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

        log_context.info(f"\nagent messages: {state.function_calling_trajectory}")

        # Ensure text agent is created
        if not hasattr(self, "text_agent"):
            self.text_agent = Agent(
                name="OpenAIAgent",
                instructions=self.prompt,
                tools=self.agents_sdk_tools,
                model=self.orch_state.bot_config.llm_config.model_type_or_path,
            )

        if stream:
            answer = await self._run_agent_and_stream(state)
        else:
            result = await Runner.run(self.text_agent, state.function_calling_trajectory)
            answer = result.final_output if hasattr(result, "final_output") else ""

        state.message_flow = ""
        agent_output = OpenAIAgentOutput(response=answer)
        return state, agent_output

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
