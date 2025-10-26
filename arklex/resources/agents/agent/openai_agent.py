from typing import Any

from agents import (
    Agent,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
)
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel

from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
)
from arklex.orchestrator.types.stream_types import EventType, StreamType
from arklex.resources.agents.base.agent import BaseAgent, register_agent
from arklex.resources.agents.base.entities import PromptVariable
from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)


class OpenAIAgentData(BaseModel):
    """Data for the OpenAIAgent."""

    name: str
    prompt: str
    prompt_variables: list[PromptVariable] = []
    start_agent: bool = False
    agent_start_message: str | None = None
    handoff_description: str | None = None


class OpenAIAgentOutput(BaseModel):
    """Output for the OpenAIAgent."""

    response: str
    last_agent_name: str


@register_agent
class OpenAIAgent(BaseAgent):
    description: str = "General-purpose Arklex agent for chat or voice."

    def __init__(
        self, agent: Agent, state: OrchestratorState, start_message: str | None = None
    ) -> None:
        self.agent = agent
        self.state = state
        self.start_message = start_message
        self.last_agent_name = ""

    async def response(
        self, trajectory: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        final_response = ""
        result = await Runner.run(self.agent, trajectory)
        for new_item in result.new_items:
            agent_name = new_item.agent.name
            if isinstance(new_item, MessageOutputItem):
                final_response = ItemHelpers.text_message_output(new_item)
                log_context.info(f"{agent_name}: {final_response}")
            elif isinstance(new_item, HandoffOutputItem):
                log_context.info(
                    f"Handed off from {agent_name} to {new_item.target_agent.name}"
                )
            elif isinstance(new_item, ToolCallItem):
                log_context.info(f"{agent_name}: Calling a tool")
            elif isinstance(new_item, ToolCallOutputItem):
                log_context.info(f"{agent_name} tool call output: {new_item.output}")
            else:
                log_context.info(f"{agent_name} unknown item: {new_item}")
        new_traj = result.to_input_list()
        self.last_agent_name = agent_name
        return final_response, new_traj

    async def stream_response(
        self, trajectory: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        final_response = ""
        result = Runner.run_streamed(self.agent, trajectory)
        async for event in result.stream_events():
            # raw final response for streaming
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                await self.state.message_queue.put(
                    {"event": EventType.CHUNK.value, "message_chunk": event.data.delta}
                )
            elif event.type == "run_item_stream_event":
                new_item = event.item
                agent_name = new_item.agent.name
                if isinstance(new_item, MessageOutputItem):
                    final_response = ItemHelpers.text_message_output(new_item)
                    log_context.info(f"{agent_name}: {final_response}")
                elif isinstance(new_item, HandoffOutputItem):
                    log_context.info(
                        f"Handed off from {new_item.source_agent.name} to {new_item.target_agent.name}"
                    )
                elif isinstance(new_item, ToolCallItem):
                    log_context.info(f"{agent_name}: Calling a tool")
                elif isinstance(new_item, ToolCallOutputItem):
                    log_context.info(
                        f"{agent_name} tool call output: {new_item.output}"
                    )
                else:
                    log_context.info(f"{agent_name} unknown item: {new_item}")
        new_traj = result.to_input_list()
        self.last_agent_name = agent_name
        return final_response, new_traj

    async def execute(self) -> tuple[OrchestratorState, OpenAIAgentOutput]:
        user_message = self.state.user_message.message
        trajectory = self.state.openai_agents_trajectory.copy() or []

        if user_message == "<start>":
            if self.start_message and self.start_message.strip():
                trajectory.append({"role": "assistant", "content": self.start_message})
                self.state.openai_agents_trajectory = trajectory
                return self.state, OpenAIAgentOutput(
                    response=self.start_message, last_agent_name=""
                )
            log_context.info("No start message configured for agent")
        trajectory.append({"role": "user", "content": user_message})

        if self.state.stream_type == StreamType.NON_STREAM:
            response, new_traj = await self.response(trajectory)
        else:
            response, new_traj = await self.stream_response(trajectory)
        self.state.openai_agents_trajectory = new_traj
        return self.state, OpenAIAgentOutput(
            response=response, last_agent_name=self.last_agent_name
        )
