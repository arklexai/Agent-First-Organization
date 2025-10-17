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

from arklex.env.agents.agent import BaseAgent, register_agent
from arklex.env.agents.entities import PromptVariable
from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
)
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
        agent: Agent,
        state: OrchestratorState,
    ) -> None:
        self.agent = agent
        self.state = state

    async def response(
        self, trajectory: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        final_response = ""
        result = await Runner.run(self.agent, trajectory)
        for new_item in result.new_items:
            if isinstance(new_item, MessageOutputItem):
                final_response = ItemHelpers.text_message_output(new_item)
                log_context.info(f"{self.agent.name}: {final_response}")
            elif isinstance(new_item, HandoffOutputItem):
                log_context.info(
                    f"Handed off from {new_item.source_agent.name} to {new_item.target_agent.name}"
                )
            elif isinstance(new_item, ToolCallItem):
                log_context.info(f"{self.agent.name}: Calling a tool")
            elif isinstance(new_item, ToolCallOutputItem):
                log_context.info(
                    f"{self.agent.name} tool call output: {new_item.output}"
                )
            else:
                log_context.info(f"{self.agent.name} unknown item: {new_item}")
        new_traj = result.to_input_list()
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
                self.state.message_queue.put(
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
        return final_response, new_traj

    async def execute(self) -> tuple[OrchestratorState, OpenAIAgentOutput]:
        user_message = self.state.user_message.message
        trajectory = self.state.openai_agents_trajectory.copy() or []
        trajectory.append({"role": "user", "content": user_message})
        if self.state.stream_type == StreamType.NON_STREAM:
            response, new_traj = await self.response(trajectory)
        else:
            response, new_traj = await self.stream_response(trajectory)
        self.state.openai_agents_trajectory = new_traj
        return self.state, OpenAIAgentOutput(response=response)
