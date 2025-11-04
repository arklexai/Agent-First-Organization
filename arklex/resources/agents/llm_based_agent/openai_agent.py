from typing import Any

from agents import (
    Agent,
    HandoffOutputItem,
    InputGuardrailTripwireTriggered,
    ItemHelpers,
    MessageOutputItem,
    OutputGuardrailTripwireTriggered,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
)
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel, Field

from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
)
from arklex.orchestrator.types.stream_types import EventType, StreamType
from arklex.resources.agents.base.agent import BaseAgent, register_agent
from arklex.resources.agents.base.entities import PromptVariable
from arklex.resources.agents.llm_based_agent.guardrail_agent import (
    create_input_guardrail_function,
    create_output_guardrail_function,
)
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
    safety_response: str | None = None


class OpenAIAgentOutput(BaseModel):
    """Output for the OpenAIAgent."""

    response: str
    last_agent_name: str


class ExecuteParams(BaseModel):
    start_message: str | None
    safety_response: str | None
    input_guardrails: list = Field(default_factory=list)
    output_guardrails: list = Field(default_factory=list)


SAFETY_RESPONSE = "An error occurred while processing your request. Please try again."


@register_agent
class OpenAIAgent(BaseAgent):
    description: str = "General-purpose Arklex agent for chat or voice."

    def __init__(
        self, agent: Agent, state: OrchestratorState, **kwargs: object
    ) -> None:
        self.agent = agent
        self.state = state
        self.last_agent_name = ""
        self.execute_params = ExecuteParams.model_validate(kwargs)
        self.configure_agent(
            self.execute_params.input_guardrails, self.execute_params.output_guardrails
        )

    def configure_agent(
        self, input_guardrails: list[Agent], output_guardrails: list[Agent]
    ) -> None:
        self.agent.input_guardrails = [
            create_input_guardrail_function(input_guardrail.name, input_guardrail)
            for input_guardrail in input_guardrails
        ]
        log_context.info(
            f"Input guardrails: {[input_guardrail.name for input_guardrail in input_guardrails]} configured for agent {self.agent.name}"
        )
        self.agent.output_guardrails = [
            create_output_guardrail_function(output_guardrail.name, output_guardrail)
            for output_guardrail in output_guardrails
        ]
        log_context.info(
            f"Output guardrails: {[output_guardrail.name for output_guardrail in output_guardrails]} configured for agent {self.agent.name}"
        )

    async def response(
        self, trajectory: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        final_response = ""
        result: list = []
        is_error = False
        try:
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
                    log_context.info(
                        f"{agent_name} tool call output: {new_item.output}"
                    )
                else:
                    log_context.info(f"{agent_name} unknown item: {new_item}")
        except InputGuardrailTripwireTriggered as e:
            log_context.error(f"Input guardrail tripped: {e}")
            is_error = True
        except OutputGuardrailTripwireTriggered as e:
            log_context.error(f"Output guardrail tripped: {e}")
            is_error = True
        except Exception as e:
            log_context.error(f"Error during agent execution: {e}")
            is_error = True

        if not is_error:
            new_traj = result.to_input_list()
        else:
            agent_name = self.agent.name or ""
            new_traj = trajectory.copy()
            safety_response = self.execute_params.safety_response or SAFETY_RESPONSE
            new_traj.append({"role": "assistant", "content": safety_response})
            final_response = safety_response

        self.last_agent_name = agent_name
        return final_response, new_traj

    async def stream_response(
        self, trajectory: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        final_response = ""
        is_error = False
        result = Runner.run_streamed(self.agent, trajectory)
        try:
            async for event in result.stream_events():
                # raw final response for streaming
                if event.type == "raw_response_event" and isinstance(
                    event.data, ResponseTextDeltaEvent
                ):
                    await self.state.message_queue.put(
                        {
                            "event": EventType.CHUNK.value,
                            "message_chunk": event.data.delta,
                        }
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
        except InputGuardrailTripwireTriggered as e:
            log_context.error(f"Input guardrail tripped: {e}")
            is_error = True
        except OutputGuardrailTripwireTriggered as e:
            log_context.error(f"Output guardrail tripped: {e}")
            is_error = True
        except Exception as e:
            log_context.error(f"Error during agent execution: {e}")
            is_error = True

        if not is_error:
            new_traj = result.to_input_list()
        else:
            agent_name = self.agent.name or ""
            new_traj = trajectory.copy()
            safety_response = self.execute_params.safety_response or SAFETY_RESPONSE
            await self.state.message_queue.put(
                {
                    "event": EventType.CHUNK.value,
                    "message_chunk": safety_response,
                }
            )
            new_traj.append({"role": "assistant", "content": safety_response})
            final_response = safety_response

        self.last_agent_name = agent_name
        return final_response, new_traj

    async def execute(self) -> tuple[OrchestratorState, OpenAIAgentOutput]:
        user_message = self.state.user_message.message
        trajectory = self.state.openai_agents_trajectory.copy() or []

        if user_message == "<start>":
            if (
                self.execute_params.start_message
                and self.execute_params.start_message.strip()
            ):
                trajectory.append(
                    {"role": "assistant", "content": self.execute_params.start_message}
                )
                self.state.openai_agents_trajectory = trajectory
                return self.state, OpenAIAgentOutput(
                    response=self.execute_params.start_message, last_agent_name=""
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
