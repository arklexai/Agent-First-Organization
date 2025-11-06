import asyncio
import json
from typing import Any

import janus
from dotenv import load_dotenv

from arklex.models.llm_config import LLMConfig
from arklex.orchestrator.entities.orchestrator_param_entities import OrchestratorParams
from arklex.orchestrator.entities.orchestrator_state_entities import (
    ConvoMessage,
    OrchestratorResp,
    OrchestratorState,
)
from arklex.orchestrator.executor.executor import Executor
from arklex.orchestrator.task_graph.agent_graph import AgentGraph
from arklex.orchestrator.task_graph.nlu_graph import NLUGraph
from arklex.orchestrator.types.stream_types import StreamType
from arklex.resources.agent_loader import AgentLoader
from arklex.resources.resource_types import AgentItem
from arklex.utils.logging.logging_utils import LogContext

load_dotenv()
log_context = LogContext(__name__)


DEFAULT_AGENTS = [
    {
        "id": AgentItem.NLU_AGENT.value,
    },
    {
        "id": AgentItem.OPENAI_AGENT.value,
    },
]


class AgentOrg:
    """Agent organization orchestrator for the Arklex framework.

    This class manages the orchestration of agent interactions, task execution,
    and workflow management. It handles the flow of conversations and ensures
    proper execution of tasks.

    Attributes:
        user_prefix (str): Prefix for user messages
        config (Dict[str, Any]): Configuration settings
        llm_config (LLMConfig): Language model configuration
        nlu_graph (NLUGraph): NLU graph for conversation flow
        executor (Executor): Executor with tools and workers
    """

    def __init__(
        self,
        config: str | dict[str, Any],
        executor: Executor | None,
    ) -> None:
        """Initialize the orchestrator.

        This function initializes the orchestrator with configuration settings and environment.
        It sets up the task graph, model configuration, and other necessary components.

        Args:
            config (Union[str, Dict[str, Any]]): Configuration file path or dictionary containing
                config settings, model configuration, and other parameters.
            executor (Executor): Executor object containing tools, workers, and other resources.
            **kwargs (Any): Additional keyword arguments for customization.
        """
        if isinstance(config, dict):
            self.config: dict[str, Any] = config
        else:
            with open(config) as f:
                self.config: dict[str, Any] = json.load(f)
        self.llm_config: LLMConfig = LLMConfig.model_validate(
            self.config.get("llm_config", {})
        )
        self.executor: Executor = executor
        self.agents: dict[str, dict[str, Any]] = AgentLoader.init_agents(DEFAULT_AGENTS)
        self.nlu_graph: NLUGraph = NLUGraph(
            "nlugraph",
            self.config,
            llm_config=self.llm_config,
        )
        self.agent_graph: AgentGraph = AgentGraph(
            "agentgraph",
            self.config,
        )

    def _get_response(
        self,
        inputs: dict[str, Any],
        stream_type: StreamType | None = None,
        message_queue: janus.Queue | None = None,
    ) -> OrchestratorResp:
        nlu_agent = self.agents[AgentItem.NLU_AGENT]["agent_instance"](
            self.llm_config, self.nlu_graph, self.executor
        )
        node_response, params = nlu_agent.execute(inputs, stream_type, message_queue)
        return OrchestratorResp(
            answer=node_response.response,
            parameters=params.model_dump(),
            choice_list=node_response.choice_list,
            human_in_the_loop=params.metadata.hitl,
        )

    async def _get_agent_response(
        self,
        inputs: dict[str, Any],
        stream_type: StreamType | None = None,
        message_queue: janus.Queue | None = None,
    ) -> OrchestratorResp:
        # params initialization
        user_message = inputs["text"]
        orch_state_params = OrchestratorParams.model_validate(
            inputs.get("parameters", {})
        )
        orch_state: OrchestratorState = OrchestratorState(
            stream_type=stream_type,
            message_queue=message_queue,
            user_message=ConvoMessage(message=user_message),
            openai_agents_trajectory=orch_state_params.memory.openai_agents_trajectory.copy(),
        )
        # agent instance initialization
        agent_cls = self.agents[AgentItem.OPENAI_AGENT]["agent_instance"]
        exec_params = self.agent_graph.configure_params()
        if not orch_state_params.agentgraph.current_agent:
            agent_name = exec_params["start_agent_name"]
        else:
            agent_name = orch_state_params.agentgraph.current_agent
        agent_instance = agent_cls(
            agent=exec_params["agents"][agent_name],
            state=orch_state,
            start_message=exec_params["start_message"],
            input_guardrails=exec_params["agents_input_guardrails"][agent_name],
            output_guardrails=exec_params["agents_output_guardrails"][agent_name],
            safety_response=exec_params["agents_safety_response"][agent_name],
        )
        # agent execution
        orch_state, agent_output = await agent_instance.execute()
        orch_state_params.memory.openai_agents_trajectory = (
            orch_state.openai_agents_trajectory
        )
        orch_state_params.agentgraph.current_agent = agent_output.last_agent_name
        log_context.info(
            f"Agent after execution: {orch_state_params.agentgraph.current_agent}"
        )
        log_context.info(f"agent trajectory: {orch_state.openai_agents_trajectory}")
        return OrchestratorResp(
            answer=agent_output.response,
            parameters=orch_state_params.model_dump(),
        )

    async def get_response(
        self,
        inputs: dict[str, Any],
        stream_type: StreamType | None = None,
        message_queue: janus.Queue | None = None,
    ) -> dict[str, Any]:
        """Get a response from the orchestrator with additional metadata.

        This function wraps the _get_response method to provide additional metadata about
        the response, such as whether human intervention is required.

        Args:
            inputs (Dict[str, Any]): Dictionary containing text, chat history, and parameters.
            stream_type (Optional[StreamType]): Type of stream for the response.
            message_queue (Optional[janus.Queue]): Queue for streaming messages.

        Returns:
            Dict[str, Any]: A dictionary containing the response, parameters, and metadata.
        """
        if not stream_type:
            stream_type = StreamType.NON_STREAM

        if self.agent_graph.enabled:
            response = await self._get_agent_response(
                inputs, stream_type, message_queue.async_q if message_queue else None
            )
        else:
            response = await asyncio.to_thread(
                self._get_response,
                inputs,
                stream_type,
                message_queue.sync_q if message_queue else None,
            )
        return response.model_dump()
