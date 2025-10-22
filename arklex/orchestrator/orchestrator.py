"""Orchestrator for the Arklex framework.

This module implements the core orchestrator functionality that manages the flow of
conversation and task execution in the Arklex framework. It coordinates between
different components of the system, including NLU processing, task graph execution,
and response generation.

Key Components:
- AgentOrg: Main orchestrator class for managing conversation flow
- Task Execution: Methods for executing tasks and managing task states
- Message Processing: Methods for handling user messages and generating responses
- State Management: Methods for maintaining conversation and task states
- Resource Management: Methods for handling system resources and connections

Features:
- Comprehensive conversation flow management
- Task graph execution and state tracking
- Message processing and response generation
- Resource management and cleanup
- Error handling and recovery
- State persistence and restoration
- Streaming response handling
- Memory management
- Tool integration

Usage:
    from arklex.orchestrator import AgentOrg
    from arklex.env.env import Env

    # Initialize environment
    env = Env()

    # Load configuration
    config = {
        "role": "customer_service",
        "user_objective": "Handle customer inquiries",
        "model": {...},
        "workers": [...],
        "tools": [...]
    }

    # Create orchestrator
    orchestrator = AgentOrg(config, env)

    # Process message
    response = orchestrator.get_response({
        "text": "user message",
        "chat_history": [...],
        "parameters": {...}
    })
"""

import asyncio
import copy
import json
import time
from typing import Any

import janus
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda

from arklex.env.entities import NodeResponse
from arklex.env.env import Environment
from arklex.env.prompts import load_prompts
from arklex.env.tools.utils import ToolGenerator
from arklex.memory.entities.memory_entities import ResourceRecord
from arklex.orchestrator.entities.orchestrator_param_entities import OrchestratorParams
from arklex.orchestrator.entities.orchestrator_state_entities import (
    BotConfig,
    ConvoMessage,
    LLMConfig,
    OrchestratorResp,
    OrchestratorState,
    StatusEnum,
)
from arklex.orchestrator.entities.taskgraph_entities import (
    NodeInfo,
    PathNode,
)
from arklex.orchestrator.NLU.services.model_service import ModelService
from arklex.orchestrator.post_process import post_process_response
from arklex.orchestrator.task_graph.task_graph import AgentGraph, TaskGraph
from arklex.types.resource_types import AgentItem
from arklex.types.stream_types import StreamType
from arklex.utils.logging_utils import LogContext
from arklex.utils.utils import format_chat_history

load_dotenv()
log_context = LogContext(__name__)


class AgentOrg:
    """Agent organization orchestrator for the Arklex framework.

    This class manages the orchestration of agent interactions, task execution,
    and workflow management. It handles the flow of conversations and ensures
    proper execution of tasks.

    Attributes:
        user_prefix (str): Prefix for user messages
        worker_prefix (str): Prefix for worker messages
        environment_prefix (str): Prefix for environment messages
        product_kwargs (Dict[str, Any]): Configuration settings
        llm_config (LLMConfig): Language model configuration
        task_graph (TaskGraph): Task graph for conversation flow
        env (Environment): Environment with tools and workers
    """

    def __init__(
        self,
        config: str | dict[str, Any],
        env: Environment | None,
    ) -> None:
        """Initialize the AgentOrg orchestrator.

        This function initializes the orchestrator with configuration settings and environment.
        It sets up the task graph, model configuration, and other necessary components.

        Args:
            config (Union[str, Dict[str, Any]]): Configuration file path or dictionary containing
                product settings, model configuration, and other parameters.
            env (Environment): Environment object containing tools, workers, and other resources.
            **kwargs (Any): Additional keyword arguments for customization.
        """
        self.user_prefix: str = "user"

        if isinstance(config, dict):
            self.product_kwargs: dict[str, Any] = config
        else:
            with open(config) as f:
                self.product_kwargs: dict[str, Any] = json.load(f)
        self.llm_config: LLMConfig = LLMConfig.model_validate(
            self.product_kwargs.get("model")
        )
        self.model_service: ModelService = ModelService(self.llm_config)
        self.env: Environment = env
        self.task_graph: TaskGraph = TaskGraph(
            "taskgraph",
            self.product_kwargs,
            llm_config=self.llm_config,
        )
        self.agent_graph: AgentGraph = AgentGraph(
            "agentgraph",
            self.product_kwargs,
        )
        # Load prompts based on bot config
        bot_config = BotConfig.model_validate(self.product_kwargs.get("bot_config", {}))
        self.prompts = load_prompts(bot_config)

    def init_params(
        self, inputs: dict[str, Any]
    ) -> tuple[str, str, OrchestratorParams, OrchestratorState]:
        """Initialize parameters for a new conversation turn.

        This function processes the input text, chat history, and parameters to initialize
        the state for a new conversation turn. It updates the turn ID, function calling
        trajectory, and creates a new message state with system instructions.

        Args:
            inputs (Dict[str, Any]): Dictionary containing text, chat history, and parameters.

        Returns:
            Tuple[str, str, OrchestratorParams, MessageState]: A tuple containing the processed text,
                formatted chat history, updated parameters, and new message state.
        """
        text: str = inputs["text"]
        chat_history: list[dict[str, str]] = inputs["chat_history"]
        input_params: dict[str, Any] | None = inputs["parameters"]

        # Create base params with defaults
        params: OrchestratorParams = OrchestratorParams()

        # Update with any provided values
        if input_params:
            params = OrchestratorParams.model_validate(input_params)

        # Update specific fields
        chat_history_copy: list[dict[str, str]] = copy.deepcopy(chat_history)
        chat_history_copy.append({"role": self.user_prefix, "content": text})
        chat_history_str: str = format_chat_history(chat_history_copy)
        # Update turn_id and function_calling_trajectory
        params.metadata.turn_id += 1
        if not params.memory.function_calling_trajectory:
            params.memory.function_calling_trajectory = copy.deepcopy(chat_history_copy)
        else:
            params.memory.function_calling_trajectory.extend(chat_history_copy[-2:])

        params.memory.trajectory.append([])

        # Initialize the orchestrator state
        orch_state: OrchestratorState = OrchestratorState(
            sys_instruct=self.product_kwargs.get("sys_instruct", ""),
            bot_config=BotConfig.model_validate(
                self.product_kwargs.get("bot_config", {})
            ),
        )
        return text, chat_history_str, params, orch_state

    def check_skip_node(self, node_info: NodeInfo, chat_history_str: str) -> bool:
        """Check if a node can be skipped in the task graph.

        This function determines whether a node can be skipped based on its configuration
        and the current state of the task graph. It checks if the node is marked as skippable
        and if it has reached its execution limit.

        Args:
            node_info (NodeInfo): Information about the current node.
            params (OrchestratorParams): Current parameters and state of the conversation.

        Returns:
            bool: True if the node can be skipped, False otherwise.
        """
        if not node_info.attribute.get("can_skipped", False):
            return False

        task = node_info.attribute.get("task", "")
        if not task:
            return False

        prompt = self.prompts["check_skip_node_prompt"].format(
            chat_history_str=chat_history_str, task=task
        )
        log_context.info(f"prompt for check skip node: {prompt}")

        try:
            response_text = self.model_service.get_response(prompt)
            log_context.info(f"LLM response for task verification: {response_text}")
            response_text = str(response_text).lower().strip()
            return response_text == "yes"
        except Exception as e:
            log_context.error(f"Error in LLM task verification: {str(e)}")
            return False

    def post_process_node(
        self,
        node_info: NodeInfo,
        params: OrchestratorParams,
        update_info: dict[str, Any] = None,
    ) -> OrchestratorParams:
        """Process a node after its execution.

        This function updates the task graph path with the current node's information,
        including whether it was skipped and its flow stack.

        Args:
            node_info (NodeInfo): Information about the current node.
            params (OrchestratorParams): Current parameters and state of the conversation.
            update_info (Dict[str, Any], optional): Additional information about the node's execution.
                Defaults to an empty dictionary.

        Returns:
            OrchestratorParams: Updated parameters after processing the node.
        """
        if update_info is None:
            update_info = {}
        curr_node: str = params.taskgraph.curr_node
        node: PathNode = PathNode(
            node_id=curr_node,
            is_skipped=update_info.get("is_skipped", False),
            in_flow_stack=node_info.add_flow_stack,
            global_intent=params.taskgraph.curr_global_intent,
        )

        params.taskgraph.path.append(node)
        return params

    def perform_node(
        self,
        orch_state: OrchestratorState,
        node_info: NodeInfo,
        params: OrchestratorParams,
        text: str,
        chat_history_str: str,
        stream_type: StreamType | None,
        message_queue: janus.Queue | None,
    ) -> tuple[NodeInfo, OrchestratorState, OrchestratorParams, NodeResponse]:
        """Execute a node in the task graph.

        This function processes a node in the task graph, handling nested graph nodes,
        creating messages, and managing the conversation flow. It updates the node information,
        message state, and parameters based on the execution results.

        Args:
            message_state (MessageState): Current state of the conversation.
            node_info (NodeInfo): Information about the current node.
            params (OrchestratorParams): Current parameters and state of the conversation.
            text (str): The current user message.
            chat_history_str (str): Formatted chat history.
            stream_type (Optional[StreamType]): Type of stream for the response.
            message_queue (Optional[janus.Queue]): Queue for streaming messages.

        Returns:
            Tuple[NodeInfo, MessageState, OrchestratorParams]: A tuple containing updated node information,
                message state, and parameters.
        """
        # Create initial resource record with common info and output from trajectory
        resource_record: ResourceRecord = ResourceRecord(
            info={
                "resource": node_info.resource,
                "attribute": node_info.attribute,
                "node_id": params.taskgraph.curr_node,
            },
            intent=params.taskgraph.intent,
        )

        # Add resource record to current turn's list
        params.memory.trajectory[-1].append(resource_record)

        # Update orchestrator state
        orch_state.user_message = ConvoMessage(history=chat_history_str, message=text)
        orch_state.function_calling_trajectory = (
            params.memory.function_calling_trajectory
        )
        orch_state.trajectory = params.memory.trajectory
        orch_state.metadata = params.metadata
        orch_state.stream_type = stream_type
        orch_state.message_queue = message_queue
        # Execute the node
        response_state: OrchestratorState
        response_state, node_response = self.env.step(
            node_info.resource["id"],
            orch_state,
            node_info,
            params.taskgraph.dialog_states,
        )
        # Update params
        params.taskgraph.node_status[node_info.node_id] = node_response.status
        if node_response.slots:
            params.taskgraph.dialog_states = node_response.slots
        return node_info, response_state, params, node_response

    def _get_response(
        self,
        inputs: dict[str, Any],
        stream_type: StreamType | None = None,
        message_queue: janus.Queue | None = None,
    ) -> OrchestratorResp:
        """Get a response from the orchestrator.

        This function processes the input through the task graph, handling personalized intents,
        retrieving relevant records, and managing the conversation flow. It supports streaming
        responses and maintains the conversation state.

        Args:
            inputs (Dict[str, Any]): Dictionary containing text, chat history, and parameters.
            stream_type (Optional[StreamType]): Type of stream for the response.
            message_queue (Optional[janus.SyncQueue]): Queue for streaming messages.

        Returns:
            OrchestratorResp: The orchestrator's response containing the answer and parameters.
        """
        text: str
        chat_history_str: str
        params: OrchestratorParams
        orch_state: OrchestratorState
        text, chat_history_str, params, orch_state = self.init_params(inputs)
        # TaskGraph Chain
        taskgraph_inputs: dict[str, Any] = {
            "text": text,
            "chat_history_str": chat_history_str,
            "parameters": params,
            "allow_global_intent_switch": True,
        }

        orch_state.trajectory = params.memory.trajectory
        taskgraph_chain = RunnableLambda(self.task_graph.get_node) | RunnableLambda(
            self.task_graph.postprocess_node
        )

        n_node_performed = 0
        max_n_node_performed = 5
        while n_node_performed < max_n_node_performed:
            taskgraph_start_time = time.time()
            node_info, params = taskgraph_chain.invoke(taskgraph_inputs)
            taskgraph_inputs["allow_global_intent_switch"] = False
            params.metadata.timing.taskgraph = time.time() - taskgraph_start_time
            # Check if current node can be skipped
            can_skip = self.check_skip_node(node_info, chat_history_str)
            if can_skip:
                params = self.post_process_node(node_info, params, {"is_skipped": True})
                continue
            log_context.info(f"The current node info is : {node_info}")

            # perform node
            node_info, orch_state, params, node_response = self.perform_node(
                orch_state,
                node_info,
                params,
                text,
                chat_history_str,
                stream_type,
                message_queue,
            )
            params = self.post_process_node(node_info, params)

            n_node_performed += 1
            # If the current node is not complete, then no need to continue to the next node
            if node_response.status == StatusEnum.INCOMPLETE:
                break
            # If the current node has a response, break the loop
            if node_response.response:
                break
            # If the current node is a leaf node, break the loop
            if node_info.is_leaf is True:
                break

        if not node_response.response:
            log_context.info("No response, do context generation")
            if stream_type == StreamType.NON_STREAM:
                answer = ToolGenerator.context_generate(orch_state)
                node_response.response = answer
            else:
                answer = ToolGenerator.stream_context_generate(orch_state)
                node_response.response = answer

        node_response = post_process_response(
            orch_state,
            node_response,
        )

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
        agent_cls = self.env.agents[AgentItem.OPENAI_AGENT]["agent_instance"]
        if not orch_state_params.agentgraph.current_agent:
            agent_name = self.agent_graph.start_agent_name
        else:
            agent_name = orch_state_params.agentgraph.current_agent
        agent_instance = agent_cls(
            agent=self.agent_graph.agents[agent_name],
            state=orch_state,
            start_message=self.agent_graph.start_message,
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
