"""Environment management for the Arklex framework.

This module provides functionality for managing the environment, including
worker initialization, tool management, and slot filling integration.
"""

import uuid
from typing import Any

from arklex.env.agents.agent import BaseAgent
from arklex.env.entities import NodeResponse
from arklex.env.resource_map import RESOURCE_MAP
from arklex.env.tools.tools import Tool
from arklex.env.workers.base.base_worker import BaseWorker
from arklex.orchestrator.entities.orchestrator_state_entities import OrchestratorState
from arklex.orchestrator.entities.taskgraph_entities import NodeInfo, StatusEnum
from arklex.orchestrator.NLU.core.slot import Slot, SlotFiller
from arklex.orchestrator.NLU.services.model_service import ModelService
from arklex.types.resource_types import ToolItem, WorkerItem
from arklex.utils.llm_config import LLMConfig
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class BaseResourceInitializer:
    """Abstract base class for resource initialization.

    This class defines the interface for initializing tools and workers in the environment.
    Concrete implementations must provide methods for tool and worker initialization.
    """

    @staticmethod
    def init_tools(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Initialize tools from configuration.

        Args:
            tools: list of tool configurations

        Returns:
            dictionary mapping tool IDs to their configurations

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    @staticmethod
    def init_workers(workers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Initialize workers from configuration.

        Args:
            workers: list of worker configurations

        Returns:
            dictionary mapping worker IDs to their configurations

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError


class DefaultResourceInitializer(BaseResourceInitializer):
    """Default implementation of resource initialization.

    This class provides a default implementation for initializing tools and workers
    in the environment.
    """

    @staticmethod
    def init_tools(
        tools: list[dict[str, Any]], nodes: list[dict[str, Any]]
    ) -> dict[str, dict[str, Tool]]:
        """Initialize tools from configuration.

        Args:
            tools: list of tool configurations
            attributes_list: optional list of attributes for the tools

        Returns:
            dictionary mapping tool IDs to their configurations
        """
        tool_registry: dict[str, dict[str, Any]] = {}
        for tool in tools:
            tool_id: str = tool["id"]
            if tool_id not in [item.value for item in ToolItem]:
                log_context.warning(f"Tool {tool_id} is not in ToolItem, skipping")
                continue
            try:
                if tool_id == ToolItem.HTTP_TOOL:
                    for node in nodes:
                        node_info = node[1]
                        node_data = node_info.get("data", {})
                        if (
                            node_info.get("resource", {}).get("id") != tool_id
                            or not node_data
                        ):
                            continue
                        # Create a new tool instance for each node to avoid sharing state
                        base_tool: Tool = RESOURCE_MAP[tool_id]["item_cls"]
                        tool_instance: Tool = base_tool.copy()
                        tool_instance.auth.update(tool.get("auth", {}))
                        tool_instance.node_specific_data = node_data
                        # Load slots from node data
                        slots = node_data.get("slots", [])
                        tool_instance.load_slots(slots)
                        tool_instance.name = node_data.get("name", "")
                        tool_instance.description = node_info.get("attribute", {}).get(
                            "task", ""
                        )
                        tool_registry[tool_instance.name] = {
                            "tool_instance": tool_instance,
                        }
                else:
                    base_tool: Tool = RESOURCE_MAP[tool_id]["item_cls"]
                    tool_instance: Tool = base_tool.copy()
                    tool_instance.auth.update(tool.get("auth", {}))
                    tool_instance.node_specific_data = {}
                    for node in nodes:
                        node_info = node[1]
                        fixed_args = node_info.get("data", {}).get("fixed_args", {})
                        if (
                            node_info.get("resource", {}).get("id") != tool_id
                            or not fixed_args
                        ):
                            continue
                        tool_instance.fixed_args.update(fixed_args)
                        break
                    tool_registry[tool_id] = {
                        "tool_instance": tool_instance,
                    }
            except Exception as e:
                log_context.exception(e)
                log_context.error(f"Tool {tool_id} is not registered, error: {e}")

        return tool_registry

    @staticmethod
    def init_workers(workers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Initialize workers from configuration.

        Args:
            workers: list of worker configurations

        Returns:
            dictionary mapping worker IDs to their configurations
        """
        worker_registry: dict[str, dict[str, Any]] = {}
        for worker in workers:
            worker_id: str = worker["id"]
            try:
                worker_registry[worker_id] = {
                    "item_cls": RESOURCE_MAP[worker["id"]]["item_cls"],
                }
            except Exception as e:
                log_context.error(f"Worker {worker_id} is not registered, error: {e}")
        return worker_registry

    @staticmethod
    def init_agents(agents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Initialize agents from configuration.

        Args:
            agents: list of agent configurations

        Returns:
            dictionary mapping agent IDs to their configurations
        """
        agent_registry: dict[str, dict[str, Any]] = {}
        for agent in agents:
            agent_id: str = agent["id"]
            try:
                agent_instance: BaseAgent = RESOURCE_MAP[agent_id]["item_cls"]
                agent_registry[agent_id] = {
                    "agent_instance": agent_instance,
                }
            except Exception as e:
                log_context.error(f"Agent {agent_id} is not registered, error: {e}")
                continue

        return agent_registry


class Environment:
    """Environment management for workers and tools.

    This class manages the environment for workers and tools, including
    initialization, state management, and slot filling integration.
    """

    def __init__(
        self,
        tools: list[dict[str, Any]],
        workers: list[dict[str, Any]],
        agents: list[dict[str, Any]],
        nodes: list[dict[str, Any]],
        llm_config: LLMConfig,
        resource_initializer: BaseResourceInitializer | None = None,
    ) -> None:
        """Initialize the environment.

        Args:
            tools: list of tools to initialize
            workers: list of workers to initialize
            slotsfillapi: API endpoint for slot filling
            resource_initializer: Resource initializer instance
            planner_enabled: Whether planning is enabled
            llm_config: Language model configuration
        """
        resource_initializer = DefaultResourceInitializer()
        self.tools: dict[str, dict[str, Any]] = resource_initializer.init_tools(
            tools, nodes
        )
        self.workers: dict[str, dict[str, Any]] = resource_initializer.init_workers(
            workers
        )
        self.agents: dict[str, dict[str, Any]] = resource_initializer.init_agents(
            agents
        )
        self.model_service = ModelService(llm_config)
        self.slotfillapi: SlotFiller = SlotFiller(model_service=self.model_service)

    def step(
        self,
        id: str,
        orch_state: OrchestratorState,
        node_info: NodeInfo,
        dialog_states: dict[str, list[Slot]],
    ) -> tuple[OrchestratorState, NodeResponse]:
        """Execute a step in the environment.

        Args:
            id: Resource ID to execute
            message_state: Current message state
            params: Current parameters
            node_info: Information about the current node

        Returns:
            Tuple containing updated message state and parameters
        """
        node_response: NodeResponse
        if id in self.tools or id == ToolItem.HTTP_TOOL:
            if id == ToolItem.HTTP_TOOL:
                log_context.info(f"HTTP tool {node_info.data.get('name', '')} selected")
                tool: Tool = self.tools[node_info.data.get("name", "")]["tool_instance"]
            else:
                log_context.info(f"{id} tool selected")
                tool: Tool = self.tools[id]["tool_instance"]
            tool.init_slotfiller(self.slotfillapi)
            orch_state, tool_output = tool.execute(
                orch_state, all_slots=dialog_states, auth=tool.auth
            )
            orch_state.message_flow = tool_output.message_flow
            if id == ToolItem.SHOPIFY_SEARCH_PRODUCTS:
                node_response = NodeResponse(
                    status=tool_output.status,
                    response=tool_output.response,
                    slots=tool_output.slots,
                )
            else:
                node_response = NodeResponse(
                    status=tool_output.status,
                    slots=tool_output.slots,
                )

        elif id in self.workers:
            log_context.info(f"{id} worker selected")
            try:
                worker: BaseWorker = self.workers[id]["item_cls"]()
                orch_state, worker_output = worker.execute(
                    orch_state, node_specific_data=node_info.data
                )
                content = ""
                if id == WorkerItem.MULTIPLE_CHOICE_WORKER:
                    node_response = NodeResponse(
                        status=worker_output.status,
                        response=worker_output.response,
                        choice_list=worker_output.choice_list,
                    )
                    content = (
                        worker_output.response
                        + "\n"
                        + "\n".join(worker_output.choice_list)
                    )
                else:
                    node_response = NodeResponse(
                        status=worker_output.status,
                        response=worker_output.response,
                    )
                    content = worker_output.response
                call_id: str = str(uuid.uuid4())
                orch_state.function_calling_trajectory.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "function": {"arguments": "{}", "name": id},
                                "id": call_id,
                                "type": "function",
                            }
                        ],
                        "function_call": None,
                    }
                )
                orch_state.function_calling_trajectory.append(
                    {
                        "role": "tool",
                        "content": content,
                        "tool_call_id": call_id,
                        "id": id,
                    }
                )
            except Exception as e:
                log_context.error(f"Error in worker {id}: {e}")
                node_response = NodeResponse(
                    status=StatusEnum.INCOMPLETE,
                )

        elif id in self.agents:
            log_context.info(f"Agent {id} selected")
            agent: BaseAgent = self.agents[id]["agent_instance"](
                successors=node_info.successors,
                predecessors=node_info.predecessors,
                tools=self.tools,
            )
            orch_state, agent_output = agent.execute(
                orch_state,
                node_specific_data=node_info.data,
            )
            node_response = NodeResponse(
                status=agent_output.status,
                response=agent_output.response,
            )

        else:
            # Resource not found in any registry, use planner as fallback
            log_context.info(
                f"Resource {id} not found in registries, return orch_state directly"
            )
            node_response = NodeResponse(
                status=StatusEnum.COMPLETE,
            )

        log_context.info(f"Response state from {id}: {orch_state}")
        return orch_state, node_response
