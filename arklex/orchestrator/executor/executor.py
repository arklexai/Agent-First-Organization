"""Environment management for the Arklex framework.

This module provides functionality for managing the environment, including
worker initialization, tool management, and slot filling integration.
"""

import uuid
from typing import Any

from arklex.models.llm_config import LLMConfig
from arklex.models.model_service import ModelService
from arklex.orchestrator.entities.orchestrator_state_entities import OrchestratorState
from arklex.orchestrator.entities.taskgraph_entities import NodeInfo, StatusEnum
from arklex.orchestrator.executor.entities import NodeResponse
from arklex.orchestrator.nlu.core.slot import Slot, SlotFiller
from arklex.resources.resource_loader import ResourceLoader
from arklex.resources.resource_types import ToolItem, WorkerItem
from arklex.resources.tools.tools import Tool
from arklex.resources.workers.base.base_worker import BaseWorker
from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)


class Executor:
    """Environment management for workers and tools.

    This class manages the environment for workers and tools, including
    initialization, state management, and slot filling integration.
    """

    def __init__(
        self,
        tools: list[dict[str, Any]],
        workers: list[dict[str, Any]],
        nodes: list[dict[str, Any]],
        llm_config: LLMConfig,
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
        resource_loader = ResourceLoader()
        self.tools: dict[str, dict[str, Any]] = resource_loader.init_tools(tools, nodes)
        self.workers: dict[str, dict[str, Any]] = resource_loader.init_workers(workers)
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
                    orch_state,
                    node_specific_data={**node_info.data, **self.workers[id]["auth"]},
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
