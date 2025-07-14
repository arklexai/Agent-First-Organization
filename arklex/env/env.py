"""Environment management for the Arklex framework.

This module provides functionality for managing the environment, including
worker initialization, tool management, and slot filling integration.
"""

import importlib
import os
import uuid
from collections.abc import Callable
from functools import partial
from typing import Any

from arklex.env.agents.agent import BaseAgent
from arklex.env.planner.react_planner import DefaultPlanner, ReactPlanner
from arklex.env.tools.tools import Tool
from arklex.env.workers.worker import BaseWorker
from arklex.orchestrator.entities.msg_state_entities import MessageState
from arklex.orchestrator.entities.orchestrator_params_entities import OrchestratorParams
from arklex.orchestrator.entities.taskgraph_entities import NodeInfo
from arklex.orchestrator.NLU.core.slot import SlotFiller
from arklex.orchestrator.NLU.services.api_service import APIClientService
from arklex.orchestrator.NLU.services.model_service import (
    DummyModelService,
    ModelService,
)
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
    def init_tools(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Initialize tools from configuration.

        Args:
            tools: list of tool configurations

        Returns:
            dictionary mapping tool IDs to their configurations
        """
        tool_registry: dict[str, dict[str, Any]] = {}
        for tool in tools:
            tool_id: str = tool["id"]
            name: str = tool["name"]
            path: str = tool["path"]
            try:
                filepath: str = os.path.join("arklex.env.tools", path)
                module_name: str = filepath.replace(os.sep, ".").replace(".py", "")
                module = importlib.import_module(module_name)
                func: Callable = getattr(module, name)
                tool_instance: Tool = func()
                tool_registry[tool_id] = {
                    "name": f"{path.replace('/', '-')}-{name}",
                    "description": tool_instance.description,
                    "execute": func,
                    "fixed_args": tool.get("fixed_args", {}),
                }
            except Exception as e:
                log_context.error(f"Tool {name} is not registered, error: {e}")
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
            name: str = worker["name"]
            path: str = worker["path"]
            try:
                filepath: str = os.path.join("arklex.env.workers", path)
                module_name: str = filepath.replace(os.sep, ".").rstrip(".py")
                module = importlib.import_module(module_name)
                func: Callable = getattr(module, name)
                worker_registry[worker_id] = {
                    "name": name,
                    "description": func.description,
                    "execute": partial(func, **worker.get("fixed_args", {})),
                }
            except Exception as e:
                log_context.error(f"Worker {name} is not registered, error: {e}")
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
            name: str = agent["name"]
            path: str = agent["path"]
            try:
                filepath: str = os.path.join("arklex.env.agents", path)
                module_name: str = filepath.replace(os.sep, ".").rstrip(".py")
                module = importlib.import_module(module_name)
                func: Callable = getattr(module, name)
                agent_registry[agent_id] = {
                    "name": name,
                    "description": func.description,
                    "execute": partial(func, **agent.get("fixed_args", {})),
                }
            except Exception as e:
                log_context.error(f"Agent {name} is not registered, error: {e}")
                continue
        return agent_registry


class ModelAwareResourceInitializer(DefaultResourceInitializer):
    """Resource initializer that passes model configuration to workers.

    This class extends DefaultResourceInitializer to pass model configuration
    to workers that require it, ensuring proper model initialization.
    """

    def __init__(self, model_config: dict[str, Any] | None = None) -> None:
        """Initialize the model-aware resource initializer.

        Args:
            model_config: Model configuration to pass to workers
        """
        self.model_config = model_config

    def init_workers(self, workers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Initialize workers from configuration with model configuration.

        Args:
            workers: list of worker configurations

        Returns:
            dictionary mapping worker IDs to their configurations
        """
        worker_registry: dict[str, dict[str, Any]] = {}
        for worker in workers:
            worker_id: str = worker["id"]
            name: str = worker["name"]
            path: str = worker["path"]
            try:
                filepath: str = os.path.join("arklex.env.workers", path)
                module_name: str = filepath.replace(os.sep, ".").rstrip(".py")
                module = importlib.import_module(module_name)
                func: Callable = getattr(module, name)

                # Add model_config to fixed_args if the worker accepts it
                fixed_args = worker.get("fixed_args", {})
                if self.model_config and hasattr(func, "__init__"):
                    # Check if the worker's __init__ method accepts model_config
                    import inspect

                    sig = inspect.signature(func.__init__)
                    if "model_config" in sig.parameters:
                        fixed_args["model_config"] = self.model_config

                worker_registry[worker_id] = {
                    "name": name,
                    "description": func.description,
                    "execute": partial(func, **fixed_args),
                }
            except Exception as e:
                log_context.error(f"Worker {name} is not registered, error: {e}")
        return worker_registry


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
        slotsfillapi: str = "",
        resource_initializer: BaseResourceInitializer | None = None,
        planner_enabled: bool = False,
        model_service: ModelService | None = None,
        **kwargs: str | int | float | bool | None,
    ) -> None:
        """Initialize the environment.

        Args:
            tools: list of tools to initialize
            workers: list of workers to initialize
            slotsfillapi: API endpoint for slot filling
            resource_initializer: Resource initializer instance
            planner_enabled: Whether planning is enabled
            model_service: Model service for intent detection and slot filling
        """
        # Accept slot_fill_api as an alias for slotsfillapi for compatibility with tests
        if "slot_fill_api" in kwargs and not slotsfillapi:
            slotsfillapi = kwargs["slot_fill_api"]

        # Use ModelAwareResourceInitializer if model_service is provided
        if resource_initializer is None:
            if model_service and hasattr(model_service, "model_config"):
                resource_initializer = ModelAwareResourceInitializer(
                    model_config=model_service.model_config
                )
            else:
                resource_initializer = DefaultResourceInitializer()

        self.tools: dict[str, dict[str, Any]] = resource_initializer.init_tools(tools)
        self.workers: dict[str, dict[str, Any]] = resource_initializer.init_workers(
            workers
        )
        self.agents: dict[str, dict[str, Any]] = resource_initializer.init_agents(
            agents
        )
        self.name2id: dict[str, str] = {
            resource["name"]: id
            for id, resource in {**self.tools, **self.workers, **self.agents}.items()
        }
        self.id2name: dict[str, str] = {
            id: resource["name"]
            for id, resource in {**self.tools, **self.workers, **self.agents}.items()
        }
        self.model_service = model_service or DummyModelService(
            {
                "model_name": "dummy",
                "api_key": "dummy",
                "endpoint": "http://dummy",
                "model_type_or_path": "dummy-path",
                "llm_provider": "dummy",
            }
        )
        self.slotfillapi: SlotFiller = self.initialize_slotfillapi(slotsfillapi)
        if planner_enabled:
            self.planner: ReactPlanner | DefaultPlanner = ReactPlanner(
                tools_map=self.tools, workers_map=self.workers, name2id=self.name2id
            )
        else:
            self.planner: ReactPlanner | DefaultPlanner = DefaultPlanner(
                tools_map=self.tools, workers_map=self.workers, name2id=self.name2id
            )

    def initialize_slotfillapi(self, slotsfillapi: str) -> SlotFiller:
        """Initialize the slot filling API.

        Args:
            slotsfillapi: API endpoint for slot filling. If not a string or empty,
                         falls back to local model-based slot filling.

        Returns:
            SlotFiller: Initialized slot filler instance, either API-based or local model-based.
        """
        if isinstance(slotsfillapi, str) and slotsfillapi:
            api_service = APIClientService(base_url=slotsfillapi)
            return SlotFiller(model_service=self.model_service, api_service=api_service)
        else:
            return SlotFiller(model_service=self.model_service)

    def step(
        self,
        id: str,
        message_state: MessageState,
        params: OrchestratorParams,
        node_info: NodeInfo,
    ) -> tuple[MessageState, OrchestratorParams]:
        """Execute a step in the environment.

        Args:
            id: Resource ID to execute
            message_state: Current message state
            params: Current parameters
            node_info: Information about the current node

        Returns:
            Tuple containing updated message state and parameters
        """
        response_state: MessageState
        if id in self.tools:
            log_context.info(f"{self.tools[id]['name']} tool selected")
            tool: Tool = self.tools[id]["execute"]()
            tool.init_slotfiller(self.slotfillapi)
            attributes = getattr(node_info, "attributes", {})
            # --- Begin slot group merge logic ---
            slots = attributes.get("slots", [])
            slot_groups = attributes.get("slot_groups", [])
            group_slots = []
            for group in slot_groups:
                # Generate prompt/description for the group
                required_fields = [s["name"] for s in group.get("schema", []) if s.get("required", False)]
                prompt = (
                    f"Please provide at least one set of the following fields: {', '.join(required_fields)}."
                    if required_fields else f"Please provide a set of values for group '{group['name']}'."
                )
                description = f"Slot group '{group['name']}' with schema: {[s['name'] for s in group.get('schema', [])]}"
                group_slots.append({
                    "name": group["name"],
                    "type": "group",
                    "schema": group.get("schema", []),
                    "required": group.get("required", False),
                    "repeatable": group.get("repeatable", True),
                    "prompt": prompt,
                    "description": description,
                })
            all_slots = slots + group_slots
            tool.load_slots(all_slots)
            combined_args: dict[str, Any] = {
                **self.tools[id]["fixed_args"],
                **(node_info.additional_args or {}),
            }
            response_state = tool.execute(message_state, **combined_args)
            params.memory.function_calling_trajectory = (
                response_state.function_calling_trajectory
            )
            params.taskgraph.dialog_states = response_state.slots
            params.taskgraph.node_status[params.taskgraph.curr_node] = (
                response_state.status
            )

        elif id in self.workers:
            log_context.info(f"{self.workers[id]['name']} worker selected")
            worker: BaseWorker = self.workers[id]["execute"]()
            if hasattr(worker, "init_slotfilling"):
                worker.init_slotfilling(self.slotfillapi)
            response_state = worker.execute(
                message_state, **(node_info.additional_args or {})
            )
            call_id: str = str(uuid.uuid4())
            params.memory.function_calling_trajectory.append(
                {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {"arguments": "{}", "name": self.id2name[id]},
                            "id": call_id,
                            "type": "function",
                        }
                    ],
                    "function_call": None,
                }
            )
            params.memory.function_calling_trajectory.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": self.id2name[id],
                    "content": response_state.response
                    if response_state.response
                    else response_state.message_flow,
                }
            )
            params.taskgraph.node_status[params.taskgraph.curr_node] = (
                response_state.status
            )

        elif id in self.agents:
            log_context.info(f"{self.agents[id]['name']} agent selected")
            agent: BaseAgent = self.agents[id]["execute"](
                successors=node_info.additional_args.get("successors", []),
                predecessors=node_info.additional_args.get("predecessors", []),
                tools=self.tools,
                state=message_state,
            )
            response_state = agent.execute(message_state, **node_info.additional_args)
            call_id: str = str(uuid.uuid4())
            params.memory.function_calling_trajectory = (
                response_state.function_calling_trajectory
            )
            params.taskgraph.dialog_states = response_state.slots
            params.taskgraph.node_status[params.taskgraph.curr_node] = (
                response_state.status
            )
        else:
            # Resource not found in any registry, use planner as fallback
            log_context.info(
                f"Resource {id} not found in registries, using planner as fallback"
            )
            action: str
            response_state: MessageState
            msg_history: list[dict[str, Any]]
            action, response_state, msg_history = self.planner.execute(
                message_state, params.memory.function_calling_trajectory
            )

        log_context.info(f"Response state from {id}: {response_state}")
        return response_state, params

    def register_tool(self, name: str, tool: Tool) -> None:
        """Register a tool in the environment.

        Args:
            name: Name of the tool
            tool: Tool instance

        Raises:
            EnvironmentError: If tool registration fails
        """
        try:
            self.tools[name] = tool
            log_context.info(f"{self.tools[name]['name']} tool selected")
        except Exception as e:
            log_context.error(f"Tool {name} is not registered, error: {e}")
