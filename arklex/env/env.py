"""Environment management and resource initialization for the Arklex framework.

This module provides functionality for managing the execution environment, including tool and
worker initialization, resource registration, and step-by-step execution of actions. It
includes classes for resource initialization, environment management, and integration with
planners and slot filling systems.

Key Components:
- BaseResourceInitializer: Abstract base class for resource initialization
- DefaultResourceInitializer: Concrete implementation for tool and worker initialization
- Env: Main environment class for managing resources and execution

Key Features:
- Dynamic loading of tools and workers
- Resource registration and management
- Integration with planners and slot filling
- Step-by-step execution control
- State management and tracking
- Error handling and logging

Usage:
    # Initialize environment with tools and workers
    env = Env(
        tools=tool_configs,
        workers=worker_configs,
        slotsfillapi=slot_filling_api,
        planner_enabled=True
    )

    # Execute a step in the environment
    response_state, params = env.step(
        id=resource_id,
        message_state=current_state,
        params=parameters,
        node_info=node_information
    )
"""

import importlib
import logging
import os
import uuid
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, cast

from arklex.env.planner.react_planner import DefaultPlanner, ReactPlanner
from arklex.env.tools.tools import Tool
from arklex.env.workers.worker import BaseWorker
from arklex.orchestrator.NLU.nlu import SlotFilling
from arklex.utils.graph_state import MessageState, NodeInfo, Params

logger: logging.Logger = logging.getLogger(__name__)


class BaseResourceInitializer:
    """Abstract base class for resource initialization.

    This class defines the interface for initializing tools and workers in the environment.
    Concrete implementations must provide methods for tool and worker initialization.
    """

    @staticmethod
    def init_tools(tools: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Initialize tools from configuration.

        Args:
            tools: List of tool configurations

        Returns:
            Dictionary mapping tool IDs to their configurations

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    @staticmethod
    def init_workers(workers: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Initialize workers from configuration.

        Args:
            workers: List of worker configurations

        Returns:
            Dictionary mapping worker IDs to their configurations

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError


class DefaulResourceInitializer(BaseResourceInitializer):
    """Default implementation of resource initialization.

    This class provides concrete implementations for initializing tools and workers
    from configuration dictionaries. It handles dynamic loading of modules and
    registration of resources.
    """

    @staticmethod
    def init_tools(tools: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Initialize tools from configuration.

        This method loads tool modules dynamically and registers them in the environment.
        It handles error cases and logging for failed tool registrations.

        Args:
            tools: List of tool configurations with id, name, path, and optional fixed_args

        Returns:
            Dictionary mapping tool IDs to their configurations including name,
            description, execute function, and fixed arguments
        """
        tool_registry: Dict[str, Dict[str, Any]] = {}
        for tool in tools:
            tool_id: str = tool["id"]
            name: str = tool["name"]
            path: str = tool["path"]
            try:
                filepath: str = os.path.join("arklex.env.tools", path)
                module_name: str = filepath.replace(os.sep, ".").replace(".py", "")
                module = importlib.import_module(module_name)
                func: Callable = getattr(module, name)
            except Exception as e:
                logger.error(f"Tool {name} is not registered, error: {e}")
                continue
            tool_registry[tool_id] = {
                "name": func().name,
                "description": func().description,
                "execute": func,
                "fixed_args": tool.get("fixed_args", {}),
            }
        return tool_registry

    @staticmethod
    def init_workers(workers: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Initialize workers from configuration.

        This method loads worker modules dynamically and registers them in the environment.
        It handles error cases and logging for failed worker registrations.

        Args:
            workers: List of worker configurations with id, name, path, and optional fixed_args

        Returns:
            Dictionary mapping worker IDs to their configurations including name,
            description, execute function, and fixed arguments
        """
        worker_registry: Dict[str, Dict[str, Any]] = {}
        for worker in workers:
            worker_id: str = worker["id"]
            name: str = worker["name"]
            path: str = worker["path"]
            try:
                filepath: str = os.path.join("arklex.env.workers", path)
                module_name: str = filepath.replace(os.sep, ".").rstrip(".py")
                module = importlib.import_module(module_name)
                func: Callable = getattr(module, name)
            except Exception as e:
                logger.error(f"Worker {name} is not registered, error: {e}")
                continue
            worker_registry[worker_id] = {
                "name": name,
                "description": func.description,
                "execute": partial(func, **worker.get("fixed_args", {})),
            }
        return worker_registry


class Env:
    """Main environment class for managing resources and execution.

    This class manages the execution environment, including tool and worker initialization,
    resource registration, and step-by-step execution of actions. It integrates with
    planners and slot filling systems for comprehensive environment management.
    """

    def __init__(
        self,
        tools: List[Dict[str, Any]],
        workers: List[Dict[str, Any]],
        slotsfillapi: str = "",
        resource_inizializer: Optional[BaseResourceInitializer] = None,
        planner_enabled: bool = False,
    ) -> None:
        """Initialize the environment with tools and workers.

        Args:
            tools: List of tool configurations
            workers: List of worker configurations
            slotsfillapi: API endpoint for slot filling
            resource_inizializer: Optional custom resource initializer
            planner_enabled: Whether to enable the planner
        """
        if resource_inizializer is None:
            resource_inizializer = DefaulResourceInitializer()
        self.tools: Dict[str, Dict[str, Any]] = resource_inizializer.init_tools(tools)
        self.workers: Dict[str, Dict[str, Any]] = resource_inizializer.init_workers(
            workers
        )
        self.name2id: Dict[str, str] = {
            resource["name"]: id
            for id, resource in {**self.tools, **self.workers}.items()
        }
        self.id2name: Dict[str, str] = {
            id: resource["name"]
            for id, resource in {**self.tools, **self.workers}.items()
        }
        self.slotfillapi: SlotFilling = self.initialize_slotfillapi(slotsfillapi)

        if planner_enabled:
            self.planner: Union[ReactPlanner, DefaultPlanner] = ReactPlanner(
                tools_map=self.tools, workers_map=self.workers, name2id=self.name2id
            )
        else:
            self.planner: Union[ReactPlanner, DefaultPlanner] = DefaultPlanner(
                tools_map=self.tools, workers_map=self.workers, name2id=self.name2id
            )

    def initialize_slotfillapi(self, slotsfillapi: str) -> SlotFilling:
        """Initialize the slot filling API.

        Args:
            slotsfillapi: API endpoint for slot filling

        Returns:
            Initialized SlotFilling instance
        """
        return SlotFilling(slotsfillapi)

    def step(
        self, id: str, message_state: MessageState, params: Params, node_info: NodeInfo
    ) -> Tuple[MessageState, Params]:
        """Execute a step in the environment.

        This method handles the execution of tools, workers, or planner actions based on
        the provided ID. It manages state updates and parameter modifications.

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
            logger.info(f"{self.tools[id]['name']} tool selected")
            tool: Tool = self.tools[id]["execute"]()
            tool.init_slotfilling(self.slotfillapi)
            combined_args: Dict[str, Any] = {
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
            logger.info(f"{self.workers[id]['name']} worker selected")
            worker: BaseWorker = self.workers[id]["execute"]()
            if hasattr(worker, "init_slotfilling"):
                worker.init_slotfilling(self.slotfillapi)
            response_state = worker.execute(message_state, **node_info.additional_args)
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
                    "content": (
                        response_state.response
                        if response_state.response
                        else response_state.message_flow
                    ),
                }
            )
            params.taskgraph.node_status[params.taskgraph.curr_node] = (
                response_state.status
            )
        else:
            logger.info("planner selected")
            action: str
            response_state: MessageState
            msg_history: List[Dict[str, Any]]
            action, response_state, msg_history = self.planner.execute(
                message_state, params.memory.function_calling_trajectory
            )

        logger.info(f"Response state from {id}: {response_state}")
        return response_state, params
