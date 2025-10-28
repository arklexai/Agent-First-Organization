import importlib
from typing import Any

from arklex.resources.resource_map import RESOURCE_MAP
from arklex.resources.resource_types import ResourceType, ToolItem
from arklex.resources.tools.tools import Tool
from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)


def init_resource_map(resource_type: ResourceType) -> dict[str, dict[str, Any]]:
    resource_map: dict[str, dict[str, Any]] = {}
    for item, details in RESOURCE_MAP.items():
        details_copy = details.copy()
        function_name = details_copy["item_cls"]
        module_path = details_copy["module"]
        if resource_type and details_copy["type"] != resource_type:
            continue
        try:
            module = importlib.import_module(module_path)
            function = getattr(module, function_name)
            details_copy["item_cls"] = function
            resource_map[item] = details_copy
            log_context.info(
                f"Successfully imported {function_name} from {module_path}"
            )
        except Exception as e:
            log_context.error(
                f"Failed to import {function_name} from {module_path}: {e}"
            )
    return resource_map


class ResourceLoader:
    @staticmethod
    def init_tools(
        tools: list[dict[str, Any]], nodes: list[dict[str, Any]]
    ) -> dict[str, dict[str, Tool]]:
        """Initialize tools from configuration.

        Args:
            tools: list of tool configurations
            nodes: list of nodes configurations

        Returns:
            dictionary mapping tool IDs to their configurations
        """
        resource_map = init_resource_map(ResourceType.TOOL)
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
                        base_tool: Tool = resource_map[tool_id]["item_cls"]
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
                    base_tool: Tool = resource_map[tool_id]["item_cls"]
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
        resource_map = init_resource_map(ResourceType.WORKER)
        worker_registry: dict[str, dict[str, Any]] = {}
        for worker in workers:
            worker_id: str = worker["id"]
            try:
                worker_registry[worker_id] = {
                    "item_cls": resource_map[worker["id"]]["item_cls"],
                    "auth": worker.get("auth", {}),
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
        resource_map = init_resource_map(ResourceType.AGENT)
        agent_registry: dict[str, dict[str, Any]] = {}
        for agent in agents:
            agent_id: str = agent["id"]
            try:
                agent_instance = resource_map[agent_id]["item_cls"]
                agent_registry[agent_id] = {
                    "agent_instance": agent_instance,
                }
            except Exception as e:
                log_context.error(f"Agent {agent_id} is not registered, error: {e}")
                continue

        return agent_registry
