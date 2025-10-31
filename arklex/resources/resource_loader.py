from typing import Any

from arklex.resources.resource_map import RESOURCE_MAP
from arklex.resources.resource_types import ToolItem
from arklex.resources.tools.tools import Tool
from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)


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
                    "auth": worker.get("auth", {}),
                }
            except Exception as e:
                log_context.error(f"Worker {worker_id} is not registered, error: {e}")
        return worker_registry
