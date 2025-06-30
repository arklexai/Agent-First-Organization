"""Node formatting component for the Arklex framework.

This module provides the NodeFormatter class that handles formatting of task
nodes in the task graph.
"""

from typing import Any

from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class NodeFormatter:
    """Formats task nodes.

    This class handles the formatting of task nodes in the task graph,
    including resource, attribute, limit, and type fields.
    """

    def __init__(
        self,
        default_directed: bool = True,
    ) -> None:
        """Initialize the NodeFormatter."""
        self._default_resource = {
            "id": "26bb6634-3bee-417d-ad75-23269ac17bc3",
            "name": "MessageWorker",
        }
        self._default_directed = default_directed

    def format_node(self, task: dict[str, Any], node_id: str) -> list[Any]:
        """Format a task node in the old format.

        Args:
            task (Dict[str, Any]): Task to format
            node_id (str): ID of the node to format

        Returns:
            List[Any]: Formatted node as [id, data] pair
        """
        node_data = {
            "resource": {
                "id": task.get("task_id") or node_id,
                "name": task.get("name") or "",
            },
            "attribute": {
                "value": task.get("description") or "",
                "task": task.get("name") or "",
                "directed": self._default_directed,
            },
        }
        if "limit" in task:
            node_data["limit"] = task["limit"]
        if "type" in task:
            node_data["type"] = task["type"]
        return [node_id, node_data]

    def format_node_data(self, task: dict[str, Any]) -> dict[str, Any]:
        """Format node data in the old format.

        Args:
            task (Dict[str, Any]): Task to format

        Returns:
            Dict[str, Any]: Formatted node data
        """
        return {
            "resource": self._default_resource,
            "attribute": {
                "value": task.get("description") or "",
                "task": task.get("name") or "",
                "directed": self._default_directed,
            },
        }

    def format_node_style(self, task: dict[str, Any]) -> dict[str, Any]:
        """Format node style.

        Args:
            task (Dict[str, Any]): Task to format

        Returns:
            Dict[str, Any]: Formatted node style
        """
        # Get priority color
        priority_colors = {"high": "#ff0000", "medium": "#ffa500", "low": "#00ff00"}
        priority = task.get("priority", "medium")
        color = priority_colors.get(priority, "#808080")

        return {
            "color": color,
            "border": {"color": "#000000", "width": 2, "style": "solid", "radius": 4},
            "padding": {"top": 10, "right": 10, "bottom": 10, "left": 10},
            "text_color": "#000000",
            "background_color": "#ffffff",
            "opacity": 1.0,
        }

    def validate_node(self, node: dict[str, Any]) -> bool:
        """Validate a node.

        Args:
            node (Dict[str, Any]): Node to validate

        Returns:
            bool: True if node is valid
        """
        # Require 'id' and 'type' at the top level
        if "id" not in node or not isinstance(node["id"], str):
            return False
        if "type" not in node or not isinstance(node["type"], str):
            return False
        # Check 'data' for 'name' and 'description'
        if "data" in node:
            data = node["data"]
            if not isinstance(data, dict):
                return False
            if "name" in data and not isinstance(data["name"], str):
                return False
            if "description" in data and not isinstance(data["description"], str):
                return False
            if "priority" in data and not isinstance(data["priority"], str | int):
                return False
        return True

    def _validate_attribute(self, value: object, attr: str) -> bool:
        """Validate a node attribute.

        Args:
            value (Any): Attribute value to validate
            attr (str): Attribute name

        Returns:
            bool: True if attribute is valid
        """
        if attr == "steps":
            if not isinstance(value, list):
                return False
            for step in value:
                if not isinstance(step, dict):
                    return False
                if "step_id" not in step:
                    return False
                if "description" not in step:
                    return False
                if "order" not in step:
                    return False
        elif attr == "dependencies":
            if not isinstance(value, list):
                return False
            for dep in value:
                if not isinstance(dep, str):
                    return False
        elif attr == "required_resources":
            if not isinstance(value, list):
                return False
            for resource in value:
                if not isinstance(resource, str):
                    return False
        elif attr == "estimated_duration":
            if not isinstance(value, str):
                return False
        elif attr == "priority":
            if not isinstance(value, str):
                return False
            if value not in ["high", "medium", "low"]:
                return False

        return True
