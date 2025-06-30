"""Graph validation component for the Arklex framework.

This module provides the GraphValidator class that handles validation of task graphs.
It ensures that graphs are properly structured and contain all required components.

Key Features:
- Graph structure validation
- Node validation
- Edge validation
- Error handling and reporting
"""

from typing import Any

from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class GraphValidator:
    """Validates task graphs.

    This class handles validation of task graphs, ensuring they are properly
    structured and contain all required components.

    Attributes:
        _errors (List[str]): List of validation error messages
    """

    def __init__(self) -> None:
        """Initialize the GraphValidator."""
        self._errors: list[str] = []

    def validate_graph(self, graph: dict[str, Any]) -> bool:
        """Validate a task graph.

        Args:
            graph (Dict[str, Any]): Task graph to validate

        Returns:
            bool: True if graph is valid
        """
        self._errors = []

        # Validate graph structure
        if not isinstance(graph, dict):
            self._errors.append("Graph must be a dictionary")
            return False

        # Validate nodes
        nodes = graph.get("nodes", [])
        if not isinstance(nodes, list):
            self._errors.append("Nodes must be a list")
            return False

        # Get node IDs (nodes are now [id, data] pairs)
        node_ids = {node[0] for node in nodes}

        # Validate edges
        edges = graph.get("edges", [])
        if not isinstance(edges, list):
            self._errors.append("Edges must be a list")
            return False

        # Validate each edge
        for edge in edges:
            if not isinstance(edge, list) or len(edge) != 3:
                self._errors.append("Edge must be a list of [source, target, data]")
                continue

            source, target, data = edge
            if source not in node_ids:
                self._errors.append(f"Edge source {source} not found in nodes")
            if target not in node_ids:
                self._errors.append(f"Edge target {target} not found in nodes")
            if not isinstance(data, dict):
                self._errors.append("Edge data must be a dictionary")

        # Validate required fields
        required_fields = [
            "role",
            "user_objective",
            "builder_objective",
            "domain",
            "intro",
            "task_docs",
            "rag_docs",
            "workers",
        ]
        for field in required_fields:
            if field not in graph:
                self._errors.append(f"Missing required field: {field}")

        return len(self._errors) == 0

    def get_error_messages(self) -> list[str]:
        """Get validation error messages.

        Returns:
            List[str]: Validation error messages
        """
        return self._errors
