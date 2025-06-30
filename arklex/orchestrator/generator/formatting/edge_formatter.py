"""Edge formatting component for the Arklex framework.

This module provides the EdgeFormatter class that handles formatting of
task graph edges.
"""

from typing import Any

from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class EdgeFormatter:
    """Formats task graph edges.

    This class handles the formatting of edges in the task graph,
    including intent and attribute fields.
    """

    def __init__(
        self,
        default_intent: str = "depends_on",
        default_weight: int = 1,
    ) -> None:
        self._default_intent = default_intent
        self._default_weight = default_weight

    def format_edge(
        self,
        source_idx: str,
        target_idx: str,
        source_task: dict[str, Any],
        target_task: dict[str, Any],
    ) -> list[Any]:
        source_name = source_task.get("name", "")
        target_name = target_task.get("name", "")
        edge_data = {
            "intent": self._default_intent,
            "attribute": {
                "weight": self._default_weight,
                "pred": self._default_intent,
                "definition": f"{target_name} depends on {source_name}",
                "sample_utterances": [
                    f"I need to complete {source_name} before {target_name}",
                    f"{target_name} requires {source_name} to be done first",
                ],
            },
        }
        return [source_idx, target_idx, edge_data]

    def format_edge_data(
        self,
        source: dict[str, Any],
        target: dict[str, Any],
        type: str = "depends_on",
        weight: float = 1.0,
        label: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Format edge data in the old format.

        Args:
            source (Dict[str, Any]): Source node dict
            target (Dict[str, Any]): Target node dict
            type (str): Edge type
            weight (float): Edge weight
            label (str): Edge label
            metadata (Optional[Dict[str, Any]]): Additional metadata

        Returns:
            Dict[str, Any]: Formatted edge data
        """
        return {
            "intent": label or type,
            "attribute": {
                "weight": weight,
                "pred": True,
                "definition": "",
                "sample_utterances": [],
            },
        }

    def format_edge_style(
        self,
        source: dict[str, Any],
        target: dict[str, Any],
        type: str = "depends_on",
        weight: float = 1.0,
        label: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Format edge style.

        Args:
            source (Dict[str, Any]): Source node dict
            target (Dict[str, Any]): Target node dict
            type (str): Edge type
            weight (float): Edge weight
            label (str): Edge label
            metadata (Optional[Dict[str, Any]]): Additional metadata

        Returns:
            Dict[str, Any]: Formatted edge style
        """
        type_colors = {
            "depends_on": "#ff0000",
            "blocks": "#ffa500",
            "related_to": "#00ff00",
            "part_of": "#0000ff",
        }
        color = type_colors.get(type, "#808080")
        return {
            "color": color,
            "width": 2,
            "style": "solid",
            "arrow_size": 10,
            "arrow_style": "triangle",
            "label_color": "#000000",
            "label_font_size": 12,
            "label_font_family": "Arial",
            "label_font_weight": "normal",
            "opacity": 1.0,
        }

    def validate_edge(
        self,
        source: str | dict[str, Any],
        target: str | dict[str, Any],
        type: str = "depends_on",
        weight: float = 1.0,
        label: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Validate an edge.

        Args:
            source (Union[str, Dict[str, Any]]): Source node ID or dict
            target (Union[str, Dict[str, Any]]): Target node ID or dict
            type (str): Edge type
            weight (float): Edge weight
            label (str): Edge label
            metadata (Optional[Dict[str, Any]]): Additional metadata

        Returns:
            bool: True if edge is valid
        """
        # Handle None values
        if source is None or target is None:
            return False

        # Handle string inputs
        if isinstance(source, str):
            source_id = source
        else:
            source_id = source.get("task_id") or source.get("id")
            if not source_id:
                return False

        if isinstance(target, str):
            target_id = target
        else:
            target_id = target.get("task_id") or target.get("id")
            if not target_id:
                return False

        # Validate edge type
        if not isinstance(type, str):
            return False

        # Validate weight
        if not isinstance(weight, int | float):
            return False

        # Validate label
        if not isinstance(label, str):
            return False

        # Validate metadata if provided
        return not (metadata is not None and not isinstance(metadata, dict))
