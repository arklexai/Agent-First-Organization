"""TaskGraph format checker for validating taskgraph.json files.

This module provides comprehensive validation for taskgraph.json files generated
by the Arklex framework. It checks the structure, content, and data types to
ensure the taskgraph is properly formatted and contains all required components.

The checker validates:
- JSON structure and format
- Required fields (nodes, edges, templates, settings)
- Node structure and content
- Edge structure and content
- Template structure and content
- Data types and formats
- Referential integrity between nodes and edges
"""

import json
import os
from typing import Any

from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class TaskGraphFormatChecker:
    """Validates taskgraph.json files for proper format and structure."""

    def __init__(self) -> None:
        """Initialize the TaskGraphFormatChecker."""
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.node_ids: set[str] = set()
        self.template_ids: set[str] = set()

    def check_taskgraph_file(self, file_path: str) -> tuple[bool, list[str], list[str]]:
        """Check a taskgraph.json file for format compliance.

        Args:
            file_path (str): Path to the taskgraph.json file

        Returns:
            tuple[bool, list[str], list[str]]: (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # Check if file exists
        if not os.path.exists(file_path):
            self.errors.append(f"File does not exist: {file_path}")
            return False, self.errors, self.warnings

        # Check file extension
        if not file_path.endswith(".json"):
            self.warnings.append(f"File does not have .json extension: {file_path}")

        # Load and parse JSON
        try:
            with open(file_path, encoding="utf-8") as f:
                taskgraph = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON format: {e}")
            return False, self.errors, self.warnings
        except Exception as e:
            self.errors.append(f"Error reading file: {e}")
            return False, self.errors, self.warnings

        # Validate taskgraph structure
        is_valid = self._validate_taskgraph_structure(taskgraph)

        return is_valid, self.errors, self.warnings

    def _validate_taskgraph_structure(self, taskgraph: dict[str, Any]) -> bool:
        """Validate the overall structure of the taskgraph.

        Args:
            taskgraph (dict[str, Any]): The taskgraph dictionary

        Returns:
            bool: True if structure is valid
        """
        if not isinstance(taskgraph, dict):
            self.errors.append("TaskGraph must be a dictionary")
            return False

        # Check required top-level fields
        required_fields = ["nodes", "edges", "settings"]
        for field in required_fields:
            if field not in taskgraph:
                self.errors.append(f"Missing required field: {field}")
                return False

        # Validate each section
        nodes_valid = self._validate_nodes(taskgraph.get("nodes", []))
        edges_valid = self._validate_edges(taskgraph.get("edges", []))
        templates_valid = self._validate_templates(taskgraph.get("reusable_tasks", {}))
        settings_valid = self._validate_settings(taskgraph.get("settings", {}))

        # Check referential integrity
        referential_valid = self._check_referential_integrity(
            taskgraph.get("nodes", []), taskgraph.get("edges", [])
        )

        return all(
            [
                nodes_valid,
                edges_valid,
                templates_valid,
                settings_valid,
                referential_valid,
            ]
        )

    def _validate_nodes(self, nodes: list[Any]) -> bool:
        """Validate the nodes section of the taskgraph.

        Args:
            nodes (list[Any]): List of nodes

        Returns:
            bool: True if nodes are valid
        """
        if not isinstance(nodes, list):
            self.errors.append("Nodes must be a list")
            return False

        if not nodes:
            self.warnings.append("Nodes list is empty")

        self.node_ids.clear()
        valid_nodes = True

        for i, node in enumerate(nodes):
            if not isinstance(node, list) or len(node) != 2:
                self.errors.append(f"Node {i} must be a list of [id, data]")
                valid_nodes = False
                continue

            node_id, node_data = node

            # Validate node ID
            if not isinstance(node_id, str):
                self.errors.append(f"Node {i} ID must be a string, got {type(node_id)}")
                valid_nodes = False
                continue

            if node_id in self.node_ids:
                self.errors.append(f"Duplicate node ID: {node_id}")
                valid_nodes = False
                continue

            self.node_ids.add(node_id)

            # Validate node data
            if not isinstance(node_data, dict):
                self.errors.append(f"Node {i} data must be a dictionary")
                valid_nodes = False
                continue

            # Check for required node data fields
            if "resource" not in node_data:
                self.errors.append(f"Node {node_id} missing required field: resource")
                valid_nodes = False
                continue

            if "attribute" not in node_data:
                self.errors.append(f"Node {node_id} missing required field: attribute")
                valid_nodes = False
                continue

            # Validate resource structure
            resource = node_data["resource"]
            if not isinstance(resource, dict):
                self.errors.append(f"Node {node_id} resource must be a dictionary")
                valid_nodes = False
                continue

            if "id" not in resource or "name" not in resource:
                self.errors.append(
                    f"Node {node_id} resource must have 'id' and 'name' fields"
                )
                valid_nodes = False
                continue

            # Validate attribute structure
            attribute = node_data["attribute"]
            if not isinstance(attribute, dict):
                self.errors.append(f"Node {node_id} attribute must be a dictionary")
                valid_nodes = False
                continue

            if "value" not in attribute or "task" not in attribute:
                self.errors.append(
                    f"Node {node_id} attribute must have 'value' and 'task' fields"
                )
                valid_nodes = False
                continue

            # Check for directed field in attribute
            if "directed" in attribute and not isinstance(attribute["directed"], bool):
                self.warnings.append(
                    f"Node {node_id} attribute 'directed' should be boolean"
                )

            # Check for optional fields
            if "type" in node_data and not isinstance(node_data["type"], str):
                self.warnings.append(f"Node {node_id} 'type' should be string")

            if "limit" in node_data and not isinstance(node_data["limit"], int | float):
                self.warnings.append(f"Node {node_id} 'limit' should be numeric")

        return valid_nodes

    def _validate_edges(self, edges: list[Any]) -> bool:
        """Validate the edges section of the taskgraph.

        Args:
            edges (list[Any]): List of edges

        Returns:
            bool: True if edges are valid
        """
        if not isinstance(edges, list):
            self.errors.append("Edges must be a list")
            return False

        valid_edges = True

        for i, edge in enumerate(edges):
            if not isinstance(edge, list) or len(edge) != 3:
                self.errors.append(f"Edge {i} must be a list of [source, target, data]")
                valid_edges = False
                continue

            source, target, data = edge

            # Validate source and target
            if not isinstance(source, str):
                self.errors.append(f"Edge {i} source must be a string")
                valid_edges = False
                continue

            if not isinstance(target, str):
                self.errors.append(f"Edge {i} target must be a string")
                valid_edges = False
                continue

            # Validate edge data
            if not isinstance(data, dict):
                self.errors.append(f"Edge {i} data must be a dictionary")
                valid_edges = False
                continue

            # Check for required edge data fields
            if "intent" not in data:
                self.errors.append(f"Edge {i} missing required field: intent")
                valid_edges = False
                continue

            if "attribute" not in data:
                self.errors.append(f"Edge {i} missing required field: attribute")
                valid_edges = False
                continue

            # Validate intent (can be None, string, or other types)
            intent = data["intent"]
            if intent is not None and not isinstance(intent, str):
                self.warnings.append(
                    f"Edge {i} intent should be string or None, got {type(intent)}"
                )

            # Validate attribute structure
            attribute = data["attribute"]
            if not isinstance(attribute, dict):
                self.errors.append(f"Edge {i} attribute must be a dictionary")
                valid_edges = False
                continue

            # Check for required attribute fields
            if "weight" not in attribute:
                self.errors.append(f"Edge {i} attribute missing required field: weight")
                valid_edges = False
                continue

            if "pred" not in attribute:
                self.errors.append(f"Edge {i} attribute missing required field: pred")
                valid_edges = False
                continue

            # Validate attribute field types
            if not isinstance(attribute["weight"], int | float):
                self.errors.append(f"Edge {i} attribute 'weight' must be numeric")
                valid_edges = False
                continue

            if not isinstance(attribute["pred"], bool):
                self.errors.append(f"Edge {i} attribute 'pred' must be boolean")
                valid_edges = False
                continue

            # Check for optional attribute fields
            if "definition" in attribute and not isinstance(
                attribute["definition"], str
            ):
                self.warnings.append(
                    f"Edge {i} attribute 'definition' should be string"
                )

            if "sample_utterances" in attribute:
                sample_utterances = attribute["sample_utterances"]
                if not isinstance(sample_utterances, list):
                    self.warnings.append(
                        f"Edge {i} attribute 'sample_utterances' should be list"
                    )
                elif sample_utterances and not all(
                    isinstance(u, str) for u in sample_utterances
                ):
                    self.warnings.append(
                        f"Edge {i} attribute 'sample_utterances' should contain strings"
                    )

        return valid_edges

    def _validate_templates(self, reusable_tasks: dict[str, Any]) -> bool:
        """Validate the reusable_tasks section of the taskgraph.

        Args:
            reusable_tasks (dict[str, Any]): Dictionary of reusable tasks/templates

        Returns:
            bool: True if reusable tasks are valid
        """
        if not isinstance(reusable_tasks, dict):
            self.errors.append("Reusable tasks must be a dictionary")
            return False

        self.template_ids.clear()
        valid_templates = True

        for template_id, template_data in reusable_tasks.items():
            if not isinstance(template_id, str):
                self.errors.append(
                    f"Template ID must be string, got {type(template_id)}"
                )
                valid_templates = False
                continue

            if template_id in self.template_ids:
                self.errors.append(f"Duplicate template ID: {template_id}")
                valid_templates = False
                continue

            self.template_ids.add(template_id)

            if not isinstance(template_data, dict):
                self.errors.append(f"Template {template_id} data must be a dictionary")
                valid_templates = False
                continue

            # Check if this is a template (has template_id) or a nested graph resource
            if "template_id" in template_data:
                # This is a template - validate template structure
                # Check for required template fields
                required_fields = [
                    "template_id",
                    "name",
                    "description",
                    "steps",
                    "parameters",
                    "examples",
                    "version",
                    "category",
                ]
                for field in required_fields:
                    if field not in template_data:
                        self.errors.append(
                            f"Template {template_id} missing required field: {field}"
                        )
                        valid_templates = False
                        continue

                # Validate template_id matches the key
                if template_data["template_id"] != template_id:
                    self.errors.append(
                        f"Template {template_id} template_id field does not match key"
                    )
                    valid_templates = False
                    continue

                # Validate field types
                if not isinstance(template_data["name"], str):
                    self.errors.append(f"Template {template_id} 'name' must be string")
                    valid_templates = False
                    continue

                if not isinstance(template_data["description"], str):
                    self.errors.append(
                        f"Template {template_id} 'description' must be string"
                    )
                    valid_templates = False
                    continue

                if not isinstance(template_data["steps"], list):
                    self.errors.append(f"Template {template_id} 'steps' must be list")
                    valid_templates = False
                    continue

                if not isinstance(template_data["parameters"], dict):
                    self.errors.append(
                        f"Template {template_id} 'parameters' must be dictionary"
                    )
                    valid_templates = False
                    continue

                if not isinstance(template_data["examples"], list):
                    self.errors.append(
                        f"Template {template_id} 'examples' must be list"
                    )
                    valid_templates = False
                    continue

                if not isinstance(template_data["version"], str):
                    self.errors.append(
                        f"Template {template_id} 'version' must be string"
                    )
                    valid_templates = False
                    continue

                if not isinstance(template_data["category"], str):
                    self.errors.append(
                        f"Template {template_id} 'category' must be string"
                    )
                    valid_templates = False
                    continue

                # Validate steps structure
                steps = template_data["steps"]
                for j, step in enumerate(steps):
                    if not isinstance(step, dict):
                        self.errors.append(
                            f"Template {template_id} step {j} must be dictionary"
                        )
                        valid_templates = False
                        continue

                    # Check for required step fields
                    step_required_fields = [
                        "task",
                        "description",
                        "step_id",
                        "required_fields",
                    ]
                    for field in step_required_fields:
                        if field not in step:
                            self.errors.append(
                                f"Template {template_id} step {j} missing required field: {field}"
                            )
                            valid_templates = False
                            continue

                    # Validate step field types
                    if not isinstance(step["task"], str):
                        self.errors.append(
                            f"Template {template_id} step {j} 'task' must be string"
                        )
                        valid_templates = False
                        continue

                    if not isinstance(step["description"], str):
                        self.errors.append(
                            f"Template {template_id} step {j} 'description' must be string"
                        )
                        valid_templates = False
                        continue

                    if not isinstance(step["step_id"], str):
                        self.errors.append(
                            f"Template {template_id} step {j} 'step_id' must be string"
                        )
                        valid_templates = False
                        continue

                    if not isinstance(step["required_fields"], list):
                        self.errors.append(
                            f"Template {template_id} step {j} 'required_fields' must be list"
                        )
                        valid_templates = False
                        continue

            elif "resource" in template_data:
                # This is a nested graph resource - validate resource structure
                resource = template_data["resource"]
                if not isinstance(resource, dict):
                    self.errors.append(
                        f"Nested graph {template_id} resource must be a dictionary"
                    )
                    valid_templates = False
                    continue

                if "id" not in resource or "name" not in resource:
                    self.errors.append(
                        f"Nested graph {template_id} resource must have 'id' and 'name' fields"
                    )
                    valid_templates = False
                    continue

                # Check for optional limit field
                if "limit" in template_data and not isinstance(
                    template_data["limit"], int | float
                ):
                    self.warnings.append(
                        f"Nested graph {template_id} 'limit' should be numeric"
                    )
            else:
                # Unknown structure
                self.errors.append(
                    f"Template {template_id} has unknown structure - must be template or nested graph resource"
                )
                valid_templates = False
                continue

        return valid_templates

    def _validate_settings(self, settings: dict[str, Any]) -> bool:
        """Validate the settings section of the taskgraph.

        Args:
            settings (dict[str, Any]): Settings dictionary

        Returns:
            bool: True if settings are valid
        """
        if not isinstance(settings, dict):
            self.errors.append("Settings must be a dictionary")
            return False

        # Settings can be empty or contain any valid JSON data
        # No specific validation required for settings content
        return True

    def _check_referential_integrity(self, nodes: list[Any], edges: list[Any]) -> bool:
        """Check referential integrity between nodes and edges.

        Args:
            nodes (list[Any]): List of nodes
            edges (list[Any]): List of edges

        Returns:
            bool: True if referential integrity is maintained
        """
        valid_integrity = True

        # Extract node IDs
        node_ids = set()
        for node in nodes:
            if isinstance(node, list) and len(node) >= 1:
                node_ids.add(node[0])

        # Check edge references
        for i, edge in enumerate(edges):
            if isinstance(edge, list) and len(edge) >= 2:
                source, target = edge[0], edge[1]

                if source not in node_ids:
                    self.errors.append(
                        f"Edge {i} references non-existent source node: {source}"
                    )
                    valid_integrity = False

                if target not in node_ids:
                    self.errors.append(
                        f"Edge {i} references non-existent target node: {target}"
                    )
                    valid_integrity = False

        return valid_integrity

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the validation results.

        Returns:
            dict[str, Any]: Summary with error count, warning count, and details
        """
        return {
            "is_valid": len(self.errors) == 0,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "node_count": len(self.node_ids),
            "reusable_task_count": len(self.template_ids),
            "errors": self.errors,
            "warnings": self.warnings,
        }


def main() -> None:
    """Main function to run the taskgraph format checker."""
    import argparse

    parser = argparse.ArgumentParser(description="Check taskgraph.json format")
    parser.add_argument("file_path", help="Path to the taskgraph.json file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    checker = TaskGraphFormatChecker()
    is_valid, errors, warnings = checker.check_taskgraph_file(args.file_path)

    summary = checker.get_summary()

    print(f"TaskGraph Format Check Results for: {args.file_path}")
    print(f"Valid: {summary['is_valid']}")
    print(f"Errors: {summary['error_count']}")
    print(f"Warnings: {summary['warning_count']}")
    print(f"Nodes: {summary['node_count']}")
    print(f"Reusable Tasks: {summary['reusable_task_count']}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  ❌ {error}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")

    if not errors and not warnings:
        print("\n✅ TaskGraph format is valid!")

    exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
