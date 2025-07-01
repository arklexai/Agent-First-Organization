"""Data management utilities for task editor components.

This module contains the TaskDataManager class that handles task data operations
independently of UI components, making it easier to test and maintain.
"""

from typing import Any

from .protocols import TreeNodeProtocol, TreeProtocol


class TaskDataManager:
    """Manages task data operations independently of UI."""

    @staticmethod
    def extract_label_text(label: str | object) -> str:
        """Extract text from a label object.

        Args:
            label: The label object which may have different formats

        Returns:
            The extracted text string
        """
        if hasattr(label, "plain"):
            return label.plain
        return str(label)

    @staticmethod
    def build_tasks_from_tree(root: TreeNodeProtocol | None) -> list[dict[str, Any]]:
        """Build tasks list from tree structure.

        Args:
            root: The root node of the tree

        Returns:
            List of task dictionaries
        """
        if not root:
            return []

        updated_tasks = []
        for task_node in root.children:
            task_name = TaskDataManager.extract_label_text(task_node.label)
            task = {"name": task_name}
            steps = []

            for step_node in task_node.children:
                step_name = TaskDataManager.extract_label_text(step_node.label)
                steps.append(step_name)

            if steps:
                task["steps"] = steps
            updated_tasks.append(task)

        return updated_tasks

    @staticmethod
    def populate_tree_from_tasks(
        tree: TreeProtocol, tasks: list[dict[str, Any]]
    ) -> None:
        """Populate tree with task data.

        Args:
            tree: The tree widget to populate
            tasks: List of task dictionaries
        """
        if not tree.root:
            return

        tree.root.expand()
        tasks = tasks if tasks is not None else []

        for task in tasks:
            if tree.root:
                task_node = tree.root.add(task["name"])
                if "steps" in task and task["steps"]:
                    for step in task["steps"]:
                        if isinstance(step, dict):
                            step_text = step.get("description", str(step))
                        else:
                            step_text = str(step)
                        task_node.add_leaf(step_text)
