"""Task editor UI component for the Arklex framework.

This module provides the TaskEditor class that handles the UI for editing
task definitions.
"""

from typing import List, Dict, Any
from arklex.utils.logging_utils import LogContext

from textual.app import App, ComposeResult
from textual.widgets import Tree, Label
from textual.widgets.tree import TreeNode

from .input_modal import InputModal

log_context = LogContext(__name__)


class TaskEditorApp(App):
    """A Textual app to edit tasks and steps in a hierarchical structure.

    This class provides a text-based user interface for editing tasks and their steps.
    It supports adding, editing, and deleting tasks and steps in a tree structure.

    Attributes:
        tasks (list): List of task dictionaries containing task names and steps
        task_tree (Tree): The tree widget displaying tasks and steps

    Methods:
        compose(): Creates the main UI components
        on_mount(): Initializes the UI after mounting
        on_tree_node_selected(): Handles node selection events
        on_key(): Processes keyboard input
        action_add_node(): Adds new nodes to the tree
        show_input_modal(): Displays the input modal dialog
        update_tasks(): Updates the tasks list from the tree structure
    """

    def __init__(self, tasks: List[Dict[str, Any]]) -> None:
        """Initialize the TaskEditorApp instance.

        Args:
            tasks (List[Dict[str, Any]]): List of task dictionaries containing task names and steps
        """
        super().__init__()
        self.tasks = tasks
        self.task_tree: Tree[str] = None

    def compose(self) -> ComposeResult:
        """Create the main UI components.

        Creates the tree widget and populates it with tasks and steps, along with
        instruction labels for user interaction.

        Yields:
            ComposeResult: The composed UI elements
        """
        self.task_tree = Tree("Tasks")
        self.task_tree.root.expand()

        # Treat None as empty list
        tasks = self.tasks if self.tasks is not None else []
        # Populate the tree with tasks and steps
        for task in tasks:
            task_node = self.task_tree.root.add(task["name"], expand=True)
            for step in task["steps"]:
                label = (
                    step["description"]
                    if isinstance(step, dict) and "description" in step
                    else str(step)
                )
                task_node.add_leaf(label)

        yield self.task_tree
        yield Label(
            "Click on a task or step to edit it. Press 'a' to add new item, 'd' to delete, 's' to save and exit."
        )

    def on_mount(self):
        """Initialize the UI after mounting.

        Sets focus to the task tree widget to enable keyboard navigation.
        """
        self.task_tree.focus()

    async def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle node selection events.

        When a tree node is selected, shows an input modal for editing the node's label.

        Args:
            event (Tree.NodeSelected): The node selection event
        """
        selected_node = event.node

        def handle_modal_result(result: str, node: TreeNode) -> None:
            """Handle the result from the input modal.

            Args:
                result (str): The result from the modal input
                node (TreeNode): The tree node being edited
            """
            if result is not None:  # Check if the user submitted a valid result
                node.set_label(result)  # Update the tree node's label
                self.call_later(
                    self.update_tasks
                )  # Ensure task sync runs after UI update

        self.push_screen(
            InputModal(
                f"Edit '{selected_node.label}'",
                default=str(selected_node.label),
                node=selected_node,
                callback=handle_modal_result,
            )
        )

    async def on_key(self, event) -> None:
        """Process keyboard input.

        Handles keyboard shortcuts for adding nodes ('a'), deleting nodes ('d'),
        and saving and exiting ('s').

        Args:
            event: The keyboard event
        """
        selected_node = self.task_tree.cursor_node
        if event.key == "a" and selected_node and selected_node.parent is not None:
            await self.action_add_node(selected_node)
        elif event.key == "d" and selected_node and selected_node.parent is not None:
            selected_node.remove()
            await self.update_tasks()
        elif event.key == "s":
            self.exit(self.tasks)

    async def action_add_node(self, node: TreeNode) -> None:
        """Add new nodes to the tree.

        Determines whether to add a task or step based on the selected node and
        shows an input modal for entering the new item.

        Args:
            node (TreeNode): The currently selected tree node
        """
        # if the node is a step node
        if node.parent.parent is not None:
            leaf = True
            node = node.parent
            title = f"Add new step under '{node.label.plain}'"
        else:  # if the node is a task node
            if node.is_expanded:  # add step
                leaf = True
                node = node
                title = f"Enter new step under '{node.label.plain}'"
            else:
                leaf = False
                node = node.parent
                title = f"Add new task under '{node.label.plain}'"

        def handle_modal_result(result: str, node: TreeNode) -> None:
            """Handle the result from the input modal for adding nodes.

            Args:
                result (str): The result from the modal input
                node (TreeNode): The tree node to add the new item to
            """
            if result is not None:  # Check if the user submitted a valid result
                if leaf:
                    node.add_leaf(result)
                else:
                    node.add(result, expand=True)
                self.call_later(
                    self.update_tasks
                )  # Ensure task sync runs after UI update

        self.push_screen(
            InputModal(title, default="", node=node, callback=handle_modal_result)
        )

    def show_input_modal(self, title: str, default: str = "") -> str:
        """Display the input modal dialog.

        Args:
            title (str): Title for the modal dialog
            default (str): Default value for the input field. Defaults to "".

        Returns:
            str: The result from the modal
        """
        modal = InputModal(title, default)
        self.push_screen(modal)
        return modal.result

    async def update_tasks(self) -> None:
        """Update the tasks list from the tree structure.

        Synchronizes the internal tasks list with the current state of the tree widget,
        extracting task names and their associated steps.
        """
        self.tasks = []
        if self.task_tree is None or getattr(self.task_tree, "root", None) is None:
            return

        for task_node in self.task_tree.root.children:
            task_name = task_node.label.plain
            steps = [step.label.plain for step in task_node.children]
            self.tasks.append({"name": task_name, "steps": steps})

        log_message = f"Updated Tasks: {self.tasks}"
        log_context.debug(log_message)

    def run(self) -> List[Dict[str, Any]]:
        """Run the task editor app.

        Returns:
            List[Dict[str, Any]]: The updated tasks list
        """
        super().run()
        return self.tasks
