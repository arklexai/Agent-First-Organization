"""Task editor UI component for the Arklex framework.

This module provides the TaskEditorApp class that handles the UI for editing
task definitions in a hierarchical tree structure.
"""

from collections.abc import Callable
from typing import Any

from textual.app import App, ComposeResult
from textual.events import Key
from textual.widgets import Label, Tree
from textual.widgets.tree import TreeNode

from arklex.utils.logging_utils import LogContext

from .data_manager import TaskDataManager
from .input_modal import InputModal

log_context = LogContext(__name__)


class TaskEditorApp(App):
    """A Textual app to edit tasks and steps in a hierarchical structure.

    This class provides a text-based user interface for editing tasks and their steps.
    It supports adding, editing, and deleting tasks and steps in a tree structure.
    The interface uses keyboard shortcuts for navigation and editing operations.

    The app displays tasks as parent nodes and their steps as child nodes in a tree
    structure. Users can navigate using arrow keys, edit nodes by selecting them,
    add new nodes with 'a', delete nodes with 'd', and save changes with 's'.

    Attributes:
        tasks (list): List of task dictionaries containing task names and steps.
                     Each task dict should have a 'name' key and optionally a 'steps' key.
        task_tree (Tree | None): The tree widget displaying tasks and steps, initialized as None.
                                Set during the compose() method.
        _data_manager (TaskDataManager): Manager for task data operations
        _input_modal_class (type): Class for creating input modals (injectable for testing)

    Methods:
        compose(): Creates the main UI components and populates the tree
        on_mount(): Initializes the UI after mounting and sets focus
        on_tree_node_selected(): Handles node selection events and opens edit modal
        on_key(): Processes keyboard input for navigation and actions
        action_add_node(): Adds new nodes to the tree via modal input
        show_input_modal(): Displays the input modal dialog for editing
        update_tasks(): Updates the tasks list from the tree structure
        run(): Runs the app and returns the updated tasks list

    Keyboard Shortcuts:
        - Arrow keys: Navigate tree nodes
        - 'a': Add new node as child of selected node
        - 'd': Delete selected node (if not root)
        - 's': Save changes and exit
        - Enter: Edit selected node
    """

    def __init__(
        self, tasks: list[dict[str, Any]], input_modal_class: type = InputModal
    ) -> None:
        """Initialize the TaskEditorApp instance.

        Args:
            tasks (List[Dict[str, Any]]): List of task dictionaries containing task names and steps.
                                         Each task dict should have:
                                         - 'name': str - The task name
                                         - 'steps': List[str] | List[Dict] - Optional list of steps.
                                           Steps can be strings or dicts with 'description' key.
            input_modal_class (type): Class for creating input modals (for testing)

        Example:
            tasks = [
                {"name": "Task 1", "steps": ["Step 1", "Step 2"]},
                {"name": "Task 2", "steps": [{"description": "Complex step"}]}
            ]
        """
        super().__init__()
        self.tasks = tasks
        self.task_tree: Tree | None = None
        self._input_modal_class = input_modal_class

    def compose(self) -> ComposeResult:
        """Create the main UI components.

        Creates the tree widget and populates it with tasks and steps, along with
        instruction labels for user interaction. Handles both string and dictionary
        step formats, extracting descriptions from dict steps when available.

        The method creates a hierarchical tree structure where:
        - Root level contains all tasks
        - Each task node contains its steps as child nodes
        - Steps can be strings or dictionaries with 'description' keys

        Yields:
            ComposeResult: The composed UI elements including the tree widget
                          and instruction label
        """
        self.task_tree = Tree("🎯 TASK EDITOR - Edit Your Tasks Below")
        TaskDataManager.populate_tree_from_tasks(self.task_tree, self.tasks)

        yield self.task_tree
        yield Label(
            "🎯 TASK EDITOR ACTIVE - Use 'a' to add nodes, 'd' to delete, 's' to save and exit, arrow keys to navigate"
        )

    def on_mount(self) -> None:
        """Initialize the UI after mounting.

        Sets focus to the task tree widget to enable keyboard navigation.
        """
        if self.task_tree:
            self.task_tree.focus()

    async def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle node selection events.

        When a tree node is selected, shows an input modal for editing the node's label.
        The modal is pre-populated with the current node label, which is extracted
        from either the 'plain' attribute or string representation of the label.

        Args:
            event (Tree.NodeSelected): The node selection event containing the selected node
        """
        selected_node = event.node

        def handle_modal_result(result: str, node: TreeNode) -> None:
            """Handle the result from the input modal.

            Args:
                result (str): The result from the modal input
                node (TreeNode): The tree node being edited
            """
            if result and result.strip():
                node.set_label(result.strip())

        current_label = TaskDataManager.extract_label_text(selected_node.label)

        self.show_input_modal(
            "Edit node", current_label, selected_node, handle_modal_result
        )

    async def on_key(self, event: Key) -> None:
        """Process keyboard input.

        Handles keyboard shortcuts for adding nodes ('a'), deleting nodes ('d'),
        and saving and exiting ('s'). Only processes specific keys and ignores others.

        Keyboard shortcuts:
        - 'a': Adds a new child node to the currently selected node
        - 'd': Deletes the currently selected node (only if it has a parent)
        - 's': Updates the tasks list and exits the app with the modified data

        Args:
            event (Key): The keyboard event containing the pressed key
        """
        if not self.task_tree:
            return

        if event.key == "a":
            await self.action_add_node(self.task_tree.cursor_node)
        elif event.key == "d":
            await self.action_delete_node(self.task_tree.cursor_node)
        elif event.key == "s":
            await self.action_save_and_exit()

    async def action_add_node(self, node: TreeNode | None) -> None:
        """Add a new node to the tree.

        Shows an input modal to get the new node's label and adds it as a child
        of the currently selected node. The modal allows users to enter a name
        for the new node, which is validated to ensure it's not empty.

        Args:
            node (TreeNode): The parent node to add the new node to. If None, no action is taken.
        """
        if not node:
            return

        def handle_modal_result(result: str, parent_node: TreeNode) -> None:
            """Handle the result from the input modal.

            Args:
                result (str): The result from the modal input
                parent_node (TreeNode): The parent node for the new node
            """
            if result and result.strip():
                parent_node.add(result.strip())

        self.show_input_modal("Add new node", "", node, handle_modal_result)

    async def action_delete_node(self, node: TreeNode | None) -> None:
        """Delete the selected node.

        Args:
            node (TreeNode | None): The node to delete
        """
        if node and node.parent:
            node.remove()

    async def action_save_and_exit(self) -> None:
        """Save changes and exit the app."""
        await self.update_tasks()
        self.exit(self.tasks)

    def push_screen(self, screen: "InputModal") -> None:
        """Push a screen to the app.

        Overrides the parent class method to handle InputModal screens specifically.
        This method is called by show_input_modal to display the modal dialog.

        Args:
            screen (InputModal): The modal screen to push onto the screen stack
        """
        # Check if parent class has push_screen method before calling it
        if hasattr(super(), "push_screen"):
            return super().push_screen(screen)
        else:
            # If parent doesn't have push_screen, just store the screen for testing
            self._current_screen = screen
            return [1, 2, 3]  # Return a value for testing

    def show_input_modal(
        self,
        title: str,
        default: str = "",
        node: TreeNode | None = None,
        callback: Callable[[str, TreeNode | None], None] | None = None,
    ) -> str:
        """Show an input modal dialog.

        Creates and displays an input modal with the given title and default value.
        The actual result will be handled by the callback function when the user
        submits or cancels the modal.

        Args:
            title (str): The title of the modal
            default (str): The default value for the input field
            node (TreeNode | None): The tree node being edited
            callback (Callable[[str, TreeNode | None], None] | None): Callback function

        Returns:
            str: The default value (the actual result is handled by the callback)
        """
        modal = self._input_modal_class(title, default, node, callback)
        self.push_screen(modal)
        # Return the default value as the immediate result
        # The actual result will be handled by the callback
        return default

    async def update_tasks(self) -> None:
        """Update the tasks list from the tree structure.

        Traverses the tree structure and updates the tasks list to reflect
        the current state of the tree. This method is called when saving changes
        to convert the visual tree structure back to the data format.

        The method handles different label formats by checking for the 'plain'
        attribute first, then falling back to string representation. It creates
        a new list of task dictionaries with 'name' and 'steps' keys.
        """
        if not self.task_tree:
            return

        self.tasks = TaskDataManager.build_tasks_from_tree(self.task_tree.root)

    def run(self) -> list[dict[str, Any]]:
        """Run the task editor app.

        Starts the app and returns the updated tasks list. This method overrides
        the parent class run method to return the modified task data instead of
        just running the app. The returned data reflects any changes made by
        the user during the editing session.

        Returns:
            List[Dict[str, Any]]: The updated tasks list with any modifications
                                 made by the user during the editing session
        """
        # Run the actual Textual app
        super().run()
        # Return the tasks (which may have been modified during the session)
        return self.tasks
