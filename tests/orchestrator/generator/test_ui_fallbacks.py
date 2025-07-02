"""Test utilities and fallback classes for UI components.

This module provides fallback classes and testing utilities for the UI components
when the textual framework is not available. These are used exclusively for testing
purposes and should not be used in production code.
"""

import contextlib
from typing import Any


class FallbackApp:
    """Fallback App class for when textual is not available."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def run(self) -> None:
        pass

    def exit(self, result: object = None) -> None:
        """Exit the app (fallback implementation)."""
        pass

    def push_screen(self, screen: object) -> None:
        """Push a screen (fallback implementation)."""
        pass

    def call_later(self, callback: object) -> None:
        """Call a callback later (fallback implementation)."""
        pass


class FallbackComposeResult:
    """Fallback ComposeResult class."""

    pass


class FallbackReturnType:
    """Fallback ReturnType class."""

    pass


class FallbackKey:
    """Fallback Key class for keyboard events."""

    def __init__(self, key: object = None) -> None:
        self.key = key


class FallbackLabel:
    """Fallback Label class."""

    def __init__(self, text: str = "", **kwargs: object) -> None:
        self.text = text


class FallbackTree:
    """Fallback Tree class."""

    def __init__(self, title: str = "", **kwargs: object) -> None:
        self.title = title
        self.root = FallbackTreeNode("root")
        self.cursor_node = None

    def focus(self) -> None:
        """Focus the tree (fallback implementation)."""
        pass

    def query_one(self, selector: str, widget_type: type = None) -> object:
        """Query for a widget (fallback implementation)."""
        return None

    class NodeSelected:
        """Fallback NodeSelected event class."""

        def __init__(self, node: object = None) -> None:
            self.node = node


class FallbackTreeNode:
    """Fallback TreeNode class."""

    def __init__(self, label: str = "", **kwargs: object) -> None:
        self.label = label
        self.children = []
        self.parent = None

    def expand(self) -> None:
        """Expand the node (fallback implementation)."""
        pass

    def add(self, label: str) -> "FallbackTreeNode":
        """Add a child node (fallback implementation)."""
        child = FallbackTreeNode(label)
        child.parent = self
        self.children.append(child)
        return child

    def add_leaf(self, label: str) -> "FallbackTreeNode":
        """Add a leaf node (fallback implementation)."""
        return self.add(label)

    def set_label(self, label: str) -> None:
        """Set the node label (fallback implementation)."""
        self.label = label

    def remove(self) -> None:
        """Remove this node from its parent (fallback implementation)."""
        if self.parent and self.parent.children:
            with contextlib.suppress(ValueError):
                self.parent.children.remove(self)


class FallbackHorizontal:
    """Fallback Horizontal container class."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass


class FallbackVertical:
    """Fallback Vertical container class."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass


class FallbackScreen:
    """Fallback Screen class."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.app = None

    def query_one(self, selector: str, widget_type: type = None) -> object:
        """Query for a widget (fallback implementation)."""
        return None


class FallbackButton:
    """Fallback Button class."""

    def __init__(self, text: str = "", **kwargs: object) -> None:
        self.text = text
        self.id = kwargs.get("id", "")

    class Pressed:
        """Fallback Pressed event class."""

        def __init__(self, button: object = None) -> None:
            self.button = button


class FallbackInput:
    """Fallback Input class."""

    def __init__(self, value: str = "", **kwargs: object) -> None:
        self.value = value


class FallbackStatic:
    """Fallback Static class."""

    def __init__(self, text: str = "", **kwargs: object) -> None:
        self.text = text


class FallbackInputModal:
    """Fallback InputModal class for testing."""

    def __init__(self, title: str, default: str = "", **kwargs: object) -> None:
        self.title = title
        self.default = default
        self.result = default
        self.node = kwargs.get("node")
        self.callback = kwargs.get("callback")

    def compose(self) -> FallbackComposeResult:
        """Fallback compose method."""
        return FallbackComposeResult()

    def on_button_pressed(self, event: FallbackButton.Pressed) -> None:
        """Fallback button press handler."""
        if getattr(event.button, "id", None) == "submit":
            self.result = self.default  # Simplified for testing
        if self.callback:
            self.callback(self.result, self.node)


class FallbackTaskEditorApp:
    """Fallback TaskEditorApp class for testing."""

    def __init__(self, tasks: list[dict[str, Any]]) -> None:
        self.tasks = tasks
        self.task_tree = None

    def compose(self) -> FallbackComposeResult:
        """Fallback compose method."""
        self.task_tree = FallbackTree("Tasks")
        self.task_tree.root.expand()
        tasks = self.tasks if self.tasks is not None else []
        for task in tasks:
            task_node = self.task_tree.root.add(task["name"])
            if "steps" in task and task["steps"]:
                for step in task["steps"]:
                    if isinstance(step, dict):
                        step_text = step.get("description", str(step))
                    else:
                        step_text = str(step)
                    task_node.add_leaf(step_text)
        return FallbackComposeResult()

    def on_mount(self) -> None:
        """Fallback on_mount method."""
        if self.task_tree:
            self.task_tree.focus()

    async def on_tree_node_selected(self, event: FallbackTree.NodeSelected) -> None:
        """Fallback node selection handler."""
        pass

    async def on_key(self, event: FallbackKey) -> None:
        """Fallback keyboard handler."""
        pass

    async def action_add_node(self, node: FallbackTreeNode) -> None:
        """Fallback add node action."""
        pass

    def push_screen(self, screen: FallbackInputModal) -> None:
        """Fallback push screen method."""
        pass

    def show_input_modal(self, title: str, default: str = "") -> str:
        """Fallback show input modal method."""
        modal = FallbackInputModal(title, default)
        return modal.result

    async def update_tasks(self) -> None:
        """Fallback update tasks method."""
        if not self.task_tree or not self.task_tree.root:
            return

        updated_tasks = []
        for task_node in self.task_tree.root.children:
            task = {"name": task_node.label}
            steps = []
            for step_node in task_node.children:
                steps.append(step_node.label)
            if steps:
                task["steps"] = steps
            updated_tasks.append(task)

        self.tasks = updated_tasks

    def run(self) -> list[dict[str, Any]]:
        """Fallback run method."""
        return self.tasks


# Test utilities for patching modules
def patch_ui_modules_for_testing() -> None:
    """Patch UI modules with fallback classes for testing."""
    import sys
    from unittest.mock import MagicMock

    # Create mock modules
    mock_textual = MagicMock()
    mock_textual.app = MagicMock()
    mock_textual.app.App = FallbackApp
    mock_textual.app.ComposeResult = FallbackComposeResult
    mock_textual.app.ReturnType = FallbackReturnType
    mock_textual.events = MagicMock()
    mock_textual.events.Key = FallbackKey
    mock_textual.widgets = MagicMock()
    mock_textual.widgets.Label = FallbackLabel
    mock_textual.widgets.Tree = FallbackTree
    mock_textual.widgets.tree = MagicMock()
    mock_textual.widgets.tree.TreeNode = FallbackTreeNode
    mock_textual.containers = MagicMock()
    mock_textual.containers.Horizontal = FallbackHorizontal
    mock_textual.containers.Vertical = FallbackVertical
    mock_textual.screen = MagicMock()
    mock_textual.screen.Screen = FallbackScreen
    mock_textual.widgets.Button = FallbackButton
    mock_textual.widgets.Input = FallbackInput
    mock_textual.widgets.Static = FallbackStatic

    # Patch sys.modules
    sys.modules["textual"] = mock_textual
    sys.modules["textual.app"] = mock_textual.app
    sys.modules["textual.events"] = mock_textual.events
    sys.modules["textual.widgets"] = mock_textual.widgets
    sys.modules["textual.widgets.tree"] = mock_textual.widgets.tree
    sys.modules["textual.containers"] = mock_textual.containers
    sys.modules["textual.screen"] = mock_textual.screen


def unpatch_ui_modules() -> None:
    """Remove patches from UI modules."""
    import sys

    modules_to_remove = [
        "textual",
        "textual.app",
        "textual.events",
        "textual.widgets",
        "textual.widgets.tree",
        "textual.containers",
        "textual.screen",
    ]

    for module_name in modules_to_remove:
        if module_name in sys.modules:
            del sys.modules[module_name]
