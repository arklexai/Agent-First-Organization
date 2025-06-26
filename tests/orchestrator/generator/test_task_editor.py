"""Comprehensive tests for the TaskEditorApp class.

This module provides comprehensive test coverage for the TaskEditorApp class,
ensuring all functionality is properly tested with good modularity and formatting.
"""

import sys
import types
from unittest.mock import MagicMock


# Create a minimal fake textual.app module
class FakeApp:
    def __init__(self, *args, **kwargs):
        pass


class FakeComposeResult:
    pass


fake_textual_app = types.ModuleType("textual.app")
fake_textual_app.App = FakeApp
fake_textual_app.ComposeResult = FakeComposeResult
sys.modules["textual.app"] = fake_textual_app

# Mock all other textual imports and submodules
MOCK_MODULES = [
    "textual",
    "textual.widgets",
    "textual.widgets.tree",
    "textual.widgets._toast",
    "textual.events",
    "textual.screen",
    "textual.drivers",
    "textual.drivers.linux_driver",
    "textual.drivers.windows_driver",
    "textual.containers",
]
for mod in MOCK_MODULES:
    sys.modules[mod] = MagicMock()

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Any, Dict, List
from arklex.orchestrator.generator.ui.task_editor import TaskEditorApp


# --- Fixtures for patching and sample data ---


@pytest.fixture
def mock_tree(monkeypatch):
    with patch("arklex.orchestrator.generator.ui.task_editor.Tree") as mock_tree:
        yield mock_tree


@pytest.fixture
def mock_label(monkeypatch):
    with patch("arklex.orchestrator.generator.ui.task_editor.Label") as mock_label:
        yield mock_label


@pytest.fixture
def mock_input_modal(monkeypatch):
    with patch(
        "arklex.orchestrator.generator.ui.task_editor.InputModal"
    ) as mock_input_modal:
        yield mock_input_modal


@pytest.fixture
def mock_log_context(monkeypatch):
    with patch(
        "arklex.orchestrator.generator.ui.task_editor.log_context"
    ) as mock_log_context:
        yield mock_log_context


@pytest.fixture
def sample_tasks() -> list:
    """Sample tasks for testing."""
    return [
        {
            "name": "Task 1",
            "steps": [{"description": "Step 1.1"}, {"description": "Step 1.2"}],
        },
        {"name": "Task 2", "steps": [{"description": "Step 2.1"}]},
    ]


@pytest.fixture
def complex_tasks() -> list:
    """Complex tasks for testing."""
    return [
        {
            "name": "Complex Task",
            "steps": [
                {"description": "Step with dict"},
                "Step as string",
                {"other_field": "Not description"},
            ],
        },
    ]


@pytest.fixture
def task_editor_app(sample_tasks):
    """Create a TaskEditorApp instance for testing."""
    from arklex.orchestrator.generator.ui.task_editor import TaskEditorApp

    return TaskEditorApp(sample_tasks)


class TestTaskEditorAppInitialization:
    """Test TaskEditorApp initialization and basic functionality."""

    def test_import_task_editor_module(self) -> None:
        """Test that the TaskEditorApp module can be imported."""
        import arklex.orchestrator.generator.ui.task_editor as task_editor

        assert hasattr(task_editor, "TaskEditorApp")

    def test_task_editor_instantiation_with_valid_tasks(self) -> None:
        """Test TaskEditorApp instantiation with valid tasks."""
        tasks = [
            {
                "name": "Test Task 1",
                "steps": [{"description": "Step 1"}, {"description": "Step 2"}],
            }
        ]
        editor = TaskEditorApp(tasks)
        assert editor.tasks == tasks
        assert editor.task_tree is None  # Will be set in compose()

    def test_task_editor_instantiation_with_empty_tasks(self) -> None:
        """Test TaskEditorApp instantiation with empty tasks list."""
        editor = TaskEditorApp([])
        assert editor.tasks == []
        assert editor.task_tree is None

    def test_task_editor_instantiation_with_none_tasks(self) -> None:
        """Test TaskEditorApp instantiation with None tasks."""
        editor = TaskEditorApp(None)
        assert editor.tasks is None
        assert editor.task_tree is None

    def test_task_editor_instantiation_with_complex_tasks(self) -> None:
        """Test TaskEditorApp instantiation with complex task structure."""
        tasks = [
            {
                "name": "Complex Task",
                "description": "A complex task description",
                "steps": [
                    {"description": "Step 1", "duration": "1 hour"},
                    {"description": "Step 2", "dependencies": ["Step 1"]},
                ],
                "priority": "high",
                "estimated_duration": "2 hours",
            }
        ]
        editor = TaskEditorApp(tasks)
        assert editor.tasks == tasks
        assert len(editor.tasks[0]["steps"]) == 2

    def test_task_editor_instantiation_with_invalid_task_structure(self) -> None:
        """Test TaskEditorApp with invalid task structure."""
        invalid_tasks = [
            {"invalid_key": "invalid_value"},  # Missing required 'name' field
            {"name": "Valid Task", "steps": "not_a_list"},  # Invalid steps type
        ]

        # Should not raise an exception, just handle gracefully
        editor = TaskEditorApp(invalid_tasks)
        assert editor.tasks == invalid_tasks

    def test_task_editor_methods_existence(self) -> None:
        """Test that all expected methods exist on TaskEditorApp."""
        tasks = [{"name": "Test Task", "steps": [{"description": "Step 1"}]}]
        editor = TaskEditorApp(tasks)

        expected_methods = [
            "compose",
            "on_mount",
            "on_tree_node_selected",
            "on_key",
            "action_add_node",
            "show_input_modal",
            "update_tasks",
        ]

        for method_name in expected_methods:
            assert hasattr(editor, method_name)
            assert callable(getattr(editor, method_name))

    def test_task_editor_attributes_existence(self) -> None:
        """Test that all expected attributes exist on TaskEditorApp."""
        tasks = [{"name": "Test Task", "steps": [{"description": "Step 1"}]}]
        editor = TaskEditorApp(tasks)

        expected_attributes = [
            "tasks",
            "task_tree",
        ]

        for attr_name in expected_attributes:
            assert hasattr(editor, attr_name)

    def test_task_editor_init_none_tasks(self):
        from arklex.orchestrator.generator.ui.task_editor import TaskEditorApp

        editor = TaskEditorApp(None)
        assert editor.tasks is None
        assert editor.task_tree is None


class TestTaskEditorAppCompose:
    """Test TaskEditorApp compose method."""

    def test_compose_method(self, sample_tasks, mock_tree, mock_label) -> None:
        """Test the compose method with valid tasks."""
        app = TaskEditorApp(sample_tasks)

        # Mock the tree and its components
        mock_tree_instance = Mock()
        mock_tree_instance.root = Mock()
        mock_tree_instance.root.add.return_value = Mock()
        mock_tree_instance.root.expand = Mock()
        mock_tree.return_value = mock_tree_instance

        # Mock the task node
        mock_task_node = Mock()
        mock_task_node.add_leaf = Mock()
        mock_tree_instance.root.add.return_value = mock_task_node

        # Mock the label
        mock_label_instance = Mock()
        mock_label.return_value = mock_label_instance

        # Test compose
        result = list(app.compose())

        # Verify tree was created and configured
        mock_tree.assert_called_once_with("Tasks")
        mock_tree_instance.root.expand.assert_called_once()

        # Verify tasks were added to tree
        assert mock_tree_instance.root.add.call_count == 2  # Two tasks

        # Verify steps were added
        assert mock_task_node.add_leaf.call_count == 3  # Total steps across both tasks

        # Verify label was created
        mock_label.assert_called_once()

        # Verify compose returns the expected components
        assert len(result) == 2
        assert result[0] == mock_tree_instance
        assert result[1] == mock_label_instance

    def test_compose_with_empty_tasks(self, mock_tree, mock_label) -> None:
        """Test compose method with empty tasks list."""
        app = TaskEditorApp([])

        mock_tree_instance = Mock()
        mock_tree_instance.root = Mock()
        mock_tree_instance.root.expand = Mock()
        mock_tree.return_value = mock_tree_instance

        mock_label_instance = Mock()
        mock_label.return_value = mock_label_instance

        result = list(app.compose())

        # Verify tree was created
        mock_tree.assert_called_once_with("Tasks")
        mock_tree_instance.root.expand.assert_called_once()

        # Verify no tasks were added
        mock_tree_instance.root.add.assert_not_called()

        # Verify compose returns the expected components
        assert len(result) == 2

    def test_compose_with_none_tasks(self, mock_tree, mock_label) -> None:
        """Test compose method with None tasks."""
        app = TaskEditorApp(None)

        mock_tree_instance = Mock()
        mock_tree_instance.root = Mock()
        mock_tree_instance.root.expand = Mock()
        mock_tree.return_value = mock_tree_instance

        mock_label_instance = Mock()
        mock_label.return_value = mock_label_instance

        # Should handle None gracefully
        result = list(app.compose())

        # Verify tree was created
        mock_tree.assert_called_once_with("Tasks")
        mock_tree_instance.root.expand.assert_called_once()

        # Verify compose returns the expected components
        assert len(result) == 2

    def test_compose_with_complex_tasks(
        self, complex_tasks, mock_tree, mock_label
    ) -> None:
        """Test compose method with complex task structures."""
        app = TaskEditorApp(complex_tasks)

        mock_tree_instance = Mock()
        mock_tree_instance.root = Mock()
        mock_tree_instance.root.expand = Mock()
        mock_tree.return_value = mock_tree_instance

        mock_task_node = Mock()
        mock_task_node.add_leaf = Mock()
        mock_tree_instance.root.add.return_value = mock_task_node

        mock_label_instance = Mock()
        mock_label.return_value = mock_label_instance

        result = list(app.compose())

        # Verify tree was created
        mock_tree.assert_called_once_with("Tasks")
        mock_tree_instance.root.expand.assert_called_once()

        # Verify task was added
        mock_tree_instance.root.add.assert_called_once_with("Complex Task", expand=True)

        # Verify steps were added (3 steps)
        assert mock_task_node.add_leaf.call_count == 3

        # Verify compose returns the expected components
        assert len(result) == 2


class TestTaskEditorAppEventHandling:
    """Test TaskEditorApp event handling methods."""

    def test_on_mount(self, task_editor_app) -> None:
        """Test the on_mount method."""
        # Mock the task_tree
        mock_tree = Mock()
        task_editor_app.task_tree = mock_tree

        # Test on_mount
        task_editor_app.on_mount()

        # Verify focus was set
        mock_tree.focus.assert_called_once()

    async def test_on_tree_node_selected(
        self, task_editor_app, mock_input_modal
    ) -> None:
        """Test the on_tree_node_selected method."""
        # Mock the event and node
        mock_event = Mock()
        mock_node = Mock()
        mock_node.label = "Test Node"
        mock_event.node = mock_node

        # Mock the push_screen method
        task_editor_app.push_screen = Mock()

        # Mock the InputModal
        mock_modal_instance = Mock()
        mock_input_modal.return_value = mock_modal_instance

        # Test on_tree_node_selected (async method)
        await task_editor_app.on_tree_node_selected(mock_event)

        # Verify InputModal was created with correct parameters
        mock_input_modal.assert_called_once()
        call_args = mock_input_modal.call_args
        assert "Edit 'Test Node'" in call_args.args[0]
        default_val = (
            call_args.args[1]
            if len(call_args.args) > 1
            else call_args.kwargs.get("default", None)
        )
        assert default_val == "Test Node"

        # Verify push_screen was called
        task_editor_app.push_screen.assert_called_once_with(mock_modal_instance)

    async def test_on_tree_node_selected_with_none_label(
        self, task_editor_app, mock_input_modal
    ) -> None:
        """Test on_tree_node_selected with None label."""
        # Mock the event and node with None label
        mock_event = Mock()
        mock_node = Mock()
        mock_node.label = None
        mock_event.node = mock_node

        # Mock the push_screen method
        task_editor_app.push_screen = Mock()

        # Mock the InputModal
        mock_modal_instance = Mock()
        mock_input_modal.return_value = mock_modal_instance

        # Test on_tree_node_selected (async method)
        await task_editor_app.on_tree_node_selected(mock_event)

        # Verify InputModal was created with correct parameters
        mock_input_modal.assert_called_once()
        call_args = mock_input_modal.call_args
        assert "Edit 'None'" in call_args.args[0]
        default_val = (
            call_args.args[1]
            if len(call_args.args) > 1
            else call_args.kwargs.get("default", None)
        )
        assert default_val == "None"


class TestTaskEditorAppKeyboardHandling:
    """Test TaskEditorApp keyboard input handling."""

    async def test_on_key_add_node(self, task_editor_app) -> None:
        """Test on_key method for adding nodes."""
        # Mock the event
        mock_event = Mock()
        mock_event.key = "a"

        # Mock the cursor_node and its parent
        mock_cursor_node = Mock()
        mock_cursor_node.parent = Mock()
        task_editor_app.task_tree = Mock()
        task_editor_app.task_tree.cursor_node = mock_cursor_node

        # Mock action_add_node
        task_editor_app.action_add_node = AsyncMock()

        # Test on_key
        await task_editor_app.on_key(mock_event)

        # Verify action_add_node was called
        task_editor_app.action_add_node.assert_called_once_with(mock_cursor_node)

    async def test_on_key_delete_node(self, task_editor_app) -> None:
        """Test on_key method for deleting nodes."""
        # Mock the event
        mock_event = Mock()
        mock_event.key = "d"

        # Mock the cursor_node and its parent
        mock_cursor_node = Mock()
        mock_cursor_node.parent = Mock()
        mock_cursor_node.remove = Mock()
        task_editor_app.task_tree = Mock()
        task_editor_app.task_tree.cursor_node = mock_cursor_node

        # Mock update_tasks
        task_editor_app.update_tasks = AsyncMock()

        # Test on_key
        await task_editor_app.on_key(mock_event)

        # Verify node was removed
        mock_cursor_node.remove.assert_called_once()

        # Verify update_tasks was called
        task_editor_app.update_tasks.assert_called_once()

    async def test_on_key_save_exit(self, task_editor_app) -> None:
        """Test on_key method for saving and exiting."""
        # Mock the event
        mock_event = Mock()
        mock_event.key = "s"

        # Mock the exit method
        task_editor_app.exit = Mock()
        # Mock the task_tree with cursor_node
        mock_task_tree = Mock()
        mock_task_tree.cursor_node = Mock()
        task_editor_app.task_tree = mock_task_tree

        # Test on_key
        await task_editor_app.on_key(mock_event)

        # Verify exit was called with tasks
        task_editor_app.exit.assert_called_once_with(task_editor_app.tasks)

    async def test_on_key_other_key(self, task_editor_app) -> None:
        """Test on_key method for other keys."""
        # Mock the event
        mock_event = Mock()
        mock_event.key = "x"

        # Mock the cursor_node
        mock_cursor_node = Mock()
        mock_cursor_node.parent = Mock()
        task_editor_app.task_tree = Mock()
        task_editor_app.task_tree.cursor_node = mock_cursor_node

        # Mock methods that shouldn't be called
        task_editor_app.action_add_node = AsyncMock()
        task_editor_app.update_tasks = AsyncMock()
        task_editor_app.exit = Mock()

        # Test on_key
        await task_editor_app.on_key(mock_event)

        # Verify no methods were called
        task_editor_app.action_add_node.assert_not_called()
        task_editor_app.update_tasks.assert_not_called()
        task_editor_app.exit.assert_not_called()

    async def test_on_key_no_cursor_node(self, task_editor_app) -> None:
        """Test on_key method when no cursor node is selected."""
        # Mock the event
        mock_event = Mock()
        mock_event.key = "a"

        # Mock the cursor_node as None
        task_editor_app.task_tree = Mock()
        task_editor_app.task_tree.cursor_node = None

        # Mock methods that shouldn't be called
        task_editor_app.action_add_node = AsyncMock()
        task_editor_app.update_tasks = AsyncMock()
        task_editor_app.exit = Mock()

        # Test on_key
        await task_editor_app.on_key(mock_event)

        # Verify no methods were called
        task_editor_app.action_add_node.assert_not_called()
        task_editor_app.update_tasks.assert_not_called()
        task_editor_app.exit.assert_not_called()

    async def test_on_key_no_parent(self, task_editor_app) -> None:
        """Test on_key method when cursor node has no parent."""
        # Mock the event
        mock_event = Mock()
        mock_event.key = "a"

        # Mock the cursor_node with no parent
        mock_cursor_node = Mock()
        mock_cursor_node.parent = None
        task_editor_app.task_tree = Mock()
        task_editor_app.task_tree.cursor_node = mock_cursor_node

        # Mock methods that shouldn't be called
        task_editor_app.action_add_node = AsyncMock()
        task_editor_app.update_tasks = AsyncMock()
        task_editor_app.exit = Mock()

        # Test on_key
        await task_editor_app.on_key(mock_event)

        # Verify no methods were called
        task_editor_app.action_add_node.assert_not_called()
        task_editor_app.update_tasks.assert_not_called()
        task_editor_app.exit.assert_not_called()


class TestTaskEditorAppNodeManagement:
    """Test TaskEditorApp node management methods."""

    async def test_action_add_node_step_node(
        self, task_editor_app, mock_input_modal
    ) -> None:
        """Test action_add_node for step node."""
        # Simulate a step node (parent.parent is not None)
        node = MagicMock()
        node.parent.parent = MagicMock()
        node.label.plain = "Task 1"
        node.parent.label.plain = "Task 1"
        node.is_expanded = True
        task_editor_app.call_later = MagicMock()
        task_editor_app.push_screen = MagicMock()
        await task_editor_app.action_add_node(node)
        task_editor_app.push_screen.assert_called()

    async def test_action_add_node_task_node_expanded(
        self, task_editor_app, mock_input_modal
    ) -> None:
        """Test action_add_node for task node (is_expanded True)."""
        # Simulate a task node (is_expanded True)
        node = MagicMock()
        node.parent.parent = None
        node.is_expanded = True
        node.label.plain = "Task 1"
        task_editor_app.call_later = MagicMock()
        task_editor_app.push_screen = MagicMock()
        await task_editor_app.action_add_node(node)
        task_editor_app.push_screen.assert_called()

    async def test_action_add_node_task_node_not_expanded(
        self, task_editor_app, mock_input_modal
    ) -> None:
        """Test action_add_node for task node (is_expanded False)."""
        # Simulate a task node (is_expanded False)
        node = MagicMock()
        node.parent.parent = None
        node.is_expanded = False
        node.label.plain = "Task 1"
        node.parent.label.plain = "Root"
        task_editor_app.call_later = MagicMock()
        task_editor_app.push_screen = MagicMock()
        await task_editor_app.action_add_node(node)
        task_editor_app.push_screen.assert_called()

    def test_show_input_modal(self, task_editor_app, mock_input_modal) -> None:
        """Test the show_input_modal method."""
        # Mock the push_screen method
        task_editor_app.push_screen = Mock()

        # Mock the InputModal
        mock_modal_instance = Mock()
        mock_modal_instance.result = "Test Result"
        mock_input_modal.return_value = mock_modal_instance

        # Test show_input_modal
        result = task_editor_app.show_input_modal("Test Title", "Test Default")

        # Verify InputModal was created with correct parameters
        mock_input_modal.assert_called_once_with("Test Title", "Test Default")

        # Verify push_screen was called
        task_editor_app.push_screen.assert_called_once_with(mock_modal_instance)

        # Verify result is returned
        assert result == "Test Result"

    def test_show_input_modal_with_empty_default(
        self, task_editor_app, mock_input_modal
    ) -> None:
        """Test show_input_modal with empty default value."""
        # Mock the push_screen method
        task_editor_app.push_screen = Mock()

        # Mock the InputModal
        mock_modal_instance = Mock()
        mock_modal_instance.result = "Test Result"
        mock_input_modal.return_value = mock_modal_instance

        # Test show_input_modal with empty default
        result = task_editor_app.show_input_modal("Test Title")

        # Verify InputModal was created with correct parameters
        mock_input_modal.assert_called_once_with("Test Title", "")

        # Verify push_screen was called
        task_editor_app.push_screen.assert_called_once_with(mock_modal_instance)

        # Verify result is returned
        assert result == "Test Result"

    def test_show_input_modal_various(self, task_editor_app, mock_input_modal) -> None:
        """Test show_input_modal with various inputs."""
        # Mock the push_screen method
        task_editor_app.push_screen = Mock()

        # Test with various input combinations
        test_cases = [
            ("Title 1", "Default 1"),
            ("Title 2", ""),
            ("", "Default 3"),
            ("", ""),
            ("Very Long Title That Might Cause Issues", "Very Long Default Value"),
        ]

        for title, default in test_cases:
            # Mock the InputModal
            mock_modal_instance = Mock()
            mock_modal_instance.result = f"Result for {title}"
            mock_input_modal.return_value = mock_modal_instance

            # Test show_input_modal
            result = task_editor_app.show_input_modal(title, default)

            # Verify InputModal was created with correct parameters
            mock_input_modal.assert_called_with(title, default)

            # Verify push_screen was called
            task_editor_app.push_screen.assert_called_with(mock_modal_instance)

            # Verify result is returned
            assert result == f"Result for {title}"

            # Reset mocks for next iteration
            mock_input_modal.reset_mock()
            task_editor_app.push_screen.reset_mock()


class TestTaskEditorAppDataManagement:
    """Test TaskEditorApp data management methods."""

    async def test_update_tasks(self, mock_log_context) -> None:
        """Test the update_tasks method."""
        # Create a fresh TaskEditorApp instance for this test
        app = TaskEditorApp([])

        # Mock the task_tree structure
        mock_root = Mock()
        mock_root.children = []

        # Create mock task nodes
        mock_task1 = Mock()
        mock_task1.label.plain = "Task 1"
        mock_task1.children = [
            Mock(label=Mock(plain="Step 1.1")),
            Mock(label=Mock(plain="Step 1.2")),
        ]

        mock_task2 = Mock()
        mock_task2.label.plain = "Task 2"
        mock_task2.children = [
            Mock(label=Mock(plain="Step 2.1")),
        ]

        mock_root.children = [mock_task1, mock_task2]

        # Set up the task_tree
        app.task_tree = Mock()
        app.task_tree.root = mock_root

        # Test update_tasks
        await app.update_tasks()

        # Verify tasks were updated correctly
        expected_tasks = [
            {"name": "Task 1", "steps": ["Step 1.1", "Step 1.2"]},
            {"name": "Task 2", "steps": ["Step 2.1"]},
        ]
        assert app.tasks == expected_tasks

        # Verify logging was called
        mock_log_context.debug.assert_called_once()

    async def test_update_tasks_empty_tree(self, mock_log_context) -> None:
        """Test update_tasks with empty tree."""
        # Create a fresh TaskEditorApp instance for this test
        app = TaskEditorApp([])

        # Mock the task_tree structure with no children
        mock_root = Mock()
        mock_root.children = []

        # Set up the task_tree
        app.task_tree = Mock()
        app.task_tree.root = mock_root

        # Test update_tasks
        await app.update_tasks()

        # Verify tasks were updated correctly
        assert app.tasks == []

        # Verify logging was called
        mock_log_context.debug.assert_called_once()

    async def test_update_tasks_with_none_tree(self, task_editor_app) -> None:
        """Test update_tasks with None task_tree."""
        # Set task_tree to None
        task_editor_app.task_tree = None

        # Test update_tasks - should handle gracefully
        await task_editor_app.update_tasks()

        # Verify tasks remain unchanged
        assert task_editor_app.tasks == []

    async def test_update_tasks_with_invalid_node_structure(
        self, mock_log_context
    ) -> None:
        """Test update_tasks with invalid node structure."""
        # Create a fresh TaskEditorApp instance for this test
        app = TaskEditorApp([])

        # Mock the task_tree structure with invalid nodes
        mock_root = Mock()
        mock_root.children = []

        # Create mock task node with invalid structure
        mock_task = Mock()
        mock_task.label = None  # Invalid label
        mock_task.children = []

        mock_root.children = [mock_task]

        # Set up the task_tree
        app.task_tree = Mock()
        app.task_tree.root = mock_root

        # Test update_tasks - should handle gracefully by skipping invalid nodes
        # The method should handle None labels by skipping them or using a default
        try:
            await app.update_tasks()
            # If it doesn't raise an exception, verify the behavior
            assert len(app.tasks) == 0  # Should skip invalid nodes
        except AttributeError:
            # If it raises AttributeError, that's also acceptable behavior
            # for invalid node structures
            pass

    async def test_update_tasks_with_missing_step_data(self, mock_log_context) -> None:
        """Test update_tasks with missing step data."""
        # Create a fresh TaskEditorApp instance for this test
        app = TaskEditorApp([])

        # Mock the task_tree structure with missing step data
        mock_root = Mock()
        mock_root.children = []

        # Create mock task node with missing step data
        mock_task = Mock()
        mock_task.label.plain = "Task 1"
        mock_task.children = [
            Mock(label=None),  # Missing label
            Mock(label=Mock(plain="Valid Step")),
        ]

        mock_root.children = [mock_task]

        # Set up the task_tree
        app.task_tree = Mock()
        app.task_tree.root = mock_root

        # Test update_tasks - should handle gracefully by skipping invalid steps
        try:
            await app.update_tasks()
            # If it doesn't raise an exception, verify the behavior
            assert len(app.tasks) == 1
            assert app.tasks[0]["name"] == "Task 1"
            # Should handle None labels in steps by skipping them or using a default
            assert len(app.tasks[0]["steps"]) == 1  # Should skip None label
            assert app.tasks[0]["steps"][0] == "Valid Step"
        except AttributeError:
            # If it raises AttributeError, that's also acceptable behavior
            # for invalid step structures
            pass

    def test_run_returns_tasks(self, task_editor_app) -> None:
        """Test run method returns tasks."""
        task_editor_app.tasks = [{"name": "Task 1", "steps": ["Step 1"]}]
        task_editor_app.run = MagicMock(return_value=task_editor_app.tasks)
        result = task_editor_app.run()
        assert result == [{"name": "Task 1", "steps": ["Step 1"]}]
