"""Tests for the TaskEditorApp UI component.

This module tests the TaskEditorApp class which provides an interactive
interface for editing task graphs using Textual UI framework.
"""

import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from arklex.orchestrator.generator.ui.task_editor import TaskEditorApp


# Create a minimal fake textual.app module
class FakeApp:
    def __init__(self, *args: object, **kwargs: object) -> None:
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

# --- Mock fixtures for UI components ---


@pytest.fixture
def mock_tree(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock the Tree widget."""
    mock_tree = Mock()
    mock_tree_instance = Mock()
    mock_tree_instance.root = Mock()
    mock_tree_instance.root.add.return_value = Mock()
    mock_tree_instance.root.expand = Mock()
    mock_tree.return_value = mock_tree_instance
    return mock_tree


@pytest.fixture
def mock_label(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock the Label widget."""
    mock_label = Mock()
    mock_label_instance = Mock()
    mock_label.return_value = mock_label_instance
    return mock_label


@pytest.fixture
def mock_input_modal(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock the InputModal class."""
    mock_modal = Mock()
    mock_modal_instance = Mock()
    mock_modal.return_value = mock_modal_instance
    return mock_modal


@pytest.fixture
def mock_log_context(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock the log_context."""
    mock_log = Mock()
    return mock_log


@pytest.fixture
def sample_tasks() -> list[dict[str, Any]]:
    """Sample tasks for testing."""
    return [
        {
            "name": "Task 1",
            "steps": [{"description": "Step 1.1"}, {"description": "Step 1.2"}],
        },
        {"name": "Task 2", "steps": [{"description": "Step 2.1"}]},
    ]


@pytest.fixture
def complex_tasks() -> list[dict[str, Any]]:
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
def task_editor_app(sample_tasks: list[dict[str, Any]]) -> TaskEditorApp:
    """Create a TaskEditorApp instance for testing."""
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

    def test_task_editor_init_none_tasks(self) -> None:
        from arklex.orchestrator.generator.ui.task_editor import TaskEditorApp

        editor = TaskEditorApp(None)
        assert editor.tasks is None
        assert editor.task_tree is None


class TestTaskEditorAppCompose:
    """Test TaskEditorApp compose method."""

    def test_compose_method(
        self, sample_tasks: list[dict[str, Any]], mock_tree: Mock, mock_label: Mock
    ) -> None:
        """Test the compose method with valid tasks."""
        app = TaskEditorApp(sample_tasks)

        with (
            patch("arklex.orchestrator.generator.ui.task_editor.Tree", mock_tree),
            patch("arklex.orchestrator.generator.ui.task_editor.Label", mock_label),
        ):
            # Mock the task node
            mock_task_node = Mock()
            mock_task_node.add_leaf = Mock()
            mock_tree.return_value.root.add.return_value = mock_task_node

            # Test compose
            list(app.compose())

            # Verify tree was created and configured
            mock_tree.assert_called_once_with("Tasks")
            mock_tree.return_value.root.expand.assert_called_once()

            # Verify tasks were added to tree
            assert mock_tree.return_value.root.add.call_count == 2  # Two tasks

    def test_compose_with_empty_tasks(self, mock_tree: Mock, mock_label: Mock) -> None:
        """Test compose method with empty tasks list."""
        app = TaskEditorApp([])

        with (
            patch("arklex.orchestrator.generator.ui.task_editor.Tree", mock_tree),
            patch("arklex.orchestrator.generator.ui.task_editor.Label", mock_label),
        ):
            list(app.compose())

            # Verify tree was created
            mock_tree.assert_called_once_with("Tasks")
            mock_tree.return_value.root.expand.assert_called_once()

    def test_compose_with_none_tasks(self, mock_tree: Mock, mock_label: Mock) -> None:
        """Test compose method with None tasks."""
        app = TaskEditorApp(None)

        with (
            patch("arklex.orchestrator.generator.ui.task_editor.Tree", mock_tree),
            patch("arklex.orchestrator.generator.ui.task_editor.Label", mock_label),
        ):
            # Should handle None gracefully
            list(app.compose())

            # Verify tree was created
            mock_tree.assert_called_once_with("Tasks")
            mock_tree.return_value.root.expand.assert_called_once()

    def test_compose_with_complex_tasks(
        self, complex_tasks: list[dict[str, Any]], mock_tree: Mock, mock_label: Mock
    ) -> None:
        """Test compose method with complex task structures."""
        app = TaskEditorApp(complex_tasks)

        with (
            patch("arklex.orchestrator.generator.ui.task_editor.Tree", mock_tree),
            patch("arklex.orchestrator.generator.ui.task_editor.Label", mock_label),
        ):
            # Mock the task node
            mock_task_node = Mock()
            mock_task_node.add_leaf = Mock()
            mock_tree.return_value.root.add.return_value = mock_task_node

            list(app.compose())

            # Verify tree was created
            mock_tree.assert_called_once_with("Tasks")
            mock_tree.return_value.root.expand.assert_called_once()


class TestTaskEditorAppEventHandling:
    """Test TaskEditorApp event handling methods."""

    def test_on_mount(self, task_editor_app: TaskEditorApp) -> None:
        """Test the on_mount method."""
        # Mock the task_tree
        mock_tree = Mock()
        task_editor_app.task_tree = mock_tree

        # Test on_mount
        task_editor_app.on_mount()

        # Verify focus was set
        mock_tree.focus.assert_called_once()

    async def test_on_tree_node_selected(
        self, task_editor_app: TaskEditorApp, mock_input_modal: Mock
    ) -> None:
        """Test on_tree_node_selected event handler."""
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.InputModal", mock_input_modal
        ):
            # Mock the selected node
            mock_node = Mock()
            mock_node.label = "Test Node"
            mock_node.set_label = Mock()

            # Mock the event
            mock_event = Mock()
            mock_event.node = mock_node

            # Mock push_screen
            task_editor_app.push_screen = Mock()

            # Test the event handler
            await task_editor_app.on_tree_node_selected(mock_event)

            # Verify InputModal was called
            mock_input_modal.assert_called_once()

    async def test_on_tree_node_selected_with_none_label(
        self, task_editor_app: TaskEditorApp, mock_input_modal: Mock
    ) -> None:
        """Test on_tree_node_selected with None label."""
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.InputModal", mock_input_modal
        ):
            # Mock the selected node with None label
            mock_node = Mock()
            mock_node.label = None
            mock_node.set_label = Mock()

            # Mock the event
            mock_event = Mock()
            mock_event.node = mock_node

            # Mock push_screen
            task_editor_app.push_screen = Mock()

            # Test the event handler
            await task_editor_app.on_tree_node_selected(mock_event)

            # Verify InputModal was called
            mock_input_modal.assert_called_once()


class TestTaskEditorAppKeyboardHandling:
    """Test TaskEditorApp keyboard input handling."""

    async def test_on_key_add_node(self, task_editor_app: TaskEditorApp) -> None:
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

    async def test_on_key_delete_node(self, task_editor_app: TaskEditorApp) -> None:
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

    async def test_on_key_save_exit(self, task_editor_app: TaskEditorApp) -> None:
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

    async def test_on_key_other_key(self, task_editor_app: TaskEditorApp) -> None:
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

    async def test_on_key_no_cursor_node(self, task_editor_app: TaskEditorApp) -> None:
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

    async def test_on_key_no_parent(self, task_editor_app: TaskEditorApp) -> None:
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
        self, task_editor_app: TaskEditorApp, mock_input_modal: Mock
    ) -> None:
        """Test action_add_node for step nodes."""
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.InputModal", mock_input_modal
        ):
            # Mock a step node (has parent.parent)
            mock_node = Mock()
            mock_node.parent = Mock()
            mock_node.parent.parent = Mock()  # This makes it a step node
            mock_node.parent.label.plain = "Parent Task"

            # Mock push_screen
            task_editor_app.push_screen = Mock()

            # Test action_add_node
            await task_editor_app.action_add_node(mock_node)

            # Verify InputModal was called
            mock_input_modal.assert_called_once()

    async def test_action_add_node_task_node_expanded(
        self, task_editor_app: TaskEditorApp, mock_input_modal: Mock
    ) -> None:
        """Test action_add_node for expanded task nodes."""
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.InputModal", mock_input_modal
        ):
            # Mock an expanded task node
            mock_node = Mock()
            mock_node.parent = Mock()
            mock_node.parent.parent = None  # This makes it a task node
            mock_node.is_expanded = True
            mock_node.label.plain = "Test Task"

            # Mock push_screen
            task_editor_app.push_screen = Mock()

            # Test action_add_node
            await task_editor_app.action_add_node(mock_node)

            # Verify InputModal was called
            mock_input_modal.assert_called_once()

    async def test_action_add_node_task_node_not_expanded(
        self, task_editor_app: TaskEditorApp, mock_input_modal: Mock
    ) -> None:
        """Test action_add_node for non-expanded task nodes."""
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.InputModal", mock_input_modal
        ):
            # Mock a non-expanded task node
            mock_node = Mock()
            mock_node.parent = Mock()
            mock_node.parent.parent = None  # This makes it a task node
            mock_node.is_expanded = False
            mock_node.parent.label.plain = "Parent Task"

            # Mock push_screen
            task_editor_app.push_screen = Mock()

            # Test action_add_node
            await task_editor_app.action_add_node(mock_node)

            # Verify InputModal was called
            mock_input_modal.assert_called_once()

    def test_show_input_modal(
        self, task_editor_app: TaskEditorApp, mock_input_modal: Mock
    ) -> None:
        """Test show_input_modal method."""
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.InputModal", mock_input_modal
        ):
            # Mock push_screen
            task_editor_app.push_screen = Mock()

            # Test show_input_modal
            task_editor_app.show_input_modal("Test Title", "Default Value")

            # Verify InputModal was called
            mock_input_modal.assert_called_once_with("Test Title", "Default Value")

    def test_show_input_modal_with_empty_default(
        self, task_editor_app: TaskEditorApp, mock_input_modal: Mock
    ) -> None:
        """Test show_input_modal with empty default value."""
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.InputModal", mock_input_modal
        ):
            # Mock push_screen
            task_editor_app.push_screen = Mock()

            # Test show_input_modal with empty default
            task_editor_app.show_input_modal("Test Title")

            # Verify InputModal was called with empty default
            mock_input_modal.assert_called_once_with("Test Title", "")

    def test_show_input_modal_various(
        self, task_editor_app: TaskEditorApp, mock_input_modal: Mock
    ) -> None:
        """Test show_input_modal with various parameters."""
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.InputModal", mock_input_modal
        ):
            # Mock push_screen
            task_editor_app.push_screen = Mock()

            # Test various parameter combinations
            test_cases = [
                ("Title 1", "Default 1"),
                ("Title 2", ""),
                ("Title 3", "Complex Default Value"),
            ]

            for title, default in test_cases:
                task_editor_app.show_input_modal(title, default)
                mock_input_modal.assert_called_with(title, default)


class TestTaskEditorAppDataManagement:
    """Test TaskEditorApp data management methods."""

    async def test_update_tasks(self, mock_log_context: Mock) -> None:
        """Test update_tasks method with valid tree structure."""
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.log_context", mock_log_context
        ):
            # Create a TaskEditorApp instance
            tasks = [
                {
                    "name": "Task 1",
                    "steps": [
                        "Step 1.1",
                        "Step 1.2",
                    ],  # Steps should be strings, not dicts
                }
            ]
            app = TaskEditorApp(tasks)

            # Mock the tree structure
            mock_tree = Mock()
            mock_root = Mock()
            mock_task_node = Mock()
            mock_step_node1 = Mock()
            mock_step_node2 = Mock()

            # Set up the tree structure
            mock_tree.root = mock_root
            mock_root.children = [mock_task_node]
            mock_task_node.label.plain = "Task 1"
            mock_task_node.children = [mock_step_node1, mock_step_node2]
            mock_step_node1.label.plain = "Step 1.1"
            mock_step_node2.label.plain = "Step 1.2"

            app.task_tree = mock_tree

            # Test update_tasks
            await app.update_tasks()

            # Verify tasks were updated correctly
            assert len(app.tasks) == 1
            assert app.tasks[0]["name"] == "Task 1"
            assert len(app.tasks[0]["steps"]) == 2
            assert app.tasks[0]["steps"][0] == "Step 1.1"  # Steps are strings
            assert app.tasks[0]["steps"][1] == "Step 1.2"  # Steps are strings

    async def test_update_tasks_empty_tree(self, mock_log_context: Mock) -> None:
        """Test update_tasks with empty tree."""
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.log_context", mock_log_context
        ):
            # Create a TaskEditorApp instance
            app = TaskEditorApp([])

            # Mock empty tree
            mock_tree = Mock()
            mock_root = Mock()
            mock_tree.root = mock_root
            mock_root.children = []

            app.task_tree = mock_tree

            # Test update_tasks
            await app.update_tasks()

            # Verify tasks list is empty
            assert app.tasks == []

    async def test_update_tasks_with_none_tree(
        self, task_editor_app: TaskEditorApp
    ) -> None:
        """Test update_tasks with None task_tree."""
        # Set task_tree to None
        task_editor_app.task_tree = None

        # Test update_tasks - should handle gracefully
        await task_editor_app.update_tasks()

        # Verify tasks remain unchanged
        assert task_editor_app.tasks == []

    async def test_update_tasks_with_invalid_node_structure(
        self, mock_log_context: Mock
    ) -> None:
        """Test update_tasks with invalid node structure."""
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.log_context", mock_log_context
        ):
            # Create a TaskEditorApp instance
            app = TaskEditorApp([])

            # Mock tree with invalid structure (node without label)
            mock_tree = Mock()
            mock_root = Mock()
            mock_task_node = Mock()
            mock_task_node.label = None  # Invalid: no label

            mock_tree.root = mock_root
            mock_root.children = [mock_task_node]

            app.task_tree = mock_tree

            # Test update_tasks - should raise AttributeError for None label
            with pytest.raises(
                AttributeError, match="'NoneType' object has no attribute 'plain'"
            ):
                await app.update_tasks()

    async def test_update_tasks_with_missing_step_data(
        self, mock_log_context: Mock
    ) -> None:
        """Test update_tasks with missing step data."""
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.log_context", mock_log_context
        ):
            # Create a TaskEditorApp instance
            app = TaskEditorApp([])

            # Mock tree with step node missing label
            mock_tree = Mock()
            mock_root = Mock()
            mock_task_node = Mock()
            mock_step_node = Mock()
            mock_step_node.label = None  # Missing label

            mock_tree.root = mock_root
            mock_root.children = [mock_task_node]
            mock_task_node.label.plain = "Task 1"
            mock_task_node.children = [mock_step_node]

            app.task_tree = mock_tree

            # Test update_tasks - should raise AttributeError for None step label
            with pytest.raises(
                AttributeError, match="'NoneType' object has no attribute 'plain'"
            ):
                await app.update_tasks()

    def test_run_returns_tasks(self, task_editor_app: TaskEditorApp) -> None:
        """Test run method returns tasks."""
        task_editor_app.tasks = [{"name": "Task 1", "steps": ["Step 1"]}]
        task_editor_app.run = MagicMock(return_value=task_editor_app.tasks)
        result = task_editor_app.run()
        assert result == [{"name": "Task 1", "steps": ["Step 1"]}]
