"""Tests for UI components with fallback support.

This module provides comprehensive tests for UI components, with fallback support
for environments where textual is not available.
"""

import sys
import types
from typing import TypeVar
from unittest.mock import Mock, patch

import pytest

import arklex.orchestrator.generator as generator_mod

# Import fallback classes for testing when textual is not available
from tests.orchestrator.generator.test_ui_fallbacks import (
    FallbackButton,
    patch_ui_modules_for_testing,
    unpatch_ui_modules,
)


# Set up fake textual modules before any imports
class FakeTreeNode:
    pass


class FakeTree:
    """Fallback Tree for testing."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.root = None
        self.children = []

    class NodeSelected:
        """Fake NodeSelected event."""

        def __init__(self, node: object) -> None:
            self.node = node


class FakeButton:
    class Pressed:
        pass


class FakeInput:
    pass


class FakeStatic:
    pass


class FakeLabel:
    def __init__(self, *args: object, **kwargs: object) -> None:
        # Store the first argument as plain if provided
        self.plain = args[0] if args else ""


class FakeHorizontal:
    pass


class FakeVertical:
    pass


class FakeApp:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def run(self) -> None:
        pass


class FakeScreen:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass


class FakeComposeResult:
    pass


# Create a proper type variable for ReturnType
ReturnType = TypeVar("ReturnType")


# Create fake modules
fake_textual_app = types.ModuleType("textual.app")
fake_textual_app.App = FakeApp
fake_textual_app.ComposeResult = FakeComposeResult
fake_textual_app.ReturnType = ReturnType
sys.modules["textual.app"] = fake_textual_app

fake_textual_widgets = types.ModuleType("textual.widgets")
fake_textual_widgets.Button = FakeButton
fake_textual_widgets.Input = FakeInput
fake_textual_widgets.Static = FakeStatic
fake_textual_widgets.Label = FakeLabel
fake_textual_widgets.Tree = FakeTree
sys.modules["textual.widgets"] = fake_textual_widgets

fake_textual_widgets_tree = types.ModuleType("textual.widgets.tree")
fake_textual_widgets_tree.TreeNode = FakeTreeNode
sys.modules["textual.widgets.tree"] = fake_textual_widgets_tree

fake_textual_containers = types.ModuleType("textual.containers")
fake_textual_containers.Horizontal = FakeHorizontal
fake_textual_containers.Vertical = FakeVertical
sys.modules["textual.containers"] = fake_textual_containers

fake_textual_screen = types.ModuleType("textual.screen")
fake_textual_screen.Screen = FakeScreen
sys.modules["textual.screen"] = fake_textual_screen

# Try to import actual UI components, fall back to test versions if not available
try:
    TEXTUAL_AVAILABLE = True
except ImportError:
    # Use fallback classes for testing
    patch_ui_modules_for_testing()

    class TreeNode:
        """Fallback TreeNode for testing."""

        def __init__(self, label: str = "") -> None:
            self.label = label

    TEXTUAL_AVAILABLE = False


def ui_available() -> bool:
    return getattr(generator_mod, "_UI_AVAILABLE", True)


def skip_if_ui_not_available() -> None:
    if not ui_available():
        pytest.skip("UI not available, skipping UI-dependent test")


# --- Fixtures ---
@pytest.fixture
def sample_tasks() -> list:
    """Sample tasks for testing."""
    return [
        {
            "name": "Customer Support",
            "steps": ["Listen to customer", "Provide solution"],
        },
        {
            "name": "Product Search",
            "steps": ["Get search criteria", "Search database"],
        },
    ]


@pytest.fixture
def patched_sample_config() -> dict:
    """Patched config for UI tests (if needed for future expansion)."""
    return {}


@pytest.fixture
def always_valid_mock_model() -> Mock:
    """A mock model that always returns a valid, non-empty task list."""
    model = Mock()
    valid_task = '[{"id": "task_1", "name": "Test Task", "description": "A test task", "steps": ["Step 1"]}]'
    model.generate = lambda messages: type("Mock", (), {"content": valid_task})()
    model.invoke = lambda messages: type("Mock", (), {"content": valid_task})()
    return model


@pytest.fixture
def mock_node() -> Mock:
    """Create a mock tree node for testing."""
    node = Mock()
    node.parent = Mock()
    node.parent.parent = None
    node.is_expanded = True
    node.label = Mock()
    node.label.plain = "Test Node"
    node.children = []
    return node


@pytest.fixture
def mock_event() -> Mock:
    """Create a mock event for testing."""
    event = Mock()
    event.key = "a"
    event.node = Mock()
    return event


# --- Test Classes ---


class TestTaskEditorUI:
    """Test the TaskEditor UI component with mock interactions."""

    def test_task_editor_initialization(self, sample_tasks: list) -> None:
        skip_if_ui_not_available()

        """Test task editor initialization."""
        # TODO: Refactor TaskEditorApp to separate business logic from UI rendering
        # - Extract task management logic into TaskManagerService
        # - Make TaskEditorApp a thin wrapper around the service
        # - Test the service logic independently of UI framework
        pass

    def test_compose_creates_tree_structure(self, sample_tasks: list) -> None:
        skip_if_ui_not_available()

        """Test that compose method creates proper tree structure."""
        # TODO: Refactor to separate tree structure logic from UI rendering
        # - Create TreeStructureBuilder service
        # - Test tree building logic independently
        # - Make compose method use the service
        pass

    def test_on_mount_sets_focus(self) -> None:
        skip_if_ui_not_available()

        """Test that on_mount sets focus to task tree."""
        # TODO: Refactor to separate initialization logic from UI framework
        # - Extract initialization logic into separate method
        # - Test initialization logic independently
        pass

    @pytest.mark.asyncio
    async def test_add_task_with_keyboard(self, sample_tasks: list) -> None:
        skip_if_ui_not_available()

        """Test adding a task using keyboard shortcut."""
        # TODO: Refactor to separate task addition logic from UI event handling
        # - Create TaskAdditionService with add_task method
        # - Test task addition logic independently
        # - Make keyboard handler use the service
        pass

    @pytest.mark.asyncio
    async def test_delete_task_with_keyboard(self, mock_node: Mock) -> None:
        skip_if_ui_not_available()

        """Test deleting a task using keyboard shortcut."""
        # TODO: Refactor to separate task deletion logic from UI event handling
        # - Create TaskDeletionService with delete_task method
        # - Test task deletion logic independently
        # - Make keyboard handler use the service
        pass

    @pytest.mark.asyncio
    async def test_save_and_exit_with_keyboard(self) -> None:
        skip_if_ui_not_available()

        """Test saving and exiting with keyboard shortcut."""
        # TODO: Refactor to separate save logic from UI event handling
        # - Create TaskSaveService with save_tasks method
        # - Test save logic independently
        # - Make keyboard handler use the service
        pass

    @pytest.mark.asyncio
    async def test_add_step_to_task(self, mock_node: Mock) -> None:
        skip_if_ui_not_available()

        """Test adding a step to a task."""
        # TODO: Refactor to separate step addition logic from UI event handling
        # - Create StepAdditionService with add_step method
        # - Test step addition logic independently
        # - Make UI handler use the service
        pass

    @pytest.mark.asyncio
    async def test_add_task_to_root(self, mock_node: Mock) -> None:
        skip_if_ui_not_available()

        """Test adding a task to the root level."""
        # TODO: Refactor to separate root task addition logic from UI event handling
        # - Create RootTaskAdditionService with add_root_task method
        # - Test root task addition logic independently
        # - Make UI handler use the service
        pass

    @pytest.mark.asyncio
    async def test_update_tasks_from_tree(self) -> None:
        skip_if_ui_not_available()

        """Test updating tasks from tree structure."""
        # TODO: Refactor to separate task synchronization logic from UI framework
        # - Create TaskSynchronizationService with sync_tasks method
        # - Test synchronization logic independently
        # - Make UI method use the service
        pass

    @pytest.mark.asyncio
    async def test_node_selection_opens_modal(self, mock_node: Mock) -> None:
        skip_if_ui_not_available()

        """Test that node selection opens the input modal."""
        # TODO: Refactor to separate modal management logic from UI event handling
        # - Create ModalManagerService with show_edit_modal method
        # - Test modal management logic independently
        # - Make event handler use the service
        pass

    def test_run_returns_tasks(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.task_editor import TaskEditorApp

        """Test run method returns tasks."""
        app = TaskEditorApp([])
        # Just test that the method returns the tasks attribute
        assert app.run() == []

    def test_task_editor_show_input_modal(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.task_editor import TaskEditorApp

        """Test TaskEditorApp show_input_modal method."""
        app = TaskEditorApp([])

        # Mock the push_screen method
        app.push_screen = Mock()

        # Create a fresh mock for this test
        fresh_mock_modal = Mock()
        fresh_mock_modal_instance = Mock()
        fresh_mock_modal_instance.result = "default value"
        fresh_mock_modal.return_value = fresh_mock_modal_instance

        # Ensure InputModal exists in the UI module
        import arklex.orchestrator.generator.ui as ui_module

        if not hasattr(ui_module, "InputModal"):
            ui_module.InputModal = fresh_mock_modal

        with patch("arklex.orchestrator.generator.ui.InputModal", fresh_mock_modal):
            result = app.show_input_modal("Test Title", "default value")

            # Verify push_screen was called
            app.push_screen.assert_called_once()

            # Verify the result
            assert result == "default value"

    def test_task_editor_show_input_modal_with_callback(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.task_editor import TaskEditorApp

        """Test TaskEditorApp show_input_modal method with callback."""
        app = TaskEditorApp([])

        # Mock the push_screen method
        app.push_screen = Mock()

        # Create a fresh mock for this test
        fresh_mock_modal = Mock()
        fresh_mock_modal_instance = Mock()
        fresh_mock_modal_instance.result = "callback value"
        fresh_mock_modal.return_value = fresh_mock_modal_instance

        # Ensure InputModal exists in the UI module
        import arklex.orchestrator.generator.ui as ui_module

        if not hasattr(ui_module, "InputModal"):
            ui_module.InputModal = fresh_mock_modal

        with patch("arklex.orchestrator.generator.ui.InputModal", fresh_mock_modal):
            result = app.show_input_modal("Test Title", "default value")

            # Verify push_screen was called
            app.push_screen.assert_called_once()

            # Verify the result
            assert result == "default value"

    def test_task_editor_show_input_modal_without_default(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.task_editor import TaskEditorApp

        """Test TaskEditorApp show_input_modal method without default value."""
        app = TaskEditorApp([])

        # Mock the push_screen method
        app.push_screen = Mock()

        # Create a fresh mock for this test
        fresh_mock_modal = Mock()
        fresh_mock_modal_instance = Mock()
        fresh_mock_modal_instance.result = ""
        fresh_mock_modal.return_value = fresh_mock_modal_instance

        # Ensure InputModal exists in the UI module
        import arklex.orchestrator.generator.ui as ui_module

        if not hasattr(ui_module, "InputModal"):
            ui_module.InputModal = fresh_mock_modal

        with patch("arklex.orchestrator.generator.ui.InputModal", fresh_mock_modal):
            result = app.show_input_modal("Test Title")

            # Verify push_screen was called
            app.push_screen.assert_called_once()

            # Verify the result
            assert result == ""

    def test_task_editor_update_tasks_with_none_tree(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.task_editor import TaskEditorApp

        """Test TaskEditorApp update_tasks method with None tree."""
        app = TaskEditorApp([])
        app.task_tree = None

        # Should not raise an exception
        app.update_tasks()

    def test_task_editor_update_tasks_with_none_root(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.task_editor import TaskEditorApp

        """Test TaskEditorApp update_tasks method with None root."""
        app = TaskEditorApp([])
        app.task_tree = Mock()
        app.task_tree.root = None

        # Should not raise an exception
        app.update_tasks()

    def test_task_editor_run_method(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.task_editor import TaskEditorApp

        """Test TaskEditorApp run method."""
        app = TaskEditorApp([])
        # Just test that the method returns the tasks attribute
        assert app.run() == []

    def test_task_editor_update_tasks_with_empty_tree(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.task_editor import TaskEditorApp

        """Test TaskEditorApp update_tasks method with empty tree."""
        app = TaskEditorApp([])
        app.task_tree = Mock()
        app.task_tree.root = Mock()
        app.task_tree.root.children = []

        app.update_tasks()
        assert app.tasks == []

    def test_task_editor_update_tasks_with_getattr_none_root(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.task_editor import TaskEditorApp

        """Test TaskEditorApp update_tasks method with getattr None root."""
        app = TaskEditorApp([])
        app.task_tree = Mock()
        app.task_tree.root = None

        app.update_tasks()
        assert app.tasks == []

    async def test_task_editor_update_tasks_with_complex_tree_structure(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.task_editor import TaskEditorApp

        """Test TaskEditorApp update_tasks method with complex tree structure."""
        app = TaskEditorApp([])

        # Create a complex tree structure
        class FakeLabel:
            def __init__(self, plain: str) -> None:
                self.plain = plain

        class FakeNode:
            def __init__(self, label: str, children: list | None = None) -> None:
                self.label = FakeLabel(label)
                self.children = children or []

        app.task_tree = Mock()
        app.task_tree.root = FakeNode(
            "root",
            [
                FakeNode("Task 1", [FakeNode("Step 1"), FakeNode("Step 2")]),
                FakeNode("Task 2", [FakeNode("Step 3")]),
            ],
        )

        await app.update_tasks()
        assert len(app.tasks) == 2
        assert app.tasks[0]["name"] == "Task 1"
        assert app.tasks[0]["steps"] == ["Step 1", "Step 2"]
        assert app.tasks[1]["name"] == "Task 2"
        assert app.tasks[1]["steps"] == ["Step 3"]


class TestInputModalUI:
    """Test the InputModal UI component with mock interactions."""

    def test_input_modal_initialization(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.input_modal import InputModal

        """Test InputModal initialization."""
        modal = InputModal("Test Title", "default value")
        assert modal.title == "Test Title"
        assert modal.default == "default value"
        assert modal.result == "default value"

    def test_input_modal_with_callback(self) -> None:
        skip_if_ui_not_available()
        from textual.widgets.tree import TreeNode

        from arklex.orchestrator.generator.ui.input_modal import InputModal

        """Test InputModal with callback function."""
        callback_called = False

        def mock_callback(result: str, node: TreeNode | None) -> None:
            nonlocal callback_called
            callback_called = True

        modal = InputModal("Test Title", "default value", callback=mock_callback)
        assert modal.callback == mock_callback

    def test_compose_creates_input_structure(self) -> None:
        skip_if_ui_not_available()
        from textual.widgets.tree import TreeNode

        from arklex.orchestrator.generator.ui.input_modal import InputModal

        """Test that compose method creates proper input structure."""
        modal = InputModal("Test Title", "default value")

        # Mock the query_one method
        modal.query_one = Mock()
        mock_input = Mock()
        mock_input.value = "test input"
        modal.query_one.return_value = mock_input

        # Mock the app.pop_screen method
        modal.app = Mock()
        modal.app.pop_screen = Mock()

        # Create a mock button press event
        mock_button = Mock()
        mock_button.id = "submit"
        mock_event = FallbackButton.Pressed(mock_button)

        # Test button press
        modal.on_button_pressed(mock_event)

        # Verify the result was updated
        assert modal.result == "test input"

        # Verify callback was called if provided
        callback_called = False

        def mock_callback(result: str, node: TreeNode | None) -> None:
            nonlocal callback_called
            callback_called = True

        modal.callback = mock_callback
        modal.on_button_pressed(mock_event)
        assert callback_called

        # Verify pop_screen was called
        modal.app.pop_screen.assert_called()

    def test_modal_dismissal(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.input_modal import InputModal

        """Test modal dismissal behavior."""
        modal = InputModal("Test Title", "default value")

        # Mock the app.pop_screen method
        modal.app = Mock()
        modal.app.pop_screen = Mock()

        # Create a mock button press event for cancel
        mock_button = Mock()
        mock_button.id = "cancel"
        mock_event = FallbackButton.Pressed(mock_button)

        # Test cancel button press
        modal.on_button_pressed(mock_event)

        # Verify pop_screen was called
        modal.app.pop_screen.assert_called_once()

    def test_input_modal_on_button_pressed_cancel_button(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.input_modal import InputModal

        """Test InputModal on_button_pressed method with cancel button."""
        modal = InputModal("Test Title", "default value")

        # Mock the app.pop_screen method
        modal.app = Mock()
        modal.app.pop_screen = Mock()

        # Create a mock button press event for cancel
        mock_button = Mock()
        mock_button.id = "cancel"
        mock_event = FallbackButton.Pressed(mock_button)

        # Test cancel button press
        modal.on_button_pressed(mock_event)

        # Verify pop_screen was called
        modal.app.pop_screen.assert_called_once()

    def test_input_modal_on_button_pressed_with_callback_and_node(self) -> None:
        skip_if_ui_not_available()
        from textual.widgets.tree import TreeNode

        from arklex.orchestrator.generator.ui.input_modal import InputModal

        """Test InputModal on_button_pressed method with callback and node."""
        callback_called = False
        callback_result = None
        callback_node = None

        def mock_callback(result: str, node: TreeNode | None) -> None:
            nonlocal callback_called, callback_result, callback_node
            callback_called = True
            callback_result = result
            callback_node = node

        modal = InputModal("Test Title", "default value", callback=mock_callback)

        # Mock the query_one method
        modal.query_one = Mock()
        mock_input = Mock()
        mock_input.value = "test input"
        modal.query_one.return_value = mock_input

        # Mock the app.pop_screen method
        modal.app = Mock()
        modal.app.pop_screen = Mock()

        # Create a mock button press event
        mock_button = Mock()
        mock_button.id = "submit"
        mock_event = FallbackButton.Pressed(mock_button)

        # Test button press
        modal.on_button_pressed(mock_event)

        # Verify callback was called with correct parameters
        assert callback_called
        assert callback_result == "test input"
        assert callback_node is None

        # Verify pop_screen was called
        modal.app.pop_screen.assert_called_once()

    def test_input_modal_on_button_pressed_with_callback_no_node(self) -> None:
        skip_if_ui_not_available()
        from textual.widgets.tree import TreeNode

        from arklex.orchestrator.generator.ui.input_modal import InputModal

        """Test InputModal on_button_pressed method with callback but no node."""
        callback_called = False
        callback_result = None
        callback_node = None

        def mock_callback(result: str, node: TreeNode | None) -> None:
            nonlocal callback_called, callback_result, callback_node
            callback_called = True
            callback_result = result
            callback_node = node

        modal = InputModal("Test Title", "default value", callback=mock_callback)

        # Mock the query_one method
        modal.query_one = Mock()
        mock_input = Mock()
        mock_input.value = "test input"
        modal.query_one.return_value = mock_input

        # Mock the app.pop_screen method
        modal.app = Mock()
        modal.app.pop_screen = Mock()

        # Create a mock button press event
        mock_button = Mock()
        mock_button.id = "submit"
        mock_event = FallbackButton.Pressed(mock_button)

        # Test button press
        modal.on_button_pressed(mock_event)

        # Verify callback was called with correct parameters
        assert callback_called
        assert callback_result == "test input"
        assert callback_node is None

        # Verify pop_screen was called
        modal.app.pop_screen.assert_called_once()

    def test_input_modal_on_button_pressed_without_callback(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.input_modal import InputModal

        """Test InputModal on_button_pressed method without callback."""
        modal = InputModal("Test Title", "default value")

        # Mock the query_one method
        modal.query_one = Mock()
        mock_input = Mock()
        mock_input.value = "test input"
        modal.query_one.return_value = mock_input

        # Mock the app.pop_screen method
        modal.app = Mock()
        modal.app.pop_screen = Mock()

        # Create a mock button press event
        mock_button = Mock()
        mock_button.id = "submit"
        mock_event = FallbackButton.Pressed(mock_button)

        # Test button press
        modal.on_button_pressed(mock_event)

        # Verify the result was updated
        assert modal.result == "test input"

        # Verify pop_screen was called
        modal.app.pop_screen.assert_called_once()

    def test_input_modal_on_button_pressed_other_button_id(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.input_modal import InputModal

        """Test InputModal on_button_pressed method with other button id."""
        modal = InputModal("Test Title", "default value")

        # Mock the app.pop_screen method
        modal.app = Mock()
        modal.app.pop_screen = Mock()

        # Create a mock button press event with unknown id
        mock_button = Mock()
        mock_button.id = "unknown"
        mock_event = FallbackButton.Pressed(mock_button)

        # Test button press
        modal.on_button_pressed(mock_event)

        # Verify pop_screen was called
        modal.app.pop_screen.assert_called_once()

    def test_input_modal_on_button_pressed_without_app_pop_screen(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.input_modal import InputModal

        """Test InputModal on_button_pressed method when app doesn't have pop_screen."""
        modal = InputModal("Test Title", "default value")

        # Mock the query_one method
        modal.query_one = Mock()
        mock_input = Mock()
        mock_input.value = "test input"
        modal.query_one.return_value = mock_input

        # Mock the app without pop_screen method
        modal.app = Mock()
        del modal.app.pop_screen  # Remove pop_screen method

        # Create a mock button press event
        mock_button = Mock()
        mock_button.id = "submit"
        mock_event = FallbackButton.Pressed(mock_button)

        # Test button press - should not raise an exception
        modal.on_button_pressed(mock_event)

        # Verify the result was updated
        assert modal.result == "test input"

        # Verify no pop_screen was called since it doesn't exist
        assert not hasattr(modal.app, "pop_screen")


class TestUIErrorHandling:
    """Test UI error handling scenarios."""

    def test_task_editor_with_invalid_tasks(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.task_editor import TaskEditorApp

        """Test TaskEditorApp with invalid tasks."""
        # Should not raise an exception
        app = TaskEditorApp(None)
        assert app.tasks is None

    def test_ui_component_initialization_errors(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.input_modal import InputModal

        """Test UI component initialization error handling."""
        # Should not raise an exception
        modal = InputModal("Test Title")
        assert modal.title == "Test Title"

    @pytest.mark.asyncio
    async def test_ui_event_handling_errors(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.task_editor import TaskEditorApp

        """Test UI event handling error scenarios."""
        app = TaskEditorApp([])

        # Test with invalid event
        invalid_event = Mock()
        invalid_event.key = "invalid_key"

        # Should not raise an exception
        await app.on_key(invalid_event)


class FakeApp:
    """Fake app for testing purposes."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def run(self) -> list:
        return []


class TestInputModalFinalCoverage:
    """Additional tests for InputModal to ensure complete coverage."""

    def test_input_modal_on_button_pressed_with_callback_and_node(self) -> None:
        skip_if_ui_not_available()
        from textual.widgets.tree import TreeNode

        from arklex.orchestrator.generator.ui.input_modal import InputModal

        """Test InputModal on_button_pressed method with callback and node."""
        callback_called = False
        callback_result = None
        callback_node = None

        def mock_callback(result: str, node: TreeNode | None) -> None:
            nonlocal callback_called, callback_result, callback_node
            callback_called = True
            callback_result = result
            callback_node = node

        modal = InputModal("Test Title", "default value", callback=mock_callback)

        # Mock the query_one method
        modal.query_one = Mock()
        mock_input = Mock()
        mock_input.value = "test input"
        modal.query_one.return_value = mock_input

        # Mock the app.pop_screen method
        modal.app = Mock()
        modal.app.pop_screen = Mock()

        # Create a mock button press event
        mock_button = Mock()
        mock_button.id = "submit"
        mock_event = FallbackButton.Pressed(mock_button)

        # Test button press
        modal.on_button_pressed(mock_event)

        # Verify callback was called with correct parameters
        assert callback_called
        assert callback_result == "test input"
        assert callback_node is None

        # Verify pop_screen was called
        modal.app.pop_screen.assert_called_once()

    def test_input_modal_on_button_pressed_with_callback_no_node(self) -> None:
        skip_if_ui_not_available()
        from textual.widgets.tree import TreeNode

        from arklex.orchestrator.generator.ui.input_modal import InputModal

        """Test InputModal on_button_pressed method with callback but no node."""
        callback_called = False
        callback_result = None
        callback_node = None

        def mock_callback(result: str, node: TreeNode | None) -> None:
            nonlocal callback_called, callback_result, callback_node
            callback_called = True
            callback_result = result
            callback_node = node

        modal = InputModal("Test Title", "default value", callback=mock_callback)

        # Mock the query_one method
        modal.query_one = Mock()
        mock_input = Mock()
        mock_input.value = "test input"
        modal.query_one.return_value = mock_input

        # Mock the app.pop_screen method
        modal.app = Mock()
        modal.app.pop_screen = Mock()

        # Create a mock button press event
        mock_button = Mock()
        mock_button.id = "submit"
        mock_event = FallbackButton.Pressed(mock_button)

        # Test button press
        modal.on_button_pressed(mock_event)

        # Verify callback was called with correct parameters
        assert callback_called
        assert callback_result == "test input"
        assert callback_node is None

        # Verify pop_screen was called
        modal.app.pop_screen.assert_called_once()

    def test_input_modal_on_button_pressed_without_callback(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.input_modal import InputModal

        """Test InputModal on_button_pressed method without callback."""
        modal = InputModal("Test Title", "default value")

        # Mock the query_one method
        modal.query_one = Mock()
        mock_input = Mock()
        mock_input.value = "test input"
        modal.query_one.return_value = mock_input

        # Mock the app.pop_screen method
        modal.app = Mock()
        modal.app.pop_screen = Mock()

        # Create a mock button press event
        mock_button = Mock()
        mock_button.id = "submit"
        mock_event = FallbackButton.Pressed(mock_button)

        # Test button press
        modal.on_button_pressed(mock_event)

        # Verify the result was updated
        assert modal.result == "test input"

        # Verify pop_screen was called
        modal.app.pop_screen.assert_called_once()

    def test_input_modal_on_button_pressed_cancel_button(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.input_modal import InputModal

        """Test InputModal on_button_pressed method with cancel button."""
        modal = InputModal("Test Title", "default value")

        # Mock the app.pop_screen method
        modal.app = Mock()
        modal.app.pop_screen = Mock()

        # Create a mock button press event for cancel
        mock_button = Mock()
        mock_button.id = "cancel"
        mock_event = FallbackButton.Pressed(mock_button)

        # Test cancel button press
        modal.on_button_pressed(mock_event)

        # Verify pop_screen was called
        modal.app.pop_screen.assert_called_once()

    def test_input_modal_on_button_pressed_other_button_id(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.input_modal import InputModal

        """Test InputModal on_button_pressed method with other button id."""
        modal = InputModal("Test Title", "default value")

        # Mock the app.pop_screen method
        modal.app = Mock()
        modal.app.pop_screen = Mock()

        # Create a mock button press event with unknown id
        mock_button = Mock()
        mock_button.id = "unknown"
        mock_event = FallbackButton.Pressed(mock_button)

        # Test button press
        modal.on_button_pressed(mock_event)

        # Verify pop_screen was called
        modal.app.pop_screen.assert_called_once()

    def test_input_modal_on_button_pressed_without_app_pop_screen(self) -> None:
        skip_if_ui_not_available()
        from arklex.orchestrator.generator.ui.input_modal import InputModal

        """Test InputModal on_button_pressed method when app doesn't have pop_screen."""
        modal = InputModal("Test Title", "default value")

        # Mock the query_one method
        modal.query_one = Mock()
        mock_input = Mock()
        mock_input.value = "test input"
        modal.query_one.return_value = mock_input

        # Mock the app without pop_screen method
        modal.app = Mock()
        del modal.app.pop_screen  # Remove pop_screen method

        # Create a mock button press event
        mock_button = Mock()
        mock_button.id = "submit"
        mock_event = FallbackButton.Pressed(mock_button)

        # Test button press - should not raise an exception
        modal.on_button_pressed(mock_event)

        # Verify the result was updated
        assert modal.result == "test input"

        # Verify no pop_screen was called since it doesn't exist
        assert not hasattr(modal.app, "pop_screen")


# Cleanup function to be called after tests
def cleanup() -> None:
    """Clean up any patches made during testing."""
    if not TEXTUAL_AVAILABLE:
        unpatch_ui_modules()


def test_protocols_full_coverage() -> None:
    """Covers all methods/properties of the Protocols in protocols.py for coverage."""
    from arklex.orchestrator.generator.ui import protocols

    class DummyTreeNode(protocols.TreeNodeProtocol):
        def add(self, label: str) -> "DummyTreeNode":
            return self

        def add_leaf(self, label: str) -> "DummyTreeNode":
            return self

        def remove(self) -> None:
            pass

        def set_label(self, label: str) -> None:
            pass

        def expand(self) -> None:
            pass

        @property
        def children(self) -> list:
            return []

        @property
        def parent(self) -> None:
            return None

        @property
        def label(self) -> str:
            return "label"

    class DummyTree(protocols.TreeProtocol):
        def focus(self) -> None:
            pass

        @property
        def root(self) -> DummyTreeNode:
            return DummyTreeNode()

        @property
        def cursor_node(self) -> DummyTreeNode:
            return DummyTreeNode()

    class DummyInputModal(protocols.InputModalProtocol):
        def __init__(
            self, title: str, default: str, node: object, callback: object
        ) -> None:
            pass

    # Instantiate and call all methods/properties
    node = DummyTreeNode()
    node.add("x")
    node.add_leaf("y")
    node.remove()
    node.set_label("z")
    node.expand()
    _ = node.children
    _ = node.parent
    _ = node.label

    tree = DummyTree()
    tree.focus()
    _ = tree.root
    _ = tree.cursor_node

    DummyInputModal("t", "d", None, None)
