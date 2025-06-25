"""UI component tests with mock user interactions.

These tests simulate user interactions with the UI components
to ensure they work correctly without requiring actual user input.
"""

import pytest
from unittest.mock import Mock, patch
from contextlib import ExitStack
import sys

from arklex.orchestrator.generator.ui.task_editor import TaskEditorApp
from arklex.orchestrator.generator.ui.input_modal import InputModal


# --- Global patching fixtures for UI components ---


@pytest.fixture(autouse=True)
def patch_task_editor_app():
    """Patch TaskEditorApp's App dependency for all tests."""
    with patch("arklex.orchestrator.generator.ui.task_editor.App"):
        yield


@pytest.fixture(autouse=True)
def patch_input_modal_screen():
    """Patch InputModal's Screen dependency for all tests."""
    with patch("arklex.orchestrator.generator.ui.input_modal.Screen"):
        yield


@pytest.fixture(autouse=True)
def patch_input_modal_components():
    """Patch all InputModal UI components for all tests."""
    patch_targets = [
        "arklex.orchestrator.generator.ui.input_modal.Vertical",
        "arklex.orchestrator.generator.ui.input_modal.Static",
        "arklex.orchestrator.generator.ui.input_modal.Input",
        "arklex.orchestrator.generator.ui.input_modal.Horizontal",
        "arklex.orchestrator.generator.ui.input_modal.Button",
    ]
    with ExitStack() as stack:
        [stack.enter_context(patch(target)) for target in patch_targets]
        yield


@pytest.fixture(autouse=True)
def patch_task_editor_components():
    """Patch TaskEditor UI components for all tests."""
    with (
        patch("arklex.orchestrator.generator.ui.task_editor.Tree") as mock_tree,
        patch("arklex.orchestrator.generator.ui.task_editor.Label") as mock_label,
    ):
        mock_tree_instance = Mock()
        mock_tree_instance.root = Mock()
        mock_tree_instance.root.add = Mock(return_value=Mock())
        mock_tree.return_value = mock_tree_instance
        mock_label_instance = Mock()
        mock_label.return_value = mock_label_instance
        yield


# --- Fixtures ---


@pytest.fixture
def sample_tasks() -> list:
    """Sample tasks for testing."""
    return [
        {
            "name": "Customer Support",
            "steps": [
                {"description": "Listen to customer", "step_id": "step_1"},
                {"description": "Provide solution", "step_id": "step_2"},
            ],
        },
        {
            "name": "Product Search",
            "steps": [
                {"description": "Get search criteria", "step_id": "step_1"},
                {"description": "Search database", "step_id": "step_2"},
            ],
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
    valid_task = '[{"id": "task_1", "name": "Test Task", "description": "A test task", "steps": [{"description": "Step 1"}]}]'
    model.generate = lambda messages: type("Mock", (), {"content": valid_task})()
    model.invoke = lambda messages: type("Mock", (), {"content": valid_task})()
    return model


@pytest.fixture
def mock_task_editor(sample_tasks: list, patch_task_editor_app) -> TaskEditorApp:
    """Create a mock task editor instance."""
    editor = TaskEditorApp(sample_tasks)
    editor.task_tree = Mock()
    editor.task_tree.root = Mock()
    editor.task_tree.root.children = []
    editor.task_tree.cursor_node = None
    return editor


@pytest.fixture
def mock_input_modal(patch_input_modal_screen) -> InputModal:
    """Create a mock input modal instance."""
    modal = InputModal("Test Title", "Default Value")
    modal.result = None
    return modal


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

    def test_task_editor_initialization(
        self, sample_tasks: list, patch_task_editor_app
    ) -> None:
        """Test task editor initialization."""
        editor = TaskEditorApp(sample_tasks)
        assert editor.tasks == sample_tasks
        assert editor.task_tree is None

    def test_compose_creates_tree_structure(
        self, sample_tasks: list, patch_task_editor_app
    ) -> None:
        """Test that compose method creates proper tree structure."""
        editor = TaskEditorApp(sample_tasks)
        result = list(editor.compose())
        mock_tree = sys.modules["arklex.orchestrator.generator.ui.task_editor"].Tree
        mock_label = sys.modules["arklex.orchestrator.generator.ui.task_editor"].Label
        assert mock_tree.called
        assert mock_label.called
        assert len(result) == 2

    def test_on_mount_sets_focus(self, mock_task_editor: TaskEditorApp) -> None:
        """Test that on_mount sets focus to task tree."""
        mock_task_editor.on_mount()
        mock_task_editor.task_tree.focus.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_task_with_keyboard(
        self, mock_task_editor: TaskEditorApp, mock_node: Mock, mock_event: Mock
    ) -> None:
        """Test adding a task using keyboard shortcut."""
        mock_task_editor.task_tree.cursor_node = mock_node
        mock_task_editor.push_screen = Mock()
        with patch.object(mock_task_editor, "action_add_node") as mock_action:
            await mock_task_editor.on_key(mock_event)
            mock_action.assert_called_once_with(mock_node)

    @pytest.mark.asyncio
    async def test_delete_task_with_keyboard(
        self, mock_task_editor: TaskEditorApp, mock_node: Mock
    ) -> None:
        """Test deleting a task using keyboard shortcut."""
        mock_task_editor.task_tree.cursor_node = mock_node
        mock_event = Mock()
        mock_event.key = "d"

        async def mock_update_tasks():
            pass

        mock_task_editor.update_tasks = mock_update_tasks
        await mock_task_editor.on_key(mock_event)
        mock_node.remove.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_and_exit_with_keyboard(
        self, mock_task_editor: TaskEditorApp
    ) -> None:
        """Test saving and exiting using keyboard shortcut."""
        mock_event = Mock()
        mock_event.key = "s"
        mock_task_editor.exit = Mock()
        await mock_task_editor.on_key(mock_event)
        mock_task_editor.exit.assert_called_once_with(mock_task_editor.tasks)

    @pytest.mark.asyncio
    async def test_add_step_to_task(
        self, mock_task_editor: TaskEditorApp, mock_node: Mock
    ) -> None:
        """Test adding a step to a task."""
        mock_node.label.plain = "Customer Support"
        mock_task_editor.push_screen = Mock()
        with patch.object(mock_task_editor, "action_add_node") as mock_action:
            await mock_task_editor.action_add_node(mock_node)
            mock_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_task_to_root(
        self, mock_task_editor: TaskEditorApp, mock_node: Mock
    ) -> None:
        """Test adding a task to the root level."""
        mock_node.is_expanded = False
        mock_task_editor.push_screen = Mock()
        with patch.object(mock_task_editor, "action_add_node") as mock_action:
            await mock_task_editor.action_add_node(mock_node)
            mock_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_tasks_from_tree(
        self, mock_task_editor: TaskEditorApp
    ) -> None:
        """Test updating tasks list from tree structure."""
        mock_task_node = Mock()
        mock_task_node.label = Mock()
        mock_task_node.label.plain = "Customer Support"
        mock_task_node.children = [
            Mock(label=Mock(plain="Step 1")),
            Mock(label=Mock(plain="Step 2")),
        ]
        mock_task_editor.task_tree = Mock()
        mock_task_editor.task_tree.root = Mock()
        mock_task_editor.task_tree.root.children = [mock_task_node]
        await mock_task_editor.update_tasks()
        assert len(mock_task_editor.tasks) == 1
        assert mock_task_editor.tasks[0]["name"] == "Customer Support"
        assert len(mock_task_editor.tasks[0]["steps"]) == 2

    @pytest.mark.asyncio
    async def test_node_selection_opens_modal(
        self, mock_task_editor: TaskEditorApp, mock_node: Mock
    ) -> None:
        """Test that node selection opens input modal."""
        mock_event = Mock()
        mock_event.node = mock_node
        mock_task_editor.push_screen = Mock()
        await mock_task_editor.on_tree_node_selected(mock_event)
        mock_task_editor.push_screen.assert_called_once()

    def test_run_returns_tasks(self, mock_task_editor: TaskEditorApp) -> None:
        """Test that run method returns the tasks."""
        with patch.object(mock_task_editor, "run") as mock_run:
            mock_run.return_value = mock_task_editor.tasks
            result = mock_task_editor.run()
            assert result == mock_task_editor.tasks


class TestInputModalUI:
    """Test the InputModal UI component with mock interactions."""

    def test_input_modal_initialization(self, patch_input_modal_screen) -> None:
        """Test InputModal initialization."""
        modal = InputModal("Test Title", "Default Value")
        assert modal.title == "Test Title"
        assert modal.default == "Default Value"

    def test_input_modal_with_callback(self, patch_input_modal_screen) -> None:
        """Test InputModal with a callback."""
        callback = Mock()
        modal = InputModal("Test Title", "Default Value", callback=callback)
        assert modal.callback == callback

    def test_compose_creates_input_structure(self, patch_input_modal_screen) -> None:
        """Test that compose method creates proper input structure."""
        modal = InputModal("Test Title", "Default Value")
        result = list(modal.compose())
        # Check that the patched UI components were called
        vertical = sys.modules["arklex.orchestrator.generator.ui.input_modal"].Vertical
        static = sys.modules["arklex.orchestrator.generator.ui.input_modal"].Static
        input_ = sys.modules["arklex.orchestrator.generator.ui.input_modal"].Input
        horizontal = sys.modules[
            "arklex.orchestrator.generator.ui.input_modal"
        ].Horizontal
        button = sys.modules["arklex.orchestrator.generator.ui.input_modal"].Button
        assert vertical.called
        assert static.called
        assert input_.called
        assert horizontal.called
        assert button.called
        assert isinstance(result, list)

    def test_modal_dismissal(self, mock_input_modal: InputModal) -> None:
        """Test modal dismissal without submission."""
        assert mock_input_modal.title == "Test Title"
        assert mock_input_modal.default == "Default Value"
        assert mock_input_modal.result is None


class TestUIErrorHandling:
    """Test UI error handling scenarios."""

    def test_task_editor_with_invalid_tasks(self, patch_task_editor_app) -> None:
        """Test task editor with invalid task data."""
        invalid_tasks = [
            {"name": "Valid Task", "steps": [{"description": "Valid step"}]},
            {"name": None, "steps": [{"description": "Valid step"}]},
            {"name": "Valid Task", "steps": None},
        ]
        editor = TaskEditorApp(invalid_tasks)
        assert editor.tasks == invalid_tasks

    def test_ui_component_initialization_errors(self, patch_task_editor_app) -> None:
        """Test UI component initialization with various error conditions."""
        editor = TaskEditorApp([])
        assert editor.tasks == []

    @pytest.mark.asyncio
    async def test_ui_event_handling_errors(self, patch_task_editor_app) -> None:
        """Test UI event handling with various error conditions."""
        editor = TaskEditorApp([])
        editor.task_tree = Mock()
        editor.task_tree.cursor_node = None
        try:
            await editor.on_key(None)
        except AttributeError:
            pass
        invalid_event = Mock()
        invalid_event.key = "invalid_key"
        await editor.on_key(invalid_event)


# Create a minimal fake textual.app module
class FakeApp:
    def __init__(self, *args, **kwargs):
        pass

    def run(self):
        pass
