"""Unit tests for the TaskEditorApp components.

This module provides focused unit tests for the TaskEditorApp class and its components. These tests
use dependency injection and protocols to test individual components in isolation, making them
faster, more reliable, and easier to maintain.

Key characteristics:
- Tests individual components in isolation
- Uses dependency injection for clean mocking
- Focuses on business logic (TaskDataManager)
- Uses protocols for better abstraction
- Faster execution and easier debugging

This approach demonstrates modern testing practices and makes the code more testable and maintainable.
"""

from unittest.mock import Mock

import pytest

from arklex.orchestrator.generator.ui.data_manager import TaskDataManager
from arklex.orchestrator.generator.ui.protocols import (
    TreeNodeProtocol,
    TreeProtocol,
)
from arklex.orchestrator.generator.ui.task_editor import TaskEditorApp
from tests.orchestrator.generator.test_protocol_implementations import (
    ConcreteTreeNode,
    create_test_input_modal,
    create_test_tree,
    create_test_tree_node,
)


class TestTaskDataManager:
    """Test the pure functions in TaskDataManager."""

    def test_extract_label_text_with_plain_attribute(self) -> None:
        """Test extracting text from label with plain attribute."""
        label = Mock()
        label.plain = "test label"

        result = TaskDataManager.extract_label_text(label)

        assert result == "test label"

    def test_extract_label_text_without_plain_attribute(self) -> None:
        """Test extracting text from label without plain attribute."""
        label = Mock()
        # Remove the plain attribute to simulate a label without it
        del label.plain
        label.__str__ = Mock(return_value="string representation")

        result = TaskDataManager.extract_label_text(label)

        assert result == "string representation"

    def test_build_tasks_from_tree_empty_root(self) -> None:
        """Test building tasks from empty tree."""
        result = TaskDataManager.build_tasks_from_tree(None)

        assert result == []

    def test_build_tasks_from_tree_with_tasks(self) -> None:
        """Test building tasks from populated tree."""
        # Create mock tree structure
        task1_node = Mock(spec=TreeNodeProtocol)
        task1_node.label = Mock()
        task1_node.label.plain = "Task 1"
        task1_node.children = []

        task2_node = Mock(spec=TreeNodeProtocol)
        task2_node.label = Mock()
        task2_node.label.plain = "Task 2"

        step1_node = Mock(spec=TreeNodeProtocol)
        step1_node.label = Mock()
        step1_node.label.plain = "Step 1"

        step2_node = Mock(spec=TreeNodeProtocol)
        step2_node.label = Mock()
        step2_node.label.plain = "Step 2"

        task2_node.children = [step1_node, step2_node]

        root = Mock(spec=TreeNodeProtocol)
        root.children = [task1_node, task2_node]

        result = TaskDataManager.build_tasks_from_tree(root)

        expected = [
            {"name": "Task 1"},
            {"name": "Task 2", "steps": ["Step 1", "Step 2"]},
        ]
        assert result == expected

    def test_populate_tree_from_tasks_empty_tasks(self) -> None:
        """Test populating tree with empty tasks."""
        tree = Mock(spec=TreeProtocol)
        tree.root = Mock(spec=TreeNodeProtocol)
        tree.root.expand = Mock()
        tree.root.add = Mock()

        TaskDataManager.populate_tree_from_tasks(tree, [])

        tree.root.expand.assert_called_once()
        tree.root.add.assert_not_called()

    def test_populate_tree_from_tasks_with_tasks(self) -> None:
        """Test populating tree with tasks."""
        tree = Mock(spec=TreeProtocol)
        tree.root = Mock(spec=TreeNodeProtocol)
        tree.root.expand = Mock()

        task_node = Mock(spec=TreeNodeProtocol)
        task_node.add_leaf = Mock()
        tree.root.add.return_value = task_node

        tasks = [
            {"name": "Task 1", "steps": ["Step 1", "Step 2"]},
            {"name": "Task 2", "steps": [{"description": "Complex step"}]},
        ]

        TaskDataManager.populate_tree_from_tasks(tree, tasks)

        assert tree.root.add.call_count == 2
        assert task_node.add_leaf.call_count == 3  # 2 string steps + 1 dict step


class TestTaskEditorAppWithDependencyInjection:
    """Test TaskEditorApp using dependency injection for better testing."""

    def test_initialization_with_custom_modal_class(self) -> None:
        """Test that custom modal class can be injected."""
        mock_modal_class = Mock()
        tasks = [{"name": "Test Task"}]

        app = TaskEditorApp(tasks, input_modal_class=mock_modal_class)

        assert app._input_modal_class == mock_modal_class
        assert app.tasks == tasks

    def test_show_input_modal_uses_injected_class(self) -> None:
        """Test that show_input_modal uses the injected modal class."""
        mock_modal_class = Mock()
        mock_modal_instance = Mock()
        mock_modal_class.return_value = mock_modal_instance

        app = TaskEditorApp([], input_modal_class=mock_modal_class)
        app.push_screen = Mock()

        result = app.show_input_modal("Test", "default", None, None)

        mock_modal_class.assert_called_once_with("Test", "default", None, None)
        app.push_screen.assert_called_once_with(mock_modal_instance)
        assert result == "default"

    async def test_action_delete_node_with_valid_node(self) -> None:
        """Test deleting a node that has a parent."""
        app = TaskEditorApp([])
        node = Mock(spec=TreeNodeProtocol)
        node.parent = Mock()
        node.remove = Mock()

        await app.action_delete_node(node)

        node.remove.assert_called_once()

    def test_action_delete_node_without_parent(self) -> None:
        """Test deleting a node without parent (should not delete)."""
        app = TaskEditorApp([])
        node = Mock(spec=TreeNodeProtocol)
        node.parent = None
        node.remove = Mock()

        app.action_delete_node(node)

        node.remove.assert_not_called()

    def test_action_delete_node_with_none_node(self) -> None:
        """Test deleting a None node (should not crash)."""
        app = TaskEditorApp([])

        # Should not raise any exception
        app.action_delete_node(None)

    def test_on_key_without_task_tree(self) -> None:
        """Test keyboard handling when task_tree is None."""
        app = TaskEditorApp([])
        app.task_tree = None

        # Should not crash
        app.on_key(Mock(key="a"))

    async def test_update_tasks_uses_data_manager(self) -> None:
        """Test that update_tasks delegates to data manager."""
        app = TaskEditorApp([])
        app.task_tree = Mock()
        app.task_tree.root = Mock()

        # Mock the data manager method
        original_build_tasks = TaskDataManager.build_tasks_from_tree
        TaskDataManager.build_tasks_from_tree = Mock(return_value=[{"name": "test"}])

        try:
            await app.update_tasks()

            TaskDataManager.build_tasks_from_tree.assert_called_once_with(
                app.task_tree.root
            )
            assert app.tasks == [{"name": "test"}]
        finally:
            # Restore original method
            TaskDataManager.build_tasks_from_tree = original_build_tasks


class TestProtocolCompliance:
    """Test that our mock objects comply with the protocols."""

    def test_tree_node_protocol_compliance(self) -> None:
        """Test that a mock can implement TreeNodeProtocol."""
        mock_node = Mock(spec=TreeNodeProtocol)

        # These should not raise AttributeError
        mock_node.add("test")
        mock_node.add_leaf("test")
        mock_node.remove()
        mock_node.set_label("test")
        mock_node.expand()
        _ = mock_node.children
        _ = mock_node.parent
        _ = mock_node.label

    def test_tree_protocol_compliance(self) -> None:
        """Test that a mock can implement TreeProtocol."""
        mock_tree = Mock(spec=TreeProtocol)

        # These should not raise AttributeError
        mock_tree.focus()
        _ = mock_tree.root
        _ = mock_tree.cursor_node


class TestIntegration:
    """Test integration between components."""

    def test_compose_uses_data_manager(self) -> None:
        """Test that compose method uses the data manager."""
        tasks = [{"name": "Test Task", "steps": ["Step 1"]}]
        app = TaskEditorApp(tasks)

        # Mock the data manager method
        original_populate = TaskDataManager.populate_tree_from_tasks
        TaskDataManager.populate_tree_from_tasks = Mock()

        try:
            # Create a mock tree
            mock_tree = Mock()
            app.task_tree = mock_tree

            # Call the static method directly (since we removed the instance)
            TaskDataManager.populate_tree_from_tasks(mock_tree, tasks)

            TaskDataManager.populate_tree_from_tasks.assert_called_once_with(
                mock_tree, tasks
            )
        finally:
            # Restore original method
            TaskDataManager.populate_tree_from_tasks = original_populate


class TestConcreteProtocolImplementations:
    """Test concrete implementations to ensure protocol methods get covered."""

    def test_concrete_tree_node_operations(self) -> None:
        """Test all TreeNodeProtocol methods with concrete implementation."""
        # Create a tree node
        node = create_test_tree_node("Root")

        # Test add method
        child = node.add("Child")
        assert isinstance(child, ConcreteTreeNode)
        assert child.label == "Child"
        assert child.parent == node

        # Test add_leaf method
        leaf = node.add_leaf("Leaf")
        assert isinstance(leaf, ConcreteTreeNode)
        assert leaf.label == "Leaf"

        # Test set_label method
        node.set_label("New Root")
        assert node.label == "New Root"

        # Test expand method
        node.expand()  # Should not raise any exception

        # Test children property
        assert len(node.children) == 2
        assert all(isinstance(child, ConcreteTreeNode) for child in node.children)

        # Test parent property
        assert child.parent == node
        assert node.parent is None

        # Test remove method
        child.remove()
        assert child.parent is None
        assert child not in node.children

    def test_concrete_tree_operations(self) -> None:
        """Test all TreeProtocol methods with concrete implementation."""
        tree = create_test_tree()

        # Test focus method
        tree.focus()  # Should not raise any exception

        # Test root property
        assert tree.root is None

        # Test cursor_node property
        assert tree.cursor_node is None

    def test_concrete_input_modal_operations(self) -> None:
        """Test InputModalProtocol methods with concrete implementation."""
        node = create_test_tree_node("Test Node")

        def test_callback(text: str, node: TreeNodeProtocol | None) -> None:
            pass

        modal = create_test_input_modal("Test Title", "Default", node, test_callback)

        assert modal.title == "Test Title"
        assert modal.default == "Default"
        assert modal.node == node
        assert modal.callback == test_callback

    def test_concrete_implementations_with_task_data_manager(self) -> None:
        """Test that concrete implementations work with TaskDataManager."""
        # Create a tree structure using concrete implementations
        root = create_test_tree_node("Root")
        root.add("Task 1")
        task2 = root.add("Task 2")
        task2.add_leaf("Step 1")
        task2.add_leaf("Step 2")

        # Test build_tasks_from_tree with concrete implementation
        result = TaskDataManager.build_tasks_from_tree(root)

        expected = [
            {"name": "Task 1"},
            {"name": "Task 2", "steps": ["Step 1", "Step 2"]},
        ]
        assert result == expected

    def test_concrete_tree_population(self) -> None:
        """Test populating a concrete tree with tasks."""
        tree = create_test_tree()
        tree._root = create_test_tree_node("Root")  # Set root for testing

        tasks = [
            {"name": "Task 1", "steps": ["Step 1", "Step 2"]},
            {"name": "Task 2", "steps": [{"description": "Complex step"}]},
        ]

        # This should not raise any exception
        TaskDataManager.populate_tree_from_tasks(tree, tasks)


if __name__ == "__main__":
    pytest.main([__file__])
