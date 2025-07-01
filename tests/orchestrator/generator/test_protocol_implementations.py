"""Concrete implementations of protocols for testing purposes.

This module contains concrete implementations of the protocols defined in
arklex.orchestrator.generator.ui.protocols to enable better testing and coverage.
These implementations are only used for testing and should not be used in production code.
"""

from collections.abc import Callable

from arklex.orchestrator.generator.ui.protocols import (
    TreeNodeProtocol,
)


class ConcreteTreeNode:
    """Concrete implementation of TreeNodeProtocol for testing coverage."""

    def __init__(
        self, label: str = "", parent: "ConcreteTreeNode | None" = None
    ) -> None:
        self._label = label
        self._parent = parent
        self._children: list[ConcreteTreeNode] = []

    def add(self, label: str) -> "ConcreteTreeNode":
        """Add a child node."""
        child = ConcreteTreeNode(label, self)
        self._children.append(child)
        return child

    def add_leaf(self, label: str) -> "ConcreteTreeNode":
        """Add a leaf node."""
        return self.add(label)

    def remove(self) -> None:
        """Remove this node."""
        if self._parent:
            self._parent._children.remove(self)
            self._parent = None

    def set_label(self, label: str) -> None:
        """Set the node label."""
        self._label = label

    def expand(self) -> None:
        """Expand the node."""
        # No-op for this implementation
        pass

    @property
    def children(self) -> list["ConcreteTreeNode"]:
        """Get child nodes."""
        return self._children.copy()

    @property
    def parent(self) -> "ConcreteTreeNode | None":
        """Get parent node."""
        return self._parent

    @property
    def label(self) -> str:
        """Get node label."""
        return self._label


class ConcreteTree:
    """Concrete implementation of TreeProtocol for testing coverage."""

    def __init__(self) -> None:
        self._root: ConcreteTreeNode | None = None
        self._cursor_node: ConcreteTreeNode | None = None

    def focus(self) -> None:
        """Focus the tree."""
        # No-op for this implementation
        pass

    @property
    def root(self) -> ConcreteTreeNode | None:
        """Get root node."""
        return self._root

    @property
    def cursor_node(self) -> ConcreteTreeNode | None:
        """Get currently selected node."""
        return self._cursor_node


class ConcreteInputModal:
    """Concrete implementation of InputModalProtocol for testing coverage."""

    def __init__(
        self,
        title: str,
        default: str,
        node: TreeNodeProtocol | None,
        callback: Callable[[str, TreeNodeProtocol | None], None] | None,
    ) -> None:
        """Initialize the modal."""
        self.title = title
        self.default = default
        self.node = node
        self.callback = callback


def create_test_tree_node(label: str = "") -> ConcreteTreeNode:
    """Create a test tree node for coverage testing."""
    return ConcreteTreeNode(label)


def create_test_tree() -> ConcreteTree:
    """Create a test tree for coverage testing."""
    return ConcreteTree()


def create_test_input_modal(
    title: str = "Test",
    default: str = "",
    node: TreeNodeProtocol | None = None,
    callback: Callable[[str, TreeNodeProtocol | None], None] | None = None,
) -> ConcreteInputModal:
    """Create a test input modal for coverage testing."""
    return ConcreteInputModal(title, default, node, callback)
