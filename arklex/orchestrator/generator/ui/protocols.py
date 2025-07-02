"""Protocol definitions for the task editor components.

This module contains protocol classes that enable better testing by providing
interfaces for dependency injection and mocking.
"""

from collections.abc import Callable
from typing import Protocol


class TreeNodeProtocol(Protocol):  # pragma: no cover
    """Protocol for tree node operations to enable better testing."""

    def add(self, label: str) -> "TreeNodeProtocol":
        """Add a child node."""
        pass

    def add_leaf(self, label: str) -> "TreeNodeProtocol":
        """Add a leaf node."""
        pass

    def remove(self) -> None:
        """Remove this node."""
        pass

    def set_label(self, label: str) -> None:
        """Set the node label."""
        pass

    def expand(self) -> None:
        """Expand the node."""
        pass

    @property
    def children(self) -> list["TreeNodeProtocol"]:
        """Get child nodes."""
        pass

    @property
    def parent(self) -> "TreeNodeProtocol | None":
        """Get parent node."""
        pass

    @property
    def label(self) -> str | object:
        """Get node label."""
        pass


class TreeProtocol(Protocol):  # pragma: no cover
    """Protocol for tree operations to enable better testing."""

    def focus(self) -> None:
        """Focus the tree."""
        pass

    @property
    def root(self) -> TreeNodeProtocol | None:
        """Get root node."""
        pass

    @property
    def cursor_node(self) -> TreeNodeProtocol | None:
        """Get currently selected node."""
        pass


class InputModalProtocol(Protocol):  # pragma: no cover
    """Protocol for input modal operations."""

    def __init__(
        self,
        title: str,
        default: str,
        node: TreeNodeProtocol | None,
        callback: Callable[[str, TreeNodeProtocol | None], None] | None,
    ) -> None:
        """Initialize the modal."""
        pass
