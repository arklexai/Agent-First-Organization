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

    def add_leaf(self, label: str) -> "TreeNodeProtocol":
        """Add a leaf node."""

    def remove(self) -> None:
        """Remove this node."""

    def set_label(self, label: str) -> None:
        """Set the node label."""

    def expand(self) -> None:
        """Expand the node."""

    @property
    def children(self) -> list["TreeNodeProtocol"]:
        """Get child nodes."""

    @property
    def parent(self) -> "TreeNodeProtocol | None":
        """Get parent node."""

    @property
    def label(self) -> str | object:
        """Get node label."""


class TreeProtocol(Protocol):  # pragma: no cover
    """Protocol for tree operations to enable better testing."""

    def focus(self) -> None:
        """Focus the tree."""

    @property
    def root(self) -> TreeNodeProtocol | None:
        """Get root node."""

    @property
    def cursor_node(self) -> TreeNodeProtocol | None:
        """Get currently selected node."""


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
