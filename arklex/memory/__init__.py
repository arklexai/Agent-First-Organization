"""Memory management module for the Arklex framework.

This module provides functionality for managing conversation memory and context.
It exports the ShortTermMemory class which handles storing and retrieving
conversation trajectories, managing embeddings for semantic search, and
personalizing user intents based on conversation context.
"""

from .core import ShortTermMemory

__all__ = [
    "ShortTermMemory",
]
