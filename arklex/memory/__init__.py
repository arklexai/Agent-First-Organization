"""Memory management module for the Arklex framework.

This module provides functionality for managing conversation memory and context.
It exports the ShortTermMemory class which handles storing and retrieving
conversation trajectories, managing embeddings for semantic search, and
personalizing user intents based on conversation context.

Key Components:
- ShortTermMemory: Core class for managing conversation memory and context
  - Handles conversation trajectory storage and retrieval
  - Manages embeddings for semantic search
  - Personalizes user intents based on context
  - Provides efficient caching mechanisms

Key Features:
- Efficient memory management with trajectory tracking
- Semantic search capabilities with embedding support
- Intent personalization based on conversation context
- Caching mechanisms for improved performance
- Asynchronous processing support

Usage:
    from arklex.memory import ShortTermMemory

    # Initialize memory with conversation trajectory and chat history
    memory = ShortTermMemory(
        trajectory=conversation_trajectory,
        chat_history=formatted_chat_history,
        llm_config=llm_configuration
    )

    # Retrieve relevant records based on a query
    found, records = memory.retrieve_records(query="user query")

    # Generate personalized intents
    await memory.personalize()
"""

from .core import ShortTermMemory

__all__ = [
    "ShortTermMemory",  # Core class for managing conversation memory and context
]
