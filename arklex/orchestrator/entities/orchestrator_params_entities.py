"""Orchestrator parameters and state management.

This module defines the core parameters and state management entities used by the orchestrator
for managing conversation flow, task execution, and system state.

Key Components:
- OrchestratorParams: Represents the orchestrator's runtime parameters including metadata,
  task graph state, and memory management for conversation processing.
"""

from pydantic import BaseModel, Field

from arklex.memory.entities.memory_entities import Memory
from arklex.orchestrator.entities.msg_state_entities import Metadata
from arklex.orchestrator.entities.taskgraph_entities import Taskgraph


class OrchestratorParams(BaseModel):
    """Runtime parameters for orchestrator task processing.

    This class holds the orchestrator's runtime parameters needed for task processing,
    including metadata, task graph structure, and memory management. It serves as the
    central state container for orchestrator operations.

    The class provides:
    1. Metadata management for session tracking
    2. Task graph structure and state management
    3. Memory management for conversation history
    4. Type-safe parameter handling for orchestrator operations

    Attributes:
        metadata (Metadata): Session metadata including chat_id, turn_id, and timing.
        taskgraph (Taskgraph): Task graph structure and current processing state.
        memory (Memory): Memory management for conversation history and trajectories.
    """

    metadata: Metadata = Field(default_factory=Metadata)
    taskgraph: Taskgraph = Field(default_factory=Taskgraph)
    memory: Memory = Field(default_factory=Memory)
