"""Entities for orchestrator.

This module defines the entities used in the orchestrator,
including MessageState and Params.

Key Components:
- MessageState: Represents the state of a message as it moves through the system.
- Params: Represents the parameters for task processing.
"""

from typing import Any

from pydantic import BaseModel, Field

from arklex.memory.entities.memory_entities import Memory, ResourceRecord
from arklex.orchestrator.entities.msg_state_entities import (
    BotConfig,
    ConvoMessage,
    Metadata,
    OrchestratorMessage,
    StatusEnum,
)
from arklex.orchestrator.entities.taskgraph_entities import Taskgraph
from arklex.orchestrator.NLU.entities.slot_entities import Slot
from arklex.types import StreamType


class MessageState(BaseModel):
    """State management for message processing.

    This class manages the complete state of a message as it moves through the system,
    including configuration, processing trajectory, and response generation.

    The class provides:
    1. System configuration management
    2. Bot configuration handling
    3. Message processing tracking
    4. Response generation
    5. Status tracking
    6. Slot management
    7. Streaming support
    8. Type-safe state management

    Attributes:
        sys_instruct (str): System instructions for message processing.
        bot_config (BotConfig): Configuration for the bot.
        user_message (ConvoMessage): User's input message.
        orchestrator_message (OrchestratorMessage): Message from the orchestrator.
        function_calling_trajectory (List[Dict[str, Any]]): History of function calls.
        trajectory (List[List[ResourceRecord]]): Processing trajectory.
        message_flow (str): Flow of messages between nodes.
        response (str): Final response to the user.
        status (StatusEnum): Current status of the message processing.
        slots (Dict[str, List[Slot]]): Dialogue state slots.
        metadata (Metadata): Session metadata.
        is_stream (bool): Whether the message is being streamed.
        message_queue (Any): Queue for streaming messages.
        stream_type (str): Type of streaming being used.
        relevant_records (Optional[List[ResourceRecord]]): Relevant resource records.
    """

    # system configuration
    sys_instruct: str = Field(default="")
    # bot configuration
    bot_config: BotConfig | None = Field(default=None)
    # input message
    user_message: ConvoMessage | None = Field(default=None)
    orchestrator_message: OrchestratorMessage | None = Field(default=None)
    # action trajectory
    function_calling_trajectory: list[dict[str, Any]] | None = Field(default=None)
    trajectory: list[list[ResourceRecord]] | None = Field(default=None)
    # message flow between different nodes
    message_flow: str = Field(
        description="message flow between different nodes", default=""
    )
    # final response
    response: str = Field(default="")
    # task-related params
    status: StatusEnum = Field(default=StatusEnum.INCOMPLETE)
    slots: dict[str, list[Slot]] | None = Field(
        description="record the dialogue states of each action", default=None
    )
    metadata: Metadata | None = Field(default=None)
    # stream
    is_stream: bool = Field(default=False)
    message_queue: Any = Field(exclude=True, default=None)
    stream_type: StreamType | None = Field(default=StreamType.NON_STREAM)
    # memory records
    relevant_records: list[ResourceRecord] | None = Field(default=None)


class Params(BaseModel):
    """Parameters for task processing.

    This class holds the parameters needed for task processing, including
    metadata, task graph structure, and memory management.

    The class provides:
    1. Metadata management
    2. Task graph structure
    3. Memory management
    4. Type-safe parameter handling

    Attributes:
        metadata (Metadata): Session metadata.
        taskgraph (Taskgraph): Task graph structure.
        memory (Memory): Memory management.
    """

    metadata: Metadata = Field(default_factory=Metadata)
    taskgraph: Taskgraph = Field(default_factory=Taskgraph)
    memory: Memory = Field(default_factory=Memory)
