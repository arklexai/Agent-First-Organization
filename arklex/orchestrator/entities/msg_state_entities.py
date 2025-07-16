"""Graph state management for the Arklex framework.

This module defines the data structures and models for managing the state of conversation
graphs and task execution flows. It includes classes for representing bot configurations,
message states, task status, and various components of the conversation graph.
The module provides comprehensive state management for tracking conversation progress,
resource records, and orchestrator responses throughout the system's operation.

The module is organized into several key components:
1. Bot Configuration: Classes for managing bot and LLM settings
2. Message Handling: Classes for processing and managing conversation messages
3. Task Status: Classes for tracking task execution and timing
4. Graph Structure: Classes for managing the conversation graph and node states
5. Memory Management: Classes for maintaining conversation history and trajectories

Key Features:
- Comprehensive state management for conversation flows
- Support for multiple bot configurations and LLM providers
- Flexible message processing and handling
- Task execution tracking and timing
- Graph-based conversation structure
- Memory management for conversation history
"""

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from arklex.memory.entities.memory_entities import ResourceRecord
from arklex.orchestrator.NLU.entities.slot_entities import Slot
from arklex.types import LLMConfig, StreamType


# Bot-related classes
class BotConfig(BaseModel):
    """Configuration for bot settings.

    This class defines the overall configuration for a bot instance, including its
    identity, version, language settings, and language model configuration.

    The class provides:
    1. Bot identity and versioning
    2. Language and type settings
    3. LLM configuration integration
    4. Type-safe configuration management

    Attributes:
        bot_id (str): Unique identifier for the bot.
        version (str): Version number of the bot.
        language (str): Primary language of the bot.
        bot_type (str): Type or category of the bot.
        llm_config (LLMConfig): Language model configuration for the bot.
    """

    bot_id: str
    version: str
    language: str
    bot_type: str
    llm_config: LLMConfig


### Message-related classes
class ConvoMessage(BaseModel):
    """Represents a conversation message with history.

    This class encapsulates a single message in a conversation, including both
    the current message content and the conversation history leading up to it.

    The class provides:
    1. Message content management
    2. Conversation history tracking
    3. Type-safe message handling

    Attributes:
        history (str): Previous conversation history or its summarization.
        message (str): The current message content.
    """

    history: str  # it could be the whole original message or the summarization of the previous conversation from memory module
    message: str


class OrchestratorMessage(BaseModel):
    """Message processed by the orchestrator.

    This class represents a message that has been processed by the orchestrator,
    including both the message content and any additional attributes or metadata
    that were added during processing.

    The class provides:
    1. Message content management
    2. Attribute/metadata storage
    3. Type-safe message handling

    Attributes:
        message (str): The message content.
        attribute (dict): Additional attributes associated with the message.
    """

    message: str
    attribute: dict[str, Any]


### Task status-related classes
class StatusEnum(str, Enum):
    """Enumeration of possible task statuses.

    This enum defines the possible states that a task can be in during its execution.
    It's used throughout the system to track the progress and state of various operations.

    The enum provides:
    1. Clear status definitions
    2. Type-safe status handling
    3. Consistent status tracking

    Values:
        COMPLETE: Task has been completed successfully.
        INCOMPLETE: Task is not yet complete.
        STAY: Task should remain in its current state.
    """

    COMPLETE: str = "complete"
    INCOMPLETE: str = "incomplete"
    STAY: str = "stay"


class Timing(BaseModel):
    """Timing information for task execution.

    This class tracks timing information for various aspects of task execution,
    particularly focusing on task graph processing times.

    The class provides:
    1. Task timing tracking
    2. Graph processing time measurement
    3. Type-safe timing management

    Attributes:
        taskgraph (Optional[float]): Time taken for task graph processing.
    """

    taskgraph: float | None = None


class Metadata(BaseModel):
    """Metadata for tracking conversation and execution.

    This class maintains metadata about the conversation session, including
    identifiers, turn tracking, and timing information.

    The class provides:
    1. Session identification
    2. Turn tracking
    3. Timing information
    4. Human-in-the-loop status
    5. Type-safe metadata management

    Attributes:
        chat_id (str): Unique identifier for the chat session.
        turn_id (int): Current turn number in the conversation.
        hitl (Optional[str]): Human-in-the-loop intervention status.
        timing (Timing): Timing information for the session.
        attempts (Optional[int]): Number of attempts for HITL MC logic.
    """

    # TODO: May need to initialize the metadata(i.e. chat_id, turn_id) based on the conversation database
    chat_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    turn_id: int = 0
    hitl: str | None = Field(default=None)
    timing: Timing = Field(default_factory=Timing)
    attempts: int | None = None


class OrchestratorResp(BaseModel):
    """Response from the orchestrator.

    This class represents the response generated by the orchestrator,
    including the answer, parameters, and any additional information
    needed for the conversation flow.

    The class provides:
    1. Answer management
    2. Parameter handling
    3. Human-in-the-loop support
    4. Choice list management
    5. Type-safe response handling

    Attributes:
        answer (str): The response answer.
        parameters (Dict[str, Any]): Additional parameters.
        human_in_the_loop (Optional[str]): Human intervention status.
        choice_list (Optional[List[str]]): List of available choices.
    """

    answer: str = Field(default="")
    parameters: dict[str, Any] = Field(default_factory=dict)
    human_in_the_loop: str | None = Field(default=None)
    choice_list: list[str] | None = Field(default=[])


class HTTPParams(BaseModel):
    """Parameters for HTTP requests.

    This class defines the parameters needed for making HTTP requests,
    including endpoint, method, headers, and body data.

    The class provides:
    1. Endpoint management
    2. HTTP method handling
    3. Header management
    4. Body data handling
    5. URL parameter support
    6. Type-safe HTTP parameter management

    Attributes:
        endpoint (str): The API endpoint URL.
        method (str): HTTP method to use.
        headers (Dict[str, str]): HTTP headers.
        body (Optional[Any]): Request body.
        params (Optional[Dict[str, Any]]): URL parameters.
    """

    endpoint: str
    method: str = Field(default="GET")
    headers: dict[str, str] = Field(
        default_factory=lambda: {"Content-Type": "application/json"}
    )
    body: Any | None = Field(default=None)
    params: dict[str, Any] | None = Field(default=None)


class SlotDetail(BaseModel):
    """Detailed slot information for dialogue state management.

    This class provides detailed information about a slot, including its verification
    status and confirmed values. It's used in database operations and slot verification
    processes.

    The class provides:
    1. Slot name and type management
    2. Value storage and verification
    3. Confirmation status tracking
    4. Description and prompt management
    5. Type-safe slot detail handling

    Attributes:
        name (str): Name of the slot.
        type (str): Type of the slot (e.g., 'string', 'date', 'time').
        value (str): Current value of the slot.
        description (str): Description of what the slot represents.
        prompt (str): Prompt used to collect the slot value.
        verified_value (str): Verified value from the database.
        confirmed (bool): Whether the slot value has been confirmed.
    """

    name: str
    type: str
    value: str = ""
    description: str = ""
    prompt: str = ""
    verified_value: str = ""
    confirmed: bool = False


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
