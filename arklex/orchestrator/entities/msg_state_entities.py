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


# Bot-related classes
class LLMConfig(BaseModel):
    """Configuration for language model settings.

    This class defines the configuration parameters for language models used in the system.
    It specifies which model to use and from which provider.

    The class provides:
    1. Model selection and configuration
    2. Provider specification
    3. Type-safe configuration management

    Attributes:
        model_type_or_path (str): The model identifier or path to use.
        llm_provider (str): The provider of the language model (e.g., 'openai', 'anthropic').
    """

    model_type_or_path: str
    llm_provider: str


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
