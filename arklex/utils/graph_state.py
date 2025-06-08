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
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from arklex.types import StreamType
from arklex.utils.slot import Slot


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
    attribute: Dict[str, Any]



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

    taskgraph: Optional[float] = None


class ResourceRecord(BaseModel):
    """Record of resource usage and processing.

    This class maintains a record of how resources are used during task execution,
    including input/output data, processing steps, and intent information.

    The class provides:
    1. Resource usage tracking
    2. Input/output data management
    3. Processing step recording
    4. Intent tracking
    5. Type-safe resource management

    Attributes:
        info (Dict[str, Any]): General information about the resource.
        intent (str): The intent associated with the resource.
        input (List[Any]): Input data for the resource.
        output (str): Output data from the resource.
        steps (List[Any]): Processing steps taken.
        personalized_intent (str): User-specific intent for the resource.
    """

    info: Dict[str, Any]
    intent: str = Field(default="")
    input: List[Any] = Field(default_factory=list)
    output: str = Field(default="")
    steps: List[Any] = Field(default_factory=list)
    personalized_intent: str = Field(default="")


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
    """

    # TODO: May need to initialize the metadata(i.e. chat_id, turn_id) based on the conversation database
    chat_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    turn_id: int = 0
    hitl: Optional[str] = Field(default=None)
    timing: Timing = Field(default_factory=Timing)


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
    bot_config: Optional[BotConfig] = Field(default=None)
    # input message
    user_message: Optional[ConvoMessage] = Field(default=None)
    orchestrator_message: Optional[OrchestratorMessage] = Field(default=None)
    # action trajectory
    function_calling_trajectory: Optional[List[Dict[str, Any]]] = Field(default=None)
    trajectory: Optional[List[List[ResourceRecord]]] = Field(default=None)
    # message flow between different nodes
    message_flow: str = Field(
        description="message flow between different nodes", default=""
    )
    # final response
    response: str = Field(default="")
    # task-related params
    status: StatusEnum = Field(default=StatusEnum.INCOMPLETE)
    slots: Optional[Dict[str, List[Slot]]] = Field(
        description="record the dialogue states of each action", default=None
    )
    metadata: Optional[Metadata] = Field(default=None)
    # stream
    is_stream: bool = Field(default=False)
    message_queue: Any = Field(exclude=True, default=None)
    stream_type: (Optional[StreamType]) = Field(default="")
    # memory records
    relevant_records: Optional[List[ResourceRecord]] = Field(default=None)


class PathNode(BaseModel):
    """Node in the processing path.

    This class represents a node in the processing path of the conversation graph,
    tracking its state and relationships with other nodes.

    The class provides:
    1. Node identification
    2. State tracking
    3. Flow stack management
    4. Nested graph handling
    5. Intent tracking
    6. Type-safe node management

    Attributes:
        node_id (str): Unique identifier for the node.
        is_skipped (bool): Whether the node was skipped.
        in_flow_stack (bool): Whether the node is in the flow stack.
        nested_graph_node_value (Optional[str]): Value for nested graph nodes.
        nested_graph_leaf_jump (Optional[int]): Jump value for nested graph leaves.
        global_intent (str): Global intent associated with the node.
    """

    node_id: str
    is_skipped: bool = False
    in_flow_stack: bool = False
    nested_graph_node_value: Optional[str] = None
    nested_graph_leaf_jump: Optional[int] = None
    global_intent: str = Field(default="")


class Taskgraph(BaseModel):
    """Graph structure for task processing.

    This class manages the overall structure of the task processing graph,
    including node states, paths, and dialogue states.

    The class provides:
    1. Dialogue state management
    2. Path tracking
    3. Node status management
    4. Intent handling
    5. Node limit enforcement
    6. NLU record tracking
    7. Type-safe graph management

    Attributes:
        dialog_states (Dict[str, List[Slot]]): States of dialogue slots.
        path (List[PathNode]): Processing path nodes.
        curr_node (str): Current node identifier.
        intent (str): Current intent.
        curr_global_intent (str): Current global intent.
        node_limit (Dict[str, int]): Limits for node processing.
        nlu_records (List[Any]): Natural language understanding records.
        node_status (Dict[str, StatusEnum]): Status of each node.
        available_global_intents (List[str]): List of available global intents.
    """

    # Need add global intent
    dialog_states: Dict[str, List[Slot]] = Field(default_factory=dict)
    path: List[PathNode] = Field(default_factory=list)
    curr_node: str = Field(default="")
    intent: str = Field(default="")
    curr_global_intent: str = Field(default="")
    node_limit: Dict[str, int] = Field(default_factory=dict)
    nlu_records: List[Any] = Field(default_factory=list)
    node_status: Dict[str, StatusEnum] = Field(default_factory=dict)
    available_global_intents: List[str] = Field(default_factory=list)


class Memory(BaseModel):
    """Memory management for conversation and processing.

    This class maintains the memory of the conversation system, including
    processing trajectories and function call history.

    The class provides:
    1. Trajectory tracking
    2. Function call history
    3. Type-safe memory management

    Attributes:
        trajectory (List[List[ResourceRecord]]): Processing trajectory history.
        function_calling_trajectory (List[Dict[str, Any]]): Function call history.
    """

    trajectory: List[List[ResourceRecord]] = Field(default_factory=list)
    function_calling_trajectory: List[Dict[str, Any]] = Field(
        default_factory=list)


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


class NodeTypeEnum(str, Enum):
    """Enumeration of node types.

    This enum defines the different types of nodes that can exist in the
    conversation graph, each with specific processing requirements.

    The enum provides:
    1. Clear node type definitions
    2. Type-safe node type handling
    3. Consistent node type tracking

    Values:
        NONE: No specific type.
        START: Starting node.
        MULTIPLE_CHOICE: Node with multiple choice options.
    """

    NONE: str = ""
    START: str = "start"
    MULTIPLE_CHOICE: str = "multiple_choice"


class NodeInfo(BaseModel):
    """Information about a processing node.

    This class contains detailed information about a node in the processing graph,
    including its type, resource information, and processing attributes.

    The class provides:
    1. Node identification
    2. Type management
    3. Resource tracking
    4. Processing attributes
    5. Type-safe node information management

    Attributes:
        node_id (Optional[str]): Unique identifier for the node.
        type (str): Type of the node.
        resource_id (str): Resource identifier.
        resource_name (str): Name of the resource.
        can_skipped (bool): Whether the node can be skipped.
        is_leaf (bool): Whether the node is a leaf node.
        attributes (Dict[str, Any]): Additional node attributes.
        add_flow_stack (Optional[bool]): Whether to add to flow stack.
        additional_args (Optional[Dict[str, Any]]): Additional arguments for the node.
    """

    node_id: Optional[str] = Field(default=None)
    type: str = Field(default="")
    resource_id: str = Field(default="")
    resource_name: str = Field(default="")
    can_skipped: bool = Field(default=False)
    is_leaf: bool = Field(default=False)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    add_flow_stack: Optional[bool] = Field(default=False)
    additional_args: Optional[Dict[str, Any]] = Field(default={})


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
    parameters: Dict[str, Any] = Field(default_factory=dict)
    human_in_the_loop: Optional[str] = Field(default=None)
    choice_list: Optional[List[str]] = Field(default=[])


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
    headers: Dict[str, str] = Field(
        default_factory=lambda: {"Content-Type": "application/json"}
    )
    body: Optional[Any] = Field(default=None)
    params: Optional[Dict[str, Any]] = Field(default=None)
