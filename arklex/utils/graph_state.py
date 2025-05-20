from typing import Any, Optional, List, Dict
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from arklex.utils.slot import Slot


### Bot-related classes
class LLMConfig(BaseModel):
    """
    Configuration for the Language Model.

    Attributes:
        model_type_or_path (str): The model type or path to be used.
        llm_provider (str): The provider of the language model.
    """

    model_type_or_path: str
    llm_provider: str


class BotConfig(BaseModel):
    """
    Configuration for the Bot.

    Attributes:
        bot_id (str): Unique identifier for the bot.
        version (str): Version of the bot.
        language (str): Language the bot operates in.
        bot_type (str): Type of the bot.
        llm_config (LLMConfig): Configuration for the language model.
    """

    bot_id: str
    version: str
    language: str
    bot_type: str
    llm_config: LLMConfig


### Message-related classes


class ConvoMessage(BaseModel):
    """
    Represents a conversation message.

    Attributes:
        history (str): The whole original message or summarization of previous conversation.
        message (str): The current message content.
    """

    history: str  # it could be the whole original message or the summarization of the previous conversation from memory module
    message: str


class OrchestratorMessage(BaseModel):
    """
    Message from the orchestrator.

    Attributes:
        message (str): The message content.
        attribute (Dict[str, Any]): Additional attributes for the message.
    """

    message: str
    attribute: dict


### Task status-related classes


class StatusEnum(str, Enum):
    """
    Enum for task status.

    Attributes:
        COMPLETE (str): Task is complete.
        INCOMPLETE (str): Task is incomplete.
        STAY (str): Task should stay in current state.
    """

    COMPLETE: str = "complete"
    INCOMPLETE: str = "incomplete"
    STAY: str = "stay"


class Timing(BaseModel):
    """
    Timing information for tasks.

    Attributes:
        taskgraph (Optional[float]): Time taken for task graph processing.
    """

    taskgraph: Optional[float] = None


class ResourceRecord(BaseModel):
    """
    Record of a resource with its associated information.

    Attributes:
        info (Dict[str, Any]): General information about the resource.
        intent (str): Intent associated with the resource.
        input (List[Any]): Input parameters for the resource.
        output (str): Output from the resource.
        steps (List[Any]): Steps involved in processing the resource.
        personalized_intent (str): Personalized intent for the resource.
    """

    info: Dict[str, Any]
    intent: str = Field(default="")
    input: List[Any] = Field(default_factory=list)
    output: str = Field(default="")
    steps: List[Any] = Field(default_factory=list)
    personalized_intent: str = Field(default="")


class Metadata(BaseModel):
    """
    Metadata for tracking conversation state.

    Attributes:
        chat_id (str): Unique identifier for the chat session.
        turn_id (int): Turn number in the conversation.
        hitl (Optional[str]): Human in the loop indicator.
        timing (Timing): Timing information for the conversation.
    """

    # TODO: May need to initialize the metadata(i.e. chat_id, turn_id) based on the conversation database
    chat_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    turn_id: int = 0
    hitl: Optional[str] = Field(default=None)
    timing: Timing = Field(default_factory=Timing)


class MessageState(BaseModel):
    """
    State of a message in the conversation.

    Attributes:
        sys_instruct (str): System instructions.
        bot_config (BotConfig): Bot configuration.
        user_message (ConvoMessage): User's message.
        orchestrator_message (OrchestratorMessage): Orchestrator's message.
        function_calling_trajectory (List[Dict[str, Any]]): History of function calls.
        trajectory (List[List[ResourceRecord]]): History of resource records.
        message_flow (str): Flow of messages between nodes.
        response (str): Final response.
        status (StatusEnum): Current status.
        slots (Dict[str, List[Slot]]): Dialogue states for each action.
        metadata (Metadata): Conversation metadata.
        is_stream (bool): Whether the message is being streamed.
        message_queue (Any): Queue for streamed messages.
        relevant_records (Optional[List[ResourceRecord]]): Relevant memory records.
    """

    # system configuration
    sys_instruct: str = Field(default="")
    # bot configuration
    bot_config: BotConfig = Field(default=None)
    # input message
    user_message: ConvoMessage = Field(default=None)
    orchestrator_message: OrchestratorMessage = Field(default=None)
    # action trajectory
    function_calling_trajectory: List[Dict[str, Any]] = Field(default=None)
    trajectory: List[List[ResourceRecord]] = Field(default=None)
    # message flow between different nodes
    message_flow: str = Field(
        description="message flow between different nodes", default=""
    )
    # final response
    response: str = Field(default="")
    # task-related params
    status: StatusEnum = Field(default=StatusEnum.INCOMPLETE)
    slots: Dict[str, List[Slot]] = Field(
        description="record the dialogue states of each action", default=None
    )
    metadata: Metadata = Field(default=None)
    # stream
    is_stream: bool = Field(default=False)
    message_queue: Any = Field(exclude=True, default=None)
    # memory records
    relevant_records: Optional[List[ResourceRecord]] = Field(default=None)


class PathNode(BaseModel):
    """
    Represents a node in the execution path.

    Attributes:
        node_id (str): Unique identifier for the node.
        is_skipped (bool): Whether the node was skipped.
        in_flow_stack (bool): Whether the node is in the flow stack.
        nested_graph_node_value (Optional[str]): Value for nested graph node.
        nested_graph_leaf_jump (Optional[int]): Jump value for nested graph leaf.
        global_intent (str): Global intent associated with the node.
    """

    node_id: str
    is_skipped: bool = False
    in_flow_stack: bool = False
    nested_graph_node_value: Optional[str] = None
    nested_graph_leaf_jump: Optional[int] = None
    global_intent: str = Field(default="")


class Taskgraph(BaseModel):
    """
    Represents a task graph structure.

    Attributes:
        dialog_states (Dict[str, List[Slot]]): States of dialogues.
        path (List[PathNode]): Path of nodes.
        curr_node (str): Current node identifier.
        intent (str): Current intent.
        curr_global_intent (str): Current global intent.
        node_limit (Dict[str, int]): Limits for nodes.
        nlu_records (List[Any]): NLU records.
        node_status (Dict[str, StatusEnum]): Status of nodes.
        available_global_intents (List[str]): Available global intents.
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
    """
    Represents memory state.

    Attributes:
        trajectory (List[List[ResourceRecord]]): History of resource records.
        function_calling_trajectory (List[Dict[str, Any]]): History of function calls.
    """

    trajectory: List[List[ResourceRecord]] = Field(default_factory=list)
    function_calling_trajectory: List[Dict[str, Any]] = Field(default_factory=list)


class Params(BaseModel):
    """
    Parameters for the conversation state.

    Attributes:
        metadata (Metadata): Conversation metadata.
        taskgraph (Taskgraph): Task graph structure.
        memory (Memory): Memory state.
    """

    metadata: Metadata = Field(default_factory=Metadata)
    taskgraph: Taskgraph = Field(default_factory=Taskgraph)
    memory: Memory = Field(default_factory=Memory)


class NodeTypeEnum(str, Enum):
    """
    Enum for node types.

    Attributes:
        NONE (str): No specific type.
        START (str): Start node.
        MULTIPLE_CHOICE (str): Multiple choice node.
    """

    NONE: str = ""
    START: str = "start"
    MULTIPLE_CHOICE: str = "multiple_choice"


class NodeInfo(BaseModel):
    """
    Information about a node.

    Attributes:
        node_id (Optional[str]): Unique identifier for the node.
        type (str): Type of the node.
        resource_id (str): Resource identifier.
        resource_name (str): Name of the resource.
        can_skipped (bool): Whether the node can be skipped.
        is_leaf (bool): Whether the node is a leaf node.
        attributes (Dict[str, Any]): Additional attributes.
        add_flow_stack (Optional[bool]): Whether to add to flow stack.
        additional_args (Optional[Dict[str, Any]]): Additional arguments.
    """

    node_id: Optional[str] = Field(default=None)
    type: str = Field(default="")
    resource_id: str = Field(default="")
    resource_name: str = Field(default="")
    can_skipped: bool = Field(default=False)
    is_leaf: bool = Field(default=False)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    add_flow_stack: Optional[bool] = Field(default=False)
    additional_args: Optional[dict] = Field(default={})


class OrchestratorResp(BaseModel):
    """
    Response from the orchestrator.

    Attributes:
        answer (str): The answer content.
        parameters (Dict[str, Any]): Additional parameters.
        human_in_the_loop (Optional[str]): Human in the loop indicator.
        choice_list (Optional[List[str]]): List of choices if applicable.
    """

    answer: str = Field(default="")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    human_in_the_loop: Optional[str] = Field(default=None)
    choice_list: Optional[List[str]] = Field(default=[])
