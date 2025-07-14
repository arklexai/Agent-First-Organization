"""Entities for task graph management.

This module defines the entities used in the task graph management system,
including PathNode, NodeTypeEnum, NodeInfo, and Taskgraph.

Key Components:
- PathNode: Represents a node in the processing path of the conversation graph.
- NodeTypeEnum: Represents the type of node.
- NodeInfo: Represents the information about a node.
- Taskgraph: Represents the overall structure of the task processing graph.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from arklex.orchestrator.entities.msg_state_entities import StatusEnum
from arklex.orchestrator.NLU.entities.slot_entities import Slot


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

    node_id: str | None = Field(default=None)
    type: str = Field(default="")
    resource_id: str = Field(default="")
    resource_name: str = Field(default="")
    can_skipped: bool = Field(default=False)
    is_leaf: bool = Field(default=False)
    attributes: dict[str, Any] = Field(default_factory=dict)
    add_flow_stack: bool | None = Field(default=False)
    additional_args: dict[str, Any] | None = Field(default={})


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
    nested_graph_node_value: str | None = None
    nested_graph_leaf_jump: int | None = None
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
    dialog_states: dict[str, list[Slot]] = Field(default_factory=dict)
    path: list[PathNode] = Field(default_factory=list)
    curr_node: str = Field(default="")
    intent: str = Field(default="")
    curr_global_intent: str = Field(default="")
    node_limit: dict[str, int] = Field(default_factory=dict)
    nlu_records: list[Any] = Field(default_factory=list)
    node_status: dict[str, StatusEnum] = Field(default_factory=dict)
    available_global_intents: list[str] = Field(default_factory=list)
