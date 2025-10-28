from typing import Any

from pydantic import BaseModel, Field

from arklex.orchestrator.entities.orchestrator_state_entities import StatusEnum
from arklex.orchestrator.nlu.entities.slot_entities import Slot


class NodeInfo(BaseModel):
    node_id: str = Field(default="")
    resource: dict[str, str] = Field(default_factory=dict)
    attribute: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    is_leaf: bool = Field(default=False)


class NLUGraphParams(BaseModel):
    curr_node: str = Field(default="")
    curr_global_intent: str = Field(default="")
    intent: str = Field(default="")
    available_global_intents: list[str] = Field(default_factory=list)
    node_status: dict[str, StatusEnum] = Field(default_factory=dict)
    dialog_states: dict[str, list[Slot]] = Field(default_factory=dict)
    nlu_records: list[Any] = Field(default_factory=list)


class AgentGraphParams(BaseModel):
    current_agent: str = Field(default="")
