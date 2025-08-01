from pydantic import BaseModel

from arklex.orchestrator.entities.orchestrator_state_entities import StatusEnum
from arklex.orchestrator.NLU.core.slot import Slot


class NodeResponse(BaseModel):
    """Response for a node."""

    status: StatusEnum
    response: str | None = None
    choice_list: list[str] | None = None
    slots: dict[str, list[Slot]] | None = None
