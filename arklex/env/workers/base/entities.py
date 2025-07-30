from abc import ABC

from pydantic import BaseModel

from arklex.orchestrator.entities.orchestrator_state_entities import StatusEnum


class WorkerOutput(ABC, BaseModel):
    """Base class for worker response."""

    response: str
    status: StatusEnum
