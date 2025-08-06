from abc import ABC

from pydantic import BaseModel

from arklex.orchestrator.entities.orch_state_entities import (
    OrchestratorState,
    StatusEnum,
)


class WorkerData(ABC, BaseModel):
    """Base class for worker data."""

    orch_state: OrchestratorState


class WorkerResp(ABC, BaseModel):
    """Base class for worker response."""

    response: str
    status: StatusEnum
