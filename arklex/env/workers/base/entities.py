from arklex.orchestrator.entities.orch_state_entities import (
    OrchestratorState,
    StatusEnum,
)


class WorkerData(OrchestratorState):
    """Base class for worker data."""


class WorkerResp(OrchestratorState):
    """Base class for worker response."""

    response: str
    status: StatusEnum
