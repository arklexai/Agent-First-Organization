from pydantic import BaseModel

from arklex.env.workers.base.entities import WorkerOutput
from arklex.orchestrator.entities.orchestrator_state_entities import OrchestratorState


class MessageWorkerData(BaseModel):
    """Data for the message worker."""

    orch_state: OrchestratorState
    message: str
    directed: bool


class MessageWorkerOutput(WorkerOutput):
    """Response for the message worker."""
