from pydantic import BaseModel

from arklex.env.workers.base.entities import WorkerOutput
from arklex.orchestrator.entities.orchestrator_state_entities import OrchestratorState


class SearchWorkerData(BaseModel):
    """Data for the search worker."""

    orch_state: OrchestratorState


class SearchWorkerOutput(WorkerOutput):
    """Response for the search worker."""
