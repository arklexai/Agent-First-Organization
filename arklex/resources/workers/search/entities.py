from pydantic import BaseModel

from arklex.orchestrator.entities.orchestrator_state_entities import StatusEnum
from arklex.resources.workers.base.entities import WorkerOutput


class SearchWorkerData(BaseModel):
    """Data for the search worker."""


class SearchWorkerOutput(WorkerOutput):
    """Response for the search worker."""

    response: str
    status: StatusEnum
