from pydantic import BaseModel

from arklex.env.workers.base.entities import WorkerOutput
from arklex.orchestrator.entities.orchestrator_state_entities import StatusEnum


class SearchWorkerData(BaseModel):
    """Data for the search worker."""


class SearchWorkerOutput(WorkerOutput):
    """Response for the search worker."""

    response: str
    status: StatusEnum
