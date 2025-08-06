from pydantic import BaseModel

from arklex.env.workers.base.entities import WorkerOutput
from arklex.orchestrator.entities.orchestrator_state_entities import StatusEnum


class MessageWorkerData(BaseModel):
    """Data for the message worker."""

    message: str
    directed: bool


class MessageWorkerOutput(WorkerOutput):
    """Response for the message worker."""

    response: str
    status: StatusEnum
