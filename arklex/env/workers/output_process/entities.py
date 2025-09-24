"""Entities for the answer node worker."""

from pydantic import BaseModel

from arklex.env.workers.base.entities import WorkerOutput
from arklex.orchestrator.entities.orchestrator_state_entities import StatusEnum


class OutputProcessWorkerData(BaseModel):
    """Data for the answer node worker."""

    task: str = ""
    prompt: str = ""


class OutputProcessWorkerOutput(WorkerOutput):
    """Response for the answer node worker."""

    response: str
    status: StatusEnum
