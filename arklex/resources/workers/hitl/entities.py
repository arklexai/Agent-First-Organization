from pydantic import BaseModel

from arklex.orchestrator.entities.orchestrator_state_entities import StatusEnum
from arklex.resources.workers.base.entities import WorkerOutput


class HitlWorkerData(BaseModel):
    """Data for the HITL worker."""


class HitlWorkerOutput(WorkerOutput):
    """Output for the HITL worker."""

    response: str
    status: StatusEnum
