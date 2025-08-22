from pydantic import BaseModel

from arklex.env.workers.base.entities import WorkerOutput
from arklex.orchestrator.entities.orchestrator_state_entities import StatusEnum


class HitlWorkerData(BaseModel):
    """Data for the HITL worker."""


class HitlWorkerOutput(WorkerOutput):
    """Output for the HITL worker."""

    response: str
    status: StatusEnum
