from pydantic import BaseModel

from arklex.env.workers.base.entities import WorkerOutput
from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
    StatusEnum,
)


class MultipleChoiceWorkerData(BaseModel):
    orch_state: OrchestratorState
    question: str
    choice_list: list[str]


class MultipleChoiceWorkerOutput(WorkerOutput):
    response: str
    choice_list: list[str]
    status: StatusEnum
