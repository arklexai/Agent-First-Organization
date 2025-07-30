from pydantic import BaseModel

from arklex.env.workers.base.entities import WorkerOutput
from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
)


class MultipleChoiceWorkerData(BaseModel):
    orch_state: OrchestratorState
    question: str
    choice_list: list[str]


class MultipleChoiceWorkerOutput(WorkerOutput):
    choice_list: list[str]
