from pydantic import BaseModel

from arklex.env.workers.base.entities import WorkerOutput
from arklex.orchestrator.entities.orchestrator_state_entities import StatusEnum


class MultipleChoiceWorkerData(BaseModel):
    question: str
    choice_list: list[str]


class MultipleChoiceWorkerOutput(WorkerOutput):
    choice_list: list[str]
    response: str
    status: StatusEnum
