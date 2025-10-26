from pydantic import BaseModel

from arklex.orchestrator.entities.orchestrator_state_entities import StatusEnum
from arklex.resources.workers.base.entities import WorkerOutput


class MultipleChoiceWorkerData(BaseModel):
    question: str
    choices: list[str]


class MultipleChoiceWorkerOutput(WorkerOutput):
    choice_list: list[str]
    response: str
    status: StatusEnum
