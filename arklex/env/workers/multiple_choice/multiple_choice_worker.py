from typing import Any

from arklex.env.workers.base.base_worker import BaseWorker
from arklex.env.workers.multiple_choice.entities import (
    MultipleChoiceWorkerData,
    MultipleChoiceWorkerOutput,
)
from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
    StatusEnum,
)


class MultipleChoiceWorker(BaseWorker):
    description: str = (
        "The worker that used to deliver the multiple choice options to the user."
    )

    def __init__(self) -> None:
        super().__init__()

    @property
    def worker_data(self) -> MultipleChoiceWorkerData:
        return self.multiple_choice_worker_data

    def init_worker_data(
        self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
    ) -> None:
        self.multiple_choice_worker_data: MultipleChoiceWorkerData = (
            MultipleChoiceWorkerData(
                orch_state=orch_state,
                **node_specific_data,
            )
        )

    def execute(self) -> MultipleChoiceWorkerOutput:
        return MultipleChoiceWorkerOutput(
            response=self.multiple_choice_worker_data.question,
            choice_list=self.multiple_choice_worker_data.choice_list,
            status=StatusEnum.COMPLETE,
        )
