from typing import Any

from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
    StatusEnum,
)
from arklex.resources.workers.base.base_worker import BaseWorker
from arklex.resources.workers.multiple_choice.entities import (
    MultipleChoiceWorkerData,
    MultipleChoiceWorkerOutput,
)


class MultipleChoiceWorker(BaseWorker):
    description: str = (
        "The worker that used to deliver the multiple choice options to the user."
    )

    def __init__(self) -> None:
        super().__init__()

    def init_worker_data(
        self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
    ) -> None:
        self.orch_state = orch_state
        self.multiple_choice_worker_data = MultipleChoiceWorkerData(
            **node_specific_data
        )

    def _execute(self) -> MultipleChoiceWorkerOutput:
        return MultipleChoiceWorkerOutput(
            response=self.multiple_choice_worker_data.question,
            choice_list=self.multiple_choice_worker_data.choices,
            status=StatusEnum.COMPLETE,
        )
