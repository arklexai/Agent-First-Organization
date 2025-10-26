from typing import Any

from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
    StatusEnum,
)
from arklex.resources.workers.base.base_worker import BaseWorker
from arklex.resources.workers.hitl.entities import HitlWorkerData, HitlWorkerOutput
from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)


class HITLWorker(BaseWorker):
    """Production chat worker with flag-based state management.

    This worker is designed to start live chat with another built server.
    It uses flag-based state management to track conversation flow and
    returns indicators of what type of human help is needed.

    Attributes:
        description: Description of the worker functionality
        mode: Interaction mode set to "chat"
    """

    description: str = "Human in the loop worker"
    mode: str = "chat"

    def __init__(self) -> None:
        super().__init__()

    def init_worker_data(
        self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
    ) -> None:
        self.orch_state = orch_state
        self.hitl_worker_data: HitlWorkerData = HitlWorkerData(**node_specific_data)

    def _execute(self) -> HitlWorkerOutput:
        """Execute the chat worker with flag-based state management.

        This method manages the conversation flow using metadata flags to
        track the state of the HITL interaction.

        Returns:
            HitlWorkerOutput: Output of the HITL worker
        """
        if not self.orch_state.metadata.hitl:
            self.orch_state.metadata.hitl = "live"
            return HitlWorkerOutput(
                status=StatusEnum.STAY,
                response="I'll connect you to a representative!",
            )
        else:
            self.orch_state.message_flow = "Live chat completed"
            self.orch_state.metadata.hitl = None
            return HitlWorkerOutput(
                status=StatusEnum.COMPLETE,
                response="",
            )
