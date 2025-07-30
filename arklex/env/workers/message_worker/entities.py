from arklex.env.workers.base.entities import WorkerData, WorkerResp
from arklex.orchestrator.entities.orch_state_entities import OrchestratorState


class MessageWorkerData(WorkerData):
    """Data for the message worker."""

    orch_state: OrchestratorState
    message_flow: str
    node_message: str
    directed: bool


class MessageWorkerResp(WorkerResp):
    """Response for the message worker."""
