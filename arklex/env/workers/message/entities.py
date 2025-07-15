from arklex.env.workers.base.entities import WorkerData, WorkerResp


class MessageWorkerData(WorkerData):
    """Data for the message worker."""

    message_flow: str
    node_message: str
    directed: bool


class MessageWorkerResp(WorkerResp):
    """Response for the message worker."""
