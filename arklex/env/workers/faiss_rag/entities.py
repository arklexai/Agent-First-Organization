from typing import Any

from pydantic import BaseModel

from arklex.env.workers.base.entities import WorkerOutput
from arklex.orchestrator.entities.orchestrator_state_entities import StatusEnum


class FaissRAGWorkerData(BaseModel):
    """Data for the Faiss RAG worker."""

    tags: dict[str, Any]


class FaissRAGWorkerOutput(WorkerOutput):
    """Response for the Faiss RAG worker."""

    response: str
    status: StatusEnum
