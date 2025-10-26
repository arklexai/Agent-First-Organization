from typing import Any

from pydantic import BaseModel

from arklex.orchestrator.entities.orchestrator_state_entities import StatusEnum
from arklex.resources.workers.base.entities import WorkerOutput


class FaissRAGWorkerData(BaseModel):
    """Data for the Faiss RAG worker."""

    tags: dict[str, Any]


class FaissRAGWorkerOutput(WorkerOutput):
    """Response for the Faiss RAG worker."""

    response: str
    status: StatusEnum
