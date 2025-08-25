from typing import Any

from pydantic import BaseModel

from arklex.env.workers.base.entities import WorkerOutput
from arklex.orchestrator.entities.orchestrator_state_entities import StatusEnum


class RAGMessageWorkerData(BaseModel):
    """Data for the RAG message worker."""

    message: str
    tags: dict[str, Any]


class RAGMessageWorkerOutput(WorkerOutput):
    """Output for the RAG message worker."""

    response: str
    status: StatusEnum
