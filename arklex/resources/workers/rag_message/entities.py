from typing import Any

from pydantic import BaseModel

from arklex.orchestrator.entities.orchestrator_state_entities import StatusEnum
from arklex.resources.workers.base.entities import WorkerOutput


class RAGMessageWorkerData(BaseModel):
    """Data for the RAG message worker."""

    message: str
    bot_id: str
    version: str
    collection_name: str
    tags: dict[str, Any]


class RAGMessageWorkerOutput(WorkerOutput):
    """Output for the RAG message worker."""

    response: str
    status: StatusEnum
