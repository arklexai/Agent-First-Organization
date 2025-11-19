from typing import Any

from pydantic import BaseModel

from arklex.orchestrator.entities.orchestrator_state_entities import (
    StatusEnum,
)
from arklex.resources.workers.base.entities import WorkerOutput


class MilvusRAGWorkerData(BaseModel):
    """Data for the Milvus RAG worker."""

    bot_id: str
    version: str
    collection_name: str
    tags: dict[str, Any]
    possible_tags: dict[str, list[str]] | None = None


class MilvusRAGWorkerOutput(WorkerOutput):
    """Response for the Milvus RAG worker."""

    response: str
    status: StatusEnum
