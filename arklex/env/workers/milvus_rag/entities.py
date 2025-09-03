from typing import Any

from pydantic import BaseModel

from arklex.env.workers.base.entities import WorkerOutput
from arklex.orchestrator.entities.orchestrator_state_entities import (
    StatusEnum,
)


class MilvusRAGWorkerData(BaseModel):
    """Data for the Milvus RAG worker."""

    tags: dict[str, Any]


class MilvusRAGWorkerOutput(WorkerOutput):
    """Response for the Milvus RAG worker."""

    response: str
    status: StatusEnum
