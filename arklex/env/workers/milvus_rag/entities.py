from typing import Any

from arklex.env.workers.base.entities import WorkerData, WorkerResp


class MilvusRAGWorkerData(WorkerData):
    """Data for the Milvus RAG worker."""

    tags: dict[str, Any]


class MilvusRAGWorkerResp(WorkerResp):
    """Response for the Milvus RAG worker."""
