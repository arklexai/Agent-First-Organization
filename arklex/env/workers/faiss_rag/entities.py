from typing import Any

from arklex.env.workers.base.entities import WorkerData, WorkerResp


class FaissRAGWorkerData(WorkerData):
    """Data for the Faiss RAG worker."""

    tags: dict[str, Any]


class FaissRAGWorkerResp(WorkerResp):
    """Response for the Faiss RAG worker."""
