from typing import Any

from arklex.env.workers.base.entities import WorkerOutput
from arklex.orchestrator.entities.orchestrator_state_entities import OrchestratorState


class FaissRAGWorkerData(OrchestratorState):
    """Data for the Faiss RAG worker."""

    tags: dict[str, Any]


class FaissRAGWorkerOutput(WorkerOutput):
    """Response for the Faiss RAG worker."""
