from typing import Any

from pydantic import BaseModel

from arklex.env.workers.base.entities import WorkerOutput
from arklex.orchestrator.entities.orchestrator_state_entities import OrchestratorState


class RAGMessageWorkerData(BaseModel):
    """Data for the RAG message worker."""

    orch_state: OrchestratorState
    message: str
    tags: dict[str, Any]


class RAGMessageWorkerOutput(WorkerOutput):
    """Output for the RAG message worker."""
