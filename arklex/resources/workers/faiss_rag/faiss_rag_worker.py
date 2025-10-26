"""FAISS RAG worker implementation for the Arklex framework.

This module provides a specialized worker for handling Retrieval-Augmented Generation (RAG)
tasks using FAISS for efficient similarity search. It implements a worker that can answer
user questions based on internal documentation, including policies, FAQs, and product
information. The worker supports both streaming and non-streaming responses, using a state
graph to manage the workflow of document retrieval and response generation.
"""

from typing import Any

from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
    StatusEnum,
)
from arklex.orchestrator.types.stream_types import StreamType
from arklex.resources.tools.rag.retrievers.faiss_retriever import RetrieveEngine
from arklex.resources.tools.utils import ToolGenerator
from arklex.resources.workers.base.base_worker import BaseWorker
from arklex.resources.workers.faiss_rag.entities import (
    FaissRAGWorkerData,
    FaissRAGWorkerOutput,
)
from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)


class FaissRAGWorker(BaseWorker):
    description: str = "Answer the user's questions based on the company's internal documentations (unstructured text data), such as the policies, FAQs, and product information"

    def __init__(self) -> None:
        super().__init__()

    def init_worker_data(
        self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
    ) -> None:
        self.orch_state = orch_state
        self.faiss_rag_worker_data: FaissRAGWorkerData = FaissRAGWorkerData(
            **node_specific_data,
        )

    def _execute(self) -> FaissRAGWorkerOutput:
        retrieved_text = RetrieveEngine.faiss_retrieve(self.orch_state)
        self.orch_state.message_flow = retrieved_text
        if self.orch_state.stream_type != StreamType.NON_STREAM:
            response = ToolGenerator.stream_context_generate(self.orch_state)
        else:
            response = ToolGenerator.context_generate(self.orch_state)

        return FaissRAGWorkerOutput(
            response=response,
            status=StatusEnum.COMPLETE,
        )
