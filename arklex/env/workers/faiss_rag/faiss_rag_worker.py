"""FAISS RAG worker implementation for the Arklex framework.

This module provides a specialized worker for handling Retrieval-Augmented Generation (RAG)
tasks using FAISS for efficient similarity search. It implements a worker that can answer
user questions based on internal documentation, including policies, FAQs, and product
information. The worker supports both streaming and non-streaming responses, using a state
graph to manage the workflow of document retrieval and response generation.
"""

from typing import TypedDict

from arklex.env.tools.RAG.retrievers.faiss_retriever import RetrieveEngine
from arklex.env.tools.utils import ToolGenerator
from arklex.env.workers.base.base_worker import BaseWorker, register_worker
from arklex.env.workers.faiss_rag.entities import FaissRAGWorkerData, FaissRAGWorkerResp
from arklex.orchestrator.entities.orch_state_entities import StatusEnum
from arklex.types.stream_types import StreamType
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class FaissRAGWorkerKwargs(TypedDict, total=False):
    """Type definition for kwargs used in FaissRAGWorker._execute method."""

    # Add specific worker parameters as needed
    pass


@register_worker
class FaissRAGWorker(BaseWorker):
    description: str = "Answer the user's questions based on the company's internal documentations (unstructured text data), such as the policies, FAQs, and product information"

    def __init__(self) -> None:
        super().__init__()

    def init_worker_data(self, input_data: FaissRAGWorkerData) -> None:
        self.faiss_rag_worker_data: FaissRAGWorkerData = input_data

    def _execute(self) -> FaissRAGWorkerResp:
        retrieved_text = RetrieveEngine.faiss_retrieve(self.faiss_rag_worker_data)
        self.faiss_rag_worker_data.message_flow = retrieved_text
        if self.faiss_rag_worker_data.stream_type != StreamType.NON_STREAM:
            response = ToolGenerator.stream_context_generate(self.faiss_rag_worker_data)
        else:
            response = ToolGenerator.context_generate(self.faiss_rag_worker_data)

        return FaissRAGWorkerResp(
            response=response,
            status=StatusEnum.COMPLETE,
        )
