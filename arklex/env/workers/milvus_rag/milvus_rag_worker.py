"""Milvus RAG worker implementation for the Arklex framework.

This module provides a specialized worker for handling Retrieval-Augmented Generation (RAG)
tasks using Milvus as the vector database. The MilvusRAGWorker class is responsible for
answering user questions based on internal documentation, supporting both streaming and
non-streaming responses. It integrates with Milvus for efficient similarity search and
retrieval of relevant documents.
"""

from arklex.env.tools.RAG.retrievers.milvus_retriever import RetrieveEngine
from arklex.env.tools.utils import ToolGenerator
from arklex.env.workers.base.base_worker import BaseWorker, register_worker
from arklex.env.workers.milvus_rag_worker.entities import (
    MilvusRAGWorkerData,
    MilvusRAGWorkerResp,
)
from arklex.orchestrator.entities.orch_state_entities import StatusEnum
from arklex.types.stream_types import StreamType
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


@register_worker
class MilvusRAGWorker(BaseWorker):
    description: str = "Answer the user's questions based on the company's internal documentations (unstructured text data), such as the policies, FAQs, and product information"

    def __init__(self) -> None:
        super().__init__()

    def init_worker_data(self, input_data: MilvusRAGWorkerData) -> None:
        self.milvus_rag_worker_data: MilvusRAGWorkerData = input_data

    def _execute(self) -> MilvusRAGWorkerResp:
        retrieved_text, retriever_params = RetrieveEngine.milvus_retrieve(
            self.milvus_rag_worker_data.user_message.history,
            self.milvus_rag_worker_data.bot_config,
            self.milvus_rag_worker_data.tags,
        )
        self.milvus_rag_worker_data.message_flow = retrieved_text
        # state = trace(input=retriever_params, state=state)
        if self.milvus_rag_worker_data.stream_type != StreamType.NON_STREAM:
            response = ToolGenerator.stream_context_generate(
                self.milvus_rag_worker_data
            )
        else:
            response = ToolGenerator.context_generate(self.milvus_rag_worker_data)
        return MilvusRAGWorkerResp(
            response=response,
            status=StatusEnum.COMPLETE,
        )
