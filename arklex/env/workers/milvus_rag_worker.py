"""Milvus RAG worker implementation for the Arklex framework.

This module provides a specialized worker for handling Retrieval-Augmented Generation (RAG)
tasks using Milvus as the vector database. The MilvusRAGWorker class is responsible for
answering user questions based on internal documentation, supporting both streaming and
non-streaming responses. It integrates with Milvus for efficient similarity search and
retrieval of relevant documents.
"""

from functools import partial
from typing import Any, TypedDict

from langgraph.graph import START, StateGraph

from arklex.env.tools.RAG.retrievers.milvus_retriever import RetrieveEngine
from arklex.env.tools.utils import ToolGenerator
from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.orchestrator.entities.msg_state_entities import MessageState
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class MilvusRAGWorkerKwargs(TypedDict, total=False):
    """Type definition for kwargs used in MilvusRAGWorker._execute method."""

    tags: dict[str, Any]


@register_worker
class MilvusRAGWorker(BaseWorker):
    description: str = "Answer the user's questions based on the company's internal documentations (unstructured text data), such as the policies, FAQs, and product information"

    def __init__(
        self,
        # stream_ reponse is a boolean value that determines whether the response should be streamed or not.
        # i.e in the case of RagMessageWorker it should be set to false.
        stream_response: bool = True,
    ) -> None:
        super().__init__()
        self.stream_response: bool = stream_response
        self.tags: dict[str, Any] = {}
        self.action_graph: StateGraph | None = None

    def choose_tool_generator(self, state: MessageState) -> str:
        if self.stream_response and state.is_stream:
            return "stream_tool_generator"
        return "tool_generator"

    def _create_action_graph(self, tags: dict[str, Any]) -> StateGraph:
        workflow: StateGraph = StateGraph(MessageState)
        # Create a partial function with the extra argument bound
        retriever_with_args = partial(RetrieveEngine.milvus_retrieve, tags=tags)
        # Add nodes for each worker
        workflow.add_node("retriever", retriever_with_args)
        workflow.add_node("tool_generator", ToolGenerator.context_generate)
        workflow.add_node(
            "stream_tool_generator", ToolGenerator.stream_context_generate
        )
        # Add edges
        workflow.add_edge(START, "retriever")
        workflow.add_conditional_edges("retriever", self.choose_tool_generator)
        return workflow

    def _execute(
        self, msg_state: MessageState, **kwargs: MilvusRAGWorkerKwargs
    ) -> dict[str, Any]:
        self.tags = kwargs.get("tags", {})
        self.action_graph = self._create_action_graph(self.tags)
        graph = self.action_graph.compile()
        result: dict[str, Any] = graph.invoke(msg_state)
        return result
