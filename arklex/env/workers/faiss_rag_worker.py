"""FAISS RAG worker implementation for the Arklex framework.

This module provides a specialized worker for handling Retrieval-Augmented Generation (RAG)
tasks using FAISS for efficient similarity search. It implements a worker that can answer
user questions based on internal documentation, including policies, FAQs, and product
information. The worker supports both streaming and non-streaming responses, using a state
graph to manage the workflow of document retrieval and response generation.
"""

import logging
from typing import Any, Dict

from langgraph.graph import StateGraph, START

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState
from arklex.env.tools.utils import ToolGenerator
from arklex.env.tools.RAG.retrievers.faiss_retriever import RetrieveEngine


logger = logging.getLogger(__name__)


@register_worker
class FaissRAGWorker(BaseWorker):
    description: str = "Answer the user's questions based on the company's internal documentations (unstructured text data), such as the policies, FAQs, and product information"

    def __init__(
        self,
        # stream_ reponse is a boolean value that determines whether the response should be streamed or not.
        # i.e in the case of RagMessageWorker it should be set to false.
        stream_response: bool = True,
    ) -> None:
        super().__init__()
        self.action_graph: StateGraph = self._create_action_graph()
        self.stream_response: bool = stream_response

    def choose_tool_generator(self, state: MessageState) -> str:
        if self.stream_response and state.is_stream:
            return "stream_tool_generator"
        return "tool_generator"

    def _create_action_graph(self) -> StateGraph:
        workflow: StateGraph = StateGraph(MessageState)
        # Add nodes for each worker
        workflow.add_node("retriever", RetrieveEngine.faiss_retrieve)
        workflow.add_node("tool_generator", ToolGenerator.context_generate)
        workflow.add_node(
            "stream_tool_generator", ToolGenerator.stream_context_generate
        )

        # Add edges
        workflow.add_edge(START, "retriever")
        workflow.add_conditional_edges("retriever", self.choose_tool_generator)
        return workflow

    def _execute(self, msg_state: MessageState, **kwargs: Any) -> Dict[str, Any]:
        graph = self.action_graph.compile()
        result: Dict[str, Any] = graph.invoke(msg_state)
        return result
