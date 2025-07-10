"""Search worker implementation for the Arklex framework.

This module provides a specialized worker for handling search-related tasks. It implements
a search engine worker that can answer user questions based on real-time online search
results. The worker uses a state graph to manage the workflow of search operations and
response generation, integrating with the framework's tool generation system.
"""

from typing import Any, TypedDict

from langgraph.graph import START, StateGraph

from arklex.env.tools.RAG.search import SearchEngine
from arklex.env.tools.utils import ToolGenerator
from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.orchestrator.entities.msg_state_entities import MessageState
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class SearchWorkerKwargs(TypedDict, total=False):
    """Type definition for kwargs used in SearchWorker._execute method."""

    # Add specific worker parameters as needed
    pass


@register_worker
class SearchWorker(BaseWorker):
    description: str = (
        "Answer the user's questions based on real-time online search results"
    )

    def __init__(self) -> None:
        super().__init__()
        self.action_graph: StateGraph = self._create_action_graph()

    def _create_action_graph(self) -> StateGraph:
        workflow: StateGraph = StateGraph(MessageState)
        # Add nodes for each worker
        search_engine: SearchEngine = SearchEngine()
        workflow.add_node("search_engine", search_engine.search)
        workflow.add_node("tool_generator", ToolGenerator.context_generate)
        # Add edges
        workflow.add_edge(START, "search_engine")
        workflow.add_edge("search_engine", "tool_generator")
        return workflow

    def _execute(
        self, msg_state: MessageState, **kwargs: SearchWorkerKwargs
    ) -> dict[str, Any]:
        graph = self.action_graph.compile()
        result: dict[str, Any] = graph.invoke(msg_state)
        return result
