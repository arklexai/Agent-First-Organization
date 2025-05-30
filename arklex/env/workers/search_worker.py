"""Search worker implementation for the Arklex framework.

This module provides a specialized worker for handling search-related tasks. It implements
a search engine worker that can answer user questions based on real-time online search
results. The worker uses a state graph to manage the workflow of search operations and
response generation, integrating with the framework's tool generation system.
"""

import logging
from typing import Any, Dict

from langgraph.graph import StateGraph, START


from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState
from arklex.env.tools.utils import ToolGenerator
from arklex.env.tools.RAG.search import SearchEngine


logger = logging.getLogger(__name__)


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

    def _execute(self, msg_state: MessageState, **kwargs: Any) -> Dict[str, Any]:
        graph = self.action_graph.compile()
        result: Dict[str, Any] = graph.invoke(msg_state)
        return result
