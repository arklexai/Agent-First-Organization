"""Search worker implementation for the Arklex framework.

This module provides a specialized worker for handling search-related tasks. It implements
a search engine worker that can answer user questions based on real-time online search
results. The worker uses a state graph to manage the workflow of search operations and
response generation, integrating with the framework's tool generation system.
"""

from typing import Any

from arklex.env.tools.RAG.search import SearchEngine
from arklex.env.tools.utils import ToolGenerator
from arklex.env.workers.base.base_worker import BaseWorker
from arklex.env.workers.search.entities import SearchWorkerData, SearchWorkerOutput
from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
    StatusEnum,
)
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class SearchWorker(BaseWorker):
    description: str = (
        "Answer the user's questions based on real-time online search results"
    )

    def __init__(self) -> None:
        super().__init__()

    @property
    def worker_data(self) -> SearchWorkerData:
        return self.search_worker_data

    def init_worker_data(
        self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
    ) -> None:
        self.search_worker_data: SearchWorkerData = SearchWorkerData(
            orch_state=orch_state,
            **node_specific_data,
        )

    def _execute(self) -> SearchWorkerOutput:
        search_engine: SearchEngine = SearchEngine()
        retrieved_text = search_engine.search(
            chat_history=self.search_worker_data.orch_state.user_message.history,
            bot_config=self.search_worker_data.orch_state.bot_config,
        )
        self.search_worker_data.orch_state.message_flow = retrieved_text
        response = ToolGenerator.context_generate(self.search_worker_data.orch_state)
        return SearchWorkerOutput(
            response=response,
            status=StatusEnum.COMPLETE,
        )
