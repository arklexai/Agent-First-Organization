"""Search worker implementation for the Arklex framework.

This module provides a specialized worker for handling search-related tasks. It implements
a search engine worker that can answer user questions based on real-time online search
results. The worker uses a state graph to manage the workflow of search operations and
response generation, integrating with the framework's tool generation system.
"""

from typing import Any

from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
    StatusEnum,
)
from arklex.resources.tools.rag.search import SearchEngine
from arklex.resources.tools.utils import ToolGenerator
from arklex.resources.workers.base.base_worker import BaseWorker
from arklex.resources.workers.search.entities import (
    SearchWorkerData,
    SearchWorkerOutput,
)
from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)


class SearchWorker(BaseWorker):
    description: str = (
        "Answer the user's questions based on real-time online search results"
    )

    def __init__(self) -> None:
        super().__init__()

    def init_worker_data(
        self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
    ) -> None:
        self.orch_state = orch_state
        self.search_worker_data = SearchWorkerData(**node_specific_data)

    def _execute(self) -> SearchWorkerOutput:
        search_engine: SearchEngine = SearchEngine()
        retrieved_text = search_engine.search(
            chat_history=self.orch_state.user_message.history,
            bot_config=self.orch_state.bot_config,
        )
        self.orch_state.message_flow = retrieved_text
        response = ToolGenerator.context_generate(self.orch_state)
        return SearchWorkerOutput(
            response=response,
            status=StatusEnum.COMPLETE,
        )
