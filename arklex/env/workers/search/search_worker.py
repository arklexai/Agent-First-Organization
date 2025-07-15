"""Search worker implementation for the Arklex framework.

This module provides a specialized worker for handling search-related tasks. It implements
a search engine worker that can answer user questions based on real-time online search
results. The worker uses a state graph to manage the workflow of search operations and
response generation, integrating with the framework's tool generation system.
"""

from arklex.env.tools.RAG.search import SearchEngine
from arklex.env.tools.utils import ToolGenerator
from arklex.env.workers.base.base_worker import BaseWorker, register_worker
from arklex.env.workers.search.entities import SearchWorkerData, SearchWorkerResp
from arklex.orchestrator.entities.orch_state_entities import StatusEnum
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


@register_worker
class SearchWorker(BaseWorker):
    description: str = (
        "Answer the user's questions based on real-time online search results"
    )

    def __init__(self) -> None:
        super().__init__()

    def init_worker_data(self, input_data: SearchWorkerData) -> None:
        self.search_worker_data: SearchWorkerData = input_data

    def _execute(self) -> SearchWorkerResp:
        search_engine: SearchEngine = SearchEngine()
        retrieved_text = search_engine.search(
            chat_history=self.search_worker_data.chat_history,
            bot_config=self.search_worker_data.bot_config,
        )
        self.search_worker_data.message_flow = retrieved_text
        response = ToolGenerator.context_generate(self.search_worker_data)
        return SearchWorkerResp(
            response=response,
            status=StatusEnum.COMPLETE,
        )
