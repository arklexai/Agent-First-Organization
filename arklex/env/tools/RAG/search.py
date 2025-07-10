"""Search engine implementation for the Arklex framework.

This module provides search functionality using the Tavily search engine. It includes
the SearchEngine class for handling search operations and the TavilySearchExecutor
class for executing searches and processing results. The module supports contextualized
query reformulation and result processing for integration with the RAG system.
"""

from typing import Any, Literal, TypedDict

from langchain.prompts import PromptTemplate
from langchain_community.tools import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser

from arklex.env.prompts import load_prompts
from arklex.orchestrator.entities.msg_state_entities import LLMConfig, MessageState
from arklex.utils.exceptions import SearchError
from arklex.utils.logging_utils import LogContext
from arklex.utils.provider_utils import validate_and_get_model_class

log_context = LogContext(__name__)


class SearchConfig(TypedDict, total=False):
    """Configuration parameters for Tavily search."""

    max_results: int
    search_depth: Literal["basic", "advanced"]
    include_answer: bool
    include_raw_content: bool
    include_images: bool


class SearchEngine:
    @staticmethod
    def search(state: MessageState) -> MessageState:
        tavily_search_executor: TavilySearchExecutor = TavilySearchExecutor()
        text_results: str = tavily_search_executor.search(state)
        state.message_flow = text_results
        return state


class TavilySearchExecutor:
    def __init__(
        self,
        llm_config: LLMConfig,
        **kwargs: SearchConfig,
    ) -> None:
        model_class = validate_and_get_model_class(llm_config)

        self.llm: Any = model_class(model=llm_config.model_type_or_path)
        self.search_tool: TavilySearchResults = TavilySearchResults(
            max_results=kwargs.get("max_results", 5),
            search_depth=kwargs.get("search_depth", "advanced"),
            include_answer=kwargs.get("include_answer", True),
            include_raw_content=kwargs.get("include_raw_content", True),
            include_images=kwargs.get("include_images", False),
        )

    def process_search_result(self, search_results: list[dict[str, Any]]) -> str:
        search_text: str = ""
        for res in search_results:
            search_text += f"Source: {res['url']} \n"
            search_text += f"Content: {res['content']} \n\n"
        return search_text

    def search(self, state: MessageState) -> str:
        prompts: dict[str, str] = load_prompts(state.bot_config)
        contextualize_q_prompt: PromptTemplate = PromptTemplate.from_template(
            prompts["retrieve_contextualize_q_prompt"]
        )
        ret_input_chain: Any = contextualize_q_prompt | self.llm | StrOutputParser()
        ret_input: str = ret_input_chain.invoke(
            {"chat_history": state.user_message.history}
        )
        log_context.info(f"Reformulated input for search engine: {ret_input}")
        search_results: list[dict[str, Any]] = self.search_tool.invoke(
            {"query": ret_input}
        )
        text_results: str = self.process_search_result(search_results)
        return text_results

    def load_search_tool(
        self,
        llm_config: LLMConfig,
        **kwargs: SearchConfig,
    ) -> "TavilySearchExecutor":
        return TavilySearchExecutor(llm_config, **kwargs)

    def search_documents(
        self, query: str, **kwargs: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Search for documents matching the query.

        Args:
            query: Search query
            **kwargs: Additional search parameters passed to the search tool

        Returns:
            List of matching documents

        Raises:
            SearchError: If search fails
        """
        try:
            log_context.info(f"Starting search for query: {query}")
            results = self.search_tool.invoke({"query": query, **kwargs})
            log_context.info(f"Search completed, found {len(results)} results")
            return results
        except Exception as e:
            log_context.error(f"Search failed: {e}")
            raise SearchError("Search failed") from e
