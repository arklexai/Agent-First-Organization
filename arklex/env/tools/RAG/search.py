"""Search engine implementation for the Arklex framework.

This module provides search functionality using the Tavily search engine. It includes
the SearchEngine class for handling search operations and the TavilySearchExecutor
class for executing searches and processing results. The module supports contextualized
query reformulation and result processing for integration with the RAG system.
"""

import logging
from typing import List, Dict, Any

from arklex.utils.model_provider_config import PROVIDER_MAP
from arklex.env.prompts import load_prompts
from arklex.utils.graph_state import MessageState, LLMConfig
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import TavilySearchResults


logger: logging.Logger = logging.getLogger(__name__)


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
        **kwargs: Any,
    ) -> None:
        self.llm: Any = PROVIDER_MAP.get(llm_config.llm_provider, ChatOpenAI)(
            model=llm_config.model_type_or_path
        )
        self.search_tool: TavilySearchResults = TavilySearchResults(
            max_results=kwargs.get("max_results", 5),
            search_depth=kwargs.get("search_depth", "advanced"),
            include_answer=kwargs.get("include_answer", True),
            include_raw_content=kwargs.get("include_raw_content", True),
            include_images=kwargs.get("include_images", False),
        )

    def process_search_result(self, search_results: List[Dict[str, Any]]) -> str:
        search_text: str = ""
        for res in search_results:
            search_text += f"Source: {res['url']} \n"
            search_text += f"Content: {res['content']} \n\n"
        return search_text

    def search(self, state: MessageState) -> str:
        prompts: Dict[str, str] = load_prompts(state.bot_config)
        contextualize_q_prompt: PromptTemplate = PromptTemplate.from_template(
            prompts["retrieve_contextualize_q_prompt"]
        )
        ret_input_chain: Any = contextualize_q_prompt | self.llm | StrOutputParser()
        ret_input: str = ret_input_chain.invoke(
            {"chat_history": state.user_message.history}
        )
        logger.info(f"Reformulated input for search engine: {ret_input}")
        search_results: List[Dict[str, Any]] = self.search_tool.invoke(
            {"query": ret_input}
        )
        text_results: str = self.process_search_result(search_results)
        return text_results

    def load_search_tool(
        self, llm_config: LLMConfig, **kwargs: Any
    ) -> "TavilySearchExecutor":
        return TavilySearchExecutor(llm_config, **kwargs)
