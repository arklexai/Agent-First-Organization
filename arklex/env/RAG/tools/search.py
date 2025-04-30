import logging

from arklex.utils.model_provider_config import PROVIDER_MAP
from arklex.env.prompts import load_prompts
from arklex.utils.graph_state import MessageState, LLMConfig
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import TavilySearchResults


logger = logging.getLogger(__name__)


class SearchEngine():
    @staticmethod
    def search(state: MessageState):
        tavily_search_executor = TavilySearchExecutor()
        text_results = tavily_search_executor.search(state)
        state.message_flow = text_results
        return state


class TavilySearchExecutor():
    def __init__(
            self,
            llm_config: LLMConfig,
            **kwargs,
        ):
        self.llm = PROVIDER_MAP.get(llm_config.llm_provider, ChatOpenAI)(
            model=llm_config.model_type_or_path
        )
        self.search_tool = TavilySearchResults(
            max_results=kwargs.get("max_results", 5),
            search_depth=kwargs.get("search_depth", "advanced"),
            include_answer=kwargs.get("include_answer", True),
            include_raw_content=kwargs.get("include_raw_content", True),
            include_images=kwargs.get("include_images", False),
        )

    def process_search_result(self, search_results):
        search_text = ""
        for res in search_results:
            search_text += f"Source: {res['url']} \n"
            search_text += f"Content: {res['content']} \n\n"
        return search_text

    def search(self, state: MessageState):
        prompts = load_prompts(state.bot_config)
        contextualize_q_prompt = PromptTemplate.from_template(
            prompts["retrieve_contextualize_q_prompt"]
        )
        ret_input_chain = contextualize_q_prompt | self.llm | StrOutputParser()
        ret_input = ret_input_chain.invoke({"chat_history": state.user_message.history})
        logger.info(f"Reformulated input for search engine: {ret_input}")
        search_results = self.search_tool.invoke({"query": ret_input})
        text_results = self.process_search_result(search_results)
        return text_results
    
    def load_search_tool(self, llm_config: LLMConfig, **kwargs):
        return TavilySearchExecutor(llm_config, **kwargs)
