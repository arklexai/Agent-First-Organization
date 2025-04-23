import logging

from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP
from arklex.env.prompts import load_prompts
from arklex.utils.graph_state import MessageState

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import TavilySearchResults


logger = logging.getLogger(__name__)


class SearchEngine():
    def __init__(self):
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        self.search_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=False,
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
        state.message_flow = self.process_search_result(search_results)
        return state
