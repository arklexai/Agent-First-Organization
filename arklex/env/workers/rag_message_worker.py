import logging
from typing import Any, Iterator, Union

from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.env.RAG.tools.retrievers.milvus_retriever import RetrieveEngine
from arklex.env.prompts import load_prompts
from arklex.env.workers.message_worker import MessageWorker
from arklex.env.workers.milvus_rag_worker import MilvusRAGWorker
from arklex.utils.utils import chunk_string
from arklex.utils.graph_state import MessageState
from arklex.utils.model_config import MODEL


logger = logging.getLogger(__name__)


@register_worker
class RagMsgWorker(BaseWorker):

    description = "A combination of RAG and Message Workers"

    def __init__(self):
        super().__init__()
        self.action_graph = self._create_action_graph()
        self.llm = ChatOpenAI(model=MODEL["model_type_or_path"], timeout=30000)

    def _choose_retriever(self, state: MessageState):
        prompts = load_prompts(state.bot_config)
        prompt = PromptTemplate.from_template(prompts["retrieval_needed_prompt"])
        input_prompt = prompt.invoke({"formatted_chat": state.user_message.history})
        logger.info(f"Prompt for choosing the retriever in RagMsgWorker: {input_prompt.text}")
        chunked_prompt = chunk_string(input_prompt.text, tokenizer=MODEL["tokenizer"], max_length=MODEL["context"])
        final_chain = self.llm | StrOutputParser()
        answer = final_chain.invoke(chunked_prompt)
        logger.info(f"Choose retriever in RagMsgWorker: {answer}")
        if "yes" in answer.lower():
            return "retriever"
        return "message_worker"
     
    def _create_action_graph(self):
        workflow = StateGraph(MessageState)
        # Add nodes for each worker
        msg_wkr = MessageWorker()
        workflow.add_node("retriever", RetrieveEngine.milvus_retrieve)
        workflow.add_node("message_worker", msg_wkr.execute)
        # Add edges
        workflow.add_conditional_edges(
            START, self._choose_retriever)
        workflow.add_edge("retriever", "message_worker")
        return workflow

    def _execute(self, msg_state: MessageState):
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result
