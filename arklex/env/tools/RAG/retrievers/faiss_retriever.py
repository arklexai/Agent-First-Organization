"""FAISS retriever implementation for the Arklex framework.

This module provides a FAISS-based retriever implementation for efficient similarity search
in document collections. It includes the RetrieveEngine class for handling retrieval
operations and the FaissRetrieverExecutor class for managing FAISS index creation and
document retrieval. The module supports contextualized query reformulation and
confidence-scored document retrieval.
"""

import os
import logging
from typing import List, Dict, Any, Tuple
import pickle

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings

from arklex.env.prompts import load_prompts
from arklex.utils.graph_state import MessageState, LLMConfig
from arklex.utils.model_provider_config import (
    PROVIDER_MAP,
    PROVIDER_EMBEDDINGS,
    PROVIDER_EMBEDDING_MODELS,
)
from arklex.env.tools.utils import trace


logger: logging.Logger = logging.getLogger(__name__)


class RetrieveEngine:
    @staticmethod
    def faiss_retrieve(state: MessageState) -> MessageState:
        # get the input message
        user_message = state.user_message

        # Search for the relevant documents
        prompts: Dict[str, str] = load_prompts(state.bot_config)
        docs: FaissRetrieverExecutor = FaissRetrieverExecutor.load_docs(
            database_path=os.environ.get("DATA_DIR"),
            llm_config=state.bot_config.llm_config,
        )
        retrieved_text: str
        retriever_returns: List[Dict[str, Any]]
        retrieved_text, retriever_returns = docs.search(
            user_message.history, prompts["retrieve_contextualize_q_prompt"]
        )

        state.message_flow = retrieved_text
        state = trace(input=retriever_returns, state=state)
        return state


class FaissRetrieverExecutor:
    def __init__(
        self,
        texts: List[Document],
        index_path: str,
        llm_config: LLMConfig,
    ) -> None:
        self.texts: List[Document] = texts
        self.index_path: str = index_path
        self.embedding_model = PROVIDER_EMBEDDINGS.get(
            llm_config.llm_provider, OpenAIEmbeddings
        )(
            **{"model": PROVIDER_EMBEDDING_MODELS[llm_config.llm_provider]}
            if llm_config.llm_provider != "anthropic"
            else {"model_name": PROVIDER_EMBEDDING_MODELS[llm_config.llm_provider]}
        )
        self.llm = PROVIDER_MAP.get(llm_config.llm_provider, ChatOpenAI)(
            model=llm_config.model_type_or_path
        )
        self.retriever = self._init_retriever()

    def _init_retriever(self, **kwargs: Any) -> Any:
        # initiate FAISS retriever
        docsearch: FAISS = FAISS.from_documents(self.texts, self.embedding_model)
        retriever = docsearch.as_retriever(**kwargs)
        return retriever

    def retrieve_w_score(self, query: str) -> List[Tuple[Document, float]]:
        k_value: int = (
            4
            if not self.retriever.search_kwargs.get("k")
            else self.retriever.search_kwargs.get("k")
        )
        docs_and_scores: List[Tuple[Document, float]] = (
            self.retriever.vectorstore.similarity_search_with_score(query, k=k_value)
        )
        return docs_and_scores

    def search(
        self, chat_history_str: str, contextualize_prompt: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        contextualize_q_prompt: PromptTemplate = PromptTemplate.from_template(
            contextualize_prompt
        )
        ret_input_chain = contextualize_q_prompt | self.llm | StrOutputParser()
        ret_input: str = ret_input_chain.invoke({"chat_history": chat_history_str})
        logger.info(f"Reformulated input for retriever search: {ret_input}")
        docs_and_score: List[Tuple[Document, float]] = self.retrieve_w_score(ret_input)
        retrieved_text: str = ""
        retriever_returns: List[Dict[str, Any]] = []
        for doc, score in docs_and_score:
            retrieved_text += f"{doc.page_content} \n"
            item: Dict[str, Any] = {
                "title": doc.metadata.get("title"),
                "content": doc.page_content,
                "source": doc.metadata.get("source"),
                "confidence": float(score),
            }
            retriever_returns.append(item)
        return retrieved_text, retriever_returns

    @staticmethod
    def load_docs(
        database_path: str, llm_config: LLMConfig, index_path: str = "./index"
    ) -> "FaissRetrieverExecutor":
        document_path: str = os.path.join(database_path, "chunked_documents.pkl")
        index_path: str = os.path.join(database_path, "index")
        logger.info(f"Loaded documents from {document_path}")
        with open(document_path, "rb") as fread:
            documents: List[Document] = pickle.load(fread)
        logger.info(f"Loaded {len(documents)} documents")

        return FaissRetrieverExecutor(
            texts=documents, index_path=index_path, llm_config=llm_config
        )
