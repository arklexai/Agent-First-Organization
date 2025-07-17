"""FAISS retriever implementation for the Arklex framework.

This module provides a FAISS-based retriever implementation for efficient similarity search
in document collections. It includes the RetrieveEngine class for handling retrieval
operations and the FaissRetrieverExecutor class for managing FAISS index creation and
document retrieval. The module supports contextualized query reformulation and
confidence-scored document retrieval.
"""

import os
import pickle
from typing import Any

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings

from arklex.env.prompts import load_prompts
from arklex.env.tools.utils import trace
from arklex.orchestrator.entities.msg_state_entities import LLMConfig, MessageState
from arklex.utils.logging_utils import LogContext
from arklex.utils.model_provider_config import (
    PROVIDER_EMBEDDING_MODELS,
    PROVIDER_EMBEDDINGS,
)
from arklex.utils.provider_utils import validate_and_get_model_class

log_context = LogContext(__name__)


class RetrieveEngine:
    @staticmethod
    def faiss_retrieve(state: MessageState) -> MessageState:
        # get the input message
        user_message = state.user_message

        # Search for the relevant documents
        prompts: dict[str, str] = load_prompts(state.bot_config)
        docs: FaissRetrieverExecutor = FaissRetrieverExecutor.load_docs(
            database_path=os.environ.get("DATA_DIR"),
            llm_config=state.bot_config.llm_config,
        )
        retrieved_text: str
        retriever_returns: list[dict[str, Any]]
        retrieved_text, retriever_returns = docs.search(
            user_message.history, prompts["retrieve_contextualize_q_prompt"]
        )

        state.message_flow = retrieved_text
        state = trace(input=retriever_returns, state=state)
        return state


class FaissRetrieverExecutor:
    def __init__(
        self,
        texts: list[Document],
        index_path: str,
        llm_config: LLMConfig,
    ) -> None:
        self.texts: list[Document] = texts
        self.index_path: str = index_path
        self.embedding_model = PROVIDER_EMBEDDINGS.get(
            llm_config.llm_provider, OpenAIEmbeddings
        )(
            **{"model": PROVIDER_EMBEDDING_MODELS[llm_config.llm_provider]}
            if llm_config.llm_provider != "anthropic"
            else {"model_name": PROVIDER_EMBEDDING_MODELS[llm_config.llm_provider]}
        )
        model_class = validate_and_get_model_class(llm_config)

        self.llm = model_class(model=llm_config.model_type_or_path)
        self.retriever = self._init_retriever()

    def _init_retriever(self, **kwargs: dict[str, object]) -> object:
        # initiate FAISS retriever
        docsearch: FAISS = FAISS.from_documents(self.texts, self.embedding_model)
        retriever = docsearch.as_retriever(**kwargs)
        return retriever

    def retrieve_w_score(self, query: str) -> list[tuple[Document, float]]:
        k_value: int = (
            4
            if not self.retriever.search_kwargs.get("k")
            else self.retriever.search_kwargs.get("k")
        )
        docs_and_scores: list[tuple[Document, float]] = (
            self.retriever.vectorstore.similarity_search_with_score(query, k=k_value)
        )
        return docs_and_scores

    def search(
        self, chat_history_str: str, contextualize_prompt: str
    ) -> tuple[str, list[dict[str, Any]]]:
        contextualize_q_prompt: PromptTemplate = PromptTemplate.from_template(
            contextualize_prompt
        )
        ret_input_chain = contextualize_q_prompt | self.llm | StrOutputParser()
        ret_input: str = ret_input_chain.invoke({"chat_history": chat_history_str})
        log_context.info(f"Reformulated input for retriever search: {ret_input}")
        docs_and_score: list[tuple[Document, float]] = self.retrieve_w_score(ret_input)
        retrieved_text: str = ""
        retriever_returns: list[dict[str, Any]] = []
        for doc, score in docs_and_score:
            retrieved_text += f"{doc.page_content} \n"
            item: dict[str, Any] = {
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
        log_context.info(f"Loaded documents from {document_path}")
        with open(document_path, "rb") as fread:
            documents: list[Document] = pickle.load(fread)
        log_context.info(f"Loaded {len(documents)} documents")

        return FaissRetrieverExecutor(
            texts=documents, index_path=index_path, llm_config=llm_config
        )
