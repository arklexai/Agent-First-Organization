import os
import logging
from typing import List
import pickle

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings


from arklex.utils.model_config import MODEL
from arklex.env.prompts import load_prompts
from arklex.utils.graph_state import MessageState
from arklex.utils.model_provider_config import PROVIDER_MAP, PROVIDER_EMBEDDINGS, PROVIDER_EMBEDDING_MODELS
from arklex.env.tools.utils import trace


logger = logging.getLogger(__name__)

class RetrieveEngine():
    @staticmethod
    def faiss_retrieve(state: MessageState):
        # get the input message
        user_message = state.user_message

        # Search for the relevant documents
        prompts = load_prompts(state.bot_config)
        docs = FaissRetrieverExecutor.load_docs(database_path=os.environ.get("DATA_DIR"))
        retrieved_text, retriever_returns = docs.search(user_message.history, prompts["retrieve_contextualize_q_prompt"])

        state.message_flow = retrieved_text
        state = trace(input=retriever_returns, state=state)
        return state


class FaissRetrieverExecutor:
    def __init__(
            self, 
            texts: List[Document], 
            index_path: str,
            embedding_model_name: str =  PROVIDER_EMBEDDING_MODELS[MODEL['llm_provider']]
        ):
        self.texts = texts
        self.index_path = index_path
        self.embedding_model_name = embedding_model_name
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        self.retriever = self._init_retriever()

    def _init_retriever(self, **kwargs):
        # initiate FAISS retriever
        embedding_model = PROVIDER_EMBEDDINGS.get(MODEL['llm_provider'], OpenAIEmbeddings)(
            **{ 'model': self.embedding_model_name } if MODEL['llm_provider'] != 'anthropic' else { 'model_name': self.embedding_model_name }
        )
        docsearch = FAISS.from_documents(self.texts, embedding_model)
        retriever = docsearch.as_retriever(**kwargs)
        return retriever     

    def retrieve_w_score(self, query: str):
        k_value = 4 if not self.retriever.search_kwargs.get('k') else self.retriever.search_kwargs.get('k')
        docs_and_scores = self.retriever.vectorstore.similarity_search_with_score(query, k=k_value)
        return docs_and_scores

    def search(self, chat_history_str: str, contextualize_prompt: str):
        contextualize_q_prompt = PromptTemplate.from_template(
            contextualize_prompt
        )
        ret_input_chain = contextualize_q_prompt | self.llm | StrOutputParser()
        ret_input = ret_input_chain.invoke({"chat_history": chat_history_str})
        logger.info(f"Reformulated input for retriever search: {ret_input}")
        docs_and_score = self.retrieve_w_score(ret_input)
        retrieved_text = ""
        retriever_returns = []
        for doc, score in docs_and_score:
            retrieved_text += f"{doc.page_content} \n"
            item = {
                "title": doc.metadata.get("title"),
                "content": doc.page_content,
                "source": doc.metadata.get("source"),
                "confidence": float(score),
            }
            retriever_returns.append(item)
        return retrieved_text, retriever_returns

    @staticmethod
    def load_docs(database_path: str, embeddings: str=None, index_path: str="./index"):
        document_path = os.path.join(database_path, "chunked_documents.pkl")
        index_path = os.path.join(database_path, "index")
        logger.info(f"Loaded documents from {document_path}")
        with open(document_path, 'rb') as fread:
            documents = pickle.load(fread)
        logger.info(f"Loaded {len(documents)} documents")

        return FaissRetrieverExecutor(
            texts=documents,
            index_path=index_path
        )
