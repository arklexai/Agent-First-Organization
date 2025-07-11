"""RAG Message worker implementation for the Arklex framework.

This module provides a specialized worker that combines Retrieval-Augmented Generation (RAG)
and message generation capabilities. The RagMsgWorker class intelligently decides whether
to use RAG retrieval or direct message generation based on the context, providing a flexible
approach to handling user queries that may require either factual information from documents
or conversational responses.
"""

from functools import partial
from typing import Any, TypedDict

from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph

from arklex.env.prompts import load_prompts
from arklex.env.tools.RAG.retrievers.milvus_retriever import RetrieveEngine
from arklex.env.workers.message_worker import MessageWorker
from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.orchestrator.entities.msg_state_entities import MessageState
from arklex.utils.logging_utils import LogContext
from arklex.utils.model_provider_config import PROVIDER_MAP

log_context = LogContext(__name__)


class RagMsgWorkerKwargs(TypedDict, total=False):
    """Type definition for kwargs used in RagMsgWorker._execute method."""

    tags: dict[str, Any]


@register_worker
class RagMsgWorker(BaseWorker):
    description: str = "A combination of RAG and Message Workers"

    def __init__(self) -> None:
        super().__init__()
        self.llm: BaseChatModel | None = None
        self.tags: dict[str, Any] = {}
        self.action_graph: StateGraph | None = None

    def _choose_retriever(self, state: MessageState) -> str:
        prompts: dict[str, str] = load_prompts(state.bot_config)
        prompt: PromptTemplate = PromptTemplate.from_template(
            prompts["retrieval_needed_prompt"]
        )
        input_prompt = prompt.invoke({"formatted_chat": state.user_message.history})
        log_context.info(
            f"Prompt for choosing the retriever in RagMsgWorker: {input_prompt.text}"
        )
        final_chain = self.llm | StrOutputParser()
        answer: str = final_chain.invoke(input_prompt.text)
        log_context.info(f"Choose retriever in RagMsgWorker: {answer}")
        if "yes" in answer.lower():
            return "retriever"
        return "message_worker"

    def _create_action_graph(self, tags: dict[str, Any]) -> StateGraph:
        workflow: StateGraph = StateGraph(MessageState)
        # Create a partial function with the extra argument bound
        retriever_with_args = partial(RetrieveEngine.milvus_retrieve, tags=tags)
        # Add nodes for each worker
        msg_wkr: MessageWorker = MessageWorker()
        workflow.add_node("retriever", retriever_with_args)
        workflow.add_node("message_worker", msg_wkr.execute)
        # Add edges
        workflow.add_conditional_edges(START, self._choose_retriever)
        workflow.add_edge("retriever", "message_worker")
        return workflow

    def _execute(
        self, msg_state: MessageState, **kwargs: RagMsgWorkerKwargs
    ) -> dict[str, Any]:
        self.llm = PROVIDER_MAP.get(
            msg_state.bot_config.llm_config.llm_provider, ChatOpenAI
        )(model=msg_state.bot_config.llm_config.model_type_or_path)
        self.tags = kwargs.get("tags", {})
        self.action_graph = self._create_action_graph(self.tags)
        graph = self.action_graph.compile()
        result: dict[str, Any] = graph.invoke(msg_state)
        return result


@register_worker
class RAGMessageWorker(BaseWorker):
    """RAG Message Worker for document search and response generation.

    This worker combines document search capabilities with response generation,
    providing a comprehensive solution for answering questions using RAG
    (Retrieval-Augmented Generation) techniques.
    """

    description: str = "Help the user with questions by searching through documents and providing relevant information using RAG (Retrieval-Augmented Generation)."

    def __init__(self, model_config: dict[str, Any] | None = None) -> None:
        """Initialize the RAG Message Worker.

        Args:
            model_config: Optional configuration for the language model.
        """
        super().__init__()
        self.llm: BaseChatModel | None = None
        self.model_config = model_config or {}

        # Initialize LLM if config is provided
        if self.model_config:
            provider = self.model_config.get("llm_provider", "openai")
            model = self.model_config.get("model_type_or_path", "gpt-4")
            self.llm = PROVIDER_MAP.get(provider, ChatOpenAI)(model=model)

    def search_documents(self, msg_state: MessageState) -> str:
        """Search for relevant documents based on the user's query.

        Args:
            msg_state: The current message state containing the user's query.

        Returns:
            str: Relevant document content or search results.
        """
        if msg_state.orchestrator_message is None:
            return "No query provided for document search."

        if msg_state.orchestrator_message.attribute is None:
            return "No query provided for document search."

        query = msg_state.orchestrator_message.attribute.get("query", "")
        if not query:
            return "No query provided for document search."

        # This is a placeholder implementation
        # In a real implementation, this would use a document retriever
        return f"Search results for: {query}"

    def generate_response(self, msg_state: MessageState) -> str:
        """Generate a response based on the user's query.

        Args:
            msg_state: The current message state containing the user's query.

        Returns:
            str: Generated response to the user's query.
        """
        if msg_state.orchestrator_message is None:
            return "No query provided for response generation."

        if msg_state.orchestrator_message.attribute is None:
            return "No query provided for response generation."

        query = msg_state.orchestrator_message.attribute.get("query", "")
        if not query:
            return "No query provided for response generation."

        # This is a placeholder implementation
        # In a real implementation, this would use the LLM to generate a response
        return f"Response to: {query}"

    def _execute(
        self, msg_state: MessageState, **kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute the RAG message worker.

        Args:
            msg_state: The current message state.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: The execution result.
        """
        # Initialize LLM if not already done
        if not self.llm and msg_state.bot_config:
            provider = msg_state.bot_config.llm_config.llm_provider
            model = msg_state.bot_config.llm_config.model_type_or_path
            self.llm = PROVIDER_MAP.get(provider, ChatOpenAI)(model=model)

        # Perform document search
        search_results = self.search_documents(msg_state)

        # Generate response
        response = self.generate_response(msg_state)

        # Update message state
        msg_state.response = response
        msg_state.message_flow = search_results

        return {"response": response, "search_results": search_results}
