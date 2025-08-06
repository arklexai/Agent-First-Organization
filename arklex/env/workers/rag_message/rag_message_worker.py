"""RAG Message worker implementation for the Arklex framework.

This module provides a specialized worker that combines Retrieval-Augmented Generation (RAG)
and message generation capabilities. The RagMsgWorker class intelligently decides whether
to use RAG retrieval or direct message generation based on the context, providing a flexible
approach to handling user queries that may require either factual information from documents
or conversational responses.
"""

from typing import Any

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from arklex.env.prompts import load_prompts
from arklex.env.tools.RAG.retrievers.milvus_retriever import RetrieveEngine
from arklex.env.workers.base.base_worker import BaseWorker
from arklex.env.workers.rag_message.entities import (
    RAGMessageWorkerData,
    RAGMessageWorkerOutput,
)
from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
    StatusEnum,
)
from arklex.types.stream_types import EventType, StreamType
from arklex.utils.logging_utils import LogContext
from arklex.utils.provider_utils import validate_and_get_model_class

log_context = LogContext(__name__)


class RagMsgWorker(BaseWorker):
    description: str = "A combination of RAG and Message Workers"

    def __init__(self) -> None:
        super().__init__()

    def init_worker_data(
        self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
    ) -> None:
        self.orch_state = orch_state
        self.rag_message_worker_data = RAGMessageWorkerData(**node_specific_data)
        model_class = validate_and_get_model_class(
            self.orch_state.bot_config.llm_config
        )
        self.llm: Any = model_class(
            model=self.orch_state.bot_config.llm_config.model_type_or_path,
            temperature=0.1,
        )

    def _need_retriever(self) -> str:
        prompt: PromptTemplate = PromptTemplate.from_template(
            self.prompts["retrieval_needed_prompt"]
        )
        input_prompt = prompt.invoke(
            {"formatted_chat": self.orch_state.user_message.history}
        )
        log_context.info(
            f"Prompt for choosing the retriever in RagMsgWorker: {input_prompt.text}"
        )
        final_chain = self.llm | StrOutputParser()
        answer: str = final_chain.invoke(input_prompt.text)
        log_context.info(f"Choose retriever in RagMsgWorker: {answer}")
        return "yes" in answer.lower()

    def _format_prompt(self, context: str) -> str:
        user_message = self.orch_state.user_message
        orch_message = self.rag_message_worker_data.node_message
        if context:
            if self.orch_state.stream_type == StreamType.SPEECH:
                prompt: PromptTemplate = PromptTemplate.from_template(
                    self.prompts["message_flow_generator_prompt_speech"]
                )
            else:
                prompt: PromptTemplate = PromptTemplate.from_template(
                    self.prompts["message_flow_generator_prompt"]
                )
            input_prompt = prompt.invoke(
                {
                    "sys_instruct": self.orch_state.sys_instruct,
                    "message": orch_message,
                    "formatted_chat": user_message.history,
                    "context": context,
                }
            )
        else:
            if self.orch_state.stream_type == StreamType.SPEECH:
                prompt: PromptTemplate = PromptTemplate.from_template(
                    self.prompts["message_generator_prompt_speech"]
                )
            else:
                prompt: PromptTemplate = PromptTemplate.from_template(
                    self.prompts["message_generator_prompt"]
                )
            input_prompt = prompt.invoke(
                {
                    "sys_instruct": self.orch_state.sys_instruct,
                    "message": orch_message,
                    "formatted_chat": user_message.history,
                }
            )
        log_context.info(
            f"Prompt for stream type {self.orch_state.stream_type}: {input_prompt.text}"
        )
        return input_prompt.text

    def generator(self, prompt: str) -> str:
        invoke_chain = self.llm | StrOutputParser()
        answer: str = invoke_chain.invoke(prompt)
        return answer

    def stream_generator(self, prompt: str) -> str:
        invoke_chain = self.llm | StrOutputParser()
        answer: str = ""
        for chunk in invoke_chain.stream(prompt):
            answer += chunk
            self.orch_state.message_queue.put(
                {"event": EventType.CHUNK.value, "message_chunk": chunk}
            )
        return answer

    def _execute(self) -> RAGMessageWorkerOutput:
        self.prompts: dict[str, str] = load_prompts(self.orch_state.bot_config)
        retrieve_text = ""
        if self._need_retriever():
            retrieve_text, retriever_params = RetrieveEngine.milvus_retrieve(
                self.orch_state.user_message.history,
                self.orch_state.bot_config,
                self.rag_message_worker_data.tags,
            )
            # state = trace(input=retriever_params, state=state)

        input_prompt = self._format_prompt(retrieve_text)
        if (
            self.orch_state.stream_type == StreamType.TEXT
            or self.orch_state.stream_type == StreamType.SPEECH
        ):
            answer = self.stream_generator(input_prompt)
        else:
            answer = self.generator(input_prompt)

        return RAGMessageWorkerOutput(
            response=answer,
            status=StatusEnum.COMPLETE,
        )


# class RAGMessageWorker(BaseWorker):
#     """RAG Message Worker for document search and response generation.

#     # not in used as of 16/07/2025

#     This worker combines document search capabilities with response generation,
#     providing a comprehensive solution for answering questions using RAG
#     (Retrieval-Augmented Generation) techniques.
#     """

#     description: str = "Help the user with questions by searching through documents and providing relevant information using RAG (Retrieval-Augmented Generation)."

#     def __init__(self, model_config: dict[str, Any] | None = None) -> None:
#         """Initialize the RAG Message Worker.

#         Args:
#             model_config: Optional configuration for the language model.
#         """
#         super().__init__()
#         self.llm: BaseChatModel | None = None
#         self.model_config = model_config or {}

#         # Initialize LLM if config is provided
#         if self.model_config:
#             provider = self.model_config.get("llm_provider", "openai")
#             model = self.model_config.get("model_type_or_path", "gpt-4")
#             self.llm = PROVIDER_MAP.get(provider, ChatOpenAI)(model=model)

#     def search_documents(self, msg_state: MessageState) -> str:
#         """Search for relevant documents based on the user's query.

#         Args:
#             msg_state: The current message state containing the user's query.

#         Returns:
#             str: Relevant document content or search results.
#         """
#         if msg_state.orchestrator_message is None:
#             return "No query provided for document search."

#         if msg_state.orchestrator_message.attribute is None:
#             return "No query provided for document search."

#         query = msg_state.orchestrator_message.attribute.get("query", "")
#         if not query:
#             return "No query provided for document search."

#         # This is a placeholder implementation
#         # In a real implementation, this would use a document retriever
#         return f"Search results for: {query}"

#     def generate_response(self, msg_state: MessageState) -> str:
#         """Generate a response based on the user's query.

#         Args:
#             msg_state: The current message state containing the user's query.

#         Returns:
#             str: Generated response to the user's query.
#         """
#         if msg_state.orchestrator_message is None:
#             return "No query provided for response generation."

#         if msg_state.orchestrator_message.attribute is None:
#             return "No query provided for response generation."

#         query = msg_state.orchestrator_message.attribute.get("query", "")
#         if not query:
#             return "No query provided for response generation."

#         # This is a placeholder implementation
#         # In a real implementation, this would use the LLM to generate a response
#         return f"Response to: {query}"

#     def _execute(
#         self, msg_state: MessageState, **kwargs: dict[str, Any]
#     ) -> dict[str, Any]:
#         """Execute the RAG message worker.

#         Args:
#             msg_state: The current message state.
#             **kwargs: Additional keyword arguments.

#         Returns:
#             dict[str, Any]: The execution result.
#         """
#         # Initialize LLM if not already done
#         if not self.llm and msg_state.bot_config:
#             provider = msg_state.bot_config.llm_config.llm_provider
#             model = msg_state.bot_config.llm_config.model_type_or_path
#             self.llm = PROVIDER_MAP.get(provider, ChatOpenAI)(model=model)

#         # Perform document search
#         search_results = self.search_documents(msg_state)

#         # Generate response
#         response = self.generate_response(msg_state)

#         # Update message state
#         msg_state.response = response
#         msg_state.message_flow = search_results

#         return {"response": response, "search_results": search_results}
