"""RAG Message worker implementation for the Arklex framework.

This module provides a specialized worker that combines Retrieval-Augmented Generation (RAG)
and message generation capabilities. The RagMsgWorker class intelligently decides whether
to use RAG retrieval or direct message generation based on the context, providing a flexible
approach to handling user queries that may require either factual information from documents
or conversational responses.
"""

from typing import Any

from langchain_core.prompts import PromptTemplate

from arklex.models.model_service import ModelService
from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
    StatusEnum,
)
from arklex.orchestrator.types.stream_types import EventType, StreamType
from arklex.resources.tools.rag.retrievers.milvus_retriever import (
    MilvusRetrieverExecutor,
)
from arklex.resources.tools.utils import trace
from arklex.resources.workers.base.base_worker import BaseWorker
from arklex.resources.workers.rag_message.entities import (
    RAGMessageWorkerData,
    RAGMessageWorkerOutput,
)
from arklex.utils.logging.logging_utils import LogContext
from arklex.utils.prompts import load_prompts

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
        self.model_service = ModelService(self.orch_state.bot_config.llm_config)

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
        answer: str = self.model_service.get_response(input_prompt.text)
        log_context.info(f"Choose retriever in RagMsgWorker: {answer}")
        return "yes" in answer.lower()

    def _format_prompts(self, context: str) -> tuple[str, str]:
        user_message = self.orch_state.user_message
        orch_message = self.rag_message_worker_data.message
        if context:
            if self.orch_state.stream_type == StreamType.SPEECH:
                system_prompt_template: PromptTemplate = PromptTemplate.from_template(
                    self.prompts["message_flow_generator_prompt_speech_system"]
                )
                user_prompt_template: PromptTemplate = PromptTemplate.from_template(
                    self.prompts["message_flow_generator_prompt_speech"]
                )
            else:
                system_prompt_template: PromptTemplate = PromptTemplate.from_template(
                    self.prompts["message_flow_generator_prompt_system"]
                )
                user_prompt_template: PromptTemplate = PromptTemplate.from_template(
                    self.prompts["message_flow_generator_prompt"]
                )
            system_prompt = system_prompt_template.invoke(
                {"sys_instruct": self.orch_state.sys_instruct}
            ).text
            user_prompt = user_prompt_template.invoke(
                {
                    "message": orch_message,
                    "formatted_chat": user_message.history,
                    "context": context,
                }
            ).text
        else:
            if self.orch_state.stream_type == StreamType.SPEECH:
                system_prompt_template: PromptTemplate = PromptTemplate.from_template(
                    self.prompts["message_generator_prompt_speech_system"]
                )
                user_prompt_template: PromptTemplate = PromptTemplate.from_template(
                    self.prompts["message_generator_prompt_speech"]
                )
            else:
                system_prompt_template: PromptTemplate = PromptTemplate.from_template(
                    self.prompts["message_generator_prompt_system"]
                )
                user_prompt_template: PromptTemplate = PromptTemplate.from_template(
                    self.prompts["message_generator_prompt"]
                )
            system_prompt = system_prompt_template.invoke(
                {"sys_instruct": self.orch_state.sys_instruct}
            ).text
            user_prompt = user_prompt_template.invoke(
                {
                    "message": orch_message,
                    "formatted_chat": user_message.history,
                }
            ).text
        log_context.info(
            f"System prompt for stream type {self.orch_state.stream_type}: {system_prompt}"
        )
        log_context.info(
            f"User prompt for stream type {self.orch_state.stream_type}: {user_prompt}"
        )
        
        # Print statements to show prompt structure
        print("\n" + "="*80)
        print("RAG MESSAGE WORKER: Formatted Prompts")
        print("="*80)
        print(f"Stream Type: {self.orch_state.stream_type}")
        print(f"Context provided: {bool(context)}")
        print("\n[SYSTEM PROMPT]")
        print(f"{'-'*80}")
        print(system_prompt)
        print(f"{'-'*80}")
        print("\n[USER PROMPT]")
        print(f"{'-'*80}")
        print(user_prompt)
        print(f"{'-'*80}")
        print("="*80 + "\n")
        
        return system_prompt, user_prompt

    def generator(self, system_prompt: str, user_prompt: str) -> str:
        answer: str = self.model_service.get_response(user_prompt, system_prompt)
        return answer

    def stream_generator(self, system_prompt: str, user_prompt: str) -> str:
        # Note: ModelService doesn't support streaming directly, so we'll use the underlying model
        # This maintains backward compatibility while using ModelService for non-streaming operations
        print("\n" + "="*80)
        print("RAG MESSAGE WORKER: stream_generator() - Streaming Response")
        print("="*80)
        print("Using formatted messages for streaming...")
        print("="*80 + "\n")
        
        answer: str = ""
        messages = self.model_service._format_messages(user_prompt, system_prompt)
        for chunk in self.model_service.model.stream(messages):
            answer += chunk.content
            self.orch_state.message_queue.put(
                {"event": EventType.CHUNK.value, "message_chunk": chunk.content}
            )
        
        print("\n" + "="*80)
        print("RAG MESSAGE WORKER: stream_generator() - Streaming Complete")
        print("="*80)
        print(f"Total response length: {len(answer)} characters")
        print("="*80 + "\n")
        
        return answer

    def _execute(self) -> RAGMessageWorkerOutput:
        self.prompts: dict[str, str] = load_prompts(self.orch_state.bot_config.language)
        retrieve_text = ""
        if self._need_retriever():
            milvus_retriever_executor = MilvusRetrieverExecutor(
                self.orch_state.bot_config
            )
            retrieve_text, retriever_params = milvus_retriever_executor.retrieve(
                self.orch_state.user_message.history,
                self.rag_message_worker_data.bot_id,
                self.rag_message_worker_data.version,
                self.rag_message_worker_data.collection_name,
                self.rag_message_worker_data.tags,
                self.rag_message_worker_data.possible_tags,
            )
            self.orch_state = trace(
                input=retriever_params, source="milvus_retrieve", state=self.orch_state
            )

        system_prompt, user_prompt = self._format_prompts(retrieve_text)
        if (
            self.orch_state.stream_type == StreamType.TEXT
            or self.orch_state.stream_type == StreamType.SPEECH
        ):
            answer = self.stream_generator(system_prompt, user_prompt)
        else:
            answer = self.generator(system_prompt, user_prompt)

        return RAGMessageWorkerOutput(
            response=answer,
            status=StatusEnum.COMPLETE,
        )
