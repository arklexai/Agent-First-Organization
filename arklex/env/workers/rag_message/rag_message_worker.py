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
from arklex.env.tools.utils import trace
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
        orch_message = self.rag_message_worker_data.message
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
            self.orch_state = trace(
                input=retriever_params, source="milvus_retrieve", state=self.orch_state
            )

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
