"""Message worker implementation for the Arklex framework.

This module provides a specialized worker for handling message generation and delivery
in the Arklex framework. The MessageWorker class is responsible for processing user
messages, orchestrator messages, and generating appropriate responses. It supports
both streaming and non-streaming response generation, with functionality for handling
message flows and direct responses.
"""

from typing import Any

from langchain.prompts import PromptTemplate

from arklex.models.model_service import ModelService
from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
    StatusEnum,
)
from arklex.orchestrator.types.stream_types import EventType, StreamType
from arklex.resources.tools.utils import trace
from arklex.resources.workers.base.base_worker import BaseWorker
from arklex.resources.workers.message.entities import (
    MessageWorkerData,
    MessageWorkerOutput,
)
from arklex.utils.logging.logging_utils import LogContext
from arklex.utils.prompts import load_prompts

log_context = LogContext(__name__)


class MessageWorker(BaseWorker):
    description: str = "The worker that used to deliver the message to the user, either a question or provide some information."

    def __init__(self) -> None:
        super().__init__()

    def init_worker_data(
        self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
    ) -> None:
        self.orch_state = orch_state
        self.msg_worker_data: MessageWorkerData = MessageWorkerData(
            **node_specific_data,
        )
        self.model_service = ModelService(self.orch_state.bot_config.llm_config)

    def _format_prompt(self) -> str:
        user_message = self.orch_state.user_message
        message_flow = self.orch_state.message_flow
        orch_message = self.msg_worker_data.message

        prompts: dict[str, str] = load_prompts(self.orch_state.bot_config.language)
        if message_flow:
            if self.orch_state.stream_type == StreamType.SPEECH:
                prompt: PromptTemplate = PromptTemplate.from_template(
                    prompts["message_flow_generator_prompt_speech"]
                )
            else:
                prompt: PromptTemplate = PromptTemplate.from_template(
                    prompts["message_flow_generator_prompt"]
                )
            input_prompt = prompt.invoke(
                {
                    "sys_instruct": self.orch_state.sys_instruct,
                    "message": orch_message,
                    "formatted_chat": user_message.history,
                    "context": message_flow,
                }
            )
        else:
            if self.orch_state.stream_type == StreamType.SPEECH:
                prompt: PromptTemplate = PromptTemplate.from_template(
                    prompts["message_generator_prompt_speech"]
                )
            else:
                prompt: PromptTemplate = PromptTemplate.from_template(
                    prompts["message_generator_prompt"]
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
        answer: str = self.model_service.get_response(prompt)
        return answer

    def stream_generator(self, prompt: str) -> str:
        answer: str = ""
        for chunk in self.model_service.model.stream(prompt):
            answer += chunk.content
            self.orch_state.message_queue.put(
                {"event": EventType.CHUNK.value, "message_chunk": chunk.content}
            )
        return answer

    def _execute(self) -> MessageWorkerOutput:
        self.orch_state = trace(
            input=self.msg_worker_data.message, source="message", state=self.orch_state
        )
        if self.msg_worker_data.directed:
            return MessageWorkerOutput(
                response=self.msg_worker_data.message,
                status=StatusEnum.COMPLETE,
            )

        input_prompt = self._format_prompt()
        if (
            self.orch_state.stream_type == StreamType.TEXT
            or self.orch_state.stream_type == StreamType.SPEECH
        ):
            answer = self.stream_generator(input_prompt)
        else:
            answer = self.generator(input_prompt)

        return MessageWorkerOutput(
            response=answer,
            status=StatusEnum.COMPLETE,
        )
