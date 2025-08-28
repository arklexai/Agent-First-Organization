"""Message worker implementation for the Arklex framework.

This module provides a specialized worker for handling message generation and delivery
in the Arklex framework. The MessageWorker class is responsible for processing user
messages, orchestrator messages, and generating appropriate responses. It supports
both streaming and non-streaming response generation, with functionality for handling
message flows and direct responses.
"""

from typing import Any

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from arklex.env.prompts import load_prompts
from arklex.env.tools.utils import trace
from arklex.env.workers.base.base_worker import BaseWorker
from arklex.env.workers.message.entities import (
    MessageWorkerData,
    MessageWorkerOutput,
)
from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
    StatusEnum,
)
from arklex.types.stream_types import EventType, StreamType
from arklex.utils.logging_utils import LogContext
from arklex.utils.provider_utils import validate_and_get_model_class

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

    def _format_prompt(self) -> str:
        user_message = self.orch_state.user_message
        message_flow = self.orch_state.message_flow
        orch_message = self.msg_worker_data.message

        prompts: dict[str, str] = load_prompts(self.orch_state.bot_config)
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
        model_class = validate_and_get_model_class(
            self.orch_state.bot_config.llm_config
        )
        self.llm: Any = model_class(
            model=self.orch_state.bot_config.llm_config.model_type_or_path,
            temperature=0.1,
        )
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
