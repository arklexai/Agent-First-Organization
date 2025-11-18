"""Message worker implementation for the Arklex framework.

This module provides a specialized worker for handling message generation and delivery
in the Arklex framework. The MessageWorker class is responsible for processing user
messages, orchestrator messages, and generating appropriate responses. It supports
both streaming and non-streaming response generation, with functionality for handling
message flows and direct responses.
"""

from typing import Any

from langchain_core.prompts import PromptTemplate

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

    def _format_prompts(self) -> tuple[str, str]:
        user_message = self.orch_state.user_message
        message_flow = self.orch_state.message_flow
        orch_message = self.msg_worker_data.message

        prompts: dict[str, str] = load_prompts(self.orch_state.bot_config.language)
        if message_flow:
            if self.orch_state.stream_type == StreamType.SPEECH:
                system_prompt_template: PromptTemplate = PromptTemplate.from_template(
                    prompts["message_flow_generator_prompt_speech_system"]
                )
                user_prompt_template: PromptTemplate = PromptTemplate.from_template(
                    prompts["message_flow_generator_prompt_speech"]
                )
            else:
                system_prompt_template: PromptTemplate = PromptTemplate.from_template(
                    prompts["message_flow_generator_prompt_system"]
                )
                user_prompt_template: PromptTemplate = PromptTemplate.from_template(
                    prompts["message_flow_generator_prompt"]
                )
            system_prompt = system_prompt_template.invoke(
                {"sys_instruct": self.orch_state.sys_instruct}
            ).text
            user_prompt = user_prompt_template.invoke(
                {
                    "message": orch_message,
                    "formatted_chat": user_message.history,
                    "context": message_flow,
                }
            ).text
        else:
            if self.orch_state.stream_type == StreamType.SPEECH:
                system_prompt_template: PromptTemplate = PromptTemplate.from_template(
                    prompts["message_generator_prompt_speech_system"]
                )
                user_prompt_template: PromptTemplate = PromptTemplate.from_template(
                    prompts["message_generator_prompt_speech"]
                )
            else:
                system_prompt_template: PromptTemplate = PromptTemplate.from_template(
                    prompts["message_generator_prompt_system"]
                )
                user_prompt_template: PromptTemplate = PromptTemplate.from_template(
                    prompts["message_generator_prompt"]
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
        print("MESSAGE WORKER: Formatted Prompts")
        print("="*80)
        print(f"Stream Type: {self.orch_state.stream_type}")
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
        print("\n" + "="*80)
        print("MESSAGE WORKER: stream_generator() - Streaming Response")
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
        print("MESSAGE WORKER: stream_generator() - Streaming Complete")
        print("="*80)
        print(f"Total response length: {len(answer)} characters")
        print("="*80 + "\n")
        
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

        system_prompt, user_prompt = self._format_prompts()
        if (
            self.orch_state.stream_type == StreamType.TEXT
            or self.orch_state.stream_type == StreamType.SPEECH
        ):
            answer = self.stream_generator(system_prompt, user_prompt)
        else:
            answer = self.generator(system_prompt, user_prompt)

        return MessageWorkerOutput(
            response=answer,
            status=StatusEnum.COMPLETE,
        )
