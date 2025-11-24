"""Answer Node worker implementation for the Arklex framework.

This module provides a specialized worker for handling answer node message generation
in the Arklex framework. The AnswerNodeWorker class is responsible for processing user
messages and generating responses using the task and prompt from the node info and
conversation history. It supports both streaming and non-streaming response generation.
"""

from typing import Any

from langchain_core.prompts import PromptTemplate

from arklex.models.model_service import ModelService
from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
    StatusEnum,
)
from arklex.orchestrator.types.stream_types import EventType, StreamType
from arklex.resources.workers.base.base_worker import BaseWorker
from arklex.resources.workers.output_process.entities import (
    OutputProcessWorkerData,
    OutputProcessWorkerOutput,
)
from arklex.utils.logging.logging_utils import LogContext
from arklex.utils.prompts import load_prompts
from arklex.utils.utils import format_chat_history

log_context = LogContext(__name__)


class OutputProcessWorker(BaseWorker):
    description: str = "The worker that generates responses using the task and prompt from the node info and conversation history."

    def __init__(self) -> None:
        super().__init__()
        self.orch_state: OrchestratorState | None = None
        self.answer_worker_data: OutputProcessWorkerData | None = None
        self.llm = None

    def init_worker_data(
        self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
    ) -> None:
        """Initialize the worker data."""
        self.orch_state = orch_state
        self.answer_worker_data = OutputProcessWorkerData(**node_specific_data)

    def _format_prompts(self) -> tuple[str, str]:
        """Format the system and user prompts for the answer node worker."""
        user_message = self.orch_state.user_message
        message_flow = self.orch_state.message_flow

        # Get the task and prompt from the worker data
        task = self.answer_worker_data.task
        prompt = self.answer_worker_data.prompt

        if not task and not prompt:
            log_context.warning("No task or prompt provided in worker data")
            return "", "I don't have a specific task to perform."

        # Load prompts based on bot configuration
        prompts = load_prompts(self.orch_state.bot_config.language)

        # Create a focused, efficient prompt template
        if message_flow and message_flow.strip():
            # Use template with context from previous nodes
            system_template = PromptTemplate.from_template(
                prompts["answer_node_prompt_with_context_system"]
            )
            user_template = PromptTemplate.from_template(
                prompts["answer_node_prompt_with_context"]
            )

            system_prompt = system_template.invoke(
                {
                    "sys_instruct": self.orch_state.sys_instruct,
                    "task": task,
                    "prompt": prompt,
                }
            ).text
            # Format history for the prompt (needs string format)
            from arklex.utils.utils import format_chat_history
            formatted_history = format_chat_history(user_message.history)
            user_prompt = user_template.invoke(
                {
                    "history": formatted_history,
                    "context": message_flow,
                }
            ).text
        else:
            # Use template without context
            system_template = PromptTemplate.from_template(
                prompts["answer_node_prompt_without_context_system"]
            )
            user_template = PromptTemplate.from_template(
                prompts["answer_node_prompt_without_context"]
            )

            system_prompt = system_template.invoke(
                {
                    "sys_instruct": self.orch_state.sys_instruct,
                    "task": task,
                    "prompt": prompt,
                }
            ).text
            # Format history for the prompt (needs string format)
            from arklex.utils.utils import format_chat_history
            formatted_history = format_chat_history(user_message.history)
            user_prompt = user_template.invoke(
                {
                    "history": formatted_history,
                }
            ).text

        log_context.info(
            f"Answer Node system prompt prepared for {self.orch_state.stream_type}: {system_prompt}"
        )
        log_context.info(
            f"Answer Node user prompt prepared for {self.orch_state.stream_type}: {user_prompt}"
        )
        
        # Print statements to show prompt structure
        print("\n" + "="*80)
        print("OUTPUT PROCESS WORKER: Formatted Prompts")
        print("="*80)
        print(f"Stream Type: {self.orch_state.stream_type}")
        print(f"Task: {task}")
        print(f"Has context: {bool(message_flow and message_flow.strip())}")
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
        """Generate a response using the LLM."""
        model_service = ModelService(self.orch_state.bot_config.llm_config)
        # Get conversation history
        conversation_history = self.orch_state.user_message.history
        # Use the current user message as the prompt
        current_user_message = self.orch_state.user_message.message
        answer: str = model_service.get_response(
            current_user_message, system_prompt, conversation_history
        )
        return answer

    def stream_generator(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a streaming response using the LLM."""
        print("\n" + "="*80)
        print("OUTPUT PROCESS WORKER: stream_generator() - Streaming Response")
        print("="*80)
        print("Using formatted messages for streaming...")
        print("="*80 + "\n")
        
        model_service = ModelService(self.orch_state.bot_config.llm_config)
        # Get conversation history
        conversation_history = self.orch_state.user_message.history
        # Use the current user message as the prompt
        current_user_message = self.orch_state.user_message.message
        messages = model_service._format_messages(
            current_user_message, system_prompt, conversation_history
        )
        answer: str = ""
        for chunk in model_service.model.stream(messages):
            answer += chunk.content
            if (
                hasattr(self.orch_state, "message_queue")
                and self.orch_state.message_queue
            ):
                self.orch_state.message_queue.put(
                    {"event": EventType.CHUNK.value, "message_chunk": chunk.content}
                )
        
        print("\n" + "="*80)
        print("OUTPUT PROCESS WORKER: stream_generator() - Streaming Complete")
        print("="*80)
        print(f"Total response length: {len(answer)} characters")
        print("="*80 + "\n")
        
        return answer

    def _execute(self) -> OutputProcessWorkerOutput:
        """Execute the answer node worker."""
        # Format the prompts
        system_prompt, user_prompt = self._format_prompts()

        # Generate response based on stream type
        if (
            self.orch_state.stream_type == StreamType.TEXT
            or self.orch_state.stream_type == StreamType.SPEECH
        ):
            answer = self.stream_generator(system_prompt, user_prompt)
        else:
            answer = self.generator(system_prompt, user_prompt)

        # Clear the message flow after processing
        self.orch_state.message_flow = ""

        return OutputProcessWorkerOutput(
            response=answer,
            status=StatusEnum.COMPLETE,
        )
