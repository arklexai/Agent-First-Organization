"""Answer Node worker implementation for the Arklex framework.

This module provides a specialized worker for handling answer node message generation
in the Arklex framework. The AnswerNodeWorker class is responsible for processing user
messages and generating responses using the task and prompt from the node info and
conversation history. It supports both streaming and non-streaming response generation.
"""

from typing import Any

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from arklex.models.llm_config import load_llm
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

    def _format_prompt(self) -> str:
        """Format the prompt for the answer node worker."""
        user_message = self.orch_state.user_message
        message_flow = self.orch_state.message_flow

        # Get the task and prompt from the worker data
        task = self.answer_worker_data.task
        prompt = self.answer_worker_data.prompt

        if not task and not prompt:
            log_context.warning("No task or prompt provided in worker data")
            return "I don't have a specific task to perform."

        # Load prompts based on bot configuration
        prompts = load_prompts(self.orch_state.bot_config.language)

        # Create a focused, efficient prompt template
        if message_flow and message_flow.strip():
            # Use template with context from previous nodes
            prompt_template = PromptTemplate.from_template(
                prompts["answer_node_prompt_with_context"]
            )

            input_prompt = prompt_template.invoke(
                {
                    "sys_instruct": self.orch_state.sys_instruct,
                    "task": task,
                    "prompt": prompt,
                    "history": user_message.history,
                    "context": message_flow,
                }
            )
        else:
            # Use template without context
            prompt_template = PromptTemplate.from_template(
                prompts["answer_node_prompt_without_context"]
            )

            input_prompt = prompt_template.invoke(
                {
                    "sys_instruct": self.orch_state.sys_instruct,
                    "task": task,
                    "prompt": prompt,
                    "history": user_message.history,
                }
            )

        log_context.info(
            f"Answer Node prompt prepared for {self.orch_state.stream_type}: {input_prompt.text}"
        )
        return input_prompt.text

    def generator(self, prompt: str) -> str:
        """Generate a response using the LLM."""
        invoke_chain = self.llm | StrOutputParser()
        answer: str = invoke_chain.invoke(prompt)
        return answer

    def stream_generator(self, prompt: str) -> str:
        """Generate a streaming response using the LLM."""
        invoke_chain = self.llm | StrOutputParser()
        answer: str = ""
        for chunk in invoke_chain.stream(prompt):
            answer += chunk
            if (
                hasattr(self.orch_state, "message_queue")
                and self.orch_state.message_queue
            ):
                self.orch_state.message_queue.put(
                    {"event": EventType.CHUNK.value, "message_chunk": chunk}
                )
        return answer

    def _execute(self) -> OutputProcessWorkerOutput:
        """Execute the answer node worker."""
        # Format the prompt
        input_prompt = self._format_prompt()

        # Initialize the LLM
        self.llm = load_llm(self.orch_state.bot_config.llm_config)

        # Generate response based on stream type
        if (
            self.orch_state.stream_type == StreamType.TEXT
            or self.orch_state.stream_type == StreamType.SPEECH
        ):
            answer = self.stream_generator(input_prompt)
        else:
            answer = self.generator(input_prompt)

        # Clear the message flow after processing
        self.orch_state.message_flow = ""

        return OutputProcessWorkerOutput(
            response=answer,
            status=StatusEnum.COMPLETE,
        )
