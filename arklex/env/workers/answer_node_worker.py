"""Answer Node worker implementation for the Arklex framework.

This module provides a specialized worker for handling answer node message generation
in the Arklex framework. The AnswerNodeWorker class is responsible for processing user
messages and generating responses using the task and prompt from the node info and
conversation history. It supports both streaming and non-streaming response generation.
"""

from typing import Any

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

from arklex.env.workers.base.base_worker import BaseWorker
from arklex.env.workers.base.entities import WorkerOutput
from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
    StatusEnum,
)
from arklex.types.stream_types import EventType, StreamType
from arklex.utils.llm_config import load_llm
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class AnswerNodeWorkerData(BaseModel):
    """Data for the answer node worker."""
    
    task: str = ""
    prompt: str = ""


class AnswerNodeWorkerOutput(WorkerOutput):
    """Response for the answer node worker."""
    
    response: str
    status: StatusEnum


class AnswerNodeWorker(BaseWorker):
    description: str = "The worker that generates responses using the task and prompt from the node info and conversation history."

    def __init__(self) -> None:
        super().__init__()
        self.orch_state: OrchestratorState | None = None
        self.answer_worker_data: AnswerNodeWorkerData | None = None
        self.llm = None

    def init_worker_data(
        self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
    ) -> None:
        """Initialize the worker data."""
        self.orch_state = orch_state
        self.answer_worker_data = AnswerNodeWorkerData(**node_specific_data)

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
        
        # Create a focused, efficient prompt template
        if message_flow and message_flow.strip():
            # Use template with context from previous nodes
            prompt_template = PromptTemplate.from_template(
                """{sys_instruct}

Your specific task: {task}

{prompt}

IMPORTANT: Respond directly to the user's question based on the task and context provided. Do not give generic responses.

Conversation history:
{history}

Context from previous operations:
{context}

Response:"""
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
                """{sys_instruct}

Your specific task: {task}

{prompt}

IMPORTANT: Respond directly to the user's question based on the task provided. Do not give generic responses.

Conversation history:
{history}

Response:"""
            )
            
            input_prompt = prompt_template.invoke(
                {
                    "sys_instruct": self.orch_state.sys_instruct,
                    "task": task,
                    "prompt": prompt,
                    "history": user_message.history,
                }
            )
        
        log_context.info(f"Answer Node prompt prepared for {self.orch_state.stream_type}: {input_prompt.text}")
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
            if hasattr(self.orch_state, 'message_queue') and self.orch_state.message_queue:
                self.orch_state.message_queue.put(
                    {"event": EventType.CHUNK.value, "message_chunk": chunk}
                )
        return answer

    def _execute(self) -> AnswerNodeWorkerOutput:
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
        
        return AnswerNodeWorkerOutput(
            response=answer,
            status=StatusEnum.COMPLETE,
        ) 