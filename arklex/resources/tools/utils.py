"""Utility tools for the Arklex framework.

This module provides utility tools and helper functions for the Arklex framework,
including the ToolGenerator class for generating responses and handling streaming
outputs. It also includes functions for tracing execution flow and managing message
states. The module integrates with various language models and prompt templates to
provide flexible response generation capabilities.
"""

from typing import Any, Protocol

from langchain.prompts import PromptTemplate

from arklex.models.model_service import ModelService
from arklex.orchestrator.entities.orchestrator_state_entities import OrchestratorState
from arklex.orchestrator.types.stream_types import EventType, StreamType
from arklex.utils.logging.logging_utils import LogContext
from arklex.utils.prompts import load_prompts

log_context = LogContext(__name__)


class ToolExecutor(Protocol):
    """Protocol for objects that can execute tools."""

    tools: dict[str, Any]


def get_prompt_template(state: OrchestratorState, prompt_key: str) -> PromptTemplate:
    """Get the prompt template based on the stream type."""
    prompts: dict[str, str] = load_prompts(state.bot_config.language)

    if state.stream_type == StreamType.SPEECH:
        # Use speech prompts, but fall back to regular prompts for Chinese
        # since Chinese speech prompts are not available yet
        if state.bot_config.language == "CN":
            return PromptTemplate.from_template(prompts[prompt_key])
        else:
            return PromptTemplate.from_template(prompts[prompt_key + "_speech"])
    else:
        return PromptTemplate.from_template(prompts[prompt_key])


class ToolGenerator:
    @staticmethod
    def generate(state: OrchestratorState) -> str:
        llm_config: dict[str, Any] = state.bot_config.llm_config
        user_message: Any = state.user_message

        model_service: ModelService = ModelService(llm_config)
        prompt: PromptTemplate = get_prompt_template(state, "generator_prompt")
        input_prompt: Any = prompt.invoke(
            {"sys_instruct": state.sys_instruct, "formatted_chat": user_message.history}
        )
        log_context.info(f"Prompt: {input_prompt.text}")
        answer: str = model_service.get_response(input_prompt.text)

        return answer

    @staticmethod
    def context_generate(state: OrchestratorState) -> str:
        llm_config: dict[str, Any] = state.bot_config.llm_config

        model_service: ModelService = ModelService(llm_config)
        # get the input message
        user_message: Any = state.user_message
        message_flow: str = state.message_flow

        # Add relevant records to context if available
        if state.relevant_records:
            relevant_context: str = "\nRelevant past interactions:\n"
            for record in state.relevant_records:
                relevant_context += "Record:\n"
                if record.info:
                    relevant_context += f"- Info: {record.info}\n"
                if record.personalized_intent:
                    relevant_context += (
                        f"- Personalized User Intent: {record.personalized_intent}\n"
                    )
                if record.output:
                    relevant_context += f"- Raw Output: {record.output}\n"
                if record.steps:
                    relevant_context += "- Intermediate Steps:\n"
                    for step in record.steps:
                        if isinstance(step, dict):
                            for key, value in step.items():
                                relevant_context += f"  * {key}: {value}\n"
                        else:
                            relevant_context += f"  * {step}\n"
                relevant_context += "\n"
            message_flow = relevant_context + "\n" + message_flow

        log_context.info(
            f"Retrieved texts (from retriever/search engine to generator): {message_flow[:50]} ..."
        )

        # generate answer based on the retrieved texts
        prompt: PromptTemplate = get_prompt_template(state, "context_generator_prompt")
        input_prompt: Any = prompt.invoke(
            {
                "sys_instruct": state.sys_instruct,
                "formatted_chat": user_message.history,
                "context": message_flow,
            }
        )
        log_context.info(f"Prompt: {input_prompt.text}")
        answer: str = model_service.get_response(input_prompt.text)
        state.message_flow = ""
        # state = trace(input=answer, state=state)
        return answer

    @staticmethod
    def stream_context_generate(state: OrchestratorState) -> str:
        llm_config: dict[str, Any] = state.bot_config.llm_config

        model_service: ModelService = ModelService(llm_config)
        # get the input message
        user_message: Any = state.user_message
        message_flow: str = state.message_flow
        # Add relevant records to context if available
        if state.relevant_records:
            relevant_context: str = "\nRelevant past interactions:\n"
            for record in state.relevant_records:
                relevant_context += "Record:\n"
                if record.info:
                    relevant_context += f"- Info: {record.info}\n"
                if record.personalized_intent:
                    relevant_context += (
                        f"- Personalized User Intent: {record.personalized_intent}\n"
                    )
                if record.output:
                    relevant_context += f"- Raw Output: {record.output}\n"
                if record.steps:
                    relevant_context += "- Intermediate Steps:\n"
                    for step in record.steps:
                        if isinstance(step, dict):
                            for key, value in step.items():
                                relevant_context += f"  * {key}: {value}\n"
                        else:
                            relevant_context += f"  * {step}\n"
                relevant_context += "\n"
            message_flow = relevant_context + "\n" + message_flow
        log_context.info(
            f"Retrieved texts (from retriever/search engine to generator): {message_flow[:50]} ..."
        )

        # generate answer based on the retrieved texts
        prompt: PromptTemplate = get_prompt_template(state, "context_generator_prompt")

        input_prompt: Any = prompt.invoke(
            {
                "sys_instruct": state.sys_instruct,
                "formatted_chat": user_message.history,
                "context": message_flow,
            }
        )
        log_context.info(f"Prompt: {input_prompt.text}")
        answer: str = ""
        for chunk in model_service.model.stream(input_prompt.text):
            answer += chunk.content
            state.message_queue.put(
                {"event": EventType.CHUNK.value, "message_chunk": chunk.content}
            )

        state.message_flow = ""
        # state = trace(input=answer, state=state)
        return answer

    @staticmethod
    def stream_generate(state: OrchestratorState) -> str:
        user_message: Any = state.user_message

        llm_config: dict[str, Any] = state.bot_config.llm_config
        model_service: ModelService = ModelService(llm_config)

        prompt: PromptTemplate = get_prompt_template(state, "generator_prompt")
        input_prompt: Any = prompt.invoke(
            {"sys_instruct": state.sys_instruct, "formatted_chat": user_message.history}
        )
        answer: str = ""
        for chunk in model_service.model.stream(input_prompt.text):
            answer += chunk.content
            state.message_queue.put(
                {"event": EventType.CHUNK.value, "message_chunk": chunk.content}
            )

        return answer


def trace(input: str, source: str, state: OrchestratorState) -> OrchestratorState:
    response_meta: dict[str, str] = {source: input}
    state.trajectory[-1][-1].steps.append(response_meta)
    return state
