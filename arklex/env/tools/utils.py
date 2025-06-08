"""Utility tools for the Arklex framework.

This module provides utility tools and helper functions for the Arklex framework,
including the ToolGenerator class for generating responses and handling streaming
outputs. It also includes functions for tracing execution flow and managing message
states. The module integrates with various language models and prompt templates to
provide flexible response generation capabilities.
"""

import inspect
import logging
from typing import Dict, Any, List, Optional, Union, TypeVar, Generic

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from arklex.env.prompts import load_prompts
from arklex.types import EventType, StreamType
from arklex.utils.graph_state import MessageState
from arklex.utils.model_provider_config import PROVIDER_MAP

logger: logging.Logger = logging.getLogger(__name__)


def get_prompt_template(state: MessageState, prompt_key: str) -> PromptTemplate:
    """Get the prompt template based on the stream type."""
    prompts: Dict[str, str] = load_prompts(state.bot_config)

    if state.stream_type == StreamType.SPEECH:
        if state.bot_config.language == "CN":
            # no speech prompts for Chinese yet
            # TODO(Vishruth): add speech prompts for Chinese
            return PromptTemplate.from_template(prompts[prompt_key])
        else:
            return PromptTemplate.from_template(prompts[prompt_key + "_speech"])

    # the regular text prompts are used for both text stream and text without stream
    return PromptTemplate.from_template(prompts[prompt_key])


class ToolGenerator:
    @staticmethod
    def generate(state: MessageState) -> MessageState:
        llm_config: Dict[str, Any] = state.bot_config.llm_config
        user_message: Any = state.user_message

        llm: Any = PROVIDER_MAP.get(llm_config.llm_provider, ChatOpenAI)(
            model=llm_config.model_type_or_path, temperature=0.1
        )
        prompt: PromptTemplate = get_prompt_template(state, "generator_prompt")
        input_prompt: Any = prompt.invoke(
            {"sys_instruct": state.sys_instruct, "formatted_chat": user_message.history}
        )
        logger.info(f"Prompt: {input_prompt.text}")
        final_chain: Any = llm | StrOutputParser()
        answer: str = final_chain.invoke(input_prompt.text)

        state.response = answer
        return state

    @staticmethod
    def context_generate(state: MessageState) -> MessageState:
        llm_config: Dict[str, Any] = state.bot_config.llm_config
        llm: Any = PROVIDER_MAP.get(llm_config.llm_provider, ChatOpenAI)(
            model=llm_config.model_type_or_path, temperature=0.1
        )
        # get the input message
        user_message: Any = state.user_message
        message_flow: str = state.message_flow

        # Add relevant records to context if available
        if state.relevant_records:
            relevant_context: str = "\nRelevant past interactions:\n"
            for record in state.relevant_records:
                relevant_context += f"Record:\n"
                if record.info:
                    relevant_context += f"- Info: {record.info}\n"
                if record.personalized_intent:
                    relevant_context += (
                        f"- Personalized User Intent: {record.personalized_intent}\n"
                    )
                if record.output:
                    relevant_context += f"- Raw Output: {record.output}\n"
                if record.steps:
                    relevant_context += f"- Intermediate Steps:\n"
                    for step in record.steps:
                        if isinstance(step, dict):
                            for key, value in step.items():
                                relevant_context += f"  * {key}: {value}\n"
                        else:
                            relevant_context += f"  * {step}\n"
                relevant_context += "\n"
            message_flow = relevant_context + "\n" + message_flow

        logger.info(
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
        final_chain: Any = llm | StrOutputParser()
        logger.info(f"Prompt: {input_prompt.text}")
        answer: str = final_chain.invoke(input_prompt.text)
        state.message_flow = ""
        state.response = answer
        state = trace(input=answer, state=state)
        return state

    @staticmethod
    def stream_context_generate(state: MessageState) -> MessageState:
        llm_config: Dict[str, Any] = state.bot_config.llm_config
        llm: Any = PROVIDER_MAP.get(llm_config.llm_provider, ChatOpenAI)(
            model=llm_config.model_type_or_path, temperature=0.1
        )
        # get the input message
        user_message: Any = state.user_message
        message_flow: str = state.message_flow
        # Add relevant records to context if available
        if state.relevant_records:
            relevant_context: str = "\nRelevant past interactions:\n"
            for record in state.relevant_records:
                relevant_context += f"Record:\n"
                if record.info:
                    relevant_context += f"- Info: {record.info}\n"
                if record.personalized_intent:
                    relevant_context += (
                        f"- Personalized User Intent: {record.personalized_intent}\n"
                    )
                if record.output:
                    relevant_context += f"- Raw Output: {record.output}\n"
                if record.steps:
                    relevant_context += f"- Intermediate Steps:\n"
                    for step in record.steps:
                        if isinstance(step, dict):
                            for key, value in step.items():
                                relevant_context += f"  * {key}: {value}\n"
                        else:
                            relevant_context += f"  * {step}\n"
                relevant_context += "\n"
            message_flow = relevant_context + "\n" + message_flow
        logger.info(
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
        final_chain: Any = llm | StrOutputParser()
        logger.info(f"Prompt: {input_prompt.text}")
        answer: str = ""
        for chunk in final_chain.stream(input_prompt.text):
            answer += chunk
            state.message_queue.put(
                {"event": EventType.CHUNK.value, "message_chunk": chunk}
            )

        state.message_flow = ""
        state.response = answer
        state = trace(input=answer, state=state)
        return state

    @staticmethod
    def stream_generate(state: MessageState) -> MessageState:
        user_message: Any = state.user_message

        llm_config: Dict[str, Any] = state.bot_config.llm_config
        llm: Any = PROVIDER_MAP.get(llm_config.llm_provider, ChatOpenAI)(
            model=llm_config.model_type_or_path, temperature=0.1
        )
        prompt: PromptTemplate = get_prompt_template(state, "generator_prompt")
        input_prompt: Any = prompt.invoke(
            {"sys_instruct": state.sys_instruct, "formatted_chat": user_message.history}
        )
        final_chain: Any = llm | StrOutputParser()
        answer: str = ""
        for chunk in final_chain.stream(input_prompt.text):
            answer += chunk
            state.message_queue.put(
                {"event": EventType.CHUNK.value, "message_chunk": chunk}
            )

        state.response = answer
        return state


def trace(input: str, state: MessageState) -> MessageState:
    current_frame: Optional[inspect.FrameInfo] = inspect.currentframe()
    previous_frame: Optional[inspect.FrameInfo] = (
        current_frame.f_back if current_frame else None
    )
    previous_function_name: str = (
        previous_frame.f_code.co_name if previous_frame else "unknown"
    )
    response_meta: Dict[str, str] = {previous_function_name: input}
    state.trajectory[-1][-1].steps.append(response_meta)
    return state
