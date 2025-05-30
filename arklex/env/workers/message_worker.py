"""Message worker implementation for the Arklex framework.

This module provides a specialized worker for handling message generation and delivery
in the Arklex framework. The MessageWorker class is responsible for processing user
messages, orchestrator messages, and generating appropriate responses. It supports
both streaming and non-streaming response generation, with functionality for handling
message flows and direct responses.
"""

import logging
from typing import Any, Dict, Optional

from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph

from arklex.env.prompts import load_prompts
from arklex.env.tools.utils import trace
from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.types import EventType, StreamType
from arklex.utils.graph_state import MessageState
from arklex.utils.model_provider_config import PROVIDER_MAP

logger = logging.getLogger(__name__)


@register_worker
class MessageWorker(BaseWorker):
    description: str = "The worker that used to deliver the message to the user, either a question or provide some information."

    def __init__(self) -> None:
        super().__init__()
        self.action_graph: StateGraph = self._create_action_graph()
        self.llm: Optional[BaseChatModel] = None

    def generator(self, state: MessageState) -> MessageState:
        # get the input message
        user_message = state.user_message
        orchestrator_message = state.orchestrator_message
        message_flow: str = state.response + "\n" + state.message_flow

        # get the orchestrator message content
        orch_msg_content: str = (
            "None" if not orchestrator_message.message else orchestrator_message.message
        )
        orch_msg_attr: Dict[str, Any] = orchestrator_message.attribute
        direct_response: bool = orch_msg_attr.get("direct_response", False)
        if direct_response:
            state.message_flow = ""
            state.response = orch_msg_content
            return state

        prompts: Dict[str, str] = load_prompts(state.bot_config)
        if message_flow and message_flow != "\n":
            prompt: PromptTemplate = PromptTemplate.from_template(
                prompts["message_flow_generator_prompt"]
            )
            input_prompt = prompt.invoke(
                {
                    "sys_instruct": state.sys_instruct,
                    "message": orch_msg_content,
                    "formatted_chat": user_message.history,
                    "context": message_flow,
                }
            )
        else:
            prompt: PromptTemplate = PromptTemplate.from_template(
                prompts["message_generator_prompt"]
            )
            input_prompt = prompt.invoke(
                {
                    "sys_instruct": state.sys_instruct,
                    "message": orch_msg_content,
                    "formatted_chat": user_message.history,
                }
            )
        logger.info(f"Prompt: {input_prompt.text}")
        final_chain = self.llm | StrOutputParser()
        answer: str = final_chain.invoke(input_prompt.text)

        state.message_flow = ""
        state.response = answer
        state = trace(input=answer, state=state)
        return state

    def text_stream_generator(self, state: MessageState) -> MessageState:
        # get the input message
        user_message = state.user_message
        orchestrator_message = state.orchestrator_message
        message_flow: str = state.response + "\n" + state.message_flow

        # get the orchestrator message content
        orch_msg_content: str = (
            "None" if not orchestrator_message.message else orchestrator_message.message
        )
        orch_msg_attr: Dict[str, Any] = orchestrator_message.attribute
        direct_response: bool = orch_msg_attr.get("direct_response", False)
        if direct_response:
            state.message_flow = ""
            state.response = orch_msg_content
            return state

        prompts: Dict[str, str] = load_prompts(state.bot_config)
        if message_flow and message_flow != "\n":
            prompt: PromptTemplate = PromptTemplate.from_template(
                prompts["message_flow_generator_prompt"]
            )
            input_prompt = prompt.invoke(
                {
                    "sys_instruct": state.sys_instruct,
                    "message": orch_msg_content,
                    "formatted_chat": user_message.history,
                    "context": message_flow,
                }
            )
        else:
            prompt: PromptTemplate = PromptTemplate.from_template(
                prompts["message_generator_prompt"]
            )
            input_prompt = prompt.invoke(
                {
                    "sys_instruct": state.sys_instruct,
                    "message": orch_msg_content,
                    "formatted_chat": user_message.history,
                }
            )
        logger.info(f"Prompt: {input_prompt.text}")
        final_chain = self.llm | StrOutputParser()
        answer: str = ""
        for chunk in final_chain.stream(input_prompt.text):
            answer += chunk
            state.message_queue.put(
                {"event": EventType.CHUNK.value, "message_chunk": chunk}
            )

        state.message_flow = ""
        state.response = answer
        return state

    def speech_stream_generator(self, state: MessageState) -> MessageState:
        # get the input message
        user_message = state.user_message
        orchestrator_message = state.orchestrator_message
        message_flow = state.response + "\n" + state.message_flow

        # get the orchestrator message content
        orch_msg_content = (
            "None" if not orchestrator_message.message else orchestrator_message.message
        )
        orch_msg_attr = orchestrator_message.attribute
        direct_response = orch_msg_attr.get("direct_response", False)
        if direct_response:
            state.message_flow = ""
            state.response = orch_msg_content
            return state

        prompts = load_prompts(state.bot_config)
        if message_flow and message_flow != "\n":
            prompt = PromptTemplate.from_template(
                prompts["message_flow_generator_prompt_speech"]
            )
            input_prompt = prompt.invoke(
                {
                    "sys_instruct": state.sys_instruct,
                    "message": orch_msg_content,
                    "formatted_chat": user_message.history,
                    "context": message_flow,
                }
            )
        else:
            prompt = PromptTemplate.from_template(
                prompts["message_generator_prompt_speech"]
            )
            input_prompt = prompt.invoke(
                {
                    "sys_instruct": state.sys_instruct,
                    "message": orch_msg_content,
                    "formatted_chat": user_message.history,
                }
            )
        logger.info(f"Prompt: {input_prompt.text}")
        final_chain = self.llm | StrOutputParser()
        answer = ""
        for chunk in final_chain.stream(input_prompt.text):
            answer += chunk
            state.message_queue.put(
                {"event": EventType.CHUNK.value, "message_chunk": chunk}
            )

        state.message_flow = ""
        state.response = answer
        return state

    def choose_generator(self, state: MessageState) -> str:
        if state.bot_config.language == "CN" and state.stream_type == StreamType.SPEECH:
            # we do not have seperate speech and text prompts for Chinese yet
            # TODO(Vishruth): add speech prompt for Chinese
            return "text_stream_generator"
        if (
            state.stream_type == StreamType.TEXT
            or state.stream_type == StreamType.AUDIO
        ):
            return "text_stream_generator"
        elif state.stream_type == StreamType.SPEECH:
            return "speech_stream_generator"
        return "generator"

    def _create_action_graph(self):
        workflow = StateGraph(MessageState)
        # Add nodes for each worker
        workflow.add_node("generator", self.generator)
        workflow.add_node("text_stream_generator", self.text_stream_generator)
        workflow.add_node("speech_stream_generator", self.speech_stream_generator)

        # Add edges
        # workflow.add_edge(START, "generator")
        workflow.add_conditional_edges(START, self.choose_generator)
        return workflow

    def _execute(self, msg_state: MessageState, **kwargs: Any) -> Dict[str, Any]:
        self.llm = PROVIDER_MAP.get(
            msg_state.bot_config.llm_config.llm_provider, ChatOpenAI
        )(model=msg_state.bot_config.llm_config.model_type_or_path)
        graph = self.action_graph.compile()
        result: Dict[str, Any] = graph.invoke(msg_state)
        return result
