import inspect
import logging

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from arklex.env.prompts import load_prompts
from arklex.types import EventType, StreamType
from arklex.utils.graph_state import MessageState
from arklex.utils.model_provider_config import PROVIDER_MAP

logger = logging.getLogger(__name__)


def get_prompt_template(state: MessageState, prompt_key: str) -> PromptTemplate:
    """Get the prompt template based on the stream type."""
    prompts = load_prompts(state.bot_config)
    
    if state.bot_config.language == "CN" and state.is_stream:
        # is_stream is True and language is Chinese
        # we do not have separate Chinese prompts for streaming text and audio
        # so we use the same prompt for both
        return PromptTemplate.from_template(prompts[prompt_key])
    
    if state.stream_type == StreamType.TEXT:
        # is_stream is True and stream_type is TEXT
        return PromptTemplate.from_template(prompts[prompt_key])
    elif state.stream_type == StreamType.AUDIO:
        # is stream is True and stream_type is AUDIO
        return PromptTemplate.from_template(prompts[prompt_key + "_speech"])
    else:
        # is_stream is False and the response is not streamed, therefore defaults to TEXT
        # because AUDIO is always streamed
        return PromptTemplate.from_template(prompts[prompt_key])


class ToolGenerator:
    @staticmethod
    def generate(state: MessageState):
        llm_config = state.bot_config.llm_config
        user_message = state.user_message

        llm = PROVIDER_MAP.get(llm_config.llm_provider, ChatOpenAI)(
            model=llm_config.model_type_or_path, temperature=0.1
        )
        prompt = get_prompt_template(state, "generator_prompt")
        input_prompt = prompt.invoke(
            {"sys_instruct": state.sys_instruct, "formatted_chat": user_message.history}
        )
        logger.info(f"Prompt: {input_prompt.text}")
        final_chain = llm | StrOutputParser()
        answer = final_chain.invoke(input_prompt.text)

        state.response = answer
        return state

    @staticmethod
    def context_generate(state: MessageState):
        llm_config = state.bot_config.llm_config
        llm = PROVIDER_MAP.get(llm_config.llm_provider, ChatOpenAI)(
            model=llm_config.model_type_or_path, temperature=0.1
        )
        # get the input message
        user_message = state.user_message
        message_flow = state.message_flow

        # Add relevant records to context if available
        if state.relevant_records:
            relevant_context = "\nRelevant past interactions:\n"
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
        prompt = get_prompt_template(state, "context_generator_prompt")
        input_prompt = prompt.invoke(
            {
                "sys_instruct": state.sys_instruct,
                "formatted_chat": user_message.history,
                "context": message_flow,
            }
        )
        final_chain = llm | StrOutputParser()
        logger.info(f"Prompt: {input_prompt.text}")
        answer = final_chain.invoke(input_prompt.text)
        state.message_flow = ""
        state.response = answer
        state = trace(input=answer, state=state)
        return state

    @staticmethod
    def stream_context_generate(state: MessageState):
        llm_config = state.bot_config.llm_config
        llm = PROVIDER_MAP.get(llm_config.llm_provider, ChatOpenAI)(
            model=llm_config.model_type_or_path, temperature=0.1
        )
        # get the input message
        user_message = state.user_message
        message_flow = state.message_flow
        # Add relevant records to context if available
        if state.relevant_records:
            relevant_context = "\nRelevant past interactions:\n"
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
        prompts = load_prompts(state.bot_config)
        if state.stream_type == StreamType.TEXT:
            prompt = PromptTemplate.from_template(prompts["context_generator_prompt"])
        elif state.stream_type == StreamType.AUDIO:
            prompt = PromptTemplate.from_template(
                prompts["context_generator_prompt_speech"]
            )
        else:
            raise ValueError(f"Unsupported stream type: {state.stream_type}")

        input_prompt = prompt.invoke(
            {
                "sys_instruct": state.sys_instruct,
                "formatted_chat": user_message.history,
                "context": message_flow,
            }
        )
        final_chain = llm | StrOutputParser()
        logger.info(f"Prompt: {input_prompt.text}")
        answer = ""
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
    def stream_generate(state: MessageState):
        user_message = state.user_message

        llm_config = state.bot_config.llm_config
        llm = PROVIDER_MAP.get(llm_config.llm_provider, ChatOpenAI)(
            model=llm_config.model_type_or_path, temperature=0.1
        )
        prompt = get_prompt_template(state, "generator_prompt")
        input_prompt = prompt.invoke(
            {"sys_instruct": state.sys_instruct, "formatted_chat": user_message.history}
        )
        final_chain = llm | StrOutputParser()
        answer = ""
        for chunk in final_chain.stream(input_prompt.text):
            answer += chunk
            state.message_queue.put(
                {"event": EventType.CHUNK.value, "message_chunk": chunk}
            )

        state.response = answer
        return state


def trace(input, state):
    current_frame = inspect.currentframe()
    previous_frame = current_frame.f_back if current_frame else None
    previous_function_name = (
        previous_frame.f_code.co_name if previous_frame else "unknown"
    )
    response_meta = {previous_function_name: input}
    state.trajectory[-1][-1].steps.append(response_meta)
    return state
