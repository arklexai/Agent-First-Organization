import logging
import inspect

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from arklex.env.prompts import load_prompts
from arklex.types import EventType
from arklex.utils.graph_state import MessageState
from arklex.utils.model_provider_config import PROVIDER_MAP


logger = logging.getLogger(__name__)
    

class ToolGenerator():
    @staticmethod
    def generate(state: MessageState):
        llm_config = state.bot_config.llm_config
        user_message = state.user_message
        
        prompts = load_prompts(state.bot_config)
        llm = PROVIDER_MAP.get(llm_config.llm_provider, ChatOpenAI)(
            model=llm_config.model_type_or_path, temperature=0.1
        )
        prompt = PromptTemplate.from_template(prompts["generator_prompt"])
        input_prompt = prompt.invoke({"sys_instruct": state.sys_instruct, "formatted_chat": user_message.history})
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
        logger.info(f"Retrieved texts (from retriever/search engine to generator): {message_flow[:50]} ...")
        
        # generate answer based on the retrieved texts
        prompts = load_prompts(state.bot_config)
        prompt = PromptTemplate.from_template(prompts["context_generator_prompt"])
        input_prompt = prompt.invoke({"sys_instruct": state.sys_instruct, "formatted_chat": user_message.history, "context": message_flow})
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
        logger.info(f"Retrieved texts (from retriever/search engine to generator): {message_flow[:50]} ...")
        
        # generate answer based on the retrieved texts
        prompts = load_prompts(state.bot_config)
        prompt = PromptTemplate.from_template(prompts["context_generator_prompt"])
        input_prompt = prompt.invoke({"sys_instruct": state.sys_instruct, "formatted_chat": user_message.history, "context": message_flow})
        final_chain = llm | StrOutputParser()
        logger.info(f"Prompt: {input_prompt.text}")
        answer = ""
        for chunk in final_chain.stream(input_prompt.text):
            answer += chunk
            state.message_queue.put({"event": EventType.CHUNK.value, "message_chunk": chunk})

        state.message_flow = ""
        state.response = answer
        state = trace(input=answer, state=state)
        return state
    
    @staticmethod
    def stream_generate(state: MessageState):
        user_message = state.user_message
        
        prompts = load_prompts(state.bot_config)
        llm_config = state.bot_config.llm_config
        llm = PROVIDER_MAP.get(llm_config.llm_provider, ChatOpenAI)(
            model=llm_config.model_type_or_path, temperature=0.1
        )
        prompt = PromptTemplate.from_template(prompts["generator_prompt"])
        input_prompt = prompt.invoke({"sys_instruct": state.sys_instruct, "formatted_chat": user_message.history})
        final_chain = llm | StrOutputParser()
        answer = ""
        for chunk in final_chain.stream(input_prompt.text):
            answer += chunk
            state.message_queue.put({"event": EventType.CHUNK.value, "message_chunk": chunk})

        state.response = answer
        return state


def trace(input, state):
    current_frame = inspect.currentframe()
    previous_frame = current_frame.f_back if current_frame else None
    previous_function_name = previous_frame.f_code.co_name if previous_frame else "unknown"
    response_meta = {previous_function_name: input}
    state.trajectory[-1][-1].steps.append(response_meta)
    return state