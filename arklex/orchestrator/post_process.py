import logging
import re
from typing import Any

from arklex.env.prompts import load_prompts
from arklex.env.workers.hitl_worker import HITLWorkerChatFlag
from arklex.utils.graph_state import MessageState, Params, ResourceRecord
from arklex.utils.model_provider_config import PROVIDER_MAP

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

RAG_NODES_STEPS = {
    "FaissRAGWorker": "faiss_retrieve",
    "milvus_rag_worker": "milvus_retrieve",
    "rag_message_worker": "milvus_retrieve",
}


def post_process_response(
    message_state: MessageState, params: Params, hitl_worker_available: bool
) -> MessageState:
    context_links = _build_context(message_state)
    response_links = _extract_links(message_state.response)
    missing_links = response_links - context_links
    if missing_links:
        logger.info(
            f"Some answer links are NOT present in the context. Missing: {missing_links}"
        )
        message_state.response = _remove_invalid_links(
            message_state.response, missing_links
        )
        message_state.response = _rephrase_answer(message_state)

    return message_state


def _build_context(message_state: MessageState) -> set:
    context_links = _extract_links(message_state.sys_instruct)
    for resource_group in message_state.trajectory:
        for resource in resource_group:
            if _include_resource(resource):
                context_links.update(_extract_links(resource.output))
            rag_step_type = RAG_NODES_STEPS.get(resource.info.get("id"))
            if rag_step_type:
                for step in resource.steps:
                    try:
                        if rag_step_type in step:
                            step_links = _extract_links_from_nested_dict(
                                step[rag_step_type]
                            )
                            context_links.update(step_links)
                    except Exception as e:
                        logger.warning(
                            f"Error extracting links from step: {e} — step: {step}"
                        )
    return context_links


def _include_resource(resource: ResourceRecord) -> bool:
    """Determines whether a ResourceRecord's output should be included in context.

    Excludes any output where a 'context_generate' flag is present in steps.
    """
    return not any(step.get("context_generate") for step in resource.steps)


def _extract_links(text: str) -> set:
    markdown_links = re.findall(r"\[[^\]]+\]\((https?://[^\s)]+)\)", text)
    cleaned_text = re.sub(r"\[[^\]]+\]\((https?://[^\s)]+)\)", "", text)
    raw_links = re.findall(r"(?:https?://|www\.)[^\s)\"']+", cleaned_text)
    all_links = set(markdown_links + raw_links)
    return {link.rstrip(".,;)!?\"'") for link in all_links}


def _extract_links_from_nested_dict(step: Any) -> set:
    links = set()

    def _recurse(val: Any) -> None:
        if isinstance(val, str):
            links.update(_extract_links(val))
        elif isinstance(val, dict):
            for v in val.values():
                _recurse(v)
        elif isinstance(val, list):
            for item in val:
                _recurse(item)

    _recurse(step)
    return links


def _remove_invalid_links(response: str, links: set) -> str:
    sorted_links = sorted([re.escape(link) for link in links], key=len, reverse=True)
    links_regex = "|".join(sorted_links)
    cleaned_response = re.sub(links_regex, "", response)
    return re.sub(r"\s+", " ", cleaned_response).strip()


def _rephrase_answer(state: MessageState) -> str:
    """Rephrases the answer using an LLM after link removal."""
    llm_config = state.bot_config.llm_config
    llm = PROVIDER_MAP.get(llm_config.llm_provider, ChatOpenAI)(
        model=llm_config.model_type_or_path, temperature=0.1
    )
    prompt: PromptTemplate = PromptTemplate.from_template(
        load_prompts(state.bot_config)["regenerate_response"]
    )
    input_prompt = prompt.invoke(
        {
            "sys_instruct": state.sys_instruct,
            "original_answer": state.response,
            "formatted_chat": state.user_message.history,
        }
    )
    final_chain = llm | StrOutputParser()
    logger.info(f"Prompt: {input_prompt.text}")
    answer: str = final_chain.invoke(input_prompt.text)
    return answer
