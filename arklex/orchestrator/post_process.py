import re

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from arklex.env.prompts import load_prompts
from arklex.memory.entities.memory_entities import ResourceRecord
from arklex.orchestrator.entities.msg_state_entities import MessageState
from arklex.orchestrator.entities.orchestrator_params_entities import OrchestratorParams
from arklex.utils.logging_utils import LogContext
from arklex.utils.provider_utils import validate_and_get_model_class

log_context = LogContext(__name__)

RAG_NODES_STEPS = {
    "FaissRAGWorker": "faiss_retrieve",
    "milvus_rag_worker": "milvus_retrieve",
    "rag_message_worker": "milvus_retrieve",
}

RAG_CONFIDENCE_THRESHOLD = {
    "FaissRAGWorker": 0.35,
    "milvus_rag_worker": 70.0,
    "rag_message_worker": 70.0,
}

TRIGGER_LIVE_CHAT_PROMPT = "Sorry, I'm not certain about the answer, would you like to connect to a human assistant?"


def post_process_response(
    message_state: MessageState,
    params: OrchestratorParams,
    hitl_worker_available: bool,
    hitl_proposal_enabled: bool,
) -> MessageState:
    """
    Post-processes the chatbot's response to ensure content quality and determine whether human takeover is needed.

    This function performs the following steps:
    1. **Link Validation**: Compares links in the bot's response against links present in the context.
    If the response includes invalid links, they are removed and the response is optionally regenerated via LLM.
    2. **HITL Proposal Trigger**: If HITL proposal is enabled and a HITL worker is available, determines
    whether to suggest a handoff to a human assistant based on confidence and relevance heuristics.

    Args:
        message_state (MessageState): Current state of the conversation including response, context, and metadata.
        params (OrchestratorParams): Additional configuration and NLU metadata.
        hitl_worker_available (bool): Flag indicating whether HITL worker is available
        hitl_proposal_enabled (bool): Flag indicating whether proactive HITL (human-in-the-loop) routing is allowed.

    Returns:
        MessageState: The updated message state with potentially cleaned or rephrased response,
                    and possibly a human handoff suggestion.
    """
    context_links = _build_context(message_state)
    response_links = _extract_links(message_state.response)
    missing_links = response_links - context_links
    if missing_links:
        log_context.info(
            f"Some answer links are NOT present in the context. Missing: {missing_links}"
        )
        message_state.response = _remove_invalid_links(
            message_state.response, missing_links
        )
        message_state.response = _rephrase_answer(message_state)

    if (
        hitl_worker_available
        and hitl_proposal_enabled
        and not message_state.metadata.hitl
    ):
        _live_chat_verifier(message_state, params)

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
                        log_context.warning(
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


def _extract_links_from_nested_dict(step: dict | list | str) -> set:
    links = set()

    def _recurse(val: dict | list | str) -> None:
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
    model_class = validate_and_get_model_class(llm_config)

    llm = model_class(model=llm_config.model_type_or_path, temperature=0.1)
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
    log_context.info(f"Prompt: {input_prompt.text}")
    answer: str = final_chain.invoke(input_prompt.text)
    return answer


def _live_chat_verifier(
    message_state: MessageState, params: OrchestratorParams
) -> None:
    """
    Determines if a live chat takeover is needed.
    Triggers handover if bot doesn't know the answer AND is NOT asking a clarifying question,
    and a HITL worker is available.
    """
    # early detection of confident bot response
    # if response has valid, verified link
    if _extract_links(message_state.response):
        return

    # check for relevance of the user's question
    if not _is_question_relevant(params):
        log_context.info(
            "User's question is not relevant. Skipping live chat initiation."
        )
        return

    # look at RAG confidence scores
    rag_confidence = 0.0
    num_of_docs = 0
    rag_confidence_threshold = 0.0

    if len(message_state.trajectory) >= 2:
        for resource in message_state.trajectory[-2]:
            rag_step_type = RAG_NODES_STEPS.get(resource.info.get("id"))
            if rag_step_type:
                for step in resource.steps:
                    try:
                        if rag_step_type in step:
                            rag_confidence_threshold = RAG_CONFIDENCE_THRESHOLD.get(
                                resource.info.get("id")
                            )
                            confidence, docs = _extract_confidence_from_nested_dict(
                                step
                            )
                            rag_confidence += confidence
                            num_of_docs += docs
                    except Exception as e:
                        log_context.warning(
                            f"Error extracting confidence from step: {e} — step: {step}"
                        )
    try:
        rag_avg_confidence = rag_confidence / num_of_docs
    except ZeroDivisionError:
        rag_avg_confidence = 0.0

    # confident in answer generated from RAG
    if rag_avg_confidence >= rag_confidence_threshold:
        return

    if should_trigger_handoff(message_state):
        message_state.response = TRIGGER_LIVE_CHAT_PROMPT


def _extract_confidence_from_nested_dict(step: dict | list | str) -> tuple[float, int]:
    confidence = 0.0
    num_of_docs = 0

    def _recurse(val: dict | list | str) -> None:
        if isinstance(val, dict):
            nonlocal confidence, num_of_docs
            if "confidence" in val and isinstance(val["confidence"], int | float):
                confidence += val["confidence"]
                num_of_docs += 1
            for v in val.values():
                _recurse(v)
        elif isinstance(val, list):
            for item in val:
                _recurse(item)

    _recurse(step)
    return confidence, num_of_docs


def _is_question_relevant(params: OrchestratorParams) -> bool:
    """Returns True if a question is relevant (no_intent is False), False otherwise.
    To be improved in the future to be more robust
    """
    return params.taskgraph.nlu_records and not params.taskgraph.nlu_records[-1].get(
        "no_intent", False
    )


def should_trigger_handoff(state: MessageState) -> bool:
    input_prompt = f"""
    You are an AI assistant evaluating a chatbot's response to determine if human intervention is needed.

    Chatbot's Response to User:
    \"\"\"{state.response}\"\"\"

    Does this response indicate the chatbot:
    1.  **Does NOT know the answer** (e.g., it's generic, evasive, or explicitly states lack of information)?
    2.  **Is NOT attempting to ask a clarifying question** (e.g., asking for more details, or offering specific options to narrow down the query)?

    Respond "YES" if BOTH conditions are met (bot is stuck and not trying to clarify).
    Otherwise, respond "NO".
    Your response must be "YES" or "NO" only.
    """

    llm_config = state.bot_config.llm_config
    model_class = validate_and_get_model_class(llm_config)

    llm = model_class(model=llm_config.model_type_or_path, temperature=0.1)

    final_chain = llm | StrOutputParser()
    result: str = final_chain.invoke(input_prompt)

    return result.strip().lower() == "yes"
