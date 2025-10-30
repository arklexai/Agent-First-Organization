import re

from langchain.prompts import PromptTemplate

from arklex.memory.entities.memory_entities import ResourceRecord
from arklex.models.model_service import ModelService
from arklex.orchestrator.entities.orchestrator_state_entities import OrchestratorState
from arklex.orchestrator.executor.entities import NodeResponse
from arklex.resources.resource_types import WorkerItem
from arklex.utils.logging.logging_utils import LogContext
from arklex.utils.prompts import load_prompts

log_context = LogContext(__name__)

RAG_NODES_STEPS = {
    WorkerItem.MESSAGE_WORKER.value: "message",
    WorkerItem.FAISS_RAG_WORKER.value: "faiss_retrieve",
    WorkerItem.MILVUS_RAG_WORKER.value: "milvus_retrieve",
    WorkerItem.RAG_MESSAGE_WORKER.value: "milvus_retrieve",
}


def post_process_response(
    orch_state: OrchestratorState,
    node_response: NodeResponse,
) -> NodeResponse:
    """
    Post-processes the chatbot's response to ensure content quality and determine whether human takeover is needed.

    This function performs the following steps:
    1. **Link Validation**: Compares links in the bot's response against links present in the context.
    If the response includes invalid links, they are removed and the response is optionally regenerated via LLM.
    2. **HITL Proposal Trigger**: If HITL proposal is enabled and a HITL worker is available, determines
    whether to suggest a handoff to a human assistant based on confidence and relevance heuristics.

    Args:
        orch_state (OrchestratorState): Current state of the conversation including response, context, and metadata.
        node_response (NodeResponse): Response from the current node.

    Returns:
        NodeResponse: The updated node response with potentially cleaned or rephrased response,
                    and possibly a human handoff suggestion.
    """
    context_links = _build_context(orch_state.sys_instruct, orch_state.trajectory)
    response_links = _extract_links(node_response.response)
    missing_links = response_links - context_links
    if missing_links:
        log_context.info(
            f"Some answer links are NOT present in the context. Missing: {missing_links}"
        )
        node_response.response = _remove_invalid_links(
            node_response.response, missing_links
        )
        node_response.response = _rephrase_answer(orch_state, node_response.response)

    return node_response


def _build_context(sys_instruct: str, trajectory: list[list[ResourceRecord]]) -> set:
    context_links = _extract_links(sys_instruct)
    for resource_group in trajectory:
        for resource in resource_group:
            if _include_resource(resource):
                context_links.update(_extract_links(resource.output))
            rag_step_type = RAG_NODES_STEPS.get(
                resource.info.get("resource", {}).get("id")
            )
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


def _rephrase_answer(orch_state: OrchestratorState, response: str) -> str:
    """Rephrases the answer using an LLM after link removal."""
    model_service = ModelService(orch_state.bot_config.llm_config)
    prompt: PromptTemplate = PromptTemplate.from_template(
        load_prompts(orch_state.bot_config.language)["regenerate_response"]
    )
    input_prompt = prompt.invoke(
        {
            "sys_instruct": orch_state.sys_instruct,
            "original_answer": response,
            "formatted_chat": orch_state.user_message.history,
        }
    )
    log_context.info(f"Prompt: {input_prompt.text}")
    answer: str = model_service.get_response(input_prompt.text)
    return answer
