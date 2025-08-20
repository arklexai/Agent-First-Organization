"""Milvus retriever tool."""

from typing import Any, TypedDict

from arklex.env.tools.RAG.retrievers.milvus_retriever import RetrieveEngine
from arklex.env.tools.tools import register_tool
from arklex.orchestrator.entities.orchestrator_state_entities import BotConfig
from arklex.types.model_types import LLMConfig
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class RetrieverParams(TypedDict, total=False):
    """Parameters for the retriever tool."""

    collection_name: str
    bot_id: str
    version: str


description = "Retrieve relevant information required to answer a user's question. example: product price, product details, things for sale, company information, etc."

slots = [
    {
        "name": "query",
        "type": "str",
        "description": "The query to search for in the knowledge base",
        "prompt": "Please provide the minimum time to query the busy times",
        "required": True,
    }
]


@register_tool(description, slots)
def retriever(query: str, auth: RetrieverParams, **kwargs: dict[str, Any]) -> str:
    collection_name = auth.get("collection_name")
    bot_id = auth.get("bot_id")
    version = auth.get("version")

    log_context.info(
        f"Retrieving from collection {collection_name} for bot {bot_id} version {version} with query {query}"
    )

    # Build LLMConfig from provided model settings
    model_type_or_path = kwargs.get("model_type_or_path")
    llm_provider = kwargs.get("llm_provider")
    if not model_type_or_path or not llm_provider:
        raise ValueError(
            "model_type_or_path and llm_provider must be provided via llm_config"
        )

    if not bot_id or not version:
        raise ValueError("bot_id and version must be provided in auth")

    llm_config = LLMConfig(
        model_type_or_path=str(model_type_or_path), llm_provider=str(llm_provider)
    )
    bot_config = BotConfig(
        bot_id=str(bot_id), version=str(version), language="en", llm_config=llm_config
    )

    # Perform retrieval
    retrieved_text, _retriever_params = RetrieveEngine.milvus_retrieve(
        chat_history=query, bot_config=bot_config, tags=None
    )

    return retrieved_text
