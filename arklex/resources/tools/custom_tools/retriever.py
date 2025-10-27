"""Milvus retriever tool."""

from typing import Any, TypedDict

from arklex.resources.tools.rag.retrievers.milvus_retriever import MilvusRetriever
from arklex.resources.tools.tools import register_tool
from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)


class RetrieverParams(TypedDict, total=False):
    """Parameters for the retriever tool."""

    collection_name: str
    bot_id: str
    version: str


description = "Retrieve relevant information required to answer an user's question. example: product price, product details, things for sale, company information, etc."

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
    with MilvusRetriever() as retriever:
        retriever_results = retriever.search(collection_name, bot_id, version, query)

    retrieved_str = ""
    for doc in retriever_results:
        if doc.metadata.get("title"):
            retrieved_str += "title: " + doc.metadata["title"] + "\n"
        if doc.metadata.get("source"):
            retrieved_str += "source: " + doc.metadata["source"] + "\n"
        retrieved_str += "content: " + doc.text + "\n\n"

    return retrieved_str
