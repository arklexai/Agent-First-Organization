import time
import uuid

import tiktoken

from arklex.resources.tools.rag.retrievers.milvus_retriever import MilvusRetriever
from arklex.resources.tools.rag.retrievers.retriever_document import (
    DEFAULT_CHUNK_ENCODING,
    RetrieverDocument,
    get_bot_uid,
)

faq_str = "[Question]: What is your company's main focus?\n[Answer] Arklex focuses on building AI agents and agent evaluation frameworks."


def insert_faq_documents(
    bot_id: str, version: str, collection_name: str, doc_str: str, tags: dict = None
) -> None:
    if tags is None:
        tags = {}
    encoding = tiktoken.get_encoding(DEFAULT_CHUNK_ENCODING)
    tokens = encoding.encode(doc_str)
    ret_dic = RetrieverDocument.faq_retreiver_doc(
        id=str(uuid.uuid4()),
        text=doc_str,
        metadata={"title": "faq", "tags": tags},
        bot_uid=get_bot_uid(bot_id, version),
        num_tokens=len(tokens),
        timestamp=time.time(),
    )
    with MilvusRetriever() as retriever:
        retriever.add_documents(collection_name, bot_id, version, [ret_dic])


def main() -> None:
    bot_id = "arklex"
    version = "v1"
    collection_name = "test_collection"

    with MilvusRetriever() as retriever:
        if not retriever.has_collection(collection_name):
            retriever.create_collection_with_partition_key(collection_name)

    insert_faq_documents(bot_id, version, collection_name, faq_str)


if __name__ == "__main__":
    main()
