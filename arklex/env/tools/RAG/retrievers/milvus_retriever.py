"""Milvus retriever implementation for the Arklex framework.

This module provides a Milvus-based retriever implementation for efficient vector similarity
search in document collections. It includes the RetrieveEngine class for handling retrieval
operations, the MilvusRetriever class for managing Milvus collections and document operations,
and the MilvusRetrieverExecutor class for executing retrieval tasks. The module supports
document embedding, vector storage, and similarity search with metadata filtering.
"""

# TODO: Implement multi-tag support for document retrieval
# TODO: Add token counting for RAG document processing

# NOTE: Only support one tag for now
# TODO: get num_tokens for functions inside milvus_retriever.py and retriever_document.py (with classmethod RetrieverDocument.faq_retreiver_doc); influence token migrations

import os
import time
from collections import defaultdict
from multiprocessing.pool import Pool

import numpy as np
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pymilvus import Collection, DataType, MilvusClient, connections

from arklex.env.prompts import load_prompts
from arklex.env.tools.RAG.retrievers.retriever_document import (
    RetrieverDocument,
    RetrieverDocumentType,
    RetrieverResult,
    embed,
    embed_retriever_document,
)
from arklex.env.tools.utils import trace
from arklex.orchestrator.entities.msg_state_entities import MessageState
from arklex.utils.logging_utils import LogContext
from arklex.utils.mysql import mysql_pool
from arklex.utils.provider_utils import validate_and_get_model_class

EMBED_DIMENSION = 1536
MAX_TEXT_LENGTH = 65535
CHUNK_NEIGHBOURS = 3

log_context = LogContext(__name__)


class RetrieveEngine:
    @staticmethod
    def milvus_retrieve(
        state: MessageState, tags: dict[str, object] | None = None
    ) -> MessageState:
        # get the input message
        user_message = state.user_message

        # Search for the relevant documents
        milvus_retriever = MilvusRetrieverExecutor(state.bot_config)
        if tags is None:
            tags = {}
        retrieved_text, retriever_params = milvus_retriever.retrieve(
            user_message.history, tags
        )

        state.message_flow = retrieved_text
        state = trace(input=retriever_params, state=state)
        return state


class MilvusRetriever:
    def __enter__(self) -> "MilvusRetriever":
        self.uri = os.getenv("MILVUS_URI", "")
        self.token = os.getenv("MILVUS_TOKEN", "")
        self.client = MilvusClient(uri=self.uri, token=self.token)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.client.close()

    def get_bot_uid(self, bot_id: str, version: str) -> str:
        return f"{bot_id}__{version}"

    def create_collection_with_partition_key(self, collection_name: str) -> None:
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
            partition_key_field="bot_uid",
            num_partitions=16,
        )
        schema.add_field(
            field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100
        )
        schema.add_field(
            field_name="qa_doc_id", datatype=DataType.VARCHAR, max_length=40
        )
        schema.add_field(
            field_name="bot_uid", datatype=DataType.VARCHAR, max_length=100
        )
        schema.add_field(field_name="chunk_id", datatype=DataType.INT32)
        schema.add_field(
            field_name="qa_doc_type", datatype=DataType.VARCHAR, max_length=10
        )
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        schema.add_field(
            field_name="text", datatype=DataType.VARCHAR, max_length=MAX_TEXT_LENGTH
        )
        schema.add_field(
            field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=EMBED_DIMENSION
        )
        schema.add_field(field_name="timestamp", datatype=DataType.INT64)
        # schema.add_field(field_name="num_tokens", datatype=DataType.INT64)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="id")
        index_params.add_index(field_name="qa_doc_id")
        index_params.add_index(field_name="bot_uid")
        index_params.add_index(
            field_name="embedding", index_type="FLAT", metric_type="L2"
        )

        self.client.create_collection(
            collection_name=collection_name, schema=schema, index_params=index_params
        )

    def delete_documents_by_qa_ids(
        self, collection_name: str, qa_ids: list[str]
    ) -> dict[str, object]:
        log_context.info(
            f"Deleting vector db documents by qa_ids: {qa_ids} from collection: {collection_name}"
        )
        quoted_ids = ",".join([f"'{qa_id}'" for qa_id in qa_ids])
        filter_expr = f"id in [{quoted_ids}]"
        res = self.client.delete(
            collection_name=collection_name, filter=filter_expr
        )
        return res

    def delete_documents_by_qa_doc_id(
        self, collection_name: str, qa_doc_id: str
    ) -> dict[str, object]:
        log_context.info(
            f"Deleting vector db documents by qa_doc_id: {qa_doc_id} from collection: {collection_name}"
        )
        res = self.client.delete(
            collection_name=collection_name, filter=f"qa_doc_id=='{qa_doc_id}'"
        )
        return res

    def add_documents_dicts(
        self, documents: list[dict], collection_name: str, upsert: bool = False
    ) -> list[dict[str, object]]:
        log_context.info(
            f"Celery sub task for adding {len(documents)} documents to collection: {collection_name}."
        )
        retriever_documents = [RetrieverDocument.from_dict(doc) for doc in documents]
        documents_to_insert = []

        if not upsert:
            # check if the document already exists in the collection
            for doc in retriever_documents:
                res = self.client.get(collection_name=collection_name, ids=doc.id)
                if len(res) == 0:
                    documents_to_insert.append(doc)
            log_context.info(f"Exisiting documents: {len(documents_to_insert)}")
        else:
            documents_to_insert = retriever_documents

        res = []
        for doc in documents_to_insert:
            data = doc.to_milvus_schema_dict_and_embed()
            try:
                res.append(
                    self.client.upsert(collection_name=collection_name, data=[data])
                )
            except Exception as e:
                log_context.error(f"Error adding document id: {data['id']} error: {e}")
                raise e
        return res

    def update_tag_by_qa_doc_id(
        self, collection_name: str, qa_doc_id: str, tags: dict
    ) -> dict[str, object]:
        """
        Updates tags for all vector entries associated with a specific qa_doc_id.

        Args:
            collection_name: The name of the Milvus collection.
            qa_doc_id: The qa_doc_id to identify the vectors.
            tags: The new tags dictionary to apply.
        """
        log_context.info(
            f"Updating metadata for qa_doc_id {qa_doc_id} in collection {collection_name}"
        )

        # Query all vectors matching the qa_doc_id
        res = self.client.query(
            collection_name=collection_name,
            filter=f"qa_doc_id == '{qa_doc_id}'",
            output_fields=[
                "id",
                "qa_doc_id",
                "bot_uid",
                "chunk_id",
                "qa_doc_type",
                "metadata",
                "text",
                "embedding",
                "timestamp",
            ],
        )

        if len(res) == 0:
            log_context.error(
                f"No vectors found for qa_doc_id {qa_doc_id} in collection {collection_name}. No update performed."
            )
            raise ValueError(
                f"No vectors found for qa_doc_id {qa_doc_id} in collection {collection_name}. No update performed."
            )

        log_context.info(
            f"Found {len(res)} vectors for qa_doc_id {qa_doc_id}. Preparing update."
        )

        updated_vectors = []
        for vector_data in res:
            updated_vector = vector_data
            if updated_vector.get("metadata", {}):
                updated_vector["metadata"]["tags"] = tags
            else:
                updated_vector["metadata"] = {"tags": tags}
            updated_vectors.append(updated_vector)

        # Upsert the updated vectors
        try:
            res = self.client.upsert(
                collection_name=collection_name, data=updated_vectors
            )
            log_context.info(
                f"Successfully upserted {len(updated_vectors)} vectors with new tags {tags} for qa_doc_id {qa_doc_id}. Upsert result: {res}"
            )
            return res
        except Exception as e:
            log_context.error(
                f"Failed to upsert updated vectors for qa_doc_id {qa_doc_id}: {e}"
            )
            raise ValueError(
                f"Failed to upsert updated vectors with new tags {tags} for qa_doc_id {qa_doc_id}: {e}"
            ) from e

    def add_documents_parallel(
        self,
        collection_name: str,
        bot_id: str,
        version: str,
        documents: list[RetrieverDocument],
        process_pool: Pool,
        upsert: bool = False,
    ) -> list[dict[str, object]]:
        log_context.info(
            f"Adding {len(documents)} vector db documents to collection '{collection_name}' for bot_id: {bot_id} version: {version}"
        )
        if not self.client.has_collection(collection_name):
            log_context.info(
                f"No collection found hence creating collection: {collection_name}"
            )
            self.create_collection_with_partition_key(collection_name)

        documents_to_insert = []

        if not upsert:
            # check if the document already exists in the collection
            for doc in documents:
                res = self.client.get(collection_name=collection_name, ids=doc.id)
                if len(res) == 0:
                    documents_to_insert.append(doc)
            log_context.info(f"Exisiting documents: {len(documents_to_insert)}")
        else:
            documents_to_insert = documents

        res = []
        # process 100 documents at a time
        count = 0
        for i in range(0, len(documents_to_insert), 100):
            batch_docs = documents_to_insert[i : i + 100]
            embedded_batch_docs = process_pool.map(embed_retriever_document, batch_docs)

            res.extend(
                self.client.upsert(
                    collection_name=collection_name, data=embedded_batch_docs
                )
            )
            count += len(batch_docs)
            log_context.info(f"Added {count}/{len(documents_to_insert)} docs")

        return res

    def add_documents(
        self,
        collection_name: str,
        bot_id: str,
        version: str,
        documents: list[RetrieverDocument],
        upsert: bool = False,
    ) -> list[dict[str, object]]:
        log_context.info(
            f"Adding {len(documents)} vector db documents to collection {collection_name} for bot_id: {bot_id} version: {version}"
        )

        if not self.client.has_collection(collection_name):
            self.create_collection_with_partition_key(collection_name)

        documents_to_insert = []

        if not upsert:
            # check if the document already exists in the collection
            for doc in documents:
                res = self.client.get(collection_name=collection_name, ids=doc.id)
                if len(res) == 0:
                    documents_to_insert.append(doc)
            log_context.info(f"Exisiting documents: {len(documents_to_insert)}")
        else:
            documents_to_insert = documents

        res = []
        for doc in documents_to_insert:
            data = doc.to_milvus_schema_dict_and_embed()
            try:
                res.append(
                    self.client.upsert(collection_name=collection_name, data=[data])
                )
            except Exception as e:
                log_context.error(f"Error adding document id: {data['id']} error: {e}")
                raise e
        return res

    def search(
        self,
        collection_name: str,
        bot_id: str,
        version: str,
        query: str,
        tags: dict[str, object] | None = None,
        top_k: int = 4,
    ) -> list[RetrieverResult]:
        log_context.info(
            f"Retreiver search for query: {query} on collection {collection_name} for bot_id: {bot_id} version: {version}"
        )

        if tags is None:
            tags = {}

        partition_key = self.get_bot_uid(bot_id, version)
        query_embedding = embed(query)
        filter = f'bot_uid == "{partition_key}"'
        if tags:
            # NOTE: Only support one tag for now
            for key, value in tags.items():
                filter += f' and metadata["tags"]["{key}"] == "{value}"'
                break
        res = self.client.search(
            collection_name=collection_name,
            data=[query_embedding],
            limit=top_k,
            filter=filter,
            output_fields=["qa_doc_id", "chunk_id", "qa_doc_type", "metadata", "text"],
        )

        ret_results: list[RetrieverResult] = []
        for r in res[0]:
            log_context.info(f"Milvus search result: {r}")
            qa_doc_id = r["entity"]["qa_doc_id"]
            chunk_id = r["entity"]["chunk_id"]
            text = r["entity"]["text"]
            log_context.info(f"Retrieved qa_doc_id: {qa_doc_id} chunk_id: {chunk_id}")

            ret_results.append(
                RetrieverResult(
                    qa_doc_id=qa_doc_id,
                    qa_doc_type=RetrieverDocumentType(r["entity"]["qa_doc_type"]),
                    text=text,
                    metadata=r["entity"]["metadata"],
                    distance=r["distance"],
                    start_chunk_idx=chunk_id,
                    end_chunk_idx=chunk_id,
                )
            )

        return ret_results

    ## TODO: get num_tokens for functions inside milvus_retriever.py and retriever_document.py (with classmethod RetrieverDocument.faq_retreiver_doc); influence token migrations
    def get_qa_docs(
        self,
        collection_name: str,
        bot_id: str,
        version: str,
        qa_doc_type: RetrieverDocumentType,
    ) -> list[RetrieverDocument]:
        connections.connect(
            uri=self.uri,
            token=self.token,
        )
        collection = Collection(collection_name)

        partition_key = self.get_bot_uid(bot_id, version)
        iterator = collection.query_iterator(
            batch_size=1000,
            expr=f"qa_doc_type=='{qa_doc_type.value}' and bot_uid=='{partition_key}'",
            output_fields=[
                "id",
                "qa_doc_id",
                "chunk_id",
                "qa_doc_type",
                "text",
                "metadata",
                "bot_uid",
                "timestamp",
            ],
        )

        qa_docs = []
        qa_doc_map = defaultdict(list)
        while True:
            result = iterator.next()
            if len(result) == 0:
                iterator.close()
                break

            for r in result:
                if qa_doc_type == RetrieverDocumentType.FAQ:
                    qa_doc = RetrieverDocument.faq_retreiver_doc(
                        id=r["id"],
                        text=r["text"],
                        metadata=r["metadata"],
                        bot_uid=r["bot_uid"],
                        timestamp=r["timestamp"],
                    )
                    qa_docs.append(qa_doc)
                else:
                    qa_doc_map[r["qa_doc_id"]].append(r)
                    qa_doc = RetrieverDocument(
                        r["id"],
                        r["qa_doc_id"],
                        r["chunk_id"],
                        qa_doc_type,
                        r["text"],
                        r["metadata"],
                        is_chunked=True,
                        bot_uid=r["bot_uid"],
                        embedding=None,
                        timestamp=r["timestamp"],
                    )

        if qa_doc_type != RetrieverDocumentType.FAQ:
            for qa_doc_id, docs in qa_doc_map.items():
                sorted_docs = sorted(docs, key=lambda x: x["chunk_id"])
                txt = "".join([d["text"] for d in sorted_docs])
                ret_doc = RetrieverDocument.unchunked_retreiver_doc(
                    qa_doc_id,
                    qa_doc_type,
                    txt,
                    sorted_docs[0]["metadata"],
                    sorted_docs[0]["bot_uid"],
                    sorted_docs[0]["timestamp"],
                )
                qa_docs.append(ret_doc)

        return qa_docs

    def get_qa_doc(self, collection_name: str, qa_doc_id: str) -> RetrieverDocument:
        log_context.info(
            f"Getting qa doc with id {qa_doc_id} from collection {collection_name}"
        )
        res = self.client.query(
            collection_name=collection_name,
            filter=f"qa_doc_id=='{qa_doc_id}'",
            output_fields=[
                "qa_doc_id",
                "chunk_id",
                "qa_doc_type",
                "text",
                "metadata",
                "bot_uid",
                "timestamp",
            ],
        )

        if len(res) == 0:
            return None

        sorted_res = sorted(res, key=lambda x: x["chunk_id"])

        if sorted_res[0]["qa_doc_type"] == RetrieverDocumentType.FAQ.value:
            return RetrieverDocument.faq_retreiver_doc(
                id=sorted_res[0]["qa_doc_id"],
                text=sorted_res[0]["text"],
                metadata=sorted_res[0]["metadata"],
                bot_uid=sorted_res[0]["bot_uid"],
                timestamp=sorted_res[0]["timestamp"],
            )
        else:
            txt = "".join([d["text"] for d in sorted_res])
            return RetrieverDocument.unchunked_retreiver_doc(
                sorted_res[0]["qa_doc_id"],
                RetrieverDocumentType(sorted_res[0]["qa_doc_type"]),
                txt,
                sorted_res[0]["metadata"],
                sorted_res[0]["bot_uid"],
                sorted_res[0]["timestamp"],
            )

    def get_qa_doc_ids(
        self,
        collection_name: str,
        bot_id: str,
        version: str,
        qa_doc_type: RetrieverDocumentType,
    ) -> list[dict]:
        log_context.info(
            f"Getting all qa_doc_ids from collection '{collection_name}' for bot_id: {bot_id}, version: {version}"
        )
        partition_key = self.get_bot_uid(bot_id, version)
        connections.connect(
            uri=self.uri,
            token=self.token,
        )
        collection = Collection(collection_name)

        iterator = collection.query_iterator(
            batch_size=1000,
            expr=f"qa_doc_type == '{qa_doc_type.value}' and bot_uid == '{partition_key}'",
            output_fields=["id", "qa_doc_id"],
        )

        qa_doc_ids = set()
        while True:
            result = iterator.next()
            if len(result) == 0:
                iterator.close()
                break

            for r in result:
                qa_doc_ids.add(r["qa_doc_id"])

        return list(qa_doc_ids)

    def has_collection(self, collection_name: str) -> bool:
        return self.client.has_collection(collection_name)

    def load_collection(self, collection_name: str) -> None:
        if self.client.has_collection(collection_name):
            self.client.load_collection(collection_name)
            return
        else:
            raise ValueError(f"Milvus Collection {collection_name} does not exist")

    def release_collection(self, collection_name: str) -> dict[str, object]:
        return self.client.release_collection(collection_name)

    def drop_collection(self, collection_name: str) -> dict[str, object]:
        return self.client.drop_collection(collection_name)

    def get_all_vectors(self, collection_name: str) -> list[dict[str, object]]:
        connections.connect(
            uri=self.uri,
            token=self.token,
        )
        collection = Collection(collection_name)

        iterator = collection.query_iterator(
            batch_size=16000,
            output_fields=[
                "id",
                "qa_doc_id",
                "chunk_id",
                "qa_doc_type",
                "num_tokens",
                "metadata",
                "text",
                "embedding",
                "timestamp",
            ],
        )

        vectors = []
        count = 0
        while True:
            result = iterator.next()
            if len(result) == 0:
                iterator.close()
                break

            for r in result:
                vectors.append(r)
                count += 1

        log_context.info(f"collection {collection_name} Total vectors: {count}")
        return vectors

    def add_vectors_parallel(
        self,
        collection_name: str,
        bot_id: str,
        version: str,
        vectors: list[dict],
        upsert: bool = False,
    ) -> list[dict[str, object]]:
        log_context.info(
            f"Adding {len(vectors)} vector db documents to institution {collection_name} for bot_id: {bot_id} version: {version}"
        )
        if not self.client.has_collection(collection_name):
            log_context.info(
                f"No colelction found hence creating collection: {collection_name}"
            )
            self.create_collection_with_partition_key(collection_name)

        vectors_to_insert = []

        if not upsert:
            # check if the document already exists in the collection
            for vec in vectors:
                res = self.client.query(
                    collection_name=collection_name,
                    ids=vec["id"],
                )
                if len(res) == 0:
                    vectors_to_insert.append(vec)
            log_context.info(f"New vectors to insert: {len(vectors_to_insert)}")
        else:
            vectors_to_insert = vectors

        for vec in vectors_to_insert:
            vec["bot_uid"] = self.get_bot_uid(bot_id, version)

        res = []
        # process 100 documents at a time
        count = 0
        for i in range(0, len(vectors_to_insert), 100):
            batch_vectors = vectors_to_insert[i : i + 100]

            res.extend(
                self.client.upsert(collection_name=collection_name, data=batch_vectors)
            )
            count += len(batch_vectors)
            log_context.info(
                f"{collection_name}: Added {count}/{len(vectors_to_insert)} docs"
            )
        return res

    def is_collection_loaded(self, collection_name: str) -> bool:
        state = self.client.get_load_state(collection_name)
        print("loaded state: ", state)
        return state["state"].__str__() == "Loaded"

    def delete_vectors_by_partition_key(
        self, collection_name: str, bot_id: str, version: str
    ) -> dict[str, object]:
        partition_key = self.get_bot_uid(bot_id, version)
        res = self.client.delete(
            collection_name=collection_name, filter=f"bot_uid=='{partition_key}'"
        )
        log_context.info(
            f"Deleted {len(res)} vectors from collection {collection_name} for bot_id: {bot_id} version: {version}"
        )

        # check if the collection is empty
        res = self.client.query(
            collection_name=collection_name, output_fields=["count(*)"]
        )
        if res[0]["count(*)"] == 0:
            log_context.info(f"Collection {collection_name} is empty.")

        return res

    def get_vector_count_for_bot(
        self, collection: str, bot_id: str, version: str
    ) -> int:
        res = self.client.query(
            collection_name=collection, filter=f"bot_uid=='{bot_id}__{version}'"
        )
        return len(res)

    # def get_token_count_for_bot(self, collection_name: str, bot_id: str, version: str):
    #     log_context.info(f"Counting tokens in collection {collection_name} for bot_id: {bot_id}, version: {version}")
    #     partition_key = self.get_bot_uid(bot_id, version)
    #     res = self.client.query(
    #         collection_name=collection_name,
    #         filter=f"bot_uid=='{partition_key}'",
    #         output_fields=["num_tokens"],
    #     )
    #     return sum([r.get("num_tokens", 0) for r in res])

    def get_collection_size(self, collection_name: str) -> int:
        # real time vector count for the collection
        return self.client.query(
            collection_name=collection_name, output_fields=["count(*)"]
        )[0]["count(*)"]

    def migrate_vectors(
        self,
        old_collection_name: str,
        bot_id: str,
        version: str,
        new_collection_name: str,
    ) -> int:
        partition_key = self.get_bot_uid(bot_id, version)
        connections.connect(
            uri=self.uri,
            token=self.token,
        )
        collection = Collection(old_collection_name)

        iterator = collection.query_iterator(
            batch_size=16000,
            expr=f"bot_uid=='{partition_key}'",
            output_fields=[
                "id",
                "bot_uid",
                "qa_doc_id",
                "chunk_id",
                "num_tokens",
                "qa_doc_type",
                "metadata",
                "text",
                "embedding",
                "timestamp",
            ],
        )

        vectors = []
        count = 0
        while True:
            result = iterator.next()
            if len(result) == 0:
                iterator.close()
                break

            for r in result:
                vectors.append(r)
                count += 1

        log_context.info(
            f"migrating {count} vectors for bot {bot_id} version {version}"
        )

        # add vectors to new collection
        if not self.has_collection(new_collection_name):
            log_context.info(
                f"No collection found hence creating collection: {new_collection_name}"
            )
            self.create_collection_with_partition_key(new_collection_name)
        self.add_vectors_parallel(new_collection_name, bot_id, version, vectors)

        # delete vectors from old collection
        self.delete_vectors_by_partition_key(old_collection_name, bot_id, version)

        log_context.info(
            f"moved {count} vectors from {old_collection_name} to {new_collection_name}"
        )
        return count

    def list_collections(self) -> list[str]:
        return self.client.list_collections()


class MilvusRetrieverExecutor:
    def __init__(self, bot_config: object) -> None:
        self.bot_config = bot_config
        model_class = validate_and_get_model_class(bot_config.llm_config)

        self.llm = model_class(model=bot_config.llm_config.model_type_or_path)

    def generate_thought(self, retriever_results: list[RetrieverResult]) -> str:
        # post process list of documents into str
        retrieved_str = ""
        for doc in retriever_results:
            if doc.metadata.get("title"):
                retrieved_str += "title: " + doc.metadata["title"] + "\n"
            if doc.metadata.get("source"):
                retrieved_str += "source: " + doc.metadata["source"] + "\n"
            retrieved_str += "content: " + doc.text + "\n\n"
        return retrieved_str

    def _gaussian_similarity(self, distance: float, sigma: float = 0.5) -> float:
        similarity = np.exp(-(distance**2) / (2 * sigma**2)) * 100
        return round(float(similarity), 2)

    def postprocess(
        self, retriever_results: list[RetrieverResult]
    ) -> dict[str, object]:
        retriever_returns = []
        for doc in retriever_results:
            confidence_score = self._gaussian_similarity(doc.distance)
            item = {
                "qa_doc_id": doc.qa_doc_id,
                "qa_doc_type": doc.qa_doc_type.value,
                "title": doc.metadata.get("title"),
                "content": doc.text,
                "source": doc.metadata.get("source"),
                "raw_score": round(float(doc.distance), 4),
                "confidence": confidence_score,
            }
            retriever_returns.append(item)
        return {"retriever": retriever_returns}

    def retrieve(
        self, chat_history_str: str, tags: dict[str, object] | None = None
    ) -> tuple[str, dict[str, object]]:
        """Given a chat history, retrieve relevant information from the database."""
        if tags is None:
            tags = {}
        st = time.time()
        prompts = load_prompts(self.bot_config)
        contextualize_q_prompt = PromptTemplate.from_template(
            prompts.get("retrieve_contextualize_q_prompt", "")
        )
        ret_input_chain = contextualize_q_prompt | self.llm | StrOutputParser()
        ret_input = ret_input_chain.invoke({"chat_history": chat_history_str})
        rit = time.time() - st

        ret_results: list[RetrieverResult] = []
        st = time.time()
        milvus_db = mysql_pool.fetchone(
            "SELECT collection_name FROM qa_bot WHERE id=%s AND version=%s",
            (self.bot_config.bot_id, self.bot_config.version),
        )
        with MilvusRetriever() as retriever:
            ret_results = retriever.search(
                milvus_db["collection_name"],
                self.bot_config.bot_id,
                self.bot_config.version,
                ret_input,
                tags,
            )
        rt = time.time() - st
        log_context.info(f"MilvusRetriever search took {rt} seconds")
        retriever_params = self.postprocess(ret_results)
        retriever_params["timing"] = {"retriever_input": rit, "retriever_search": rt}
        thought = self.generate_thought(ret_results)
        return thought, retriever_params
