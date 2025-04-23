import json
from enum import Enum
from typing import List, Dict
from openai import OpenAI
import logging

import tiktoken
from arklex.utils.mysql import mysql_pool
from langchain.text_splitter import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_ENCODING = "cl100k_base"

logger = logging.getLogger(__name__)

def embed(text: str):
    client = OpenAI()
    try:
        response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    except Exception as e:
        logger.error(f"Error embedding text of length {len(text)}")
        logger.error(text[:1000])
        logger.exception(e)
        raise e
    return response.data[0].embedding    

class RetrieverDocumentType(Enum):
    WEBSITE = "website"
    FAQ = "faq"
    OTHER = "other"


class RetrieverResult:
    def __init__(
        self,
        qa_doc_id: str,
        qa_doc_type: RetrieverDocumentType,
        distance: float,
        metadata: dict,
        text: str,
        start_chunk_idx: int,
        end_chunk_idx: int,
    ):
        self.qa_doc_id = qa_doc_id
        self.distance = distance
        if isinstance(metadata, str):
            self.metadata = json.loads(metadata)
        else:
            self.metadata = metadata
        self.text = text
        self.start_chunk_idx = start_chunk_idx
        self.end_chunk_idx = end_chunk_idx
        self.qa_doc_type = qa_doc_type


class RetrieverDocument:
    def __init__(
        self,
        id: str,
        qa_doc_id: str,
        chunk_idx: int,
        qa_doc_type: RetrieverDocumentType,
        text: str,
        metadata: dict,
        is_chunked: bool,
        bot_uid: str,
        # num_tokens: int = None,
        embedding=None,
        timestamp: int = None,
    ):
        self.id = id
        self.qa_doc_id = qa_doc_id
        self.chunk_idx = int(chunk_idx)
        self.qa_doc_type = qa_doc_type
        self.text = text
        if isinstance(metadata, str):
            self.metadata = json.loads(metadata)
        else:
            self.metadata = metadata
        # self.num_tokens = num_tokens
        self.embedding = embedding
        self.is_chunked = is_chunked
        self.timestamp = int(timestamp)
        self.bot_uid = bot_uid

    def chunk(self, chunk_encoding=DEFAULT_CHUNK_ENCODING) -> List["RetrieverDocument"]:
        if self.is_chunked:
            raise ValueError("Document is already chunked")
        elif self.qa_doc_type == RetrieverDocumentType.FAQ:
            raise ValueError("Cannot chunk FAQ document")
        encoding = tiktoken.get_encoding(chunk_encoding)
        chunked_texts = RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name=chunk_encoding, chunk_size=400, chunk_overlap=50).split_text(self.text.strip())
        chunked_docs = []
        tokens = encoding.encode(self.text)
        logger.info(f"Original text token length: {len(tokens)}, Chunked to {len(chunked_texts)} chunks")
        for i, chunk in enumerate(chunked_texts):
            chunk = chunk.strip()
            tokens = encoding.encode(chunk)
            doc = RetrieverDocument(
                id=str(f"{self.id}__{i}"),
                qa_doc_id=self.qa_doc_id,
                chunk_idx=i,
                qa_doc_type=self.qa_doc_type,
                text=chunk,
                metadata=self.metadata,
                # num_tokens=len(tokens),
                embedding=None,
                bot_uid=self.bot_uid,
                is_chunked=True,
                timestamp=self.timestamp,
            )
            chunked_docs.append(doc)

        return chunked_docs
    
    def to_dict(self):
        return {
            "id": self.id,
            "qa_doc_id": self.qa_doc_id,
            "chunk_idx": self.chunk_idx,
            "qa_doc_type": self.qa_doc_type.value,
            "text": self.text,
            "metadata": self.metadata,
            # "num_tokens": self.num_tokens,
            "embedding": self.embedding,
            "is_chunked": self.is_chunked,
            "timestamp": self.timestamp,
            "bot_uid": self.bot_uid,
        }
    
    def to_milvus_schema_dict_and_embed(self) -> Dict:
        # check if values exists
        if (
            self.id is None
            or self.qa_doc_id is None
            or self.chunk_idx is None
            or self.qa_doc_type is None
            or self.text is None
            # or self.num_tokens is None
            or self.metadata is None
            or self.timestamp is None
            or self.bot_uid is None
        ):
            raise ValueError("Missing values")

        return {
            "id": self.id,
            "qa_doc_id": self.qa_doc_id,
            "chunk_id": self.chunk_idx,
            "qa_doc_type": self.qa_doc_type.value,
            "text": self.text,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            # "num_tokens": self.num_tokens,
            "embedding": embed(self.text),
            "bot_uid": self.bot_uid,
        }
    
    @classmethod
    def from_dict(cls, doc_dict):
        return cls(
            id=doc_dict["id"],
            qa_doc_id=doc_dict["qa_doc_id"],
            chunk_idx=doc_dict["chunk_idx"],
            qa_doc_type=RetrieverDocumentType(doc_dict["qa_doc_type"]),
            text=doc_dict["text"],
            metadata=doc_dict["metadata"],
            # num_tokens=doc_dict["num_tokens"],
            embedding=doc_dict["embedding"],
            is_chunked=doc_dict["is_chunked"],
            timestamp=doc_dict["timestamp"],
            bot_uid=doc_dict["bot_uid"],
        )

    @classmethod
    def faq_retreiver_doc(
    #     cls, id: str, text: str, metadata: dict, bot_uid: str, timestamp: int = None, num_tokens: int = None
    # ):
        cls, id: str, text: str, metadata: dict, bot_uid: str, timestamp: int = None
    ):
        return cls(
            id,
            id,
            0,
            RetrieverDocumentType.FAQ,
            text,
            metadata,
            is_chunked=False,
            # num_tokens=num_tokens,
            embedding=None,
            timestamp=timestamp,
            bot_uid=bot_uid
        )

    @classmethod
    def unchunked_retreiver_doc(
        cls,
        id: str,
        qa_doc_type: RetrieverDocumentType,
        text: str,
        metadata: dict,
        bot_uid: str,
        timestamp: int = None,
    ):
        return cls(
            id,
            id,
            0,
            qa_doc_type,
            text,
            metadata,
            is_chunked=False,
            embedding=None,
            timestamp=timestamp,
            bot_uid=bot_uid,
        )

    @classmethod
    def chunked_retriever_docs_from_db_docs(
        cls, db_docs: List[dict], doc_type: RetrieverDocumentType, bot_uid: str,
    ) -> List["RetrieverDocument"]:
        chunked_db_docs: List[RetrieverDocument] = []
        for doc in db_docs:
            doc_id = doc["id"]
            metadata = doc["metadata"]
            if doc["content"] is None:
                continue
            text = doc["content"].strip()
            timestamp = doc["timestamp"]

            ret_doc = cls.unchunked_retreiver_doc(
                doc_id, doc_type, text, metadata, bot_uid, timestamp
            )
            chunked_db_docs.extend(ret_doc.chunk())

        return chunked_db_docs

    @classmethod
    def load_all_chunked_docs_from_mysql(
        cls, bot_id: str, version: str
    ) -> List["RetrieverDocument"]:
        faq_db_docs = mysql_pool.fetchall(
            "SELECT id, content, metadata, unix_timestamp(updated_at) as timestamp FROM qa_doc_faq WHERE qa_bot_id=%s and qa_bot_version=%s;",
            (bot_id, version),
        )
        faq_docs = []
        for doc in faq_db_docs:
            faq_docs.append(
                cls.faq_retreiver_doc(
                    id=doc["id"], text=doc["content"], metadata=doc["metadata"], bot_uid=get_bot_uid(bot_id, version), timestamp=doc["timestamp"]
                )
            )
        logger.info(f"Loaded {len(faq_docs)} faq docs")

        other_db_docs = mysql_pool.fetchall(
            "SELECT id, metadata, content, unix_timestamp(updated_at) as timestamp FROM qa_doc_other WHERE qa_bot_id=%s and qa_bot_version=%s;",
            (bot_id, version),
        )
        logger.info(f"Loaded {len(other_db_docs)} other docs")
        chunked_other_docs = cls.chunked_retriever_docs_from_db_docs(
            other_db_docs, RetrieverDocumentType.OTHER, get_bot_uid(bot_id, version)
        )
        logger.info(f"Chunked to {len(chunked_other_docs)} other docs")

        website_db_docs = mysql_pool.fetchall(
            "SELECT id, metadata, content, unix_timestamp(updated_at) as timestamp FROM qa_doc_website WHERE qa_bot_id=%s and qa_bot_version=%s and is_crawled=1 and is_error=0;",
            (bot_id, version),
        )
        logger.info(f"Loaded {len(website_db_docs)} website docs")
        chunked_website_docs = cls.chunked_retriever_docs_from_db_docs(
            website_db_docs, RetrieverDocumentType.WEBSITE, get_bot_uid(bot_id, version)
        )
        logger.info(f"Chunked to {len(chunked_website_docs)} website docs")

        return chunked_website_docs + chunked_other_docs + faq_docs

def embed_retriever_document(retriever_document: RetrieverDocument):
    return retriever_document.to_milvus_schema_dict_and_embed()

def get_bot_uid(bot_id: str, version: str):
    return f"{bot_id}__{version}"