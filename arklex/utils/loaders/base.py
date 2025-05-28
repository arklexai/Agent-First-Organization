from abc import ABC
from abc import abstractmethod
import pickle
from typing import List, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class Loader(ABC):
    def __init__(self) -> None:
        pass

    @staticmethod
    def save(filepath: str, data: Any) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @abstractmethod
    def load(self, filepath: str) -> List[Document]:
        raise NotImplementedError

    @abstractmethod
    def chunk(self, document_objs: List[Any]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=200, chunk_overlap=40
        )
        docs = []
        langchain_docs = []
        for doc in document_objs:
            splitted_text = text_splitter.split_text(doc.content)
            for i, txt in enumerate(splitted_text):
                docs.append(doc)
                langchain_docs.append(
                    Document(page_content=txt, metadata={"source": doc.title})
                )
        return langchain_docs
