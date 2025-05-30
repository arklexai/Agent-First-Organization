"""Base loader implementation for the Arklex framework.

This module provides the base abstract class for document loaders in the framework.
It defines the interface for loading and processing documents, including methods for
saving data, loading documents, and chunking content into smaller pieces. The module
serves as a foundation for implementing specific document loaders for different
file formats and data sources.
"""

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
        """Save data to a file.

        This function serializes and saves data to a file using pickle.

        Args:
            filepath (str): Path where to save the data.
            data (Any): Data to save.
        """
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @abstractmethod
    def load(self, filepath: str) -> List[Document]:
        """Load documents from a file.

        This abstract method should be implemented by subclasses to load documents
        from a specific file format or data source.

        Args:
            filepath (str): Path to the file to load.

        Returns:
            List[Document]: List of loaded documents.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def chunk(self, document_objs: List[Any]) -> List[Document]:
        """Split documents into smaller chunks.

        This method splits documents into smaller, more manageable chunks while
        preserving their metadata and structure.

        Args:
            document_objs (List[Any]): List of document objects to chunk.

        Returns:
            List[Document]: List of chunked documents.
        """
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
