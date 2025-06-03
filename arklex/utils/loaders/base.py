"""Base loader implementation for the Arklex framework.

This module provides the base abstract class for document loaders in the framework.
It defines the interface for loading and processing documents, including methods for
saving data, loading documents, and chunking content into smaller pieces. The module
serves as a foundation for implementing specific document loaders for different
file formats and data sources.

Key Components:
1. Loader (ABC): Abstract base class for document loaders
   - save: Static method for serializing and saving data
   - load: Abstract method for loading documents
   - chunk: Abstract method for splitting documents

Features:
- Abstract base class for document loading
- Data serialization using pickle
- Document chunking with configurable parameters
- Support for metadata preservation
- Integration with LangChain document format

Usage:
    from arklex.utils.loaders.base import Loader

    class CustomLoader(Loader):
        def load(self, filepath: str) -> List[Document]:
            # Implement custom loading logic
            pass

        def chunk(self, document_objs: List[Any]) -> List[Document]:
            # Implement custom chunking logic
            pass
"""

from abc import ABC
from abc import abstractmethod
import pickle
from typing import List, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class Loader(ABC):
    """Abstract base class for document loaders in the Arklex framework.

    This class defines the interface for loading and processing documents,
    providing methods for saving data, loading documents, and chunking content.
    Subclasses must implement the abstract methods to provide specific loading
    and chunking functionality for different file formats and data sources.

    Attributes:
        None

    Methods:
        save: Static method for saving data to a file
        load: Abstract method for loading documents
        chunk: Abstract method for splitting documents
    """

    def __init__(self) -> None:
        """Initialize the Loader instance."""
        pass

    @staticmethod
    def save(filepath: str, data: Any) -> None:
        """Save data to a file using pickle serialization.

        This function serializes and saves data to a file using pickle.
        It's useful for persisting document data or intermediate processing results.

        Args:
            filepath (str): Path where to save the data.
            data (Any): Data to save. Must be pickle-serializable.

        Raises:
            IOError: If the file cannot be written.
            pickle.PickleError: If the data cannot be serialized.
        """
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @abstractmethod
    def load(self, filepath: str) -> List[Document]:
        """Load documents from a file.

        This abstract method should be implemented by subclasses to load documents
        from a specific file format or data source. The implementation should handle
        the specific file format and convert the content into LangChain Document objects.

        Args:
            filepath (str): Path to the file to load.

        Returns:
            List[Document]: List of loaded documents in LangChain Document format.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file format is invalid or unsupported.
        """
        raise NotImplementedError

    @abstractmethod
    def chunk(self, document_objs: List[Any]) -> List[Document]:
        """Split documents into smaller chunks.

        This method splits documents into smaller, more manageable chunks while
        preserving their metadata and structure. It uses the RecursiveCharacterTextSplitter
        with tiktoken encoding for efficient text splitting.

        The chunking process:
        1. Initializes a text splitter with specified parameters
        2. Processes each document in the input list
        3. Splits the content into chunks
        4. Preserves document metadata
        5. Creates new LangChain Document objects for each chunk

        Args:
            document_objs (List[Any]): List of document objects to chunk.

        Returns:
            List[Document]: List of chunked documents in LangChain Document format.

        Note:
            The current implementation uses:
            - cl100k_base encoding
            - 200 token chunk size
            - 40 token chunk overlap
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
