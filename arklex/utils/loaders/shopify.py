"""Shopify document loader implementation for the Arklex framework.

This module provides a specialized document loader for Shopify data, implementing
the base Loader interface. It includes functionality for loading product information
from Shopify's GraphQL API and processing the data into document chunks suitable
for further processing. The loader handles product descriptions, titles, and metadata,
making it easy to integrate Shopify product data into the framework's document
processing pipeline.

Key Components:
1. ShopifyLoader: Specialized loader for Shopify product data
   - load: Loads product data from Shopify's GraphQL API
   - chunk: Splits product documents into manageable chunks

Features:
- Integration with Shopify's GraphQL API
- Product data extraction and processing
- Metadata preservation during chunking
- Configurable text splitting parameters
- Support for product descriptions and titles

Usage:
    from arklex.utils.loaders.shopify import ShopifyLoader

    # Initialize the loader
    loader = ShopifyLoader()

    # Load product data
    products = loader.load()

    # Split into chunks
    chunks = loader.chunk(products)
"""

import json
from typing import List

import shopify

from arklex.utils.loaders.base import Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class ShopifyLoader(Loader):
    """Specialized loader for Shopify product data.

    This class implements the base Loader interface to provide functionality
    for loading and processing Shopify product data. It retrieves product
    information through Shopify's GraphQL API and processes it into document
    chunks suitable for further analysis or processing.

    The loader handles:
    - Product titles and descriptions
    - Product tags and metadata
    - Inventory information
    - Document chunking with metadata preservation

    Attributes:
        None

    Methods:
        load: Loads product data from Shopify's GraphQL API
        chunk: Splits product documents into manageable chunks
    """

    def __init__(self) -> None:
        """Initialize the ShopifyLoader instance."""
        pass

    def load(self) -> List[Document]:
        """Load product data from Shopify's GraphQL API.

        This function retrieves product information from Shopify's GraphQL API,
        including titles, descriptions, tags, and inventory data. It processes
        the data into Document objects suitable for further processing.

        The function:
        1. Executes a GraphQL query to fetch product data
        2. Processes the response into Document objects
        3. Preserves product metadata
        4. Returns a list of processed documents

        Returns:
            List[Document]: List of documents containing product information.
            Each document contains:
            - page_content: Product description
            - metadata: Product title

        Raises:
            shopify.ShopifyError: If there's an error accessing the Shopify API
            json.JSONDecodeError: If the API response cannot be parsed
        """
        docs = []
        response = shopify.GraphQL().execute("""
            {
                products(first: 23) {
                    edges {
                        node {
                            title
                            tags
                            description
                            totalInventory
                        }
                    }
                }
            }
            """)
        product_docs = json.loads(response)["data"]["products"]["edges"]
        for product_doc in product_docs:
            docs.append(
                Document(
                    page_content=product_doc["node"]["description"],
                    metadata={"title": product_doc["node"]["title"]},
                )
            )
        return docs

    def chunk(self, document_objs: List[Document]) -> List[Document]:
        """Split product documents into smaller chunks.

        This function splits product documents into smaller, more manageable chunks
        while preserving their metadata. It uses a text splitter to create chunks
        of appropriate size for processing.

        The chunking process:
        1. Initializes a text splitter with specified parameters
        2. Processes each product document
        3. Splits the content into chunks
        4. Preserves product metadata for each chunk
        5. Creates new Document objects for each chunk

        Args:
            document_objs (List[Document]): List of product documents to chunk.
                Each document should contain:
                - page_content: Product description
                - metadata: Product metadata

        Returns:
            List[Document]: List of chunked product documents.
            Each chunk preserves the original document's metadata.

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
            splitted_text = text_splitter.split_text(doc.page_content)
            for i, txt in enumerate(splitted_text):
                docs.append(doc)
                langchain_docs.append(Document(page_content=txt, metadata=doc.metadata))
        return langchain_docs
