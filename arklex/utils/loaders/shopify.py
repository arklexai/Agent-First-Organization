"""Shopify document loader implementation for the Arklex framework.

This module provides a specialized document loader for Shopify data, implementing
the base Loader interface. It includes functionality for loading product information
from Shopify's GraphQL API and processing the data into document chunks suitable
for further processing. The loader handles product descriptions, titles, and metadata,
making it easy to integrate Shopify product data into the framework's document
processing pipeline.
"""

import json
from typing import List

import shopify

from arklex.utils.loaders.base import Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class ShopifyLoader(Loader):
    def __init__(self) -> None:
        pass

    def load(self) -> List[Document]:
        """Load product data from Shopify's GraphQL API.

        This function retrieves product information from Shopify's GraphQL API,
        including titles, descriptions, tags, and inventory data. It processes
        the data into Document objects suitable for further processing.

        Returns:
            List[Document]: List of documents containing product information.
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

        Args:
            document_objs (List[Document]): List of product documents to chunk.

        Returns:
            List[Document]: List of chunked product documents.
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
