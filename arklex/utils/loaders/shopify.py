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
