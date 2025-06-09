"""Document loading and processing utilities for evaluation in the Arklex framework.

This module provides functionality for loading and processing documents from various sources
during evaluation, including web URLs, local files, and text content. It includes utilities
for domain information extraction, document loading with caching, and handling different
document types (web, file, text) with appropriate processing methods.
"""

import os
import sys
import json
from os.path import dirname, abspath
from typing import List, Dict, Any, Optional

sys.path.insert(0, dirname(dirname(abspath(__file__))))
from arklex.utils.loader import Loader, CrawledObject, SourceType


def get_domain_info(documents: List[Dict[str, str]]) -> Optional[str]:
    """Retrieve the domain information from a list of documents.

    This function searches for a document with the URL "summary" in the list and returns its content.

    Args:
        documents (List[Dict[str, str]]): List of documents containing URL and content.

    Returns:
        Optional[str]: Content of the summary document, or None if no summary document is found.
    """
    summary: Optional[str] = None
    for doc in documents:
        if doc["URL"] == "summary":
            summary = doc["content"]
            break
    return summary


def load_docs(
    document_dir: Optional[str], doc_config: Dict[str, Any], limit: int = 10
) -> List[Dict[str, str]]:
    """Load documents from specified sources.

    This function loads documents from the specified directory or configuration and returns them as a list of dictionaries.

    Args:
        document_dir (Optional[str]): Directory containing documents.
        doc_config (Dict[str, Any]): Configuration settings containing document source information.
        limit (int): Maximum number of documents to load.

    Returns:
        List[Dict[str, str]]: List of loaded documents.
    """
    if document_dir is not None:
        try:
            if "rag_docs" not in doc_config:
                if "task_docs" not in doc_config:
                    raise ValueError(
                        "The config json file must have a key 'rag_docs' or 'task_docs' with a list of documents to load."
                    )
            else:
                rag_docs: List[Dict[str, Any]] = doc_config["task_docs"]
                filename: str = "task_documents.pkl"
            filepath: str = os.path.join(document_dir, filename)
            total_num_docs: int = sum(
                [doc.get("num") if doc.get("num") else 1 for doc in rag_docs]
            )
            loader: Loader = Loader()
            if os.path.exists(filepath):
                docs: List[CrawledObject] = pickle.load(
                    open(os.path.join(document_dir, filename), "rb")
                )
            else:
                docs: List[CrawledObject] = []
                for doc in rag_docs:
                    source: str = doc.get("source")
                    if doc.get("type") == "url":
                        num_docs: int = doc.get("num") if doc.get("num") else 1
                        urls: List[str] = loader.get_all_urls(source, num_docs)
                        crawled_urls: List[CrawledObject] = loader.to_crawled_url_objs(
                            urls
                        )
                        docs.extend(crawled_urls)
                    elif doc.get("type") == "file":
                        file_list: List[str] = [
                            os.path.join(source, f) for f in os.listdir(source)
                        ]
                        docs.extend(loader.to_crawled_local_objs(file_list))
                    elif doc.get("type") == "text":
                        docs.extend(loader.to_crawled_text([source]))
                    else:
                        # TODO: how to handle when type is not provided
                        raise Exception(
                            "type must be one of [url, file, text] and it must be provided"
                        )
                Loader.save(filepath, docs)
            if total_num_docs > 50:
                limit = total_num_docs // 5
            else:
                limit = 10
            if isinstance(docs[0], CrawledObject):
                documents: List[Dict[str, str]] = []
                # Get candidate websites for only web urls
                web_docs: List[CrawledObject] = list(
                    filter(lambda x: x.source_type == SourceType.WEB, docs)
                )
                file_docs: List[CrawledObject] = list(
                    filter(lambda x: x.source_type == SourceType.FILE, docs)
                )
                text_docs: List[CrawledObject] = list(
                    filter(lambda x: x.source_type == SourceType.TEXT, docs)
                )
                documents.extend(loader.get_candidates_websites(web_docs, limit))
                documents.extend(file_docs)
                documents.extend(text_docs)
                documents = [doc.to_dict() for doc in documents]
            else:
                raise ValueError(
                    "The documents must be a list of CrawledObject objects."
                )
        except Exception as e:
            print(f"Error loading documents: {e}")
            documents: List[Dict[str, str]] = []
    else:
        documents: List[Dict[str, str]] = []
    return documents


if __name__ == "__main__":
    doc_config: Dict[str, Any] = json.load(open("./temp_files/richtech_config.json"))
    docs: List[Dict[str, str]] = load_docs("./temp_files", doc_config, 10)
