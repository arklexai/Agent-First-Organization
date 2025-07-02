"""Document loading and processing utilities for evaluation in the Arklex framework.

This module provides functionality for loading and processing documents from various sources
during evaluation, including web URLs, local files, and text content. It includes utilities
for domain information extraction, document loading with caching, and handling different
document types (web, file, text) with appropriate processing methods.
"""

import json
import os
import pickle
import sys
from os.path import abspath, dirname
from typing import Any

sys.path.insert(0, dirname(dirname(abspath(__file__))))
from arklex.utils.loader import CrawledObject, Loader, SourceType


def get_domain_info(documents: list[dict[str, str]]) -> str | None:
    """Retrieve the domain information from a list of documents.

    This function searches for a document with the URL "summary" in the list and returns its content.

    Args:
        documents (List[Dict[str, str]]): List of documents containing URL and content.

    Returns:
        Optional[str]: Content of the summary document, or None if no summary document is found.
    """
    summary: str | None = None
    for doc in documents:
        if doc["URL"] == "summary":
            summary = doc["content"]
            break
    return summary


def load_docs(
    document_dir: str | None, doc_config: dict[str, Any], limit: int = 10
) -> list[dict[str, str]]:
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
            if "rag_docs" in doc_config:
                rag_docs: list[dict[str, Any]] = doc_config["rag_docs"]
                filename: str = "rag_documents.pkl"
            elif "task_docs" in doc_config:
                rag_docs: list[dict[str, Any]] = doc_config["task_docs"]
                filename: str = "task_documents.pkl"
            else:
                raise ValueError(
                    "The config json file must have a key 'rag_docs' or 'task_docs' with a list of documents to load."
                )

            # If the docs array is empty, return empty list
            if not rag_docs:
                return []

            filepath: str = os.path.join(document_dir, filename)
            total_num_docs: int = sum(
                [doc.get("num") if doc.get("num") else 1 for doc in rag_docs]
            )
            loader: Loader = Loader()
            if os.path.exists(filepath):
                with open(os.path.join(document_dir, filename), "rb") as f:
                    docs: list[CrawledObject] = pickle.load(f)
            else:
                docs: list[CrawledObject] = []
                for doc in rag_docs:
                    source: str = doc.get("source")
                    if doc.get("type") == "url":
                        num_docs: int = doc.get("num") if doc.get("num") else 1
                        urls: list[str] = loader.get_all_urls(source, num_docs)
                        crawled_urls: list[CrawledObject] = loader.to_crawled_url_objs(
                            urls
                        )
                        docs.extend(crawled_urls)
                    elif doc.get("type") == "file":
                        file_list: list[str] = [
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
            limit = total_num_docs // 5 if total_num_docs > 50 else 10
            if len(docs) > 0 and isinstance(docs[0], CrawledObject):
                documents: list[dict[str, str]] = []
                # Get candidate websites for only web urls
                web_docs: list[CrawledObject] = list(
                    filter(lambda x: x.source_type == SourceType.WEB, docs)
                )
                file_docs: list[CrawledObject] = list(
                    filter(lambda x: x.source_type == SourceType.FILE, docs)
                )
                text_docs: list[CrawledObject] = list(
                    filter(lambda x: x.source_type == SourceType.TEXT, docs)
                )
                documents.extend(loader.get_candidates_websites(web_docs, limit))
                documents.extend(file_docs)
                documents.extend(text_docs)
                documents = [doc.to_dict() for doc in documents]
            elif len(docs) > 0:
                raise ValueError(
                    "The documents must be a list of CrawledObject objects."
                )
            else:
                documents: list[dict[str, str]] = []
        except ValueError as e:
            # Re-raise ValueError exceptions
            raise e
        except Exception as e:
            print(f"Error loading documents: {e}")
            documents: list[dict[str, str]] = []
    else:
        documents: list[dict[str, str]] = []
    return documents


if __name__ == "__main__":
    with open("./temp_files/richtech_config.json") as f:
        doc_config: dict[str, Any] = json.load(f)
    docs: list[dict[str, str]] = load_docs("./temp_files", doc_config, 10)
