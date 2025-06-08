"""Document and content loading utilities for the Arklex framework.

This module provides functionality for loading and processing various types of content,
including web pages, local files, and text data. It includes classes and methods for
web crawling, document parsing, and content chunking. The module supports multiple
file formats and provides utilities for handling different types of content sources,
ensuring consistent processing and storage of loaded content.
"""

import logging
import time
from pathlib import Path
from typing import List
import requests
import pickle
import uuid
from enum import Enum
import os

from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import networkx as nx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from mistralai import Mistral
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
)
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

CHROME_DRIVER_VERSION = "125.0.6422.7"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        logger.error(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        logger.error(f"Error: {e}")
        return None


class SourceType(Enum):
    WEB = 1
    FILE = 2
    TEXT = 3


class DocObject:
    def __init__(self, id: str, source: str):
        self.id = id
        self.source = source


class CrawledObject(DocObject):
    def __init__(
        self,
        id: str,
        source: str,
        content: str,
        metadata={},
        is_chunk=False,
        is_error=False,
        error_message=None,
        source_type=SourceType.WEB,
    ):
        super().__init__(id, source)
        self.content = content
        self.metadata = metadata
        self.is_chunk = is_chunk
        self.is_error = is_error
        self.error_message = error_message
        self.source_type = source_type

    def to_dict(self):
        return {
            "id": self.id,
            "source": self.source,
            "content": self.content,
            "metadata": self.metadata,
            "is_chunk": self.is_chunk,
            "is_error": self.is_error,
            "error_message": self.error_message,
            "source_type": self.source_type,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            id=data["id"],
            source=data["source"],
            content=data["content"],
            metadata=data["metadata"],
            is_chunk=data["is_chunk"],
            is_error=data["is_error"],
            error_message=data["error_message"],
            source_type=data["source_type"],
        )


class Loader:
    def __init__(self):
        pass

    def to_crawled_url_objs(self, url_list: List[str]) -> List[CrawledObject]:
        """Convert a list of URLs to CrawledObject instances.

        This function takes a list of URLs and converts them into CrawledObject instances
        by first creating DocObject instances and then crawling the URLs.

        Args:
            url_list (List[str]): List of URLs to convert.

        Returns:
            List[CrawledObject]: List of CrawledObject instances containing crawled content.
        """
        url_objs = [DocObject(str(uuid.uuid4()), url) for url in url_list]
        crawled_url_objs = self.crawl_urls(url_objs)
        return crawled_url_objs

    def crawl_urls(self, url_objects: list[DocObject]) -> List[CrawledObject]:
        """Crawl a list of URLs and extract their content.

        This function uses Selenium WebDriver to crawl a list of URLs and extract their
        content. It handles both successful and failed crawls, creating appropriate
        CrawledObject instances for each case.

        Args:
            url_objects (list[DocObject]): List of DocObject instances containing URLs to crawl.

        Returns:
            List[CrawledObject]: List of CrawledObject instances containing crawled content.
        """
        logger.info(f"Start crawling {len(url_objects)} urls")
        options = webdriver.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--headless")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-infobars")
        options.add_argument("--remote-debugging-pipe")
        chrome_driver_path = Path(
            ChromeDriverManager(driver_version=CHROME_DRIVER_VERSION).install()
        )
        options.binary_location = str(chrome_driver_path.parent.absolute())
        logger.info(f"chrome binary location: {options.binary_location}")
        driver = webdriver.Chrome(options=options)

        docs: List[CrawledObject] = []
        for url_obj in url_objects:
            try:
                logger.info(f"loading url: {url_obj.source}")
                driver.get(url_obj.source)
                time.sleep(2)
                html = driver.page_source
                soup = BeautifulSoup(html, "html.parser")

                text_list = []
                for string in soup.strings:
                    if string.find_parent("a"):
                        href = urljoin(
                            url_obj.source, string.find_parent("a").get("href")
                        )
                        if href.startswith(url_obj.source):
                            text = f"{string} {href}"
                            text_list.append(text)
                    elif string.strip():
                        text_list.append(string)
                text_output = "\n".join(text_list)

                title = url_obj.source
                for title in soup.find_all("title"):
                    title = title.get_text()
                    break

                docs.append(
                    CrawledObject(
                        id=url_obj.id,
                        source=url_obj.source,
                        content=text_output,
                        metadata={"title": title, "source": url_obj.source},
                        source_type=SourceType.WEB,
                    )
                )

            except Exception as err:
                logger.info(f"error crawling {url_obj}")
                logger.error(err)
                docs.append(
                    CrawledObject(
                        id=url_obj.id,
                        source=url_obj.source,
                        content=None,
                        metadata={"title": url_obj.source, "source": url_obj.source},
                        is_error=True,
                        error_message=str(err),
                        source_type=SourceType.WEB,
                    )
                )
        driver.quit()
        return docs

    def get_all_urls(self, base_url: str, max_num: int) -> List[str]:
        """Get all URLs from a base URL up to a maximum number.

        This function performs a breadth-first search of URLs starting from a base URL,
        collecting all valid URLs up to the specified maximum number.

        Args:
            base_url (str): The starting URL to crawl from.
            max_num (int): Maximum number of URLs to collect.

        Returns:
            List[str]: List of collected URLs, sorted alphabetically.
        """
        logger.info(
            f"Getting all pages for base url: {base_url}, maximum number is: {max_num}"
        )
        urls_visited = []
        base_url = base_url.split("#")[0].rstrip("/")
        urls_to_visit = [base_url]

        while urls_to_visit:
            if len(urls_visited) >= max_num:
                break
            current_url = urls_to_visit.pop(0)
            if current_url not in urls_visited:
                urls_visited.append(current_url)
                new_urls = self.get_outsource_urls(current_url, base_url)
                urls_to_visit.extend(new_urls)
                urls_to_visit = list(set(urls_to_visit))
        logger.info(f"URLs visited: {urls_visited}")
        return sorted(urls_visited[:max_num])

    def get_outsource_urls(self, curr_url: str, base_url: str):
        """Get outsource URLs from a given URL.

        This function extracts URLs from a webpage that point to external resources.
        It filters and validates the URLs to ensure they are relevant to the base URL.

        Args:
            curr_url (str): The current URL to extract links from.
            base_url (str): The base URL for filtering and validation.

        Returns:
            List[str]: List of valid outsource URLs.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"
        }
        new_urls = list()
        try:
            response = requests.get(curr_url, headers=headers, timeout=10)
            # Check if the request was successful
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.find_all("a"):
                    try:
                        full_url = urljoin(curr_url, link.get("href"))
                        full_url = full_url.split("#")[0].rstrip("/")
                        if self._check_url(full_url, base_url):
                            new_urls.append(full_url)
                    except Exception as err:
                        logger.error(
                            f"Fail to process sub-url {link.get('href')}: {err}"
                        )
            else:
                logger.error(
                    f"Failed to retrieve page {curr_url}, status code: {response.status_code}"
                )
        except Exception as err:
            logger.error(f"Fail to get the page from {curr_url}: {err}")
        return list(set(new_urls))

    def _check_url(self, full_url, base_url):
        """Check if a URL is valid and belongs to the base URL.

        This function validates a URL by checking if it is properly formatted and
        belongs to the specified base URL domain.

        Args:
            full_url (str): The URL to check.
            base_url (str): The base URL for validation.

        Returns:
            bool: True if the URL is valid and belongs to the base URL, False otherwise.
        """
        kw_list = [".pdf", ".jpg", ".png", ".docx", ".xlsx", ".pptx", ".zip", ".jpeg"]
        if (
            full_url.startswith(base_url)
            and full_url
            and not any(kw in full_url for kw in kw_list)
            and full_url != base_url
        ):
            return True
        return False

    def get_candidates_websites(
        self, urls: List[CrawledObject], top_k: int
    ) -> List[CrawledObject]:
        """Get candidate websites based on content relevance.

        This function analyzes the content of crawled URLs and selects the most
        relevant ones based on their content and metadata.

        Args:
            urls (List[CrawledObject]): List of crawled URL objects.
            top_k (int): Number of top candidates to return.

        Returns:
            List[CrawledObject]: List of selected candidate websites.
        """

        nodes = []
        edges = []
        url_to_id_mapping = {}
        for url in urls:
            url_to_id_mapping[url.source] = url.id

        for url in urls:
            if url.is_error:
                continue
            for url_key in url_to_id_mapping:
                if url_key in url.content:
                    edge = [url.id, url_to_id_mapping[url_key]]
                    edges.append(edge)

            node = [url.id, url.to_dict()]
            nodes.append(node)

        self.graph = nx.DiGraph(name="website graph")
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)
        pr = nx.pagerank(self.graph, alpha=0.9)
        # sort the pagerank values in descending order
        sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"pagerank results: {sorted_pr}")
        # get the top websites
        top_k_websites = sorted_pr[:top_k]
        urls_candidates = [self.graph.nodes[url_id] for url_id, _ in top_k_websites]
        urls_cleaned = [CrawledObject.from_dict(doc) for doc in urls_candidates if doc]
        return urls_cleaned

    def to_crawled_text(self, text_list: List[str]) -> List[CrawledObject]:
        """Convert a list of text strings to CrawledObject instances.

        This function creates CrawledObject instances from a list of text strings,
        assigning unique IDs and appropriate metadata.

        Args:
            text_list (List[str]): List of text strings to convert.

        Returns:
            List[CrawledObject]: List of CrawledObject instances.
        """
        crawled_local_objs = []
        for text in text_list:
            crawled_obj = CrawledObject(
                id=str(uuid.uuid4()),
                source="text",
                content=text,
                metadata={},
                source_type=SourceType.TEXT,
            )
            crawled_local_objs.append(crawled_obj)
        return crawled_local_objs

    def to_crawled_local_objs(self, file_list: List[str]) -> List[CrawledObject]:
        """Convert a list of local files to CrawledObject instances.

        This function processes local files and creates CrawledObject instances
        for each file, handling different file formats and extracting content.

        Args:
            file_list (List[str]): List of file paths to process.

        Returns:
            List[CrawledObject]: List of CrawledObject instances.
        """
        local_objs = [DocObject(str(uuid.uuid4()), file) for file in file_list]
        crawled_local_objs = [self.crawl_file(local_obj) for local_obj in local_objs]
        return crawled_local_objs

    def crawl_file(self, local_obj: DocObject) -> CrawledObject:
        """Crawl a local file and extract its content.

        This function reads and processes a local file, extracting its content
        and metadata based on the file type. It supports various file formats
        including PDF, Word, Excel, Markdown, and text files.

        Args:
            local_obj (DocObject): The local file object to process.

        Returns:
            CrawledObject: A CrawledObject instance containing the file's content and metadata.
        """
        file_path = Path(local_obj.source)
        file_type = file_path.suffix.lstrip(".")
        file_name = file_path.name

        try:
            if not file_type:
                err_msg = f"No file type detected for file: {str(file_path)}"
                raise FileNotFoundError(err_msg)

            if file_type in ["pdf", "png", "jpg", "jpeg"] and (
                MISTRAL_API_KEY is not None
                and MISTRAL_API_KEY != "<your-mistral-api-key>"
            ):
                # Call the Mistral API to extract data.
                client = Mistral(api_key=MISTRAL_API_KEY)
                if file_type == "pdf":
                    # For pdf's
                    uploaded_pdf = client.files.upload(
                        file={
                            "file_name": file_name,
                            "content": open(file_path, "rb"),
                        },
                        purpose="ocr",
                    )
                    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
                    ocr_response = client.ocr.process(
                        model="mistral-ocr-latest",
                        document={
                            "type": "document_url",
                            "document_url": signed_url.url,
                        },
                    )
                else:
                    # For image files
                    base64_image = encode_image(file_path)
                    ocr_response = client.ocr.process(
                        model="mistral-ocr-latest",
                        document={
                            "type": "image_url",
                            "image_url": f"data:image/{file_type};base64,{base64_image}",
                        },
                    )
                doc_text = ""
                for page in ocr_response.pages:
                    doc_text += page.markdown

                logger.info("Mistral PDF extractor worked as expected.")
                return CrawledObject(
                    id=local_obj.id,
                    source=local_obj.source,
                    content=doc_text,
                    metadata={"title": file_name, "source": local_obj.source},
                    source_type=SourceType.FILE,
                )
            elif file_type == "html":
                # TODO : Consider replacing this logic with the Unstructured HTML Loader.
                # Would need to be done in crawl_urls too.
                html = open(file_path, "r", encoding="utf-8").read()
                soup = BeautifulSoup(html, "html.parser")

                text_list = []
                for string in soup.strings:
                    if string.find_parent("a"):
                        href = string.find_parent("a").get("href")
                        text = f"{string} {href}"
                        text_list.append(text)
                    elif string.strip():
                        text_list.append(string)
                doc_text = "\n".join(text_list)

                title = file_name
                for title in soup.find_all("title"):
                    title = title.get_text()
                    break

                return CrawledObject(
                    id=local_obj.id,
                    source=local_obj.source,
                    content=doc_text,
                    metadata={"title": title, "source": local_obj.source},
                    source_type=SourceType.FILE,
                )
            elif file_type == "pdf":
                # Since Mistral API key is absent, we default to basic pdf parser
                logger.info(
                    "MISTRAL_API_KEY env variable not set, hence defaulting to static parsing."
                )
                loader = PyPDFLoader(file_path)
            elif file_type == "doc" or file_type == "docx":
                loader = UnstructuredWordDocumentLoader(file_path, mode="single")
            elif file_type == "xlsx" or file_type == "xls":
                loader = UnstructuredExcelLoader(file_path, mode="single")
            elif file_type == "txt":
                loader = TextLoader(file_path)
            elif file_type == "md":
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                err_msg = "Unsupported file type. If you are trying to upload a pdf, make sure it is less than 50MB. Images are only supported with the advanced parser."
                raise NotImplementedError(err_msg)

            doc_text = "\n".join(
                [
                    document.to_json()["kwargs"]["page_content"]
                    for document in loader.load()
                ]
            )
            return CrawledObject(
                id=local_obj.id,
                source=local_obj.source,
                content=doc_text,
                metadata={"title": file_name, "source": local_obj.source},
                source_type=SourceType.FILE,
            )

        except Exception as err_msg:
            logger.info(f"error processing file: {err_msg}")
            return CrawledObject(
                id=local_obj.id,
                source=local_obj.source,
                content=None,
                metadata={"title": file_name},
                source_type=SourceType.FILE,
                is_error=True,
                error_message=str(err_msg),
            )

    @staticmethod
    def save(file_path: str, docs: List[CrawledObject]):
        """Save a list of CrawledObject instances to a file.

        This function serializes and saves CrawledObject instances to a file
        for later use.

        Args:
            file_path (str): Path where to save the objects.
            docs (List[CrawledObject]): List of CrawledObject instances to save.
        """
        with open(file_path, "wb") as f:
            pickle.dump(docs, f)

    @classmethod
    def chunk(cls, doc_objs: List[CrawledObject]) -> List[CrawledObject]:
        """Split documents into smaller chunks.

        This function splits large documents into smaller, more manageable chunks
        while preserving their metadata and structure.

        Args:
            doc_objs (List[CrawledObject]): List of CrawledObject instances to chunk.

        Returns:
            List[CrawledObject]: List of chunked CrawledObject instances.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=200, chunk_overlap=40
        )
        docs = []
        langchain_docs = []
        for doc_obj in doc_objs:
            if doc_obj.is_error or doc_obj.content is None:
                logger.error(
                    f"Skip source: {doc_obj.source} because of error or no content"
                )
                continue
            elif doc_obj.is_chunk:
                logger.error(
                    f"Skip source: {doc_obj.source} because it has been chunked"
                )
                docs.append(doc_obj)
                continue
            splitted_text = text_splitter.split_text(doc_obj.content)
            for i, txt in enumerate(splitted_text):
                doc = CrawledObject(
                    id=doc_obj.id + "_" + str(i),
                    source=doc_obj.source,
                    content=txt,
                    metadata=doc_obj.metadata,
                    is_chunk=True,
                    source_type=doc_obj.source_type,
                )
                docs.append(doc)
                langchain_docs.append(
                    Document(page_content=txt, metadata={"source": doc_obj.source})
                )
        return langchain_docs
