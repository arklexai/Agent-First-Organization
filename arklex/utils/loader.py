import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
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
import filetype
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
)
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

CHROME_DRIVER_VERSION: str = "125.0.6422.7"
MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")


def encode_image(image_path: str) -> Optional[str]:
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
    def __init__(self, id: str, source: str) -> None:
        self.id: str = id
        self.source: str = source


class CrawledObject(DocObject):
    def __init__(
        self,
        id: str,
        source: str,
        content: Optional[str],
        metadata: Dict[str, Any] = {},
        is_chunk: bool = False,
        is_error: bool = False,
        error_message: Optional[str] = None,
        source_type: SourceType = SourceType.WEB,
    ) -> None:
        super().__init__(id, source)
        self.content: Optional[str] = content
        self.metadata: Dict[str, Any] = metadata
        self.is_chunk: bool = is_chunk
        self.is_error: bool = is_error
        self.error_message: Optional[str] = error_message
        self.source_type: SourceType = source_type

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "CrawledObject":
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
    def __init__(self) -> None:
        pass

    def to_crawled_url_objs(self, url_list: List[str]) -> List[CrawledObject]:
        url_objs: List[DocObject] = [
            DocObject(str(uuid.uuid4()), url) for url in url_list
        ]
        crawled_url_objs: List[CrawledObject] = self.crawl_urls(url_objs)
        return crawled_url_objs

    def crawl_urls(self, url_objects: List[DocObject]) -> List[CrawledObject]:
        logger.info(f"Start crawling {len(url_objects)} urls")
        options: webdriver.ChromeOptions = webdriver.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--headless")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-infobars")
        options.add_argument("--remote-debugging-pipe")
        chrome_driver_path: Path = Path(
            ChromeDriverManager(driver_version=CHROME_DRIVER_VERSION).install()
        )
        options.binary_location = str(chrome_driver_path.parent.absolute())
        logger.info(f"chrome binary location: {options.binary_location}")
        driver: webdriver.Chrome = webdriver.Chrome(options=options)

        docs: List[CrawledObject] = []
        for url_obj in url_objects:
            try:
                logger.info(f"loading url: {url_obj.source}")
                driver.get(url_obj.source)
                time.sleep(2)
                html: str = driver.page_source
                soup: BeautifulSoup = BeautifulSoup(html, "html.parser")

                text_list: List[str] = []
                for string in soup.strings:
                    if string.find_parent("a"):
                        href: str = urljoin(
                            url_obj.source, string.find_parent("a").get("href")
                        )
                        if href.startswith(url_obj.source):
                            text: str = f"{string} {href}"
                            text_list.append(text)
                    elif string.strip():
                        text_list.append(string)
                text_output: str = "\n".join(text_list)

                title: str = url_obj.source
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
        logger.info(
            f"Getting all pages for base url: {base_url}, maximum number is: {max_num}"
        )
        urls_visited: List[str] = []
        base_url: str = base_url.split("#")[0].rstrip("/")
        urls_to_visit: List[str] = [base_url]

        while urls_to_visit:
            if len(urls_visited) >= max_num:
                break
            current_url: str = urls_to_visit.pop(0)
            if current_url not in urls_visited:
                urls_visited.append(current_url)
                new_urls: List[str] = self.get_outsource_urls(current_url, base_url)
                urls_to_visit.extend(new_urls)
                urls_to_visit = list(set(urls_to_visit))
        logger.info(f"URLs visited: {urls_visited}")
        return sorted(urls_visited[:max_num])

    def get_outsource_urls(self, curr_url: str, base_url: str) -> List[str]:
        headers: Dict[str, str] = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"
        }
        new_urls: List[str] = list()
        try:
            response: requests.Response = requests.get(
                curr_url, headers=headers, timeout=10
            )
            # Check if the request was successful
            if response.status_code == 200:
                soup: BeautifulSoup = BeautifulSoup(response.text, "html.parser")
                for link in soup.find_all("a"):
                    try:
                        full_url: str = urljoin(curr_url, link.get("href"))
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

    def _check_url(self, full_url: str, base_url: str) -> bool:
        kw_list: List[str] = [
            ".pdf",
            ".jpg",
            ".png",
            ".docx",
            ".xlsx",
            ".pptx",
            ".zip",
            ".jpeg",
        ]
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
        """Based on the pagerank algorithm of the crawled websites, return the top k websites.
        The reason why we can do that is because we have the hreqs of the including <a> tags in the content of the website.
        So we can use that to construct the edges and then use the tool from networkx to get the pagerank of the websites.
        """

        nodes: List[List[Any]] = []
        edges: List[List[str]] = []
        url_to_id_mapping: Dict[str, str] = {}
        for url in urls:
            url_to_id_mapping[url.source] = url.id

        for url in urls:
            if url.is_error:
                continue
            for url_key in url_to_id_mapping:
                if url_key in url.content:
                    edge: List[str] = [url.id, url_to_id_mapping[url_key]]
                    edges.append(edge)

            node: List[Any] = [url.id, url.to_dict()]
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
        text_objs: List[DocObject] = [
            DocObject(str(uuid.uuid4()), text) for text in text_list
        ]
        crawled_text_objs: List[CrawledObject] = []
        for text_obj in text_objs:
            crawled_text_objs.append(
                CrawledObject(
                    id=text_obj.id,
                    source=text_obj.source,
                    content=text_obj.source,
                    metadata={"title": text_obj.source, "source": text_obj.source},
                    source_type=SourceType.TEXT,
                )
            )
        return crawled_text_objs

    def to_crawled_local_objs(self, file_list: List[str]) -> List[CrawledObject]:
        file_objs: List[DocObject] = [
            DocObject(str(uuid.uuid4()), file) for file in file_list
        ]
        crawled_file_objs: List[CrawledObject] = [
            self.crawl_file(file_obj) for file_obj in file_objs
        ]
        return crawled_file_objs

    def crawl_file(self, local_obj: DocObject) -> CrawledObject:
        try:
            file_path: str = local_obj.source
            file_type: Optional[str] = filetype.guess(file_path)
            if file_type is None:
                return CrawledObject(
                    id=local_obj.id,
                    source=local_obj.source,
                    content=None,
                    metadata={"title": local_obj.source, "source": local_obj.source},
                    is_error=True,
                    error_message="File type not supported",
                    source_type=SourceType.FILE,
                )

            file_extension: str = file_type.extension
            if file_extension == "pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == "txt":
                loader = TextLoader(file_path)
            elif file_extension == "docx":
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension == "xlsx":
                loader = UnstructuredExcelLoader(file_path)
            elif file_extension == "md":
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                return CrawledObject(
                    id=local_obj.id,
                    source=local_obj.source,
                    content=None,
                    metadata={"title": local_obj.source, "source": local_obj.source},
                    is_error=True,
                    error_message=f"File type {file_extension} not supported",
                    source_type=SourceType.FILE,
                )

            docs: List[Document] = loader.load()
            content: str = "\n".join([doc.page_content for doc in docs])

            return CrawledObject(
                id=local_obj.id,
                source=local_obj.source,
                content=content,
                metadata={"title": local_obj.source, "source": local_obj.source},
                source_type=SourceType.FILE,
            )

        except Exception as err:
            logger.error(f"Error crawling file {local_obj.source}: {err}")
            return CrawledObject(
                id=local_obj.id,
                source=local_obj.source,
                content=None,
                metadata={"title": local_obj.source, "source": local_obj.source},
                is_error=True,
                error_message=str(err),
                source_type=SourceType.FILE,
            )

    @staticmethod
    def save(file_path: str, docs: List[CrawledObject]) -> None:
        with open(file_path, "wb") as f:
            pickle.dump(docs, f)

    @classmethod
    def chunk(cls, doc_objs: List[CrawledObject]) -> List[CrawledObject]:
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        chunked_docs: List[CrawledObject] = []
        for doc in doc_objs:
            if doc.is_error or doc.is_chunk:
                continue

            chunks: List[str] = text_splitter.split_text(doc.content)
            for i, chunk in enumerate(chunks):
                chunked_docs.append(
                    CrawledObject(
                        id=f"{doc.id}_chunk_{i}",
                        source=doc.source,
                        content=chunk,
                        metadata=doc.metadata,
                        is_chunk=True,
                        source_type=doc.source_type,
                    )
                )
        return chunked_docs
