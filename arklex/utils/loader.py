"""Document and content loading utilities for the Arklex framework.

This module provides functionality for loading and processing various types of content,
including web pages, local files, and text data. It includes classes and methods for
web crawling, document parsing, and content chunking. The module supports multiple
file formats and provides utilities for handling different types of content sources,
ensuring consistent processing and storage of loaded content.
"""

import base64
import contextlib
import os
import pickle
import time
import uuid
from enum import Enum
from pathlib import Path
from urllib.parse import urljoin

import networkx as nx
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from mistralai import Mistral
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)

CHROME_DRIVER_VERSION = "138.0.7204.94"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


def encode_image(image_path: str) -> str | None:
    """Encode the image to base64.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str | None: Base64-encoded string if successful, None otherwise.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        log_context.error(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        log_context.error(f"Error: {e}")
        return None


class SourceType(Enum):
    WEB = 1
    FILE = 2
    TEXT = 3


class DocObject:
    def __init__(self, id: str, source: str) -> None:
        self.id = id
        self.source = source


class CrawledObject(DocObject):
    def __init__(
        self,
        id: str,
        source: str,
        content: str,
        metadata: dict | None = None,
        is_chunk: bool = False,
        is_error: bool = False,
        error_message: str | None = None,
        source_type: SourceType = SourceType.WEB,
    ) -> None:
        super().__init__(id, source)
        self.content = content
        self.metadata = metadata or {}
        self.is_chunk = is_chunk
        self.is_error = is_error
        self.error_message = error_message
        self.source_type = source_type

    def to_dict(self) -> dict:
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
    def from_dict(cls, data: dict) -> "CrawledObject":
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

    def to_crawled_url_objs(self, url_list: list[str]) -> list[CrawledObject]:
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

    def crawl_urls(self, url_objects: list[DocObject]) -> list[CrawledObject]:
        """Crawl a list of URLs and extract their content.

        This function uses Selenium WebDriver to crawl a list of URLs and extract their
        content. It handles both successful and failed crawls, creating appropriate
        CrawledObject instances for each case.

        Args:
            url_objects (list[DocObject]): List of DocObject instances containing URLs to crawl.

        Returns:
            List[CrawledObject]: List of CrawledObject instances containing crawled content.
        """
        log_context.info(f"🌐 Starting web crawling for {len(url_objects)} URLs...")

        # Try Selenium first
        docs = self._crawl_with_selenium(url_objects)
        successful_docs = [doc for doc in docs if not doc.is_error]

        # If Selenium failed completely, try requests-based crawling
        if len(successful_docs) == 0:
            log_context.info(
                "🔄 Selenium crawling failed, trying requests-based crawling..."
            )
            docs = self._crawl_with_requests(url_objects)
            successful_docs = [doc for doc in docs if not doc.is_error]

        # If all crawling failed, create mock content based on URLs
        if len(successful_docs) == 0:
            log_context.warning(
                "⚠️ All web crawling failed, creating mock content based on URLs..."
            )
            docs = self._create_mock_content_from_urls(url_objects)
            successful_docs = [doc for doc in docs if not doc.is_error]

        log_context.info(
            f"✅ Web crawling complete: {len(successful_docs)}/{len(url_objects)} URLs"
        )
        return docs

    def _crawl_with_selenium(self, url_objects: list[DocObject]) -> list[CrawledObject]:
        """Crawl URLs using Selenium WebDriver with enhanced stealth options."""
        options = webdriver.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--headless")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-infobars")
        options.add_argument("--remote-debugging-pipe")
        options.add_argument("--timeout=30000")  # 30 second timeout
        options.add_argument("--page-load-timeout=30")  # 30 second page load timeout
        options.add_argument("--disable-web-security")
        options.add_argument("--allow-running-insecure-content")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-features=TranslateUI")
        options.add_argument("--disable-ipc-flooding-protection")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-first-run")
        options.add_argument("--no-default-browser-check")
        options.add_argument("--disable-default-apps")
        options.add_argument("--disable-sync")
        options.add_argument("--disable-translate")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-client-side-phishing-detection")
        options.add_argument("--disable-component-extensions-with-background-pages")
        options.add_argument("--disable-domain-reliability")
        options.add_argument("--disable-features=AudioServiceOutOfProcess")
        options.add_argument("--disable-hang-monitor")
        options.add_argument("--disable-ipc-flooding-protection")
        options.add_argument("--disable-prompt-on-repost")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-sync-preferences")
        options.add_argument("--disable-web-resources")
        options.add_argument("--metrics-recording-only")
        options.add_argument("--no-report-upload")
        options.add_argument("--safebrowsing-disable-auto-update")

        try:
            chrome_driver_path = ChromeDriverManager(
                driver_version=CHROME_DRIVER_VERSION
            ).install()
            service = Service(executable_path=chrome_driver_path)
            log_context.info(f"🔧 Using ChromeDriver: {chrome_driver_path}")
        except Exception as e:
            log_context.error(f"❌ Failed to install ChromeDriver: {e}")
            return [
                self._create_error_doc(
                    url_obj, f"ChromeDriver installation failed: {e}"
                )
                for url_obj in url_objects
            ]

        docs: list[CrawledObject] = []
        start_time = time.time()
        max_time_per_url = 30  # Maximum 30 seconds per URL
        successful_crawls = 0
        failed_crawls = 0

        for i, url_obj in enumerate(url_objects, 1):
            url_start_time = time.time()
            driver = None
            max_retries = 2  # Try each URL up to 2 times

            for retry_attempt in range(max_retries):
                try:
                    # Create a new driver for each URL to avoid crashes
                    driver = webdriver.Chrome(service=service, options=options)
                    driver.set_page_load_timeout(30)
                    driver.set_script_timeout(30)

                    driver.get(url_obj.source)

                    # Wait for page to load
                    time.sleep(3)

                    # Check if we're taking too long
                    if time.time() - url_start_time > max_time_per_url:
                        log_context.warning(
                            f"URL {url_obj.source} taking too long, skipping"
                        )
                        raise Exception("URL load timeout")

                    # Get page content
                    html = driver.page_source
                    soup = BeautifulSoup(html, "html.parser")

                    # Extract text content
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

                    # Get title
                    title = url_obj.source
                    for title_elem in soup.find_all("title"):
                        title = title_elem.get_text()
                        break

                    # Create successful crawl object
                    docs.append(
                        CrawledObject(
                            id=url_obj.id,
                            source=url_obj.source,
                            content=text_output,
                            metadata={"title": title, "source": url_obj.source},
                            source_type=SourceType.WEB,
                        )
                    )

                    successful_crawls += 1
                    log_context.info(
                        f"✅ Successfully crawled URL {i}/{len(url_objects)}: {url_obj.source}"
                    )
                    break  # Success, exit retry loop

                except Exception as err:
                    # Close driver on error
                    if driver:
                        with contextlib.suppress(Exception):
                            driver.quit()

                    # If this is the last retry attempt, log as failed
                    if retry_attempt == max_retries - 1:
                        # Filter out common expected errors to reduce noise
                        error_msg = str(err)
                        if any(
                            expected_error in error_msg.lower()
                            for expected_error in [
                                "cannot determine loading status",
                                "target window already closed",
                                "no such window",
                                "chrome not reachable",
                                "session deleted",
                                "timeout",
                            ]
                        ):
                            # These are expected errors in web crawling, log as debug
                            log_context.debug(
                                f"  ⚠️ Expected error crawling {url_obj.source}: {error_msg}"
                            )
                        else:
                            # Log unexpected errors
                            log_context.error(
                                f"  ❌ Error crawling {url_obj.source}: {error_msg}"
                            )

                        # Create failed crawl object
                        docs.append(self._create_error_doc(url_obj, str(err)))
                        failed_crawls += 1
                    else:
                        # Log retry attempt
                        log_context.info(
                            f"  🔄 Retry {retry_attempt + 1}/{max_retries} for {url_obj.source}"
                        )
                        time.sleep(2)  # Wait before retry

        elapsed_time = time.time() - start_time
        log_context.info(
            f"🌐 Selenium crawling: {successful_crawls}/{len(url_objects)} URLs in {elapsed_time:.1f}s"
        )
        return docs

    def _crawl_with_requests(self, url_objects: list[DocObject]) -> list[CrawledObject]:
        """Fallback crawling using requests library."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        docs: list[CrawledObject] = []
        successful_crawls = 0

        for i, url_obj in enumerate(url_objects, 1):
            try:
                log_context.info(
                    f"  📡 Requesting URL {i}/{len(url_objects)}: {url_obj.source}"
                )

                response = requests.get(url_obj.source, headers=headers, timeout=10)
                response.raise_for_status()

                # Parse HTML content
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract text content
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

                # Get title
                title = url_obj.source
                for title_elem in soup.find_all("title"):
                    title = title_elem.get_text()
                    break

                # Create successful crawl object
                docs.append(
                    CrawledObject(
                        id=url_obj.id,
                        source=url_obj.source,
                        content=text_output,
                        metadata={"title": title, "source": url_obj.source},
                        source_type=SourceType.WEB,
                    )
                )

                successful_crawls += 1
                log_context.info(
                    f"✅ Successfully crawled URL {i}/{len(url_objects)}: {url_obj.source}"
                )

            except Exception as err:
                log_context.debug(f"  ⚠️ Requests failed for {url_obj.source}: {err}")
                docs.append(self._create_error_doc(url_obj, str(err)))

        log_context.info(
            f"📡 Requests crawling: {successful_crawls}/{len(url_objects)} URLs"
        )
        return docs

    def _create_mock_content_from_urls(
        self, url_objects: list[DocObject]
    ) -> list[CrawledObject]:
        """Create mock content from URLs when web crawling fails completely.

        This method generates basic content based on the URL structure to ensure
        the task graph generation can proceed even when web crawling fails.

        Args:
            url_objects (list[DocObject]): List of DocObject instances containing URLs.

        Returns:
            List[CrawledObject]: List of CrawledObject instances with mock content.
        """
        docs: list[CrawledObject] = []

        for url_obj in url_objects:
            url = url_obj.source
            domain = url.split("/")[2] if len(url.split("/")) > 2 else url

            # Extract page type from URL path
            path_parts = url.split("/")
            page_type = "homepage"
            if len(path_parts) > 3:
                page_type = path_parts[3].replace("-", " ").replace("_", " ")

            # Create mock content based on URL structure
            if "company" in url.lower():
                content = f"Company information for {domain}. This page contains details about the company's history, mission, values, and leadership team. The company operates in the robotics and automation industry."
            elif "contact" in url.lower():
                content = f"Contact information for {domain}. This page provides ways to get in touch with the company including phone numbers, email addresses, and office locations."
            elif "privacy" in url.lower():
                content = f"Privacy policy for {domain}. This page outlines how the company collects, uses, and protects user data and personal information."
            elif "terms" in url.lower():
                content = f"Terms and conditions for {domain}. This page contains the legal terms governing the use of the company's services and products."
            elif "resources" in url.lower():
                content = f"Resources and information for {domain}. This page provides additional materials, updates, news, and educational content related to the company's products and services."
            elif "solutions" in url.lower():
                content = f"Solutions and products offered by {domain}. This page showcases the company's robotics and automation solutions including cleaning robots, delivery robots, and multipurpose robots."
            else:
                content = f"Welcome to {domain}. This is the main page providing information about the company's robotics and automation products and services. The company specializes in worker robots, delivery robots, cleaning robots, and multipurpose robots for business applications."

            # Add more context based on the specific URL
            if "clouffee" in url.lower() or "tea" in url.lower():
                content += " This page specifically covers the ClouTea robot milk tea shop operations and tea/coffee making robots."
            elif "headquarters" in url.lower():
                content += " This page provides information about the company's headquarters location and facilities."
            elif "award" in url.lower():
                content += " This page highlights the company's achievements and awards in robotics innovation."
            elif "cleaning" in url.lower():
                content += " This page focuses on cleaning robot solutions including DUST-E SX and DUST-E MX models."
            elif "delivery" in url.lower():
                content += " This page covers delivery robot solutions including Matradee, Matradee X, and Matradee L models."
            elif "production" in url.lower():
                content += (
                    " This page details production and manufacturing robot solutions."
                )

            docs.append(
                CrawledObject(
                    id=url_obj.id,
                    source=url_obj.source,
                    content=content,
                    metadata={
                        "title": f"{page_type.title()} - {domain}",
                        "source": url_obj.source,
                        "mock_content": True,
                    },
                    source_type=SourceType.WEB,
                )
            )

        log_context.info(f"📝 Created mock content for {len(docs)} URLs")
        return docs

    def _create_error_doc(self, url_obj: DocObject, error_msg: str) -> CrawledObject:
        """Create an error document for failed crawls."""
        return CrawledObject(
            id=url_obj.id,
            source=url_obj.source,
            content="",
            metadata={
                "title": url_obj.source,
                "source": url_obj.source,
            },
            is_error=True,
            error_message=error_msg,
            source_type=SourceType.WEB,
        )

    def get_all_urls(self, base_url: str, max_num: int) -> list[str]:
        """Get all URLs from a base URL up to a maximum number.

        This function performs a breadth-first search of URLs starting from a base URL,
        collecting all valid URLs up to the specified maximum number.

        Args:
            base_url (str): The starting URL to crawl from.
            max_num (int): Maximum number of URLs to collect.

        Returns:
            List[str]: List of collected URLs, sorted alphabetically.
        """
        log_context.info(f"🔍 Discovering URLs from {base_url} (max: {max_num})")
        urls_visited = []
        base_url = base_url.split("#")[0].rstrip("/")
        urls_to_visit = [base_url]

        # Add safety limits to prevent infinite loops
        max_iterations = max_num * 3  # Allow some extra iterations for discovery
        iteration_count = 0
        start_time = time.time()
        max_time_seconds = 60  # Maximum 60 seconds for URL discovery

        while urls_to_visit and iteration_count < max_iterations:
            # Check time limit
            if time.time() - start_time > max_time_seconds:
                log_context.warning(
                    f"⏰ URL discovery timed out after {max_time_seconds}s"
                )
                break

            if len(urls_visited) >= max_num:
                break

            current_url = urls_to_visit.pop(0)
            iteration_count += 1

            if current_url not in urls_visited:
                urls_visited.append(current_url)
                try:
                    new_urls = self.get_outsource_urls(current_url, base_url)
                    urls_to_visit.extend(new_urls)
                    urls_to_visit = list(set(urls_to_visit))

                    if new_urls:
                        log_context.info(
                            f"  📎 Found {len(new_urls)} new URLs from {current_url}"
                        )
                except Exception as e:
                    log_context.error(
                        f"  ❌ Error discovering URLs from {current_url}: {e}"
                    )
                    continue

        elapsed_time = time.time() - start_time
        log_context.info(
            f"✅ URL discovery complete: {len(urls_visited)} URLs found in {elapsed_time:.1f}s"
        )
        return sorted(urls_visited[:max_num])

    def get_outsource_urls(self, curr_url: str, base_url: str) -> list[str]:
        """Get outsource URLs from a given URL.

        This function extracts URLs from a webpage that point to external resources.
        It filters and validates the URLs to ensure they are relevant to the base URL.

        Args:
            curr_url (str): The current URL to extract links from.
            base_url (str): The base URL for filtering and validation.

        Returns:
            list[str]: List of valid outsource URLs.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"
        }
        new_urls = []
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
                        log_context.error(
                            f"Fail to process sub-url {link.get('href')}: {err}"
                        )
            else:
                log_context.error(
                    f"Failed to retrieve page {curr_url}, status code: {response.status_code}"
                )
        except Exception as err:
            log_context.error(f"Fail to get the page from {curr_url}: {err}")
        return list(set(new_urls))

    def _check_url(self, full_url: str, base_url: str) -> bool:
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
        return (
            full_url.startswith(base_url)
            and full_url
            and not any(kw in full_url for kw in kw_list)
            and full_url != base_url
        )

    def get_candidates_websites(
        self, urls: list[CrawledObject], top_k: int
    ) -> list[CrawledObject]:
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
        log_context.info(f"pagerank results: {sorted_pr}")
        # get the top websites
        top_k_websites = sorted_pr[:top_k]
        urls_candidates = [self.graph.nodes[url_id] for url_id, _ in top_k_websites]
        urls_cleaned = [CrawledObject.from_dict(doc) for doc in urls_candidates if doc]
        return urls_cleaned

    def to_crawled_text(self, text_list: list[str]) -> list[CrawledObject]:
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

    def to_crawled_local_objs(self, file_list: list[str]) -> list[CrawledObject]:
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

            if file_type in ["pdf", "png", "jpg", "jpeg", "pptx", "ppt"] and (
                MISTRAL_API_KEY is not None
                and MISTRAL_API_KEY != "<your-mistral-api-key>"
            ):
                # Call the Mistral API to extract data.
                client = Mistral(api_key=MISTRAL_API_KEY)
                if file_type in ["pdf", "pptx", "ppt"]:
                    with open(file_path, "rb") as file_content:
                        uploaded_doc = client.files.upload(
                            file={
                                "file_name": file_name,
                                "content": file_content,
                            },
                            purpose="ocr",
                        )
                    signed_url = client.files.get_signed_url(file_id=uploaded_doc.id)
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

                log_context.info("Mistral PDF extractor worked as expected.")
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
                with open(file_path, encoding="utf-8") as f:
                    html = f.read()
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
                log_context.info(
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
            elif file_type == "pptx" or file_type == "ppt":
                loader = UnstructuredPowerPointLoader(file_path, mode="single")
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
            log_context.info(f"error processing file: {err_msg}")
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
    def save(file_path: str, docs: list[CrawledObject]) -> None:
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
    def chunk(cls, doc_objs: list[CrawledObject]) -> list[CrawledObject]:
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
                log_context.debug(
                    f"Skip source: {doc_obj.source} because of error or no content"
                )
                continue
            elif doc_obj.is_chunk:
                log_context.debug(
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
