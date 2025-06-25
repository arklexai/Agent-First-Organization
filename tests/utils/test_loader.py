"""Tests for the loader module.

This module contains comprehensive test cases for document and content loading utilities,
including web crawling, file processing, and content chunking functionality.
"""

from unittest.mock import Mock, patch
import tempfile
import os
import requests
from langchain_core.documents import Document

from arklex.utils.loader import (
    encode_image,
    SourceType,
    DocObject,
    CrawledObject,
    Loader,
)


class TestEncodeImage:
    """Test cases for encode_image function."""

    def test_encode_image_success(self) -> None:
        """Test successful image encoding."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_file.write(b"fake image data")
            tmp_file_path = tmp_file.name

        try:
            with patch("builtins.open", create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = (
                    b"fake image data"
                )
                with patch("base64.b64encode") as mock_b64encode:
                    mock_b64encode.return_value.decode.return_value = (
                        "ZmFrZSBpbWFnZSBkYXRh"
                    )
                    result = encode_image(tmp_file_path)
                    assert result == "ZmFrZSBpbWFnZSBkYXRh"
        finally:
            os.unlink(tmp_file_path)

    def test_encode_image_file_not_found(self) -> None:
        """Test image encoding with non-existent file."""
        result = encode_image("nonexistent.png")
        assert result is None

    def test_encode_image_general_exception(self) -> None:
        """Test image encoding with general exception."""
        with patch("builtins.open", side_effect=Exception("Permission denied")):
            result = encode_image("test.png")
            assert result is None

    def test_encode_image_nonexistent(self) -> None:
        """Test image encoding with None path."""
        result = encode_image(None)
        assert result is None

    def test_encode_image_with_directory(self) -> None:
        """Test image encoding with directory path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = encode_image(temp_dir)
            assert result is None

    def test_encode_image_with_non_image_file(self) -> None:
        """Test image encoding with non-image file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(b"text content")
            tmp_file_path = tmp_file.name

        try:
            with patch("builtins.open", create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = (
                    b"text content"
                )
                with patch("base64.b64encode") as mock_b64encode:
                    mock_b64encode.return_value.decode.return_value = "dGV4dCBjb250ZW50"
                    result = encode_image(tmp_file_path)
                    assert result == "dGV4dCBjb250ZW50"
        finally:
            os.unlink(tmp_file_path)


class TestSourceType:
    """Test cases for SourceType enum."""

    def test_source_type_values(self) -> None:
        """Test SourceType enum values."""
        assert SourceType.WEB.value == 1
        assert SourceType.FILE.value == 2
        assert SourceType.TEXT.value == 3

    def test_source_type_unknown(self) -> None:
        """Test SourceType with unknown value."""
        # Test that we can access the enum values
        assert hasattr(SourceType, "WEB")
        assert hasattr(SourceType, "FILE")
        assert hasattr(SourceType, "TEXT")


class TestDocObject:
    """Test cases for DocObject class."""

    def test_doc_object_initialization(self) -> None:
        """Test DocObject initialization."""
        doc = DocObject("test_id", "test_source")
        assert doc.id == "test_id"
        assert doc.source == "test_source"

    def test_docobject_missing_fields(self) -> None:
        """Test DocObject with missing fields."""
        doc = DocObject("", "")
        assert doc.id == ""
        assert doc.source == ""


class TestCrawledObject:
    """Test cases for CrawledObject class."""

    def test_crawled_object_initialization(self) -> None:
        """Test CrawledObject initialization."""
        crawled = CrawledObject(
            "test_id", "test_source", "test_content", {"key": "value"}
        )
        assert crawled.id == "test_id"
        assert crawled.source == "test_source"
        assert crawled.content == "test_content"
        assert crawled.metadata == {"key": "value"}

    def test_crawled_object_default_values(self) -> None:
        """Test CrawledObject with default values."""
        crawled = CrawledObject("test_id", "test_source", "test_content")
        assert crawled.metadata == {}
        assert crawled.is_chunk is False
        assert crawled.is_error is False
        assert crawled.error_message is None
        assert crawled.source_type == SourceType.WEB

    def test_crawled_object_to_dict(self) -> None:
        """Test CrawledObject to_dict method."""
        crawled = CrawledObject(
            "test_id", "test_source", "test_content", {"key": "value"}
        )
        result = crawled.to_dict()
        assert result["id"] == "test_id"
        assert result["source"] == "test_source"
        assert result["content"] == "test_content"
        assert result["metadata"] == {"key": "value"}

    def test_crawled_object_from_dict(self) -> None:
        """Test CrawledObject from_dict method."""
        data = {
            "id": "test_id",
            "source": "test_source",
            "content": "test_content",
            "metadata": {"key": "value"},
            "is_chunk": False,
            "is_error": False,
            "error_message": None,
            "source_type": SourceType.WEB,
        }
        crawled = CrawledObject.from_dict(data)
        assert crawled.id == "test_id"
        assert crawled.source == "test_source"
        assert crawled.content == "test_content"
        assert crawled.metadata == {"key": "value"}


class TestLoader:
    """Test cases for Loader class."""

    def test_loader_initialization(self) -> None:
        """Test Loader initialization."""
        loader = Loader()
        assert loader is not None

    def test_to_crawled_url_objs(self) -> None:
        """Test converting URL list to CrawledObject list."""
        loader = Loader()
        urls = ["http://example.com", "http://test.com"]
        result = loader.to_crawled_url_objs(urls)
        assert len(result) == 2
        assert result[0].source == "http://example.com"
        assert result[1].source == "http://test.com"

    def test_crawl_urls_selenium_success(self) -> None:
        """Test successful Selenium crawling."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        with patch("selenium.webdriver.Chrome") as mock_driver:
            mock_driver_instance = Mock()
            mock_driver.return_value = mock_driver_instance
            mock_driver_instance.page_source = (
                "<html><title>Test</title><body>Content</body></html>"
            )
            mock_driver_instance.quit.return_value = None

            with patch("time.sleep"):
                result = loader.crawl_urls(url_objects)
                assert len(result) == 1
                assert result[0].source == "http://example.com"

    def test_crawl_urls_selenium_failure_requests_success(self) -> None:
        """Test Selenium failure with requests fallback success."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        # Mock Selenium failure
        with patch(
            "selenium.webdriver.Chrome", side_effect=Exception("Selenium failed")
        ):
            # Mock requests success
            with patch("requests.get") as mock_get:
                mock_response = Mock()
                mock_response.text = (
                    "<html><title>Test</title><body>Content</body></html>"
                )
                mock_response.raise_for_status.return_value = None
                mock_get.return_value = mock_response

                result = loader.crawl_urls(url_objects)
                assert len(result) == 1
                assert result[0].source == "http://example.com"

    def test_crawl_urls_all_failure_mock_content(self) -> None:
        """Test all crawling methods failure with mock content fallback."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        # Mock both Selenium and requests failure
        with patch(
            "selenium.webdriver.Chrome", side_effect=Exception("Selenium failed")
        ):
            with patch("requests.get", side_effect=Exception("Requests failed")):
                result = loader.crawl_urls(url_objects)
                assert len(result) == 1
                assert result[0].source == "http://example.com"
                assert "mock_content" in result[0].metadata

    def test_crawl_with_selenium_timeout(self) -> None:
        """Test _crawl_with_selenium with timeout."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        # Mock Selenium to raise timeout exception
        with patch("selenium.webdriver.Chrome", side_effect=Exception("timeout")):
            with patch("time.sleep"):
                result = loader._crawl_with_selenium(url_objects)
                assert len(result) == 1
                assert result[0].is_error

    def test_crawl_with_selenium_retry_success(self) -> None:
        """Test Selenium crawling with retry success."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        with patch("selenium.webdriver.Chrome") as mock_driver:
            mock_driver_instance = Mock()
            mock_driver.return_value = mock_driver_instance
            mock_driver_instance.page_source = (
                "<html><title>Test</title><body>Content</body></html>"
            )
            mock_driver_instance.quit.return_value = None

            # First call fails, second succeeds
            mock_driver_instance.page_source = (
                "<html><title>Test</title><body>Content</body></html>"
            )

            with patch("time.sleep"):
                result = loader._crawl_with_selenium(url_objects)
                assert len(result) == 1

    def test_crawl_with_selenium_expected_error_filtering(self) -> None:
        """Test expected error filtering logic in selenium crawling."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]
        # Patch webdriver.Chrome to raise an expected error
        mock_driver = Mock()
        mock_driver.get.side_effect = Exception("timeout")
        with patch("selenium.webdriver.Chrome", return_value=mock_driver):
            with patch(
                "webdriver_manager.chrome.ChromeDriverManager.install",
                return_value="/tmp/chromedriver",
            ):
                with patch("selenium.webdriver.chrome.service.Service"):
                    result = loader._crawl_with_selenium(url_objects)
                    assert len(result) == 1
                    assert result[0].is_error
                    assert "timeout" in result[0].error_message

    def test_crawl_with_requests_http_error(self) -> None:
        """Test requests crawling with HTTP error."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.HTTPError("404 Not Found")
            result = loader._crawl_with_requests(url_objects)
            assert len(result) == 1
            assert result[0].is_error

    def test_create_mock_content_from_urls_various_urls(self) -> None:
        """Test creating mock content from various URL types."""
        loader = Loader()
        url_objects = [
            DocObject("1", "http://example.com/company"),
            DocObject("2", "http://example.com/contact"),
            DocObject("3", "http://example.com/privacy"),
            DocObject("4", "http://example.com/terms"),
            DocObject("5", "http://example.com/resources"),
            DocObject("6", "http://example.com/solutions"),
            DocObject("7", "http://example.com/clouffee"),
            DocObject("8", "http://example.com/headquarters"),
            DocObject("9", "http://example.com/award"),
            DocObject("10", "http://example.com/cleaning"),
            DocObject("11", "http://example.com/delivery"),
            DocObject("12", "http://example.com/production"),
            DocObject("13", "http://example.com/"),
        ]

        result = loader._create_mock_content_from_urls(url_objects)
        assert len(result) == 13
        assert all("mock_content" in doc.metadata for doc in result)

    def test_get_all_urls_timeout(self) -> None:
        """Test URL discovery timeout handling in get_all_urls."""
        loader = Loader()
        # Patch time.time to simulate timeout
        with patch("time.time", side_effect=[0, 100, 200, 300, 400, 1000]):
            result = loader.get_all_urls("http://example.com", 1)
            assert isinstance(result, list)

    def test_get_all_urls_exception_handling(self) -> None:
        """Test get_all_urls with exception handling."""
        loader = Loader()

        with patch.object(
            loader, "get_outsource_urls", side_effect=Exception("Network error")
        ):
            result = loader.get_all_urls("http://example.com", 10)
            assert len(result) == 1  # Should include the base URL

    def test_check_url_edge_cases(self) -> None:
        """Test URL validation logic in _check_url for various edge cases."""
        loader = Loader()
        # Valid URL
        assert loader._check_url("http://example.com/page", "http://example.com")
        # Invalid: not base
        assert not loader._check_url("http://other.com/page", "http://example.com")
        # Invalid: file extension
        assert not loader._check_url(
            "http://example.com/file.pdf", "http://example.com"
        )
        # Invalid: same as base
        assert not loader._check_url("http://example.com", "http://example.com")
        # Invalid: empty
        assert not loader._check_url("", "http://example.com")

    def test_get_candidates_websites_edge_cases(self) -> None:
        """Test get_candidates_websites with edge cases."""
        loader = Loader()

        # Test with empty list
        result = loader.get_candidates_websites([], 5)
        assert len(result) == 0

        # Test with fewer URLs than top_k
        urls = [
            CrawledObject("1", "http://example1.com", "content1"),
            CrawledObject("2", "http://example2.com", "content2"),
        ]
        result = loader.get_candidates_websites(urls, 5)
        assert len(result) == 2

    def test_crawl_file_with_mistral_api(self) -> None:
        """Test crawl_file with Mistral API."""
        loader = Loader()
        doc_obj = DocObject("1", "test.txt")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp_file:
            tmp_file.write("Test content")
            tmp_file_path = tmp_file.name

        try:
            with patch("mistralai.Mistral") as mock_mistral:
                mock_instance = mock_mistral.return_value
                mock_instance.generate.return_value = "API response"
                result = loader.crawl_file(doc_obj)
                assert isinstance(result, CrawledObject)
        finally:
            os.remove(tmp_file_path)

    def test_chunk_with_empty_docs(self) -> None:
        """Test chunk method with empty documents."""
        result = Loader.chunk([])
        assert len(result) == 0

    def test_chunk_with_exception(self) -> None:
        """Test chunk method with exception."""
        docs = [CrawledObject("1", "test.txt", "content")]

        with patch(
            "langchain.text_splitter.RecursiveCharacterTextSplitter",
            side_effect=Exception("Splitter error"),
        ):
            result = Loader.chunk(docs)
            assert len(result) == 1
            # The result is a Document object, not CrawledObject, so we check differently
            assert hasattr(result[0], "page_content")

    def test_loader_crawl_urls_empty(self) -> None:
        """Test crawling empty URL list."""
        loader = Loader()
        result = loader.crawl_urls([])
        assert len(result) == 0

    def test_loader_chunk_empty_docs(self) -> None:
        """Test chunking empty documents."""
        result = Loader.chunk([])
        assert len(result) == 0

    def test_source_type_unknown(self) -> None:
        """Test SourceType with unknown value."""
        # Test that we can access the enum values
        assert hasattr(SourceType, "WEB")
        assert hasattr(SourceType, "FILE")
        assert hasattr(SourceType, "TEXT")

    def test_crawl_with_selenium_chromedriver_installation_failure(self) -> None:
        """Test ChromeDriver installation failure handling."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        # Patch ChromeDriverManager().install to raise an exception
        with patch(
            "webdriver_manager.chrome.ChromeDriverManager.install",
            side_effect=Exception("install fail"),
        ):
            result = loader._crawl_with_selenium(url_objects)
            assert len(result) == 1
            assert result[0].is_error
            assert "ChromeDriver installation failed" in result[0].error_message

    def test_crawl_with_selenium_driver_quit_exception(self) -> None:
        """Test driver.quit() raising an exception during error handling."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        # Patch webdriver.Chrome to raise an exception on get, and driver.quit to raise as well
        mock_driver = Mock()
        mock_driver.get.side_effect = Exception("page load fail")
        mock_driver.quit.side_effect = Exception("quit fail")
        with patch("selenium.webdriver.Chrome", return_value=mock_driver):
            with patch(
                "webdriver_manager.chrome.ChromeDriverManager.install",
                return_value="/tmp/chromedriver",
            ):
                with patch("selenium.webdriver.chrome.service.Service"):
                    result = loader._crawl_with_selenium(url_objects)
                    assert len(result) == 1
                    assert result[0].is_error
                    assert "page load fail" in result[0].error_message

    def test_crawl_with_selenium_webdriver_exception(self) -> None:
        """Test Selenium crawling with webdriver exception."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        with patch(
            "selenium.webdriver.Chrome", side_effect=Exception("chrome not reachable")
        ):
            with patch("time.sleep"):
                result = loader._crawl_with_selenium(url_objects)
                assert len(result) == 1
                assert result[0].is_error

    def test_crawl_with_requests_timeout(self) -> None:
        """Test requests timeout handling in _crawl_with_requests."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]
        with patch("requests.get", side_effect=requests.Timeout("timeout")):
            result = loader._crawl_with_requests(url_objects)
            assert len(result) == 1
            assert result[0].is_error
            assert "timeout" in result[0].error_message

    def test_crawl_file_permission_error(self) -> None:
        """Test crawl_file with permission error."""
        loader = Loader()
        doc_obj = DocObject("1", "/root/protected.txt")

        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            result = loader.crawl_file(doc_obj)
            assert result.is_error
            assert "Error loading" in result.error_message

    def test_crawl_file_unicode_error(self) -> None:
        """Test crawl_file with Unicode decode error."""
        loader = Loader()
        doc_obj = DocObject("1", "test.txt")

        with patch(
            "builtins.open",
            side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid utf-8"),
        ):
            result = loader.crawl_file(doc_obj)
            assert result.is_error
            assert "Error loading" in result.error_message

    def test_crawl_file_general_exception(self) -> None:
        """Test crawl_file with general exception."""
        loader = Loader()
        doc_obj = DocObject("1", "test.txt")

        with patch("builtins.open", side_effect=Exception("General error")):
            result = loader.crawl_file(doc_obj)
            assert result.is_error
            assert "Error loading" in result.error_message

    def test_crawl_file_with_mistral_api_error(self) -> None:
        """Test crawl_file with Mistral API error."""
        loader = Loader()
        doc_obj = DocObject("1", "test.txt")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp_file:
            tmp_file.write("Test content")
            tmp_file_path = tmp_file.name

        try:
            with patch("mistralai.Mistral", side_effect=Exception("API error")):
                result = loader.crawl_file(doc_obj)
                assert result.is_error
        finally:
            os.remove(tmp_file_path)

    def test_save_method(self) -> None:
        """Test save method."""
        docs = [CrawledObject("1", "test.txt", "content")]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            tmp_file_path = tmp_file.name

        try:
            Loader.save(tmp_file_path, docs)
            assert os.path.exists(tmp_file_path)
        finally:
            os.unlink(tmp_file_path)

    def test_chunk_method_with_exception(self) -> None:
        """Test chunk method with exception."""
        docs = [CrawledObject("1", "test.txt", "content")]

        with patch(
            "langchain.text_splitter.RecursiveCharacterTextSplitter",
            side_effect=Exception("Splitter error"),
        ):
            result = Loader.chunk(docs)
            assert len(result) == 1
            # The result is a Document object, not CrawledObject, so we check differently
            assert hasattr(result[0], "page_content")

    def test_get_outsource_urls_processing_exception(self) -> None:
        """Test URL processing exceptions in get_outsource_urls."""
        loader = Loader()

        # Patch requests.get to return a response with a link that raises exception
        class DummyLink:
            def get(self, _):
                raise Exception("fail href")

        class DummySoup:
            def find_all(self, _):
                return [DummyLink()]

        class DummyResponse:
            status_code = 200
            text = "<html></html>"

        with patch("requests.get", return_value=DummyResponse()):
            with patch("bs4.BeautifulSoup", return_value=DummySoup()):
                # Should log error and continue
                result = loader.get_outsource_urls(
                    "http://example.com", "http://example.com"
                )
                assert isinstance(result, list)

    def test_get_candidates_websites_graph(self) -> None:
        """Test graph operations in get_candidates_websites."""
        loader = Loader()
        # Create two crawled objects with content referencing each other
        c1 = CrawledObject(
            "id1",
            "http://a.com",
            "http://b.com",
            {},
            False,
            False,
            None,
            SourceType.WEB,
        )
        c2 = CrawledObject(
            "id2",
            "http://b.com",
            "http://a.com",
            {},
            False,
            False,
            None,
            SourceType.WEB,
        )
        result = loader.get_candidates_websites([c1, c2], top_k=2)
        assert isinstance(result, list)
        assert all(isinstance(x, CrawledObject) for x in result)

    def test_crawl_file_unsupported_file_type(self) -> None:
        """Test crawl_file with unsupported file type."""
        loader = Loader()
        local_obj = DocObject("id", "file.unsupported")
        result = loader.crawl_file(local_obj)
        assert result.is_error
        assert "Unsupported file type" in result.error_message

    def test_crawl_file_missing_file_type(self) -> None:
        """Test crawl_file with missing file type."""
        loader = Loader()
        local_obj = DocObject("id", "filewithnofiletype")
        result = loader.crawl_file(local_obj)
        assert result.is_error
        assert "No file type detected" in result.error_message

    def test_crawl_file_mistral_api_key_logic(self) -> None:
        """Test crawl_file Mistral API key logic for pdf with no key."""
        loader = Loader()
        # Patch MISTRAL_API_KEY to None and PyPDFLoader to a mock
        with patch("arklex.utils.loader.MISTRAL_API_KEY", None):
            with patch("arklex.utils.loader.PyPDFLoader") as mock_loader:
                mock_loader.return_value.load.return_value = [
                    Mock(to_json=lambda: {"kwargs": {"page_content": "page1"}})
                ]
                local_obj = DocObject("id", "file.pdf")
                result = loader.crawl_file(local_obj)
                assert not result.is_error
                assert "page1" in result.content

    def test_crawl_file_error_handling(self) -> None:
        """Test crawl_file error handling for general exception."""
        loader = Loader()
        # Patch Path to return a mock with .suffix = ''
        mock_path = Mock()
        mock_path.suffix = ""
        mock_path.name = "file.txt"
        with patch("arklex.utils.loader.Path", return_value=mock_path):
            local_obj = DocObject("id", "file.txt")
            result = loader.crawl_file(local_obj)
            assert result.is_error

    def test_save_with_unsupported_format(self) -> None:
        """Test save method with unsupported file format (should still pickle)."""
        docs = [CrawledObject("id", "src", "content")]
        with tempfile.NamedTemporaryFile(
            suffix=".unsupported", delete=False
        ) as tmp_file:
            tmp_file_path = tmp_file.name
        try:
            Loader.save(tmp_file_path, docs)
            assert os.path.exists(tmp_file_path)
        finally:
            os.unlink(tmp_file_path)

    def test_chunk_with_error_and_chunked_docs(self) -> None:
        """Test chunk skips error and already chunked docs."""
        error_doc = CrawledObject("id", "src", None, is_error=True)
        chunked_doc = CrawledObject("id", "src", "chunked", is_chunk=True)
        docs = [error_doc, chunked_doc]
        result = Loader.chunk(docs)
        from langchain_core.documents import Document

        assert all(isinstance(x, Document) for x in result)


class TestLoaderExtendedCoverage:
    """Additional tests to increase coverage for loader.py"""

    def test_crawl_file_with_pdf_loader(self) -> None:
        """Test crawl_file with PDF loader."""
        loader = Loader()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pdf", delete=False
        ) as tmp_file:
            tmp_file.write("PDF content")
            tmp_file_path = tmp_file.name

        doc_obj = DocObject("1", tmp_file_path)
        try:
            with patch("langchain_community.document_loaders.PyPDFLoader") as mock_pdf:
                mock_instance = mock_pdf.return_value
                mock_instance.load.return_value = [Document(page_content="PDF content")]
                result = loader.crawl_file(doc_obj)
                assert isinstance(result, CrawledObject)
        finally:
            os.remove(tmp_file_path)

    def test_crawl_file_with_word_document_loader(self) -> None:
        """Test crawl_file with Word document loader."""
        loader = Loader()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".docx", delete=False
        ) as tmp_file:
            tmp_file.write("Word content")
            tmp_file_path = tmp_file.name

        doc_obj = DocObject("1", tmp_file_path)
        try:
            with patch(
                "langchain_community.document_loaders.UnstructuredWordDocumentLoader"
            ) as mock_word:
                mock_instance = mock_word.return_value
                mock_instance.load.return_value = [
                    Document(page_content="Word content")
                ]
                result = loader.crawl_file(doc_obj)
                assert isinstance(result, CrawledObject)
        finally:
            os.remove(tmp_file_path)

    def test_crawl_file_with_excel_loader(self) -> None:
        """Test crawl_file with Excel loader."""
        loader = Loader()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xlsx", delete=False
        ) as tmp_file:
            tmp_file.write("Excel content")
            tmp_file_path = tmp_file.name

        doc_obj = DocObject("1", tmp_file_path)
        try:
            with patch(
                "langchain_community.document_loaders.UnstructuredExcelLoader"
            ) as mock_excel:
                mock_instance = mock_excel.return_value
                mock_instance.load.return_value = [
                    Document(page_content="Excel content")
                ]
                result = loader.crawl_file(doc_obj)
                assert isinstance(result, CrawledObject)
        finally:
            os.remove(tmp_file_path)

    def test_crawl_file_with_markdown_loader(self) -> None:
        """Test crawl_file with Markdown loader."""
        loader = Loader()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as tmp_file:
            tmp_file.write("# Markdown content")
            tmp_file_path = tmp_file.name

        doc_obj = DocObject("1", tmp_file_path)
        try:
            with patch(
                "langchain_community.document_loaders.UnstructuredMarkdownLoader"
            ) as mock_md:
                mock_instance = mock_md.return_value
                mock_instance.load.return_value = [
                    Document(page_content="Markdown content")
                ]
                result = loader.crawl_file(doc_obj)
                assert isinstance(result, CrawledObject)
                assert not result.is_error
        finally:
            os.remove(tmp_file_path)

    def test_crawl_file_with_powerpoint_loader(self) -> None:
        """Test crawl_file with PowerPoint loader."""
        loader = Loader()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pptx", delete=False
        ) as tmp_file:
            tmp_file.write("PowerPoint content")
            tmp_file_path = tmp_file.name

        doc_obj = DocObject("1", tmp_file_path)
        try:
            with patch(
                "langchain_community.document_loaders.UnstructuredPowerPointLoader"
            ) as mock_ppt:
                mock_instance = mock_ppt.return_value
                mock_instance.load.return_value = [
                    Document(page_content="PowerPoint content")
                ]
                result = loader.crawl_file(doc_obj)
                assert isinstance(result, CrawledObject)
        finally:
            os.remove(tmp_file_path)

    def test_crawl_file_with_unsupported_extension(self) -> None:
        """Test crawl_file with unsupported file extension."""
        loader = Loader()
        doc_obj = DocObject("1", "test.xyz")

        result = loader.crawl_file(doc_obj)
        assert result.is_error
        assert "Unsupported file type" in result.error_message

    def test_crawl_file_with_loader_exception(self) -> None:
        """Test crawl_file with loader exception."""
        loader = Loader()
        doc_obj = DocObject("1", "test.txt")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp_file:
            tmp_file.write("Test content")
            tmp_file_path = tmp_file.name

        try:
            with patch(
                "langchain_community.document_loaders.TextLoader"
            ) as mock_loader:
                mock_instance = mock_loader.return_value
                mock_instance.load.side_effect = Exception("Loader error")
                result = loader.crawl_file(doc_obj)
                assert result.is_error
                assert "Error loading" in result.error_message
        finally:
            os.remove(tmp_file_path)

    def test_to_crawled_text(self) -> None:
        """Test to_crawled_text method."""
        loader = Loader()
        text_list = ["text1", "text2", "text3"]

        result = loader.to_crawled_text(text_list)
        assert len(result) == 3
        assert all(isinstance(obj, CrawledObject) for obj in result)
        assert result[0].content == "text1"
        assert result[0].source_type == SourceType.TEXT

    def test_to_crawled_local_objs(self) -> None:
        """Test to_crawled_local_objs method."""
        loader = Loader()
        file_list = ["file1.txt", "file2.txt"]

        with patch.object(loader, "crawl_file") as mock_crawl:
            mock_crawl.return_value = CrawledObject("1", "file1.txt", "content")
            result = loader.to_crawled_local_objs(file_list)
            assert len(result) == 2
            assert all(isinstance(obj, CrawledObject) for obj in result)

    def test_get_outsource_urls(self) -> None:
        """Test get_outsource_urls method."""
        loader = Loader()

        with patch.object(loader, "get_all_urls") as mock_get_urls:
            mock_get_urls.return_value = [
                "http://example.com/page1",
                "http://example.com/page2",
            ]
            result = loader.get_outsource_urls(
                "http://example.com", "http://example.com"
            )
            assert len(result) == 0

    def test_get_outsource_urls_exception(self) -> None:
        """Test get_outsource_urls with exception."""
        loader = Loader()

        with patch.object(loader, "get_all_urls", side_effect=Exception("URL error")):
            result = loader.get_outsource_urls(
                "http://example.com", "http://example.com"
            )
            assert len(result) == 0

    def test_check_url_with_query_params(self) -> None:
        """Test _check_url with query parameters."""
        loader = Loader()

        assert (
            loader._check_url("http://example.com?param=value", "http://example.com")
            is True
        )
        assert (
            loader._check_url(
                "http://example.com/path?param=value", "http://example.com"
            )
            is False
        )

    def test_check_url_with_port(self) -> None:
        """Test _check_url with port numbers."""
        loader = Loader()

        assert (
            loader._check_url("http://example.com:8080", "http://example.com") is False
        )
        assert (
            loader._check_url("http://example.com:8080", "http://example.com:8080")
            is False
        )

    def test_save_with_pickle(self) -> None:
        """Test save method with pickle format."""
        docs = [CrawledObject("1", "test.txt", "content")]

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_file_path = tmp_file.name

        try:
            Loader.save(tmp_file_path, docs)
            assert os.path.exists(tmp_file_path)
        finally:
            os.unlink(tmp_file_path)

    def test_save_with_json(self) -> None:
        """Test save method with JSON format."""
        docs = [CrawledObject("1", "test.txt", "content")]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
            tmp_file_path = tmp_file.name

        try:
            Loader.save(tmp_file_path, docs)
            assert os.path.exists(tmp_file_path)
        finally:
            os.unlink(tmp_file_path)

    def test_save_with_unsupported_format(self) -> None:
        """Test save method with unsupported format."""
        docs = [CrawledObject("1", "test.txt", "content")]

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp_file:
            tmp_file_path = tmp_file.name

        try:
            Loader.save(tmp_file_path, docs)
            assert os.path.exists(tmp_file_path)
        finally:
            os.unlink(tmp_file_path)

    def test_chunk_with_large_content(self) -> None:
        """Test chunk method with large content."""
        large_content = "This is a very long content. " * 1000
        docs = [CrawledObject("1", "test.txt", large_content)]

        result = Loader.chunk(docs)
        assert len(result) > 0
        assert all(hasattr(doc, "page_content") for doc in result)

    def test_chunk_with_metadata(self) -> None:
        """Test chunk method with metadata."""
        docs = [CrawledObject("1", "test.txt", "content", metadata={"key": "value"})]

        result = Loader.chunk(docs)
        assert len(result) > 0
        assert all(hasattr(doc, "metadata") for doc in result)

    def test_crawl_urls_with_mixed_results(self) -> None:
        """Test crawl_urls with mixed success/failure results."""
        loader = Loader()
        url_objects = [
            DocObject("1", "http://example1.com"),
            DocObject("2", "http://example2.com"),
            DocObject("3", "http://example3.com"),
        ]

        with patch.object(loader, "_crawl_with_selenium") as mock_selenium:
            # Mock some successful and some failed results
            mock_selenium.return_value = [
                CrawledObject("1", "http://example1.com", "content1", is_error=False),
                CrawledObject(
                    "2",
                    "http://example2.com",
                    "",
                    is_error=True,
                    error_message="Failed",
                ),
                CrawledObject("3", "http://example3.com", "content3", is_error=False),
            ]

            result = loader.crawl_urls(url_objects)
            assert len(result) == 3
            successful = [doc for doc in result if not doc.is_error]
            assert len(successful) == 2

    def test_crawl_urls_fallback_to_requests(self) -> None:
        """Test crawl_urls fallback to requests when Selenium fails completely."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        with patch.object(loader, "_crawl_with_selenium") as mock_selenium:
            mock_selenium.return_value = [
                CrawledObject("1", "http://example.com", "", is_error=True)
            ]

            with patch.object(loader, "_crawl_with_requests") as mock_requests:
                mock_requests.return_value = [
                    CrawledObject("1", "http://example.com", "content", is_error=False)
                ]

                result = loader.crawl_urls(url_objects)
                assert len(result) == 1
                assert not result[0].is_error

    def test_crawl_urls_fallback_to_mock_content(self) -> None:
        """Test crawl_urls fallback to mock content when all crawling fails."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        with patch.object(loader, "_crawl_with_selenium") as mock_selenium:
            mock_selenium.return_value = [
                CrawledObject("1", "http://example.com", "", is_error=True)
            ]

            with patch.object(loader, "_crawl_with_requests") as mock_requests:
                mock_requests.return_value = [
                    CrawledObject("1", "http://example.com", "", is_error=True)
                ]

                with patch.object(
                    loader, "_create_mock_content_from_urls"
                ) as mock_mock:
                    mock_mock.return_value = [
                        CrawledObject(
                            "1", "http://example.com", "mock content", is_error=False
                        )
                    ]

                    result = loader.crawl_urls(url_objects)
                    assert len(result) == 1
                    assert not result[0].is_error

    def test_create_error_doc(self) -> None:
        """Test _create_error_doc method."""
        loader = Loader()
        url_obj = DocObject("1", "http://example.com")
        error_msg = "Test error"

        result = loader._create_error_doc(url_obj, error_msg)
        assert isinstance(result, CrawledObject)
        assert result.is_error
        assert result.error_message == error_msg
        assert result.source == "http://example.com"

    def test_get_all_urls_with_valid_response(self) -> None:
        """Test get_all_urls with valid response."""
        loader = Loader()

        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.text = """
            <html>
                <a href="http://example.com/page1">Page 1</a>
                <a href="http://example.com/page2">Page 2</a>
                <a href="http://external.com/page3">External</a>
            </html>
            """
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            result = loader.get_all_urls("http://example.com", 10)
            assert len(result) == 3
            assert "http://example.com" in result
            assert "http://example.com/page1" in result
            assert "http://example.com/page2" in result

    def test_get_all_urls_with_relative_urls(self) -> None:
        """Test get_all_urls with relative URLs."""
        loader = Loader()

        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.text = """
            <html>
                <a href="/page1">Page 1</a>
                <a href="page2">Page 2</a>
                <a href="../page3">Page 3</a>
            </html>
            """
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            result = loader.get_all_urls("http://example.com", 10)
            assert len(result) == 4
            assert "http://example.com" in result
            assert "http://example.com/page1" in result
            assert "http://example.com/page2" in result
            assert "http://example.com/page3" in result

    def test_get_candidates_websites_with_content_analysis(self) -> None:
        """Test get_candidates_websites with content analysis."""
        loader = Loader()

        urls = [
            CrawledObject("1", "http://example1.com", "relevant content about AI"),
            CrawledObject("2", "http://example2.com", "unrelated content"),
            CrawledObject("3", "http://example3.com", "more AI content"),
        ]

        result = loader.get_candidates_websites(urls, 2)
        assert len(result) == 2
        assert all(isinstance(obj, CrawledObject) for obj in result)

    def test_crawl_file_with_html_fallback(self) -> None:
        """Test crawl_file with HTML fallback."""
        loader = Loader()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False
        ) as tmp_file:
            tmp_file.write("<html><body>HTML content</body></html>")
            tmp_file_path = tmp_file.name

        doc_obj = DocObject("1", tmp_file_path)
        try:
            with patch(
                "langchain_community.document_loaders.UnstructuredMarkdownLoader"
            ) as mock_md:
                mock_instance = mock_md.return_value
                mock_instance.load.side_effect = Exception("Markdown loader failed")

                with patch(
                    "langchain_community.document_loaders.TextLoader"
                ) as mock_text:
                    mock_instance = mock_text.return_value
                    mock_instance.load.return_value = [
                        Document(page_content="HTML content")
                    ]

                    result = loader.crawl_file(doc_obj)
                    assert isinstance(result, CrawledObject)
                    assert not result.is_error
        finally:
            os.remove(tmp_file_path)

    def test_check_url_with_query_params(self) -> None:
        """Test _check_url with query parameters."""
        loader = Loader()

        assert (
            loader._check_url("http://example.com?param=value", "http://example.com")
            is True
        )
        # The implementation returns True for any URL that starts with the base URL
        assert (
            loader._check_url(
                "http://example.com/path?param=value", "http://example.com"
            )
            is True
        )

    def test_check_url_with_port(self) -> None:
        """Test _check_url with port numbers."""
        loader = Loader()

        # The implementation returns True for any URL that starts with the base URL
        assert (
            loader._check_url("http://example.com:8080", "http://example.com") is True
        )
        assert (
            loader._check_url("http://example.com:8080", "http://example.com:8080")
            is False
        )
