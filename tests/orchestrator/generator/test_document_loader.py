"""Tests for document loading components.

This module provides comprehensive tests for the document loading, processing,
and validation components of the Arklex framework.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from arklex.orchestrator.generator.docs.document_loader import DocumentLoader
from arklex.orchestrator.generator.docs.document_processor import DocumentProcessor
from arklex.orchestrator.generator.docs.document_validator import DocumentValidator

# --- Test Utilities ---


class NoReadTextPath(type(Path())):
    """Path subclass that raises AttributeError for read_text to test fallback logic."""

    def __getattribute__(self, name: str) -> object:
        if name == "read_text":
            raise AttributeError("no read_text")
        return super().__getattribute__(name)


# --- Centralized Mock Fixtures ---


@pytest.fixture
def sample_document() -> dict:
    """Sample document for testing."""
    return {
        "title": "Product Creation Guide",
        "sections": [
            {
                "name": "Product Details",
                "content": "Required product information",
                "requirements": ["Name", "Description", "Price"],
            },
            {
                "name": "Pricing Guidelines",
                "content": "Pricing strategy guidelines",
                "requirements": ["Market research", "Profit margin"],
            },
        ],
        "metadata": {"version": "1.0", "last_updated": "2024-03-20"},
    }


@pytest.fixture
def sample_task_doc() -> dict:
    """Sample task document for testing."""
    return {
        "task_id": "task1",
        "name": "Create Product",
        "description": "Create a new product in the store",
        "steps": [
            {
                "step_id": "step1",
                "description": "Enter product details",
                "required_fields": ["name", "description", "price"],
            },
            {
                "step_id": "step2",
                "description": "Set pricing strategy",
                "required_fields": ["market_price", "profit_margin"],
            },
        ],
    }


@pytest.fixture
def sample_instruction_doc() -> dict:
    """Sample instruction document for testing."""
    return {
        "instruction_id": "inst1",
        "title": "Product Creation Instructions",
        "content": "Step-by-step guide for creating products",
        "sections": [
            {
                "section_id": "sec1",
                "title": "Product Information",
                "steps": ["Enter product name", "Add description", "Set price"],
            },
            {
                "section_id": "sec2",
                "title": "Pricing Strategy",
                "steps": ["Research market prices", "Calculate profit margin"],
            },
        ],
    }


@pytest.fixture
def mock_file_system(sample_document: dict) -> dict:
    """Mock file system for testing."""
    with (
        patch("pathlib.Path.exists") as mock_exists,
        patch("pathlib.Path.read_text") as mock_read_text,
        patch("json.loads") as mock_json_loads,
    ):
        mock_exists.return_value = True
        mock_read_text.return_value = json.dumps(sample_document)
        mock_json_loads.return_value = sample_document
        yield {
            "exists": mock_exists,
            "read_text": mock_read_text,
            "json_loads": mock_json_loads,
        }


@pytest.fixture
def mock_file_not_found() -> None:
    """Mock file system for file not found scenarios."""
    with patch("pathlib.Path.exists", return_value=False):
        yield


@pytest.fixture
def mock_invalid_json() -> None:
    """Mock file system for invalid JSON scenarios."""
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.read_text", return_value="not json"),
        patch("json.loads", side_effect=json.JSONDecodeError("msg", "doc", 0)),
    ):
        yield


@pytest.fixture
def mock_html_content() -> str:
    """Mock file system for HTML content scenarios."""
    html_content = "<html><head><title>Test</title></head><body><p>Step 1</p><p>Step 2</p></body></html>"
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.read_text", return_value=html_content),
        patch("json.loads", side_effect=json.JSONDecodeError("msg", "doc", 0)),
    ):
        yield html_content


@pytest.fixture
def mock_html_content_no_title() -> str:
    """Mock file system for HTML content without title scenarios."""
    html_content = "<html><body><p>Step 1</p><p>Step 2</p></body></html>"
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.read_text", return_value=html_content),
        patch("json.loads", side_effect=json.JSONDecodeError("msg", "doc", 0)),
    ):
        yield html_content


@pytest.fixture
def mock_requests_get(sample_task_doc: dict) -> MagicMock:
    """Mock requests.get for URL handling scenarios."""
    mock_response = MagicMock()
    mock_response.text = json.dumps(sample_task_doc)
    mock_response.raise_for_status.return_value = None
    with patch("requests.get", return_value=mock_response):
        yield mock_response


@pytest.fixture
def mock_beautiful_soup_error() -> None:
    """Mock BeautifulSoup for parsing error scenarios."""
    with patch(
        "arklex.orchestrator.generator.docs.document_loader.BeautifulSoup",
        side_effect=Exception("Parsing error"),
    ):
        yield


@pytest.fixture
def mock_validation_disabled() -> None:
    """Mock file system for validation disabled scenarios."""
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.read_text", return_value='{"invalid": "structure"}'),
    ):
        yield


# --- Component Fixtures ---


@pytest.fixture
def document_loader(mock_file_system: dict) -> DocumentLoader:
    """DocumentLoader instance for testing."""
    return DocumentLoader(cache_dir=Path("/tmp/cache"), validate_documents=True)


@pytest.fixture
def document_loader_no_validation() -> DocumentLoader:
    """DocumentLoader instance with validation disabled for testing."""
    return DocumentLoader(cache_dir=Path("/tmp/cache"), validate_documents=False)


@pytest.fixture
def document_processor() -> DocumentProcessor:
    """DocumentProcessor instance for testing."""
    return DocumentProcessor()


@pytest.fixture
def document_validator() -> DocumentValidator:
    """DocumentValidator instance for testing."""
    return DocumentValidator()


# --- Test Classes ---


class TestDocumentLoader:
    """Test suite for the DocumentLoader class."""

    def test_load_document(
        self, document_loader: DocumentLoader, mock_file_system: dict
    ) -> None:
        """Test loading a document from file."""
        doc_path = Path("/path/to/document.json")
        document = document_loader.load_document(doc_path)
        assert isinstance(document, dict)
        assert document["title"] == "Product Creation Guide"
        assert len(document["sections"]) == 2

    def test_load_task_document(
        self,
        document_loader: DocumentLoader,
        mock_file_system: dict,
        sample_task_doc: dict,
    ) -> None:
        """Test loading a task document."""
        doc_path = Path("/path/to/task.json")
        mock_file_system["json_loads"].return_value = sample_task_doc
        task_doc = document_loader.load_task_document(doc_path)
        assert isinstance(task_doc, dict)
        assert task_doc["task_id"] == "task1"
        assert len(task_doc["steps"]) == 2

    def test_load_instruction_document(
        self,
        document_loader: DocumentLoader,
        mock_file_system: dict,
        sample_instruction_doc: dict,
    ) -> None:
        """Test loading an instruction document."""
        doc_path = Path("/path/to/instruction.json")
        mock_file_system["json_loads"].return_value = sample_instruction_doc
        instruction_doc = document_loader.load_instruction_document(doc_path)
        assert isinstance(instruction_doc, dict)
        assert instruction_doc["instruction_id"] == "inst1"
        assert len(instruction_doc["sections"]) == 2

    def test_cache_document(
        self,
        document_loader: DocumentLoader,
        mock_file_system: dict,
        sample_document: dict,
    ) -> None:
        """Test document caching."""
        doc_path = Path("/path/to/document.json")
        document_loader.cache_document(doc_path, sample_document)
        cached_doc = document_loader.load_document(doc_path)
        assert cached_doc == sample_document

    def test_validate_document(
        self, document_loader: DocumentLoader, sample_document: dict
    ) -> None:
        """Test document validation."""
        is_valid = document_loader.validate_document(sample_document)
        assert is_valid

    def test_load_document_file_not_found(
        self, document_loader: DocumentLoader, mock_file_not_found: None
    ) -> None:
        """Test loading document when file is not found."""
        doc_path = Path("/not/found.json")
        with pytest.raises(FileNotFoundError):
            document_loader.load_document(doc_path)

    def test_load_document_invalid_json(
        self, document_loader: DocumentLoader, mock_invalid_json: None
    ) -> None:
        """Test loading document with invalid JSON."""
        doc_path = Path("/invalid.json")
        with pytest.raises(json.JSONDecodeError):
            document_loader.load_document(doc_path)

    def test_load_task_document_invalid_structure(
        self, document_loader: DocumentLoader, mock_file_system: dict
    ) -> None:
        """Test loading task document with invalid structure."""
        doc_path = Path("/invalid_task.json")
        mock_file_system["json_loads"].return_value = {"bad": "structure"}
        with pytest.raises(ValueError):
            document_loader.load_task_document(doc_path)

    def test_load_instruction_document_invalid_structure(
        self, document_loader: DocumentLoader, mock_file_system: dict
    ) -> None:
        """Test loading instruction document with invalid structure."""
        doc_path = Path("/invalid_instruction.json")
        mock_file_system["json_loads"].return_value = {"bad": "structure"}
        with pytest.raises(ValueError):
            document_loader.load_instruction_document(doc_path)

    def test_load_task_document_html_fallback(
        self, document_loader: DocumentLoader, mock_html_content: str
    ) -> None:
        """Test HTML fallback for task document loading."""
        doc_path = Path("/html_doc.html")
        doc = document_loader.load_task_document(doc_path)
        assert doc["name"] == "Test"
        assert len(doc["steps"]) == 2

    def test_load_task_document_url_handling(
        self,
        document_loader: DocumentLoader,
        mock_requests_get: MagicMock,
        sample_task_doc: dict,
    ) -> None:
        """Test loading task document from URL."""
        url = "http://example.com/task.json"
        with patch.object(
            document_loader, "_validate_task_document", return_value=True
        ):
            document_loader._cache[url] = sample_task_doc
            doc = document_loader.load_task_document(url)
            assert doc["task_id"] == "task1"
            assert len(doc["steps"]) == 2

    def test_load_task_document_html_fallback_without_title(
        self, document_loader: DocumentLoader, mock_html_content_no_title: str
    ) -> None:
        """Test HTML fallback when document has no title."""
        doc_path = Path("/html_doc.html")
        doc = document_loader.load_task_document(doc_path)
        assert doc["name"] == "HTML Document"
        assert len(doc["steps"]) == 2

    def test_load_task_document_html_fallback_parsing_error(
        self, document_loader: DocumentLoader, mock_beautiful_soup_error: None
    ) -> None:
        """Test HTML fallback when parsing fails."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html") as tmpfile:
            tmpfile.write("not html content")
            tmpfile.flush()
            tmp_path = NoReadTextPath(tmpfile.name)
        with (
            patch("json.loads", side_effect=json.JSONDecodeError("msg", "doc", 0)),
            patch.object(document_loader, "_validate_task_document", return_value=True),
            pytest.raises(ValueError, match="neither valid JSON nor parseable HTML"),
        ):
            document_loader.load_task_document(tmp_path)
        Path(tmpfile.name).unlink()

    def test_load_task_document_without_read_text_method(
        self, sample_task_doc: dict
    ) -> None:
        """Test loading task document when Path doesn't have read_text method."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmpfile:
            tmpfile.write(json.dumps(sample_task_doc))
            tmpfile.flush()
            tmpfile.close()
            tmp_path = NoReadTextPath(tmpfile.name)
        loader = DocumentLoader(cache_dir=Path("/tmp/cache"), validate_documents=True)
        with patch.object(loader, "_validate_task_document", return_value=True):
            doc = loader.load_task_document(tmp_path)
            assert doc["task_id"] == sample_task_doc["task_id"]
        Path(tmpfile.name).unlink()

    def test_load_instruction_document_without_read_text_method(
        self, sample_instruction_doc: dict
    ) -> None:
        """Test loading instruction document when Path doesn't have read_text method."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmpfile:
            tmpfile.write(json.dumps(sample_instruction_doc))
            tmpfile.flush()
            tmpfile.close()
            tmp_path = NoReadTextPath(tmpfile.name)
        loader = DocumentLoader(cache_dir=Path("/tmp/cache"), validate_documents=True)
        with patch.object(loader, "_validate_instruction_document", return_value=True):
            doc = loader.load_instruction_document(tmp_path)
            assert doc["instruction_id"] == sample_instruction_doc["instruction_id"]
        Path(tmpfile.name).unlink()

    def test_load_document_with_validation_disabled(
        self,
        document_loader_no_validation: DocumentLoader,
        mock_validation_disabled: None,
    ) -> None:
        """Test loading document with validation disabled."""
        doc_path = Path("/invalid_structure.json")
        doc = document_loader_no_validation.load_document(doc_path)
        assert doc == {"invalid": "structure"}

    @pytest.mark.parametrize(
        "invalid_doc,expected",
        [
            ("not a dict", False),
            ({"title": "Test"}, False),
            ({"title": "Test", "sections": "not a list"}, False),
            ({"title": "Test", "sections": ["not a dict"]}, False),
            ({"title": "Test", "sections": [{"name": "Test"}]}, False),
            (
                {
                    "title": "Test",
                    "sections": [
                        {
                            "name": "Test",
                            "content": "Test",
                            "requirements": "not a list",
                        }
                    ],
                },
                False,
            ),
        ],
    )
    def test_validate_document_invalid_cases(
        self, document_loader: DocumentLoader, invalid_doc: object, expected: object
    ) -> None:
        """Test document validation with various invalid cases."""
        is_valid = document_loader.validate_document(invalid_doc)
        assert is_valid == expected

    @pytest.mark.parametrize(
        "invalid_doc,expected",
        [
            ("not a dict", False),
            ({"task_id": "test"}, False),
            (
                {
                    "task_id": "test",
                    "name": "test",
                    "description": "test",
                    "steps": "not a list",
                },
                False,
            ),
            (
                {
                    "task_id": "test",
                    "name": "test",
                    "description": "test",
                    "steps": ["not a dict"],
                },
                False,
            ),
            (
                {
                    "task_id": "test",
                    "name": "test",
                    "description": "test",
                    "steps": [{"step_id": "test"}],
                },
                False,
            ),
            (
                {
                    "task_id": "test",
                    "name": "test",
                    "description": "test",
                    "steps": [
                        {
                            "step_id": "test",
                            "description": "test",
                            "required_fields": "not a list",
                        }
                    ],
                },
                False,
            ),
        ],
    )
    def test_validate_task_document_invalid_cases(
        self, document_loader: DocumentLoader, invalid_doc: object, expected: object
    ) -> None:
        """Test task document validation with various invalid cases."""
        is_valid = document_loader._validate_task_document(invalid_doc)
        assert is_valid == expected

    @pytest.mark.parametrize(
        "invalid_doc,expected",
        [
            ("not a dict", False),
            ({"instruction_id": "test"}, False),
            (
                {
                    "instruction_id": "test",
                    "title": "test",
                    "content": "test",
                    "sections": "not a list",
                },
                False,
            ),
            (
                {
                    "instruction_id": "test",
                    "title": "test",
                    "content": "test",
                    "sections": ["not a dict"],
                },
                False,
            ),
            (
                {
                    "instruction_id": "test",
                    "title": "test",
                    "content": "test",
                    "sections": [{"section_id": "test"}],
                },
                False,
            ),
            (
                {
                    "instruction_id": "test",
                    "title": "test",
                    "content": "test",
                    "sections": [
                        {"section_id": "test", "title": "test", "steps": "not a list"}
                    ],
                },
                False,
            ),
        ],
    )
    def test_validate_instruction_document_invalid_cases(
        self, document_loader: DocumentLoader, invalid_doc: object, expected: object
    ) -> None:
        """Test instruction document validation with various invalid cases."""
        is_valid = document_loader._validate_instruction_document(invalid_doc)
        assert is_valid == expected

    def test_load_document_without_read_text_method_fallback(
        self, sample_document: dict
    ) -> None:
        """Test load_document when Path doesn't have read_text method."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_document, f)
            temp_path = f.name

        try:
            # Create a Path-like object without read_text method
            class NoReadTextPath:
                def __init__(self, path: str) -> None:
                    self.path = path

                def exists(self) -> bool:
                    return True

                def __str__(self) -> str:
                    return self.path

                def __fspath__(self) -> str:
                    return self.path

            doc_path = NoReadTextPath(temp_path)
            loader = DocumentLoader(
                cache_dir=Path("/tmp/cache"), validate_documents=False
            )

            # This should use the fallback open() method
            result = loader.load_document(doc_path)
            assert result == sample_document
        finally:
            import os

            os.unlink(temp_path)

    def test_load_document_validation_failure(self) -> None:
        """Test load_document when validation fails."""
        from arklex.orchestrator.generator.docs.document_loader import DocumentLoader

        # Create a document that's missing required fields
        invalid_document = {"sections": []}  # Missing "title" field

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_document, f)
            temp_path = f.name

        try:
            loader = DocumentLoader(
                cache_dir=Path("/tmp/cache"), validate_documents=True
            )
            # The loader has validation enabled by default, so this should fail
            with pytest.raises(ValueError, match="Invalid document structure"):
                loader.load_document(Path(temp_path))
        finally:
            import os

            os.unlink(temp_path)

    def test_load_task_document_url_with_extension(self, sample_task_doc: dict) -> None:
        """Test load_task_document with URL that has an extension."""
        mock_response = MagicMock()
        mock_response.text = json.dumps(sample_task_doc)
        mock_response.raise_for_status.return_value = None

        with (
            patch("requests.get", return_value=mock_response),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value=json.dumps(sample_task_doc)),
            patch("builtins.open", create=True) as mock_open,
            patch("os.path.join", return_value="/tmp/cache/url_abc123.json"),
            patch("os.path.splitext", return_value=(".json", ".json")),
            patch("hashlib.md5") as mock_md5,
        ):
            mock_md5.return_value.hexdigest.return_value = "abc123"

            loader = DocumentLoader(cache_dir=Path("/tmp/cache"))
            result = loader.load_task_document("https://example.com/doc.json")

            assert result == sample_task_doc
            mock_open.assert_called()

    def test_load_task_document_url_without_extension(
        self, sample_task_doc: dict
    ) -> None:
        """Test load_task_document with URL that doesn't have an extension."""
        mock_response = MagicMock()
        mock_response.text = json.dumps(sample_task_doc)
        mock_response.raise_for_status.return_value = None

        with (
            patch("requests.get", return_value=mock_response),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value=json.dumps(sample_task_doc)),
            patch("builtins.open", create=True) as mock_open,
            patch("os.path.join", return_value="/tmp/cache/url_abc123.html"),
            patch("os.path.splitext", return_value=("", "")),
            patch("hashlib.md5") as mock_md5,
        ):
            mock_md5.return_value.hexdigest.return_value = "abc123"

            loader = DocumentLoader(cache_dir=Path("/tmp/cache"))
            result = loader.load_task_document("https://example.com/doc")

            assert result == sample_task_doc
            mock_open.assert_called()

    def test_load_task_document_string_path_conversion(
        self, sample_task_doc: dict
    ) -> None:
        """Test load_task_document with string path that gets converted to Path."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value=json.dumps(sample_task_doc)),
        ):
            loader = DocumentLoader(cache_dir=Path("/tmp/cache"))
            result = loader.load_task_document("/path/to/document.json")

            assert result == sample_task_doc

    def test_load_task_document_without_read_text_method_fallback(
        self, sample_task_doc: dict
    ) -> None:
        """Test load_task_document when Path doesn't have read_text method."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_task_doc, f)
            temp_path = f.name

        try:
            # Create a Path-like object without read_text method
            class NoReadTextPath:
                def __init__(self, path: str) -> None:
                    self.path = path

                def exists(self) -> bool:
                    return True

                def __str__(self) -> str:
                    return self.path

                def __fspath__(self) -> str:
                    return self.path

            doc_path = NoReadTextPath(temp_path)
            loader = DocumentLoader(cache_dir=Path("/tmp/cache"))

            # This should use the fallback open() method
            result = loader.load_task_document(doc_path)
            assert result == sample_task_doc
        finally:
            import os

            os.unlink(temp_path)

    def test_load_instruction_document_cached(
        self, document_loader: DocumentLoader, sample_instruction_doc: dict
    ) -> None:
        """Test load_instruction_document when document is already cached."""
        # First, cache the document
        doc_path = Path("/tmp/test_instruction.json")
        document_loader.cache_document(doc_path, sample_instruction_doc)

        # Now load it - should return from cache without file operations
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False  # File doesn't exist, but should use cache

            result = document_loader.load_instruction_document(doc_path)
            assert result == sample_instruction_doc
            mock_exists.assert_not_called()

    def test_load_instruction_document_file_not_found(
        self, document_loader: DocumentLoader
    ) -> None:
        """Test load_instruction_document when file doesn't exist."""
        doc_path = Path("/nonexistent/instruction.json")

        # Mock the file to not exist, but also mock the JSON loading to return invalid structure
        # so we get the FileNotFoundError instead of the validation error
        with (
            patch("pathlib.Path.exists", return_value=False),
            pytest.raises(FileNotFoundError, match="Document not found"),
        ):
            document_loader.load_instruction_document(doc_path)

    def test_load_instruction_document_without_read_text_method_fallback(
        self, sample_instruction_doc: dict
    ) -> None:
        """Test load_instruction_document when Path doesn't have read_text method."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_instruction_doc, f)
            temp_path = f.name

        try:
            # Create a Path-like object without read_text method
            class NoReadTextPath:
                def __init__(self, path: str) -> None:
                    self.path = path

                def exists(self) -> bool:
                    return True

                def __str__(self) -> str:
                    return self.path

                def __fspath__(self) -> str:
                    return self.path

            doc_path = NoReadTextPath(temp_path)
            loader = DocumentLoader(cache_dir=Path("/tmp/cache"))

            # This should use the fallback open() method
            result = loader.load_instruction_document(doc_path)
            assert result == sample_instruction_doc
        finally:
            import os

            os.unlink(temp_path)

    def test_load_task_document_html_fallback_no_title(self) -> None:
        """Test load_task_document with HTML content without title."""
        # Create HTML content without title
        html_content = """
        <html>
            <body>
                <p>This is step 1 description</p>
                <p>This is step 2 description</p>
            </body>
        </html>
        """

        with (
            patch("pathlib.Path.read_text", return_value=html_content),
            patch("pathlib.Path.exists", return_value=True),
        ):
            document_loader = DocumentLoader(cache_dir=Path("/tmp/cache"))
            doc_path = Path("/path/to/document.html")
            document = document_loader.load_task_document(doc_path)

            # Should create a document with default title
            assert document["task_id"] == "html_task"
            assert document["name"] == "HTML Document"  # Default name when no title
            assert document["description"] == "Document parsed from HTML"
            assert len(document["steps"]) == 2

    def test_load_task_document_html_fallback_exception_handling(self) -> None:
        """Test HTML fallback exception handling in load_task_document."""
        from pathlib import Path

        from arklex.orchestrator.generator.docs.document_loader import DocumentLoader

        # Create a document loader
        loader = DocumentLoader(cache_dir=Path("/tmp/cache"))

        # Create a mock HTML content that will cause BeautifulSoup to fail
        html_content = "<html><body><p>Invalid HTML with unclosed tags<p>"

        # Mock the file system to return HTML content
        with (
            patch("builtins.open", mock_open(read_data=html_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value=html_content),
            patch(
                "json.loads", side_effect=json.JSONDecodeError("Invalid JSON", "", 0)
            ),
            patch(
                "arklex.orchestrator.generator.docs.document_loader.BeautifulSoup"
            ) as mock_soup,
        ):
            # Make BeautifulSoup raise an exception
            mock_soup.side_effect = Exception("BeautifulSoup parsing error")

            # Test that the method raises a ValueError when HTML parsing fails
            with pytest.raises(
                ValueError,
                match="Content at .* is neither valid JSON nor parseable HTML",
            ):
                loader.load_task_document(Path("/path/to/document.html"))

    def test_load_task_document_html_fallback_with_paragraph_extraction(self) -> None:
        """Test HTML fallback with paragraph extraction in load_task_document."""
        from pathlib import Path

        from arklex.orchestrator.generator.docs.document_loader import DocumentLoader

        # Create a document loader
        loader = DocumentLoader(cache_dir=Path("/tmp/cache"))

        # Create HTML content with paragraphs
        html_content = """
        <html>
        <head><title>Test Document</title></head>
        <body>
            <p>First paragraph content</p>
            <p>Second paragraph content</p>
            <p>Third paragraph content</p>
        </body>
        </html>
        """

        # Mock the file system to return HTML content
        with (
            patch("builtins.open", mock_open(read_data=html_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value=html_content),
            patch(
                "json.loads", side_effect=json.JSONDecodeError("Invalid JSON", "", 0)
            ),
        ):
            # Test that the method successfully parses HTML and extracts paragraphs
            result = loader.load_task_document(Path("/path/to/document.html"))

            # Verify the result structure
            assert result["task_id"] == "html_task"
            assert result["name"] == "Test Document"
            assert result["description"] == "Document parsed from HTML"
            assert len(result["steps"]) == 3
            assert result["steps"][0]["step_id"] == 1
            assert result["steps"][0]["description"] == "First paragraph content"
            assert result["steps"][1]["step_id"] == 2
            assert result["steps"][1]["description"] == "Second paragraph content"
            assert result["steps"][2]["step_id"] == 3
            assert result["steps"][2]["description"] == "Third paragraph content"


class TestDocumentProcessor:
    """Test suite for the DocumentProcessor class."""

    def test_process_document(
        self, document_processor: DocumentProcessor, sample_document: dict
    ) -> None:
        """Test document processing."""
        processed_doc = document_processor.process_document(sample_document)
        assert isinstance(processed_doc, dict)
        assert "processed_sections" in processed_doc
        assert len(processed_doc["processed_sections"]) == 2

    def test_extract_requirements(
        self, document_processor: DocumentProcessor, sample_document: dict
    ) -> None:
        """Test requirement extraction."""
        requirements = document_processor.extract_requirements(sample_document)
        assert isinstance(requirements, list)
        assert len(requirements) > 0
        assert all(isinstance(req, str) for req in requirements)

    def test_format_for_tasks(
        self, document_processor: DocumentProcessor, sample_document: dict
    ) -> None:
        """Test document formatting for tasks."""
        formatted_doc = document_processor.format_for_tasks(sample_document)
        assert isinstance(formatted_doc, dict)
        assert "formatted_sections" in formatted_doc
        assert len(formatted_doc["formatted_sections"]) == 2

    def test_handle_specific_requirements(
        self, document_processor: DocumentProcessor
    ) -> None:
        """Test handling specific requirements."""
        requirements = ["Name", "Price", "Description"]
        handled_reqs = document_processor.handle_specific_requirements(requirements)
        assert isinstance(handled_reqs, list)
        assert len(handled_reqs) == len(requirements)

    def test_document_processor_process_section_method(self) -> None:
        """Test the _process_section method of DocumentProcessor."""
        from arklex.orchestrator.generator.docs.document_processor import (
            DocumentProcessor,
        )

        processor = DocumentProcessor()
        section = {
            "name": "Test Section",
            "content": "Test content with extra spaces   ",
            "requirements": ["req1", "req2"],
        }

        result = processor._process_section(section)

        assert result["name"] == "Test Section"
        assert result["content"] == "Test content with extra spaces   "
        assert result["requirements"] == ["req1", "req2"]
        assert result["processed_content"] == "Test content with extra spaces"

    def test_document_processor_process_content_method(self) -> None:
        """Test the _process_content method of DocumentProcessor."""
        from arklex.orchestrator.generator.docs.document_processor import (
            DocumentProcessor,
        )

        processor = DocumentProcessor()

        # Test with content that has extra whitespace
        content = "  Test content with spaces  \n\n"
        result = processor._process_content(content)
        assert result == "Test content with spaces"

        # Test with empty content
        result = processor._process_content("")
        assert result == ""

        # Test with content that has leading/trailing spaces
        result = processor._process_content("  content  ")
        assert result == "content"

    def test_document_processor_extract_steps_with_dict_steps(self) -> None:
        """Test _extract_steps method with dictionary steps."""
        from arklex.orchestrator.generator.docs.document_processor import (
            DocumentProcessor,
        )

        processor = DocumentProcessor()
        section = {
            "steps": [
                {"task": "Step 1", "description": "First step"},
                {"task": "Step 2", "description": "Second step"},
            ]
        }

        result = processor._extract_steps(section)
        assert len(result) == 2
        assert result[0]["task"] == "Step 1"
        assert result[1]["task"] == "Step 2"

    def test_document_processor_extract_steps_with_string_steps(self) -> None:
        """Test _extract_steps method with string steps."""
        from arklex.orchestrator.generator.docs.document_processor import (
            DocumentProcessor,
        )

        processor = DocumentProcessor()
        section = {"steps": ["Step 1", "Step 2", "Step 3"]}

        result = processor._extract_steps(section)
        assert len(result) == 3
        assert result[0]["task"] == "Step 1"
        assert result[1]["task"] == "Step 2"
        assert result[2]["task"] == "Step 3"

    def test_document_processor_extract_steps_with_mixed_steps(self) -> None:
        """Test _extract_steps method with mixed step types."""
        from arklex.orchestrator.generator.docs.document_processor import (
            DocumentProcessor,
        )

        processor = DocumentProcessor()
        section = {
            "steps": [
                {"task": "Step 1", "description": "First step"},
                "Step 2",
                {"task": "Step 3", "description": "Third step"},
            ]
        }

        result = processor._extract_steps(section)
        assert len(result) == 3
        assert result[0]["task"] == "Step 1"
        assert result[1]["task"] == "Step 2"
        assert result[2]["task"] == "Step 3"

    def test_document_processor_extract_steps_with_no_steps(self) -> None:
        """Test _extract_steps method with no steps."""
        from arklex.orchestrator.generator.docs.document_processor import (
            DocumentProcessor,
        )

        processor = DocumentProcessor()
        section = {}

        result = processor._extract_steps(section)
        assert result == []

    def test_document_processor_extract_steps_with_non_dict_non_string(self) -> None:
        """Test _extract_steps method with non-dict, non-string steps."""
        from arklex.orchestrator.generator.docs.document_processor import (
            DocumentProcessor,
        )

        processor = DocumentProcessor()
        section = {"steps": [123, True, None]}

        result = processor._extract_steps(section)
        assert len(result) == 3
        assert result[0]["task"] == "123"
        assert result[1]["task"] == "True"
        assert result[2]["task"] == "None"


class TestDocumentValidator:
    """Test suite for the DocumentValidator class."""

    def test_validate_structure(
        self, document_validator: DocumentValidator, sample_document: dict
    ) -> None:
        """Test document structure validation."""
        is_valid = document_validator.validate_structure(sample_document)
        assert is_valid

    def test_validate_required_fields(
        self, document_validator: DocumentValidator, sample_document: dict
    ) -> None:
        """Test required fields validation."""
        is_valid = document_validator.validate_required_fields(sample_document)
        assert is_valid

    def test_validate_consistency(
        self, document_validator: DocumentValidator, sample_document: dict
    ) -> None:
        """Test document consistency validation."""
        is_valid = document_validator.validate_consistency(sample_document)
        assert is_valid

    def test_get_error_messages(
        self, document_validator: DocumentValidator, sample_document: dict
    ) -> None:
        """Test error message generation."""
        errors = document_validator.get_error_messages(sample_document)
        assert isinstance(errors, list)
        assert len(errors) == 0  # No errors in valid document


# --- Integration Tests ---


@pytest.mark.integration
def test_integration_document_pipeline(
    mock_file_system: dict, sample_document: dict
) -> None:
    """Test the complete document handling pipeline integration."""
    # Initialize components
    document_loader = DocumentLoader(
        cache_dir=Path("/tmp/cache"), validate_documents=True
    )
    document_processor = DocumentProcessor()
    document_validator = DocumentValidator()

    # Load document
    doc_path = Path("/path/to/document.json")
    document = document_loader.load_document(doc_path)
    assert isinstance(document, dict)

    # Validate document
    is_valid = document_validator.validate_structure(document)
    assert is_valid

    # Process document
    processed_doc = document_processor.process_document(document)
    assert isinstance(processed_doc, dict)

    # Extract requirements
    requirements = document_processor.extract_requirements(processed_doc)
    assert isinstance(requirements, list)
    assert len(requirements) > 0

    # Format for tasks
    formatted_doc = document_processor.format_for_tasks(processed_doc)
    assert isinstance(formatted_doc, dict)
    assert "formatted_sections" in formatted_doc

    # Verify integration
    assert all("name" in section for section in formatted_doc["formatted_sections"])
    assert all("content" in section for section in formatted_doc["formatted_sections"])


def test_load_document_invalid_json_error(tmp_path: Path) -> None:
    import json

    import pytest

    # Create a file with invalid JSON
    file_path = tmp_path / "invalid.json"
    file_path.write_text("not a json")
    # Should raise JSONDecodeError
    # Create a new document loader with validation disabled to ensure JSONDecodeError is raised
    loader = DocumentLoader(cache_dir=tmp_path, validate_documents=False)

    with pytest.raises(json.JSONDecodeError) as exc_info:
        loader.load_document(file_path)

    # Verify it's a JSONDecodeError (even if re-raised with custom message)
    assert isinstance(exc_info.value, json.JSONDecodeError)
