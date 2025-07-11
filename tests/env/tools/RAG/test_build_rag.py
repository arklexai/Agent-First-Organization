import os
import pickle
import tempfile
import uuid
import zipfile
from unittest.mock import Mock, patch

import pytest
from _pytest.logging import LogCaptureFixture

from arklex.env.tools.RAG.build_rag import build_rag

# Set test environment
os.environ["ARKLEX_TEST_ENV"] = "local"


# Create a simple object that can be pickled and matches CrawledObject interface
class SimpleDoc:
    def __init__(self, content: str, source: str) -> None:
        self.id = str(uuid.uuid4())  # Generate a unique ID
        self.content = content
        self.source = source
        self.is_error = False
        self.is_chunk = False
        self.metadata = {"source": source}
        self.error_message = None
        self.source_type = 1  # SourceType.WEB = 1


class TestBuildRAG:
    """Test class for build_rag function."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_folder_path = os.path.join(self.temp_dir, "test_rag")

    def teardown_method(self) -> None:
        """Clean up test fixtures after each test method."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_build_rag_creates_folder_if_not_exists(self) -> None:
        """Test that build_rag creates the folder if it doesn't exist."""
        # Arrange
        folder_path = os.path.join(self.temp_dir, "new_folder")
        rag_docs = []

        # Act
        build_rag(folder_path, rag_docs)

        # Assert
        assert os.path.exists(folder_path)
        assert os.path.isdir(folder_path)

    def test_build_rag_loads_existing_documents(self) -> None:
        """Test that build_rag loads existing documents from pickle file."""
        # Arrange
        os.makedirs(self.test_folder_path, exist_ok=True)
        existing_docs = [SimpleDoc("test content", "test source")]
        filepath = os.path.join(self.test_folder_path, "documents.pkl")

        with open(filepath, "wb") as f:
            pickle.dump(existing_docs, f)

        rag_docs = []

        # Act
        build_rag(self.test_folder_path, rag_docs)

        # Assert
        assert os.path.exists(filepath)
        with open(filepath, "rb") as f:
            loaded_docs = pickle.load(f)
        assert len(loaded_docs) == 1
        assert loaded_docs[0].content == "test content"

    @patch("arklex.env.tools.RAG.build_rag.Loader")
    def test_build_rag_processes_url_documents(self, mock_loader_class: Mock) -> None:
        """Test that build_rag processes URL type documents correctly."""
        # Arrange
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        # Mock the loader methods
        mock_loader.get_all_urls.return_value = ["http://example.com"]
        mock_loader.to_crawled_url_objs.return_value = [
            Mock(content="url content", source="http://example.com")
        ]

        rag_docs = [{"source": "http://example.com", "type": "url", "num": 5}]

        # Act
        build_rag(self.test_folder_path, rag_docs)

        # Assert
        mock_loader.get_all_urls.assert_called_once_with("http://example.com", 5)
        mock_loader.to_crawled_url_objs.assert_called_once()

    @patch("arklex.env.tools.RAG.build_rag.Loader")
    def test_build_rag_processes_url_documents_with_default_num(
        self, mock_loader_class: Mock
    ) -> None:
        """Test that build_rag uses default num=1 when num is not provided for URL documents."""
        # Arrange
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        mock_loader.get_all_urls.return_value = ["http://example.com"]
        mock_loader.to_crawled_url_objs.return_value = [
            Mock(content="url content", source="http://example.com")
        ]

        rag_docs = [{"source": "http://example.com", "type": "url"}]

        # Act
        build_rag(self.test_folder_path, rag_docs)

        # Assert
        mock_loader.get_all_urls.assert_called_once_with("http://example.com", 1)

    @patch("arklex.env.tools.RAG.build_rag.Loader")
    def test_build_rag_processes_file_documents_single_file(
        self, mock_loader_class: Mock
    ) -> None:
        """Test that build_rag processes single file documents correctly."""
        # Arrange
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        # Create a temporary test file
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        mock_loader.to_crawled_local_objs.return_value = [
            Mock(content="file content", source=test_file)
        ]

        rag_docs = [{"source": test_file, "type": "file"}]

        # Act
        build_rag(self.test_folder_path, rag_docs)

        # Assert
        mock_loader.to_crawled_local_objs.assert_called_once_with([test_file])

    @patch("arklex.env.tools.RAG.build_rag.Loader")
    def test_build_rag_processes_file_documents_directory(
        self, mock_loader_class: Mock
    ) -> None:
        """Test that build_rag processes directory documents correctly."""
        # Arrange
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        # Create a temporary test directory with files
        test_dir = os.path.join(self.temp_dir, "test_dir")
        os.makedirs(test_dir, exist_ok=True)

        test_file1 = os.path.join(test_dir, "file1.txt")
        test_file2 = os.path.join(test_dir, "file2.txt")

        with open(test_file1, "w") as f:
            f.write("content 1")
        with open(test_file2, "w") as f:
            f.write("content 2")

        mock_loader.to_crawled_local_objs.return_value = [
            Mock(content="file content 1", source=test_file1),
            Mock(content="file content 2", source=test_file2),
        ]

        rag_docs = [{"source": test_dir, "type": "file"}]

        # Act
        build_rag(self.test_folder_path, rag_docs)

        # Assert
        # Sort files to handle non-deterministic order
        expected_files = sorted([test_file1, test_file2])
        actual_call_args = mock_loader.to_crawled_local_objs.call_args[0][0]
        assert sorted(actual_call_args) == expected_files

    @patch("arklex.env.tools.RAG.build_rag.Loader")
    def test_build_rag_processes_zip_file(self, mock_loader_class: Mock) -> None:
        """Test that build_rag processes zip files correctly."""
        # Arrange
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        # Create a temporary zip file
        zip_file = os.path.join(self.temp_dir, "test.zip")
        with zipfile.ZipFile(zip_file, "w") as zip_ref:
            zip_ref.writestr("file1.txt", "content 1")
            zip_ref.writestr("file2.txt", "content 2")

        mock_loader.to_crawled_local_objs.return_value = [
            Mock(content="zip content 1", source="temp/file1.txt"),
            Mock(content="zip content 2", source="temp/file2.txt"),
        ]

        rag_docs = [{"source": zip_file, "type": "file"}]

        # Act
        build_rag(self.test_folder_path, rag_docs)

        # Assert
        # The loader should be called with the extracted files
        mock_loader.to_crawled_local_objs.assert_called_once()
        call_args = mock_loader.to_crawled_local_objs.call_args[0][0]
        assert len(call_args) == 2
        assert any("file1.txt" in arg for arg in call_args)
        assert any("file2.txt" in arg for arg in call_args)

    @patch("arklex.env.tools.RAG.build_rag.Loader")
    def test_build_rag_processes_text_documents(self, mock_loader_class: Mock) -> None:
        """Test that build_rag processes text type documents correctly."""
        # Arrange
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        mock_loader.to_crawled_text.return_value = [
            Mock(content="text content", source="test text")
        ]

        rag_docs = [{"source": "test text content", "type": "text"}]

        # Act
        build_rag(self.test_folder_path, rag_docs)

        # Assert
        mock_loader.to_crawled_text.assert_called_once_with(["test text content"])

    def test_build_rag_raises_file_not_found_error_for_nonexistent_file(self) -> None:
        """Test that build_rag raises FileNotFoundError for nonexistent file."""
        # Arrange
        rag_docs = [{"source": "/nonexistent/file.txt", "type": "file"}]

        # Act & Assert
        with pytest.raises(
            FileNotFoundError,
            match="Source path '/nonexistent/file.txt' does not exist",
        ):
            build_rag(self.test_folder_path, rag_docs)

    def test_build_rag_raises_exception_for_invalid_type(self) -> None:
        """Test that build_rag raises exception for invalid document type."""
        # Arrange
        rag_docs = [{"source": "test", "type": "invalid_type"}]

        # Act & Assert
        with pytest.raises(
            Exception,
            match="type must be one of \\[url, file, text\\] and it must be provided",
        ):
            build_rag(self.test_folder_path, rag_docs)

    def test_build_rag_raises_exception_for_missing_type(self) -> None:
        """Test that build_rag raises exception when type is not provided."""
        # Arrange
        rag_docs = [{"source": "test"}]

        # Act & Assert
        with pytest.raises(
            Exception,
            match="type must be one of \\[url, file, text\\] and it must be provided",
        ):
            build_rag(self.test_folder_path, rag_docs)

    @patch("arklex.env.tools.RAG.build_rag.Loader")
    def test_build_rag_saves_documents_and_chunks(
        self, mock_loader_class: Mock
    ) -> None:
        """Test that build_rag saves both documents and chunked documents."""
        # Arrange
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        # Mock the loader methods
        mock_loader.to_crawled_text.return_value = [
            Mock(content="text content", source="test text")
        ]
        mock_loader.chunk.return_value = [
            Mock(content="chunk content", source="test text")
        ]

        rag_docs = [{"source": "test text content", "type": "text"}]

        # Act
        build_rag(self.test_folder_path, rag_docs)

        # Assert
        # Verify save was called for both documents and chunks
        assert mock_loader_class.save.call_count == 2

        # Check that save was called with correct file paths
        save_calls = mock_loader_class.save.call_args_list
        file_paths = [call[0][0] for call in save_calls]
        assert any("documents.pkl" in path for path in file_paths)
        assert any("chunked_documents.pkl" in path for path in file_paths)

    @patch("arklex.env.tools.RAG.build_rag.Loader")
    def test_build_rag_logs_crawled_sources(
        self, mock_loader_class: Mock, caplog: LogCaptureFixture
    ) -> None:
        """Test that build_rag logs the crawled sources."""
        # Arrange
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        mock_loader.to_crawled_text.return_value = [
            Mock(content="text content", source="test source")
        ]
        mock_loader.chunk.return_value = [
            Mock(content="chunk content", source="test source")
        ]

        rag_docs = [{"source": "test text content", "type": "text"}]

        # Act
        build_rag(self.test_folder_path, rag_docs)

        # Assert
        assert "crawled sources: ['test source']" in caplog.text

    @patch("arklex.env.tools.RAG.build_rag.Loader")
    def test_build_rag_logs_content_for_new_documents(
        self, mock_loader_class: Mock, caplog: LogCaptureFixture
    ) -> None:
        """Test that build_rag logs content when processing new documents."""
        # Arrange
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        mock_loader.to_crawled_text.return_value = [
            Mock(content="text content", source="test source")
        ]
        mock_loader.chunk.return_value = [
            Mock(content="chunk content", source="test source")
        ]

        rag_docs = [{"source": "test text content", "type": "text"}]

        # Act
        build_rag(self.test_folder_path, rag_docs)

        # Assert
        assert "Content: ['text content']" in caplog.text

    @patch("arklex.env.tools.RAG.build_rag.Loader")
    def test_build_rag_warns_when_loading_existing_documents(
        self, mock_loader_class: Mock, caplog: LogCaptureFixture
    ) -> None:
        """Test that build_rag warns when loading existing documents."""
        # Arrange
        os.makedirs(self.test_folder_path, exist_ok=True)
        existing_docs = [SimpleDoc("test content", "test source")]
        filepath = os.path.join(self.test_folder_path, "documents.pkl")

        with open(filepath, "wb") as f:
            pickle.dump(existing_docs, f)

        rag_docs = []

        # Act
        build_rag(self.test_folder_path, rag_docs)

        # Assert
        assert "Loading existing documents" in caplog.text
        assert "If you want to recrawl" in caplog.text

    @patch("arklex.env.tools.RAG.build_rag.Loader")
    def test_build_rag_logs_crawling_info_for_urls(
        self, mock_loader_class: Mock, caplog: LogCaptureFixture
    ) -> None:
        """Test that build_rag logs crawling info for URL documents."""
        # Arrange
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        mock_loader.get_all_urls.return_value = ["http://example.com"]
        mock_loader.to_crawled_url_objs.return_value = [
            Mock(content="url content", source="http://example.com")
        ]

        rag_docs = [{"source": "http://example.com", "type": "url", "num": 5}]

        # Act
        build_rag(self.test_folder_path, rag_docs)

        # Assert
        assert "Crawling http://example.com" in caplog.text
