"""Tests for document loading and processing module.

This module tests the document loading utilities including domain information
extraction, document loading with caching, and handling different document types.
"""

from typing import NoReturn
from unittest.mock import MagicMock, Mock, patch

import pytest

from arklex.evaluation import get_documents
from arklex.utils.loader import CrawledObject, SourceType


class TestGetDomainInfo:
    """Test cases for get_domain_info function."""

    def test_get_domain_info_with_summary(self) -> None:
        """Test getting domain info when summary document exists."""
        documents = [
            {"URL": "https://example.com", "content": "Example content"},
            {"URL": "summary", "content": "This is the summary"},
            {"URL": "https://another.com", "content": "Another content"},
        ]
        result = get_documents.get_domain_info(documents)
        assert result == "This is the summary"

    def test_get_domain_info_without_summary(self) -> None:
        """Test getting domain info when no summary document exists."""
        documents = [
            {"URL": "https://example.com", "content": "Example content"},
            {"URL": "https://another.com", "content": "Another content"},
        ]
        result = get_documents.get_domain_info(documents)
        assert result is None

    def test_get_domain_info_empty_list(self) -> None:
        """Test getting domain info with empty document list."""
        documents = []
        result = get_documents.get_domain_info(documents)
        assert result is None

    def test_get_domain_info_multiple_summaries(self) -> None:
        """Test getting domain info with multiple summary documents (should return first)."""
        documents = [
            {"URL": "summary", "content": "First summary"},
            {"URL": "summary", "content": "Second summary"},
            {"URL": "https://example.com", "content": "Example content"},
        ]
        result = get_documents.get_domain_info(documents)
        assert result == "First summary"

    def test_get_domain_info_no_summary(self) -> None:
        """Explicitly test get_domain_info for missing summary (lines 57-58)."""
        from arklex.evaluation.get_documents import get_domain_info

        docs = [
            {"URL": "https://foo.com", "content": "foo"},
            {"URL": "https://bar.com", "content": "bar"},
        ]
        assert get_domain_info(docs) is None


class TestLoadDocs:
    """Test cases for load_docs function."""

    @patch("pickle.load")
    @patch("os.path.exists")
    @patch("builtins.open", create=True)
    @patch("arklex.utils.loader.Loader.save")
    def test_load_docs_with_existing_cache(
        self,
        mock_save: Mock,
        mock_open: Mock,
        mock_exists: Mock,
        mock_pickle_load: Mock,
    ) -> None:
        """Test loading documents with existing cache file."""
        mock_exists.return_value = True
        mock_crawled_objects = [
            CrawledObject(
                id="1",
                source="https://example.com",
                content="Example content",
                source_type=SourceType.WEB,
            ),
            CrawledObject(
                id="2",
                source="file.txt",
                content="File content",
                source_type=SourceType.FILE,
            ),
        ]
        mock_pickle_load.return_value = mock_crawled_objects
        mock_file = MagicMock()
        mock_open.return_value = mock_file

        doc_config = {
            "task_docs": [{"source": "https://example.com", "type": "url", "num": 1}]
        }
        result = get_documents.load_docs("./temp_files", doc_config, 10)

        assert len(result) == 2
        assert result[0]["source"] == "https://example.com"
        assert result[0]["content"] == "Example content"
        assert result[1]["source"] == "file.txt"
        assert result[1]["content"] == "File content"

    @patch("pickle.load")
    @patch("os.path.exists")
    @patch("builtins.open", create=True)
    @patch("arklex.utils.loader.Loader.save")
    def test_load_docs_without_cache_url_type(
        self,
        mock_save: Mock,
        mock_open: Mock,
        mock_exists: Mock,
        mock_pickle_load: Mock,
    ) -> None:
        """Test loading documents without cache for URL type."""
        mock_exists.return_value = False
        from arklex.utils.loader import CrawledObject, Loader, SourceType

        loader = Loader()
        with (
            patch.object(loader, "get_all_urls", return_value=["https://example.com"]),
            patch.object(
                loader,
                "to_crawled_url_objs",
                return_value=[
                    CrawledObject(
                        id="1",
                        source="https://example.com",
                        content="Example content",
                        source_type=SourceType.WEB,
                    )
                ],
            ),
            patch.object(
                loader,
                "get_candidates_websites",
                return_value=[
                    CrawledObject(
                        id="1",
                        source="https://example.com",
                        content="Example content",
                        source_type=SourceType.WEB,
                    )
                ],
            ),
        ):
            mock_file = MagicMock()
            mock_open.return_value = mock_file

            doc_config = {
                "task_docs": [
                    {"source": "https://example.com", "type": "url", "num": 1}
                ]
            }
            with patch("arklex.evaluation.get_documents.Loader", return_value=loader):
                result = get_documents.load_docs("./temp_files", doc_config, 10)

            assert len(result) == 1
            assert result[0]["source"] == "https://example.com"
            assert result[0]["content"] == "Example content"

    @patch("pickle.load")
    @patch("os.path.exists")
    @patch("arklex.evaluation.get_documents.Loader")
    @patch("os.listdir")
    def test_load_docs_file_type(
        self,
        mock_listdir: Mock,
        mock_loader_class: Mock,
        mock_exists: Mock,
        mock_pickle_load: Mock,
    ) -> None:
        """Test loading documents for file type."""
        mock_exists.return_value = False
        mock_listdir.return_value = ["file1.txt", "file2.txt"]
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_loader.to_crawled_local_objs.return_value = [
            CrawledObject(
                id="1",
                source="file1.txt",
                content="File 1 content",
                source_type=SourceType.FILE,
            ),
            CrawledObject(
                id="2",
                source="file2.txt",
                content="File 2 content",
                source_type=SourceType.FILE,
            ),
        ]
        mock_loader.save = MagicMock()

        doc_config = {"task_docs": [{"source": "./docs", "type": "file"}]}
        result = get_documents.load_docs("./temp_files", doc_config, 10)

        assert len(result) == 2
        assert result[0]["source"] == "file1.txt"
        assert result[0]["content"] == "File 1 content"
        assert result[1]["source"] == "file2.txt"
        assert result[1]["content"] == "File 2 content"

    @patch("pickle.load")
    @patch("os.path.exists")
    @patch("arklex.evaluation.get_documents.Loader")
    def test_load_docs_text_type(
        self, mock_loader_class: Mock, mock_exists: Mock, mock_pickle_load: Mock
    ) -> None:
        """Test loading documents for text type."""
        mock_exists.return_value = False
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_loader.to_crawled_text.return_value = [
            CrawledObject(
                id="1",
                source="text_content",
                content="This is text content",
                source_type=SourceType.TEXT,
            )
        ]
        mock_loader.save = MagicMock()

        doc_config = {"task_docs": [{"source": "This is text", "type": "text"}]}
        result = get_documents.load_docs("./temp_files", doc_config, 10)

        assert len(result) == 1
        assert result[0]["source"] == "text_content"
        assert result[0]["content"] == "This is text content"

    def test_load_docs_missing_rag_and_task_docs(self) -> None:
        """Test loading documents with missing rag_docs and task_docs."""
        doc_config = {"other_key": "value"}

        # The function raises ValueError when neither rag_docs nor task_docs is present
        with pytest.raises(
            ValueError,
            match="The config json file must have a key 'rag_docs' or 'task_docs' with a list of documents to load.",
        ):
            get_documents.load_docs("./temp_files", doc_config, 10)

    def test_load_docs_empty_rag_docs(self) -> None:
        """Test loading documents with empty rag_docs array."""
        doc_config = {"rag_docs": []}
        result = get_documents.load_docs("./temp_files", doc_config, 10)
        assert result == []

    def test_load_docs_empty_task_docs(self) -> None:
        """Test loading documents with empty task_docs array."""
        doc_config = {"task_docs": []}
        result = get_documents.load_docs("./temp_files", doc_config, 10)
        assert result == []

    def test_load_docs_none_directory(self) -> None:
        """Test loading documents with None directory."""
        doc_config = {"task_docs": [{"source": "https://example.com", "type": "url"}]}
        result = get_documents.load_docs(None, doc_config, 10)
        assert result == []

    @patch("pickle.load")
    @patch("os.path.exists")
    @patch("arklex.evaluation.get_documents.Loader")
    def test_load_docs_with_exception(
        self, mock_loader_class: Mock, mock_exists: Mock, mock_pickle_load: Mock
    ) -> None:
        """Test loading documents with exception handling."""
        mock_exists.return_value = False
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_loader.get_all_urls.side_effect = Exception("Network error")

        doc_config = {
            "task_docs": [{"source": "https://example.com", "type": "url", "num": 1}]
        }
        result = get_documents.load_docs("./temp_files", doc_config, 10)

        assert result == []

    @patch("pickle.load")
    @patch("os.path.exists")
    @patch("arklex.evaluation.get_documents.Loader")
    def test_load_docs_high_total_docs(
        self, mock_loader_class: Mock, mock_exists: Mock, mock_pickle_load: Mock
    ) -> None:
        """Test loading documents with high total document count."""
        mock_exists.return_value = False
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_loader.get_all_urls.return_value = ["https://example.com"]
        mock_loader.to_crawled_url_objs.return_value = [
            CrawledObject(
                id="1",
                source="https://example.com",
                content="Example content",
                source_type=SourceType.WEB,
            )
        ]
        mock_loader.get_candidates_websites.return_value = [
            CrawledObject(
                id="1",
                source="https://example.com",
                content="Example content",
                source_type=SourceType.WEB,
            )
        ]

        # High total docs should adjust the limit for get_candidates_websites
        doc_config = {
            "task_docs": [{"source": "https://example.com", "type": "url", "num": 100}]
        }
        result = get_documents.load_docs("./temp_files", doc_config, 10)

        assert len(result) == 1
        mock_loader.get_all_urls.assert_called_with(
            "https://example.com", 100
        )  # Uses original num value
        mock_loader.get_candidates_websites.assert_called_with(
            mock_loader.to_crawled_url_objs.return_value, 20
        )  # Uses adjusted limit

    @patch("pickle.load")
    @patch("os.path.exists")
    @patch("arklex.evaluation.get_documents.Loader")
    def test_load_docs_mixed_source_types(
        self, mock_loader_class: Mock, mock_exists: Mock, mock_pickle_load: Mock
    ) -> None:
        """Test loading documents with mixed source types."""
        mock_exists.return_value = False
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_loader.get_all_urls.return_value = ["https://example.com"]
        mock_loader.to_crawled_url_objs.return_value = [
            CrawledObject(
                id="1",
                source="https://example.com",
                content="Example content",
                source_type=SourceType.WEB,
            )
        ]
        mock_loader.to_crawled_local_objs.return_value = [
            CrawledObject(
                id="2",
                source="file.txt",
                content="File content",
                source_type=SourceType.FILE,
            )
        ]
        mock_loader.to_crawled_text.return_value = [
            CrawledObject(
                id="3",
                source="text_content",
                content="Text content",
                source_type=SourceType.TEXT,
            )
        ]
        mock_loader.get_candidates_websites.return_value = [
            CrawledObject(
                id="1",
                source="https://example.com",
                content="Example content",
                source_type=SourceType.WEB,
            )
        ]
        mock_loader.save = MagicMock()

        doc_config = {
            "task_docs": [
                {"source": "https://example.com", "type": "url", "num": 1},
                {"source": "./docs", "type": "file"},
                {"source": "This is text", "type": "text"},
            ]
        }
        result = get_documents.load_docs("./temp_files", doc_config, 10)

        assert len(result) == 3
        assert result[0]["source"] == "https://example.com"
        assert result[1]["source"] == "file.txt"
        assert result[2]["source"] == "text_content"

    def test_load_docs_invalid_type(self) -> None:
        """Test loading documents with invalid type."""
        doc_config = {
            "task_docs": [{"source": "https://example.com", "type": "invalid"}]
        }

        # The function prints an error and returns empty list, doesn't raise Exception
        result = get_documents.load_docs("./temp_files", doc_config, 10)
        assert result == []

    def test_load_docs_missing_type(self) -> None:
        """Test loading documents with missing type."""
        doc_config = {"task_docs": [{"source": "https://example.com"}]}

        # The function prints an error and returns empty list, doesn't raise Exception
        result = get_documents.load_docs("./temp_files", doc_config, 10)
        assert result == []

    @patch("pickle.load")
    @patch("os.path.exists")
    @patch("builtins.open", create=True)
    def test_load_docs_invalid_crawled_objects(
        self, mock_open: Mock, mock_exists: Mock, mock_pickle_load: Mock
    ) -> None:
        """Test load_docs raises ValueError for non-CrawledObject docs."""
        mock_exists.return_value = True
        mock_pickle_load.return_value = [{"not": "CrawledObject"}]
        mock_file = MagicMock()
        mock_open.return_value = mock_file

        doc_config = {"task_docs": [{"source": "foo", "type": "url", "num": 1}]}

        with pytest.raises(
            ValueError, match="The documents must be a list of CrawledObject objects"
        ):
            get_documents.load_docs("/tmp", doc_config, 10)

    @patch("pickle.load")
    @patch("os.path.exists")
    @patch("arklex.evaluation.get_documents.Loader")
    def test_load_docs_with_num_field(
        self, mock_loader_class: Mock, mock_exists: Mock, mock_pickle_load: Mock
    ) -> None:
        """Test loading documents with num field specified."""
        mock_exists.return_value = False
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_loader.get_all_urls.return_value = ["https://example.com"]
        mock_loader.to_crawled_url_objs.return_value = [
            CrawledObject(
                id="1",
                source="https://example.com",
                content="Example content",
                source_type=SourceType.WEB,
            )
        ]
        mock_loader.get_candidates_websites.return_value = [
            CrawledObject(
                id="1",
                source="https://example.com",
                content="Example content",
                source_type=SourceType.WEB,
            )
        ]
        mock_loader.save = MagicMock()

        doc_config = {
            "task_docs": [{"source": "https://example.com", "type": "url", "num": 5}]
        }
        result = get_documents.load_docs("./temp_files", doc_config, 10)

        assert len(result) == 1
        mock_loader.get_all_urls.assert_called_with("https://example.com", 5)

    @patch("pickle.load")
    @patch("os.path.exists")
    @patch("arklex.evaluation.get_documents.Loader")
    def test_load_docs_without_num_field(
        self, mock_loader_class: Mock, mock_exists: Mock, mock_pickle_load: Mock
    ) -> None:
        """Test loading documents without num field (should default to 1)."""
        mock_exists.return_value = False
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_loader.get_all_urls.return_value = ["https://example.com"]
        mock_loader.to_crawled_url_objs.return_value = [
            CrawledObject(
                id="1",
                source="https://example.com",
                content="Example content",
                source_type=SourceType.WEB,
            )
        ]
        mock_loader.get_candidates_websites.return_value = [
            CrawledObject(
                id="1",
                source="https://example.com",
                content="Example content",
                source_type=SourceType.WEB,
            )
        ]
        mock_loader.save = MagicMock()

        doc_config = {"task_docs": [{"source": "https://example.com", "type": "url"}]}
        result = get_documents.load_docs("./temp_files", doc_config, 10)

        assert len(result) == 1
        mock_loader.get_all_urls.assert_called_with("https://example.com", 1)

    def test_load_docs_exception_branch_and_none_dir(self) -> None:
        """Covers the except Exception branch (lines 138-140) and the else branch for document_dir is None (line 125) in load_docs."""
        from arklex.evaluation.get_documents import load_docs

        # except Exception branch
        class DummyLoader:
            def __init__(self) -> None:
                pass

            def get_all_urls(self, *a: object, **kw: object) -> NoReturn:
                raise Exception("fail")

        doc_config = {"task_docs": [{"source": "bad", "type": "url", "num": 1}]}
        with patch(
            "arklex.evaluation.get_documents.Loader", return_value=DummyLoader()
        ):
            # Should not raise, should return []
            result = load_docs("/tmp", doc_config, 10)
            assert isinstance(result, list)
        # else branch for document_dir is None
        result2 = load_docs(None, doc_config, 10)
        assert result2 == []

    def test_main_block_execution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test the main block execution (lines 137-140) for coverage."""

        # Mock file operations
        def mock_open(*args: object, **kwargs: object) -> object:
            class MockFile:
                def __enter__(self) -> "MockFile":
                    return self

                def __exit__(self, *args: object) -> None:
                    pass

                def read(self) -> str:
                    return '{"rag_docs": [{"source": "https://example.com", "type": "url", "num": 1}]}'

            return MockFile()

        # Mock the load_docs function to avoid actual execution
        def mock_load_docs(*args: object, **kwargs: object) -> list[dict[str, str]]:
            return [{"source": "https://example.com", "content": "test content"}]

        monkeypatch.setattr("builtins.open", mock_open)
        monkeypatch.setattr("arklex.evaluation.get_documents.load_docs", mock_load_docs)
        monkeypatch.setattr(
            "json.load",
            lambda f: {
                "rag_docs": [{"source": "https://example.com", "type": "url", "num": 1}]
            },
        )

        # Execute the main block by importing and running the module
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

        # Create a temporary file to simulate the config file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(
                '{"rag_docs": [{"source": "https://example.com", "type": "url", "num": 1}]}'
            )
            temp_config_path = f.name

        try:
            # Execute the main block by reading and executing the file
            with open("arklex/evaluation/get_documents.py") as f:
                content = f.read()
                # Execute the content with mocked functions
                exec_globals = {
                    "__name__": "__main__",
                    "json": __import__("json"),
                    "load_docs": mock_load_docs,
                    "open": mock_open,
                }
                exec(content, exec_globals)
        finally:
            # Clean up
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)

    def test_load_docs_empty_docs_list_edge_case(self) -> None:
        """Test load_docs with empty docs list after processing (line 125)."""

        # Test the case where document_dir is None
        result = get_documents.load_docs(
            None, {"rag_docs": [{"source": "https://example.com", "type": "url"}]}, 10
        )
        assert result == []

        # Test the case where document_dir is not None but processing results in empty list
        with (
            patch("os.path.exists", return_value=False),
            patch("arklex.evaluation.get_documents.Loader") as mock_loader_class,
        ):
            mock_loader = MagicMock()
            mock_loader_class.return_value = mock_loader
            mock_loader.get_all_urls.return_value = []
            mock_loader.to_crawled_url_objs.return_value = []
            mock_loader.get_candidates_websites.return_value = []
            mock_loader.save = MagicMock()

            doc_config = {
                "rag_docs": [{"source": "https://example.com", "type": "url", "num": 1}]
            }
            result = get_documents.load_docs("./temp_files", doc_config, 10)
            assert result == []

    def test_load_docs_zero_total_docs_edge_case(self) -> None:
        """Test load_docs with zero total documents (edge case for limit calculation)."""
        with (
            patch("os.path.exists", return_value=False),
            patch("arklex.evaluation.get_documents.Loader") as mock_loader_class,
        ):
            mock_loader = MagicMock()
            mock_loader_class.return_value = mock_loader
            mock_loader.get_all_urls.return_value = ["https://example.com"]
            mock_loader.to_crawled_url_objs.return_value = [
                CrawledObject(
                    id="1",
                    source="https://example.com",
                    content="Example content",
                    source_type=SourceType.WEB,
                )
            ]
            mock_loader.get_candidates_websites.return_value = [
                CrawledObject(
                    id="1",
                    source="https://example.com",
                    content="Example content",
                    source_type=SourceType.WEB,
                )
            ]
            mock_loader.save = MagicMock()

            # Test with zero total docs (should use default limit of 10)
            doc_config = {
                "rag_docs": [{"source": "https://example.com", "type": "url", "num": 0}]
            }
            result = get_documents.load_docs("./temp_files", doc_config, 10)
            assert len(result) == 1
            mock_loader.get_candidates_websites.assert_called_with(
                mock_loader.to_crawled_url_objs.return_value, 10
            )

    def test_load_docs_exactly_50_total_docs_edge_case(self) -> None:
        """Test load_docs with exactly 50 total documents (edge case for limit calculation)."""
        with (
            patch("os.path.exists", return_value=False),
            patch("arklex.evaluation.get_documents.Loader") as mock_loader_class,
        ):
            mock_loader = MagicMock()
            mock_loader_class.return_value = mock_loader
            mock_loader.get_all_urls.return_value = ["https://example.com"]
            mock_loader.to_crawled_url_objs.return_value = [
                CrawledObject(
                    id="1",
                    source="https://example.com",
                    content="Example content",
                    source_type=SourceType.WEB,
                )
            ]
            mock_loader.get_candidates_websites.return_value = [
                CrawledObject(
                    id="1",
                    source="https://example.com",
                    content="Example content",
                    source_type=SourceType.WEB,
                )
            ]
            mock_loader.save = MagicMock()

            # Test with exactly 50 total docs (should use default limit of 10, not 50//5=10)
            doc_config = {
                "rag_docs": [
                    {"source": "https://example.com", "type": "url", "num": 50}
                ]
            }
            result = get_documents.load_docs("./temp_files", doc_config, 10)
            assert len(result) == 1
            mock_loader.get_candidates_websites.assert_called_with(
                mock_loader.to_crawled_url_objs.return_value, 10
            )

    def test_load_docs_51_total_docs_edge_case(self) -> None:
        """Test load_docs with 51 total documents (edge case for limit calculation)."""
        with (
            patch("os.path.exists", return_value=False),
            patch("arklex.evaluation.get_documents.Loader") as mock_loader_class,
        ):
            mock_loader = MagicMock()
            mock_loader_class.return_value = mock_loader
            mock_loader.get_all_urls.return_value = ["https://example.com"]
            mock_loader.to_crawled_url_objs.return_value = [
                CrawledObject(
                    id="1",
                    source="https://example.com",
                    content="Example content",
                    source_type=SourceType.WEB,
                )
            ]
            mock_loader.get_candidates_websites.return_value = [
                CrawledObject(
                    id="1",
                    source="https://example.com",
                    content="Example content",
                    source_type=SourceType.WEB,
                )
            ]
            mock_loader.save = MagicMock()

            # Test with 51 total docs (should use adjusted limit of 51//5=10)
            doc_config = {
                "rag_docs": [
                    {"source": "https://example.com", "type": "url", "num": 51}
                ]
            }
            result = get_documents.load_docs("./temp_files", doc_config, 10)
            assert len(result) == 1
            mock_loader.get_candidates_websites.assert_called_with(
                mock_loader.to_crawled_url_objs.return_value, 10
            )
