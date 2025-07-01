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
