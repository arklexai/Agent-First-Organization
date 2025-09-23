"""Tests for get_documents module."""

import pytest
from unittest.mock import patch, mock_open
from arklex.evaluation.get_documents import get_domain_info, load_docs


class TestGetDomainInfo:
    """Test cases for get_domain_info function."""

    def test_get_domain_info_empty_documents(self):
        """Test with empty documents list."""
        documents = []
        result = get_domain_info(documents)
        assert result is None

    def test_get_domain_info_single_document(self):
        """Test with single document."""
        documents = [
            {"title": "Test Document", "content": "This is test content"}
        ]
        result = get_domain_info(documents)
        assert result is not None
        assert isinstance(result, str)
        assert "Test Document" in result
        assert "This is test content" in result

    def test_get_domain_info_multiple_documents(self):
        """Test with multiple documents."""
        documents = [
            {"title": "Doc 1", "content": "Content 1"},
            {"title": "Doc 2", "content": "Content 2"},
        ]
        result = get_domain_info(documents)
        assert result is not None
        assert isinstance(result, str)
        assert "Doc 1" in result
        assert "Doc 2" in result
        assert "Content 1" in result
        assert "Content 2" in result

    def test_get_domain_info_documents_with_missing_keys(self):
        """Test with documents missing title or content."""
        documents = [
            {"title": "Doc 1"},  # Missing content
            {"content": "Content 2"},  # Missing title
        ]
        result = get_domain_info(documents)
        assert result is not None
        assert isinstance(result, str)
        assert "Doc 1" in result
        assert "Content 2" in result

    def test_get_domain_info_documents_with_none_values(self):
        """Test with documents containing None values."""
        documents = [
            {"title": None, "content": "Content 1"},
            {"title": "Doc 2", "content": None},
        ]
        result = get_domain_info(documents)
        assert result is not None
        assert isinstance(result, str)
        assert "Content 1" in result
        assert "Doc 2" in result


class TestLoadDocs:
    """Test cases for load_docs function."""

    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_docs_with_valid_directory(self, mock_file, mock_listdir, mock_exists):
        """Test loading documents from valid directory."""
        mock_exists.return_value = True
        mock_listdir.return_value = ["doc1.txt", "doc2.txt"]
        mock_file.return_value.read.return_value = "Document content"
        
        doc_config = {"type": "file", "path": "/test/dir"}
        result = load_docs("/test/dir", doc_config, limit=10)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all("content" in doc for doc in result)

    @patch("os.path.exists")
    def test_load_docs_with_nonexistent_directory(self, mock_exists):
        """Test loading documents from nonexistent directory."""
        mock_exists.return_value = False
        
        doc_config = {"type": "file", "path": "/nonexistent/dir"}
        result = load_docs("/nonexistent/dir", doc_config, limit=10)
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_load_docs_with_none_document_dir(self):
        """Test loading documents with None document directory."""
        doc_config = {"type": "file", "path": "/test/dir"}
        result = load_docs(None, doc_config, limit=10)
        
        assert isinstance(result, list)
        assert len(result) == 0

    @patch("os.path.exists")
    @patch("os.listdir")
    def test_load_docs_with_empty_directory(self, mock_listdir, mock_exists):
        """Test loading documents from empty directory."""
        mock_exists.return_value = True
        mock_listdir.return_value = []
        
        doc_config = {"type": "file", "path": "/empty/dir"}
        result = load_docs("/empty/dir", doc_config, limit=10)
        
        assert isinstance(result, list)
        assert len(result) == 0

    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_docs_with_limit(self, mock_file, mock_listdir, mock_exists):
        """Test loading documents with limit."""
        mock_exists.return_value = True
        mock_listdir.return_value = ["doc1.txt", "doc2.txt", "doc3.txt", "doc4.txt"]
        mock_file.return_value.read.return_value = "Document content"
        
        doc_config = {"type": "file", "path": "/test/dir"}
        result = load_docs("/test/dir", doc_config, limit=2)
        
        assert isinstance(result, list)
        assert len(result) == 2

    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_docs_with_file_reading_error(self, mock_file, mock_listdir, mock_exists):
        """Test loading documents when file reading fails."""
        mock_exists.return_value = True
        mock_listdir.return_value = ["doc1.txt"]
        mock_file.side_effect = IOError("Cannot read file")
        
        doc_config = {"type": "file", "path": "/test/dir"}
        result = load_docs("/test/dir", doc_config, limit=10)
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_load_docs_with_invalid_doc_config(self):
        """Test loading documents with invalid doc_config."""
        doc_config = {}  # Missing required keys
        
        result = load_docs("/test/dir", doc_config, limit=10)
        
        assert isinstance(result, list)
        assert len(result) == 0
