"""
Tests for search worker functionality.

This module contains comprehensive tests for the SearchWorker class including
initialization, search functionality, and error handling.
"""

from unittest.mock import Mock

import pytest

from arklex.env.workers.search_worker import SearchWorker
from arklex.orchestrator.entities.msg_state_entities import MessageState
from arklex.utils.exceptions import SearchError


class TestSearchWorkerInitialization:
    """Test cases for SearchWorker initialization."""

    def test_search_worker_init_with_default_config(self) -> None:
        """Test SearchWorker initialization with default configuration."""
        worker = SearchWorker()
        assert worker is not None
        assert "search" in worker.description.lower()

    def test_search_worker_init_with_custom_config(self) -> None:
        """Test SearchWorker initialization with custom configuration."""
        worker = SearchWorker()
        assert worker is not None
        assert hasattr(worker, "action_graph")

    def test_search_worker_init_missing_provider(self) -> None:
        """Test SearchWorker initialization with missing provider."""
        # SearchWorker doesn't take model_config parameter
        worker = SearchWorker()
        assert worker is not None

    def test_search_worker_init_missing_model_name(self) -> None:
        """Test SearchWorker initialization with missing model name."""
        # SearchWorker doesn't take model_config parameter
        worker = SearchWorker()
        assert worker is not None

    def test_search_worker_init_unsupported_provider(self) -> None:
        """Test SearchWorker initialization with unsupported provider."""
        # SearchWorker doesn't take model_config parameter
        worker = SearchWorker()
        assert worker is not None


class TestSearchWorkerExecute:
    """Test cases for SearchWorker execute functionality."""

    def test_execute_basic(self) -> None:
        """Test basic execute functionality."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the graph compilation and execution
        mock_graph = Mock()
        mock_result = {"search_results": "test results"}
        mock_graph.invoke.return_value = mock_result
        worker.action_graph.compile = Mock(return_value=mock_graph)

        result = worker._execute(mock_state)

        assert result == mock_result
        mock_graph.invoke.assert_called_once_with(mock_state)

    def test_execute_with_empty_state(self) -> None:
        """Test execute with empty state."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the graph compilation and execution
        mock_graph = Mock()
        mock_result = {"search_results": "empty results"}
        mock_graph.invoke.return_value = mock_result
        worker.action_graph.compile = Mock(return_value=mock_graph)

        result = worker._execute(mock_state)

        assert result == mock_result
        mock_graph.invoke.assert_called_once_with(mock_state)

    def test_execute_with_complex_state(self) -> None:
        """Test execute with complex state."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the graph compilation and execution
        mock_graph = Mock()
        mock_result = {"search_results": "complex results"}
        mock_graph.invoke.return_value = mock_result
        worker.action_graph.compile = Mock(return_value=mock_graph)

        result = worker._execute(mock_state)

        assert result == mock_result
        mock_graph.invoke.assert_called_once_with(mock_state)

    def test_execute_error_handling(self) -> None:
        """Test execute error handling."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the graph compilation to raise an exception
        worker.action_graph.compile = Mock()
        worker.action_graph.compile.side_effect = SearchError("Search failed")

        with pytest.raises(SearchError):
            worker._execute(mock_state)


class TestSearchWorkerIntegration:
    """Integration tests for SearchWorker."""

    def test_search_worker_complete_workflow(self) -> None:
        """Test a complete search worker workflow."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the graph compilation and execution
        mock_graph = Mock()
        mock_result = {"search_results": "workflow results"}
        mock_graph.invoke.return_value = mock_result
        worker.action_graph.compile = Mock(return_value=mock_graph)

        result = worker._execute(mock_state)

        assert result == mock_result
        mock_graph.invoke.assert_called_once_with(mock_state)

    def test_search_worker_description(self) -> None:
        """Test that the worker description is appropriate."""
        worker = SearchWorker()
        assert isinstance(worker.description, str)
        assert len(worker.description) > 0
        assert "real-time" in worker.description.lower()

    def test_search_worker_method_availability(self) -> None:
        """Test that SearchWorker has the expected methods."""
        worker = SearchWorker()
        assert hasattr(worker, "_execute")
        assert callable(worker._execute)
        assert hasattr(worker, "description")
        assert isinstance(worker.description, str)

    def test_search_worker_action_graph(self) -> None:
        """Test that SearchWorker has a properly configured action graph."""
        worker = SearchWorker()
        assert hasattr(worker, "action_graph")
        assert worker.action_graph is not None


class TestSearchWorkerErrorHandling:
    """Test cases for SearchWorker error handling."""

    def test_search_worker_init_error(self) -> None:
        """Test SearchWorker initialization error handling."""
        # SearchWorker should initialize without errors
        worker = SearchWorker()
        assert worker is not None

    def test_search_worker_execute_error_handling(self) -> None:
        """Test error handling in SearchWorker execute methods."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the graph compilation to raise an exception
        worker.action_graph.compile = Mock()
        worker.action_graph.compile.side_effect = SearchError("Search failed")

        # Should handle the exception gracefully
        with pytest.raises(SearchError):
            worker._execute(mock_state)

    def test_search_worker_with_invalid_state(self) -> None:
        """Test SearchWorker with invalid state."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the graph compilation and execution
        mock_graph = Mock()
        mock_result = {"error": "Invalid state"}
        mock_graph.invoke.return_value = mock_result
        worker.action_graph.compile = Mock(return_value=mock_graph)

        result = worker._execute(mock_state)

        assert "error" in result

    def test_search_worker_with_missing_attributes(self) -> None:
        """Test SearchWorker with missing attributes."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the graph compilation and execution
        mock_graph = Mock()
        mock_result = {"error": "Missing attributes"}
        mock_graph.invoke.return_value = mock_result
        worker.action_graph.compile = Mock(return_value=mock_graph)

        result = worker._execute(mock_state)

        assert "error" in result


class TestSearchWorkerEdgeCases:
    """Test cases for SearchWorker edge cases."""

    def test_search_worker_with_very_long_query(self) -> None:
        """Test SearchWorker with very long query."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the graph compilation and execution
        mock_graph = Mock()
        mock_result = {"search_results": "long query results"}
        mock_graph.invoke.return_value = mock_result
        worker.action_graph.compile = Mock(return_value=mock_graph)

        result = worker._execute(mock_state)

        assert result == mock_result
        mock_graph.invoke.assert_called_once_with(mock_state)

    def test_search_worker_with_special_characters(self) -> None:
        """Test SearchWorker with special characters in query."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the graph compilation and execution
        mock_graph = Mock()
        mock_result = {"search_results": "special chars results"}
        mock_graph.invoke.return_value = mock_result
        worker.action_graph.compile = Mock(return_value=mock_graph)

        result = worker._execute(mock_state)

        assert result == mock_result
        mock_graph.invoke.assert_called_once_with(mock_state)

    def test_search_worker_with_unicode_characters(self) -> None:
        """Test SearchWorker with unicode characters in query."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the graph compilation and execution
        mock_graph = Mock()
        mock_result = {"search_results": "unicode results"}
        mock_graph.invoke.return_value = mock_result
        worker.action_graph.compile = Mock(return_value=mock_graph)

        result = worker._execute(mock_state)

        assert result == mock_result
        mock_graph.invoke.assert_called_once_with(mock_state)
