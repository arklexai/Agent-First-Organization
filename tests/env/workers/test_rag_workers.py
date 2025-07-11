"""
Tests for RAG workers functionality.

This module contains comprehensive tests for all RAG worker classes including
FaissRAGWorker, MilvusRAGWorker, RagMsgWorker, and RAGMessageWorker.
"""

import os
from unittest.mock import Mock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from arklex.env.workers.faiss_rag_worker import FaissRAGWorker
from arklex.env.workers.milvus_rag_worker import MilvusRAGWorker
from arklex.env.workers.rag_message_worker import (
    RAGMessageWorker,
    RagMsgWorker,
    RagMsgWorkerKwargs,
)
from arklex.orchestrator.entities.msg_state_entities import MessageState

# Set test environment
os.environ["ARKLEX_TEST_ENV"] = "local"


class TestRagMsgWorker:
    """Test cases for RagMsgWorker - the main RAG and Message Worker combination."""

    def test_rag_msg_worker_init(self) -> None:
        """Test RagMsgWorker initialization."""
        worker = RagMsgWorker()
        assert worker.description == "A combination of RAG and Message Workers"
        assert worker.llm is None
        assert worker.tags == {}
        assert worker.action_graph is None

    def test_rag_msg_worker_choose_retriever_with_yes_response(self) -> None:
        """Test _choose_retriever method when LLM responds with 'yes'."""
        worker = RagMsgWorker()
        worker.llm = Mock(spec=BaseChatModel)

        # Mock the LLM to return "yes"
        mock_llm = Mock()
        mock_llm.invoke.return_value = "yes"
        worker.llm.__or__ = lambda self, other: mock_llm

        mock_state = Mock(spec=MessageState)
        mock_state.bot_config = Mock()
        mock_state.user_message = Mock()
        mock_state.user_message.history = "test history"

        with patch(
            "arklex.env.workers.rag_message_worker.load_prompts"
        ) as mock_load_prompts:
            mock_load_prompts.return_value = {
                "retrieval_needed_prompt": "Test prompt {formatted_chat}"
            }

            result = worker._choose_retriever(mock_state)

            assert result == "retriever"
            mock_load_prompts.assert_called_once_with(mock_state.bot_config)

    def test_rag_msg_worker_choose_retriever_with_no_response(self) -> None:
        """Test _choose_retriever method when LLM responds with 'no'."""
        worker = RagMsgWorker()
        worker.llm = Mock(spec=BaseChatModel)

        # Mock the LLM to return "no"
        mock_llm = Mock()
        mock_llm.invoke.return_value = "no"
        worker.llm.__or__ = lambda self, other: mock_llm

        mock_state = Mock(spec=MessageState)
        mock_state.bot_config = Mock()
        mock_state.user_message = Mock()
        mock_state.user_message.history = "test history"

        with patch(
            "arklex.env.workers.rag_message_worker.load_prompts"
        ) as mock_load_prompts:
            mock_load_prompts.return_value = {
                "retrieval_needed_prompt": "Test prompt {formatted_chat}"
            }

            result = worker._choose_retriever(mock_state)

            assert result == "message_worker"

    def test_rag_msg_worker_create_action_graph(self) -> None:
        """Test _create_action_graph method."""
        worker = RagMsgWorker()
        tags = {"test": "value"}

        with (
            patch("arklex.env.workers.rag_message_worker.RetrieveEngine"),
            patch(
                "arklex.env.workers.rag_message_worker.MessageWorker"
            ) as mock_message_worker,
        ):
            mock_message_worker.return_value = Mock()

            result = worker._create_action_graph(tags)

            assert isinstance(result, StateGraph)
            # The partial function is created but not called during graph creation
            # We just verify that the method returns a StateGraph

    def test_rag_msg_worker_execute_success(self) -> None:
        """Test _execute method with successful execution."""
        worker = RagMsgWorker()

        # Mock dependencies
        mock_llm = Mock(spec=ChatOpenAI)
        mock_state = Mock(spec=MessageState)
        mock_state.bot_config = Mock()
        mock_state.bot_config.llm_config = Mock()
        mock_state.bot_config.llm_config.llm_provider = "openai"
        mock_state.bot_config.llm_config.model_type_or_path = "gpt-4"

        mock_graph = Mock()
        mock_graph.invoke.return_value = {"result": "success"}

        with patch(
            "arklex.env.workers.rag_message_worker.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_provider_map.get.return_value = Mock(return_value=mock_llm)

            with patch.object(worker, "_create_action_graph") as mock_create_graph:
                mock_create_graph.return_value = Mock()
                mock_create_graph.return_value.compile.return_value = mock_graph

                result = worker._execute(mock_state, tags={"test": "value"})

                assert result == {"result": "success"}
                assert worker.llm is not None
                assert worker.tags == {"test": "value"}
                mock_create_graph.assert_called_once_with({"test": "value"})

    def test_rag_msg_worker_execute_with_default_tags(self) -> None:
        """Test _execute method with default tags."""
        worker = RagMsgWorker()

        mock_llm = Mock(spec=ChatOpenAI)
        mock_state = Mock(spec=MessageState)
        mock_state.bot_config = Mock()
        mock_state.bot_config.llm_config = Mock()
        mock_state.bot_config.llm_config.llm_provider = "openai"
        mock_state.bot_config.llm_config.model_type_or_path = "gpt-4"

        mock_graph = Mock()
        mock_graph.invoke.return_value = {"result": "success"}

        with patch(
            "arklex.env.workers.rag_message_worker.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_provider_map.get.return_value = Mock(return_value=mock_llm)

            with patch.object(worker, "_create_action_graph") as mock_create_graph:
                mock_create_graph.return_value = Mock()
                mock_create_graph.return_value.compile.return_value = mock_graph

                result = worker._execute(mock_state)

                assert result == {"result": "success"}
                assert worker.tags == {}

    def test_rag_msg_worker_execute_with_custom_provider(self) -> None:
        """Test _execute method with custom LLM provider."""
        worker = RagMsgWorker()

        mock_llm = Mock(spec=ChatOpenAI)
        mock_state = Mock(spec=MessageState)
        mock_state.bot_config = Mock()
        mock_state.bot_config.llm_config = Mock()
        mock_state.bot_config.llm_config.llm_provider = "anthropic"
        mock_state.bot_config.llm_config.model_type_or_path = "claude-3"

        mock_graph = Mock()
        mock_graph.invoke.return_value = {"result": "success"}

        with patch(
            "arklex.env.workers.rag_message_worker.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_provider_map.get.return_value = Mock(return_value=mock_llm)

            with patch.object(worker, "_create_action_graph") as mock_create_graph:
                mock_create_graph.return_value = Mock()
                mock_create_graph.return_value.compile.return_value = mock_graph

                result = worker._execute(mock_state)

                assert result == {"result": "success"}
                mock_provider_map.get.assert_called_once_with("anthropic", ChatOpenAI)

    def test_rag_msg_worker_choose_retriever_logging(self) -> None:
        """Test that _choose_retriever logs appropriately."""
        worker = RagMsgWorker()
        worker.llm = Mock(spec=BaseChatModel)

        mock_llm = Mock()
        mock_llm.invoke.return_value = "yes"
        worker.llm.__or__ = lambda self, other: mock_llm

        mock_state = Mock(spec=MessageState)
        mock_state.bot_config = Mock()
        mock_state.user_message = Mock()
        mock_state.user_message.history = "test history"

        with patch(
            "arklex.env.workers.rag_message_worker.load_prompts"
        ) as mock_load_prompts:
            mock_load_prompts.return_value = {
                "retrieval_needed_prompt": "Test prompt {formatted_chat}"
            }

            with patch("arklex.env.workers.rag_message_worker.log_context") as mock_log:
                worker._choose_retriever(mock_state)

                # Verify logging calls
                assert mock_log.info.call_count == 2

    def test_rag_msg_worker_create_action_graph_with_partial(self) -> None:
        """Test _create_action_graph method with partial function binding."""
        worker = RagMsgWorker()
        tags = {"test": "value"}

        with (
            patch("arklex.env.workers.rag_message_worker.RetrieveEngine"),
            patch(
                "arklex.env.workers.rag_message_worker.MessageWorker"
            ) as mock_message_worker,
        ):
            mock_message_worker.return_value = Mock()

            result = worker._create_action_graph(tags)

            assert isinstance(result, StateGraph)
            # Verify that the method creates a StateGraph with the expected structure
            # The partial function is created but not executed during graph creation

    def test_rag_msg_worker_choose_retriever_edge_cases(self) -> None:
        """Test _choose_retriever method with edge case responses."""
        worker = RagMsgWorker()
        worker.llm = Mock(spec=BaseChatModel)

        test_cases = [
            ("YES", "retriever"),
            ("Yes", "retriever"),
            ("yes", "retriever"),
            ("NO", "message_worker"),
            ("No", "message_worker"),
            ("no", "message_worker"),
            ("maybe", "message_worker"),
            ("", "message_worker"),
            ("I don't know", "message_worker"),
        ]

        mock_state = Mock(spec=MessageState)
        mock_state.bot_config = Mock()
        mock_state.user_message = Mock()
        mock_state.user_message.history = "test history"

        with patch(
            "arklex.env.workers.rag_message_worker.load_prompts"
        ) as mock_load_prompts:
            mock_load_prompts.return_value = {
                "retrieval_needed_prompt": "Test prompt {formatted_chat}"
            }

            for response, expected in test_cases:
                mock_llm = Mock()
                mock_llm.invoke.return_value = response
                worker.llm.__or__ = lambda self, other, llm=mock_llm: llm

                result = worker._choose_retriever(mock_state)
                assert result == expected, (
                    f"Expected {expected} for response '{response}'"
                )


class TestFaissRAGWorker:
    """Test cases for FaissRAGWorker."""

    def test_faiss_rag_worker_init(self) -> None:
        """Test FaissRAGWorker initialization."""
        worker = FaissRAGWorker()
        assert worker.stream_response is True
        assert hasattr(worker, "action_graph")

    def test_faiss_rag_worker_with_custom_config(self) -> None:
        """Test FaissRAGWorker initialization with custom stream configuration."""
        worker = FaissRAGWorker(stream_response=False)
        assert worker.stream_response is False
        assert hasattr(worker, "action_graph")

    def test_faiss_rag_worker_choose_tool_generator(self) -> None:
        """Test FaissRAGWorker choose_tool_generator method."""
        worker = FaissRAGWorker()
        mock_state = Mock(spec=MessageState)

        # Test with streaming enabled and state is stream
        mock_state.is_stream = True
        result = worker.choose_tool_generator(mock_state)
        assert result == "stream_tool_generator"

        # Test with streaming enabled but state is not stream
        mock_state.is_stream = False
        result = worker.choose_tool_generator(mock_state)
        assert result == "tool_generator"

        # Test with streaming disabled
        worker.stream_response = False
        result = worker.choose_tool_generator(mock_state)
        assert result == "tool_generator"


class TestMilvusRAGWorker:
    """Test cases for MilvusRAGWorker."""

    def test_milvus_rag_worker_init(self) -> None:
        """Test MilvusRAGWorker initialization."""
        worker = MilvusRAGWorker()
        assert worker.stream_response is True
        assert worker.tags == {}
        assert hasattr(worker, "action_graph")

    def test_milvus_rag_worker_with_custom_config(self) -> None:
        """Test MilvusRAGWorker initialization with custom configuration."""
        worker = MilvusRAGWorker(stream_response=False)
        assert worker.stream_response is False
        assert worker.tags == {}

    def test_milvus_rag_worker_choose_tool_generator(self) -> None:
        """Test MilvusRAGWorker choose_tool_generator method."""
        worker = MilvusRAGWorker()
        mock_state = Mock(spec=MessageState)

        # Test with streaming enabled and state is stream
        mock_state.is_stream = True
        result = worker.choose_tool_generator(mock_state)
        assert result == "stream_tool_generator"

        # Test with streaming enabled but state is not stream
        mock_state.is_stream = False
        result = worker.choose_tool_generator(mock_state)
        assert result == "tool_generator"

        # Test with streaming disabled
        worker.stream_response = False
        result = worker.choose_tool_generator(mock_state)
        assert result == "tool_generator"


class TestRAGMessageWorker:
    """Test cases for RAGMessageWorker."""

    def test_rag_message_worker_init(self) -> None:
        """Test RAGMessageWorker initialization."""
        worker = RAGMessageWorker()
        assert hasattr(worker, "model_config")
        assert hasattr(worker, "llm")
        assert worker.model_config == {}

    def test_rag_message_worker_with_custom_config(self) -> None:
        """Test RAGMessageWorker initialization with custom configuration."""
        custom_config = {
            "llm_provider": "openai",
            "model_type_or_path": "gpt-3.5-turbo",
            "api_key": "test-api-key",
        }
        worker = RAGMessageWorker(model_config=custom_config)
        assert worker.model_config == custom_config

    def test_rag_message_worker_with_llm_initialization(self) -> None:
        """Test RAGMessageWorker initialization with LLM setup."""
        custom_config = {
            "llm_provider": "openai",
            "model_type_or_path": "gpt-3.5-turbo",
        }

        with patch(
            "arklex.env.workers.rag_message_worker.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_provider_map.get.return_value = mock_llm_class
            mock_llm_class.return_value = Mock(spec=BaseChatModel)

            worker = RAGMessageWorker(model_config=custom_config)

            assert worker.llm is not None
            mock_provider_map.get.assert_called_once_with("openai", ChatOpenAI)
            mock_llm_class.assert_called_once_with(model="gpt-3.5-turbo")

    def test_rag_message_worker_search_documents(self) -> None:
        """Test RAGMessageWorker search_documents method."""
        worker = RAGMessageWorker()
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        mock_orchestrator_message.attribute = {"query": "test query"}
        mock_state.orchestrator_message = mock_orchestrator_message

        result = worker.search_documents(mock_state)

        assert "test query" in result

    def test_rag_message_worker_search_documents_no_query(self) -> None:
        """Test RAGMessageWorker search_documents method with no query."""
        worker = RAGMessageWorker()
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        mock_orchestrator_message.attribute = {}
        mock_state.orchestrator_message = mock_orchestrator_message

        result = worker.search_documents(mock_state)

        assert "No query provided" in result

    def test_rag_message_worker_search_documents_no_orchestrator_message(self) -> None:
        """Test RAGMessageWorker search_documents method with no orchestrator message."""
        worker = RAGMessageWorker()
        mock_state = Mock(spec=MessageState)
        mock_state.orchestrator_message = None

        result = worker.search_documents(mock_state)

        assert "No query provided" in result

    def test_rag_message_worker_search_documents_no_attribute(self) -> None:
        """Test RAGMessageWorker search_documents method with no attribute."""
        worker = RAGMessageWorker()
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        mock_orchestrator_message.attribute = None
        mock_state.orchestrator_message = mock_orchestrator_message

        result = worker.search_documents(mock_state)

        assert "No query provided" in result

    def test_rag_message_worker_generate_response(self) -> None:
        """Test RAGMessageWorker generate_response method."""
        worker = RAGMessageWorker()
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        mock_orchestrator_message.attribute = {"query": "test query"}
        mock_state.orchestrator_message = mock_orchestrator_message

        result = worker.generate_response(mock_state)

        assert "test query" in result

    def test_rag_message_worker_generate_response_no_query(self) -> None:
        """Test RAGMessageWorker generate_response method with no query."""
        worker = RAGMessageWorker()
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        mock_orchestrator_message.attribute = {}
        mock_state.orchestrator_message = mock_orchestrator_message

        result = worker.generate_response(mock_state)

        assert "No query provided" in result

    def test_rag_message_worker_generate_response_no_orchestrator_message(self) -> None:
        """Test RAGMessageWorker generate_response method with no orchestrator message."""
        worker = RAGMessageWorker()
        mock_state = Mock(spec=MessageState)
        mock_state.orchestrator_message = None

        result = worker.generate_response(mock_state)

        assert "No query provided" in result

    def test_rag_message_worker_generate_response_no_attribute(self) -> None:
        """Test RAGMessageWorker generate_response method with no attribute."""
        worker = RAGMessageWorker()
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        mock_orchestrator_message.attribute = None
        mock_state.orchestrator_message = mock_orchestrator_message

        result = worker.generate_response(mock_state)

        assert "No query provided" in result

    def test_rag_message_worker_execute_with_bot_config(self) -> None:
        """Test RAGMessageWorker _execute method with bot config."""
        worker = RAGMessageWorker()
        mock_state = Mock(spec=MessageState)
        mock_state.bot_config = Mock()
        mock_state.bot_config.llm_config = Mock()
        mock_state.bot_config.llm_config.llm_provider = "openai"
        mock_state.bot_config.llm_config.model_type_or_path = "gpt-4"

        mock_orchestrator_message = Mock()
        mock_orchestrator_message.attribute = {"query": "test query"}
        mock_state.orchestrator_message = mock_orchestrator_message

        with patch(
            "arklex.env.workers.rag_message_worker.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_provider_map.get.return_value = mock_llm_class
            mock_llm_class.return_value = Mock(spec=BaseChatModel)

            result = worker._execute(mock_state)

            assert "response" in result
            assert "search_results" in result
            assert "test query" in result["response"]
            assert "test query" in result["search_results"]

            # Verify LLM was initialized
            mock_provider_map.get.assert_called_once_with("openai", ChatOpenAI)
            mock_llm_class.assert_called_once_with(model="gpt-4")

    def test_rag_message_worker_execute_without_bot_config(self) -> None:
        """Test RAGMessageWorker _execute method without bot config."""
        worker = RAGMessageWorker()
        mock_state = Mock(spec=MessageState)
        mock_state.bot_config = None

        mock_orchestrator_message = Mock()
        mock_orchestrator_message.attribute = {"query": "test query"}
        mock_state.orchestrator_message = mock_orchestrator_message

        result = worker._execute(mock_state)

        assert "response" in result
        assert "search_results" in result
        assert "test query" in result["response"]
        assert "test query" in result["search_results"]
        assert worker.llm is None  # Should not initialize LLM without bot_config

    def test_rag_message_worker_execute_with_existing_llm(self) -> None:
        """Test RAGMessageWorker _execute method with existing LLM."""
        worker = RAGMessageWorker()
        worker.llm = Mock(spec=BaseChatModel)

        mock_state = Mock(spec=MessageState)
        mock_state.bot_config = Mock()
        mock_state.bot_config.llm_config = Mock()
        mock_state.bot_config.llm_config.llm_provider = "openai"
        mock_state.bot_config.llm_config.model_type_or_path = "gpt-4"

        mock_orchestrator_message = Mock()
        mock_orchestrator_message.attribute = {"query": "test query"}
        mock_state.orchestrator_message = mock_orchestrator_message

        original_llm = worker.llm

        result = worker._execute(mock_state)

        assert "response" in result
        assert "search_results" in result
        # Should not reinitialize LLM if it already exists
        assert worker.llm is original_llm

    def test_rag_message_worker_execute_updates_state(self) -> None:
        """Test that RAGMessageWorker _execute method updates the message state."""
        worker = RAGMessageWorker()
        mock_state = Mock(spec=MessageState)
        mock_state.bot_config = Mock()
        mock_state.bot_config.llm_config = Mock()
        mock_state.bot_config.llm_config.llm_provider = "openai"
        mock_state.bot_config.llm_config.model_type_or_path = "gpt-4"

        mock_orchestrator_message = Mock()
        mock_orchestrator_message.attribute = {"query": "test query"}
        mock_state.orchestrator_message = mock_orchestrator_message

        with patch(
            "arklex.env.workers.rag_message_worker.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_provider_map.get.return_value = mock_llm_class
            mock_llm_class.return_value = Mock(spec=BaseChatModel)

            result = worker._execute(mock_state)

            # Verify state was updated
            assert mock_state.response == result["response"]
            assert mock_state.message_flow == result["search_results"]


class TestRAGWorkersIntegration:
    """Integration tests for RAG workers."""

    def test_rag_workers_description_comparison(self) -> None:
        """Test that RAG workers have appropriate descriptions."""
        # Create workers
        faiss_worker = FaissRAGWorker()
        milvus_worker = MilvusRAGWorker()
        rag_msg_worker = RagMsgWorker()
        rag_message_worker = RAGMessageWorker()

        # Test that all workers have descriptions
        assert isinstance(faiss_worker.description, str)
        assert isinstance(milvus_worker.description, str)
        assert isinstance(rag_msg_worker.description, str)
        assert isinstance(rag_message_worker.description, str)

        # Test that descriptions are not empty
        assert len(faiss_worker.description) > 0
        assert len(milvus_worker.description) > 0
        assert len(rag_msg_worker.description) > 0
        assert len(rag_message_worker.description) > 0

        # Test that descriptions contain relevant keywords
        assert "internal" in faiss_worker.description.lower()
        assert "internal" in milvus_worker.description.lower()
        assert "combination" in rag_msg_worker.description.lower()
        assert "rag" in rag_message_worker.description.lower()

    def test_rag_workers_initialization_consistency(self) -> None:
        """Test that all RAG workers initialize consistently."""
        # Create workers
        faiss_worker = FaissRAGWorker()
        milvus_worker = MilvusRAGWorker()
        rag_msg_worker = RagMsgWorker()
        rag_message_worker = RAGMessageWorker()

        # Test that all workers have descriptions
        assert hasattr(faiss_worker, "description")
        assert hasattr(milvus_worker, "description")
        assert hasattr(rag_msg_worker, "description")
        assert hasattr(rag_message_worker, "description")

        # Test that descriptions are not empty
        assert len(faiss_worker.description) > 0
        assert len(milvus_worker.description) > 0
        assert len(rag_msg_worker.description) > 0
        assert len(rag_message_worker.description) > 0

    def test_rag_workers_method_availability(self) -> None:
        """Test that RAG workers have the expected methods."""
        # Create workers
        faiss_worker = FaissRAGWorker()
        milvus_worker = MilvusRAGWorker()
        rag_msg_worker = RagMsgWorker()
        rag_message_worker = RAGMessageWorker()

        # Test that all workers have _execute method
        assert hasattr(faiss_worker, "_execute")
        assert hasattr(milvus_worker, "_execute")
        assert hasattr(rag_msg_worker, "_execute")
        assert hasattr(rag_message_worker, "_execute")

        # Test that RAGMessageWorker has search_documents and generate_response methods
        assert hasattr(rag_message_worker, "search_documents")
        assert hasattr(rag_message_worker, "generate_response")

        # Test that RagMsgWorker has specific methods
        assert hasattr(rag_msg_worker, "_choose_retriever")
        assert hasattr(rag_msg_worker, "_create_action_graph")

        # Test that FaissRAGWorker and MilvusRAGWorker have choose_tool_generator method
        assert hasattr(faiss_worker, "choose_tool_generator")
        assert hasattr(milvus_worker, "choose_tool_generator")


class TestRAGWorkersErrorHandling:
    """Test error handling for RAG workers."""

    def test_faiss_rag_worker_init_error(self) -> None:
        """Test FaissRAGWorker initialization error handling."""
        # This should not raise any errors
        worker = FaissRAGWorker()
        assert worker is not None

    def test_milvus_rag_worker_init_error(self) -> None:
        """Test MilvusRAGWorker initialization error handling."""
        # This should not raise any errors
        worker = MilvusRAGWorker()
        assert worker is not None

    def test_rag_msg_worker_init_error(self) -> None:
        """Test RagMsgWorker initialization error handling."""
        # This should not raise any errors
        worker = RagMsgWorker()
        assert worker is not None

    def test_rag_message_worker_init_error(self) -> None:
        """Test RAGMessageWorker initialization error handling."""
        # This should not raise any errors
        worker = RAGMessageWorker()
        assert worker is not None

    def test_rag_workers_search_error_handling(self) -> None:
        """Test RAG workers search error handling."""
        # Test that workers handle missing orchestrator_message gracefully
        rag_message_worker = RAGMessageWorker()
        mock_state = Mock(spec=MessageState)
        mock_state.orchestrator_message = None

        # This should not raise an error
        result = rag_message_worker.search_documents(mock_state)
        assert "No query provided" in result

        # Test with orchestrator_message but no attribute
        mock_state.orchestrator_message = Mock()
        mock_state.orchestrator_message.attribute = None

        result = rag_message_worker.search_documents(mock_state)
        assert "No query provided" in result

        # Test with orchestrator_message and empty attribute dict
        mock_state.orchestrator_message.attribute = {}

        result = rag_message_worker.search_documents(mock_state)
        assert "No query provided" in result

    def test_rag_msg_worker_choose_retriever_error_handling(self) -> None:
        """Test RagMsgWorker _choose_retriever error handling."""
        worker = RagMsgWorker()
        worker.llm = Mock(spec=BaseChatModel)

        mock_state = Mock(spec=MessageState)
        mock_state.bot_config = Mock()
        mock_state.user_message = Mock()
        mock_state.user_message.history = "test history"

        # Test with missing prompts
        with patch(
            "arklex.env.workers.rag_message_worker.load_prompts"
        ) as mock_load_prompts:
            mock_load_prompts.side_effect = RuntimeError("Prompts not found")

            with pytest.raises(RuntimeError):
                worker._choose_retriever(mock_state)

    def test_rag_msg_worker_create_action_graph_error_handling(self) -> None:
        """Test RagMsgWorker _create_action_graph error handling."""
        worker = RagMsgWorker()
        tags = {"test": "value"}

        # Test with missing dependencies - this should not raise an error
        # because the function is not called during graph creation
        with patch(
            "arklex.env.workers.rag_message_worker.RetrieveEngine"
        ) as mock_retrieve_engine:
            mock_retrieve_engine.milvus_retrieve.side_effect = RuntimeError(
                "RetrieveEngine error"
            )

            # The graph creation should succeed because the function is not called
            # during graph creation, only when the graph is executed
            result = worker._create_action_graph(tags)
            assert isinstance(result, StateGraph)

    def test_rag_message_worker_execute_error_handling(self) -> None:
        """Test RAGMessageWorker _execute error handling."""
        worker = RAGMessageWorker()
        mock_state = Mock(spec=MessageState)
        mock_state.bot_config = Mock()
        mock_state.bot_config.llm_config = Mock()
        mock_state.bot_config.llm_config.llm_provider = "invalid_provider"
        mock_state.bot_config.llm_config.model_type_or_path = "invalid_model"

        mock_orchestrator_message = Mock()
        mock_orchestrator_message.attribute = {"query": "test query"}
        mock_state.orchestrator_message = mock_orchestrator_message

        # Should handle invalid provider gracefully
        result = worker._execute(mock_state)
        assert "response" in result
        assert "search_results" in result


class TestRagMsgWorkerKwargs:
    """Test cases for RagMsgWorkerKwargs TypedDict."""

    def test_rag_msg_worker_kwargs_structure(self) -> None:
        """Test RagMsgWorkerKwargs structure."""
        kwargs: RagMsgWorkerKwargs = {"tags": {"test": "value"}}
        assert "tags" in kwargs
        assert kwargs["tags"]["test"] == "value"

    def test_rag_msg_worker_kwargs_optional_tags(self) -> None:
        """Test RagMsgWorkerKwargs with optional tags."""
        kwargs: RagMsgWorkerKwargs = {}
        assert "tags" not in kwargs
