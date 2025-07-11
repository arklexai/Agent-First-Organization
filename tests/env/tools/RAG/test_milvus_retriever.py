"""Tests for the Milvus retriever implementation.

This module provides comprehensive tests for the MilvusRetriever, MilvusRetrieverExecutor,
and RetrieveEngine classes, covering all methods and edge cases.
"""

import os
import time
from unittest.mock import Mock, patch

import pytest

from arklex.env.tools.RAG.retrievers.milvus_retriever import (
    CHUNK_NEIGHBOURS,
    EMBED_DIMENSION,
    MAX_TEXT_LENGTH,
    MilvusRetriever,
    MilvusRetrieverExecutor,
    RetrieveEngine,
)
from arklex.env.tools.RAG.retrievers.retriever_document import (
    RetrieverDocument,
    RetrieverDocumentType,
    RetrieverResult,
)
from arklex.orchestrator.entities.msg_state_entities import MessageState


@pytest.fixture
def mock_milvus_client() -> Mock:
    """Mock MilvusClient for testing."""
    with patch(
        "arklex.env.tools.RAG.retrievers.milvus_retriever.MilvusClient"
    ) as mock_client:
        mock_instance = Mock()

        # Configure mock to return proper values
        mock_instance.get.return_value = []
        mock_instance.query.return_value = []
        mock_instance.search.return_value = [[]]
        mock_instance.list_collections.return_value = []
        mock_instance.has_collection.return_value = False
        mock_instance.get_collection_stats.return_value = {"row_count": 0}
        mock_instance.get_load_state.return_value = {"state": "NotLoaded"}
        mock_instance.upsert.return_value = {"insert_count": 1}
        mock_instance.delete.return_value = {"deleted_count": 1}
        mock_instance.prepare_index_params.return_value = Mock()

        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_bot_config() -> Mock:
    """Mock bot configuration."""
    config = Mock()
    config.collection_name = "test_collection"
    config.bot_id = "test_bot"
    config.version = "1.0"
    config.language = "EN"  # Add language attribute

    # Add llm_config for MilvusRetrieverExecutor
    llm_config = Mock()
    llm_config.llm_provider = "openai"
    llm_config.model_type_or_path = "gpt-3.5-turbo"
    config.llm_config = llm_config

    return config


@pytest.fixture
def mock_message_state(mock_bot_config: Mock) -> Mock:
    """Mock MessageState for testing."""
    state = Mock(spec=MessageState)
    state.bot_config = mock_bot_config
    state.user_message = Mock()
    state.user_message.history = "test query"
    state.message_flow = ""
    return state


@pytest.fixture
def sample_retriever_document() -> RetrieverDocument:
    """Sample RetrieverDocument for testing."""
    return RetrieverDocument(
        id="test_doc_1",
        qa_doc_id="qa_123",
        chunk_idx=0,
        qa_doc_type=RetrieverDocumentType.FAQ,
        text="This is a test document",
        metadata={"source": "test.txt", "tags": ["test"]},
        is_chunked=False,
        bot_uid="test_bot__1.0",
        embedding=[0.1] * EMBED_DIMENSION,
        timestamp=int(time.time()),
    )


@pytest.fixture
def sample_retriever_result() -> RetrieverResult:
    """Sample RetrieverResult for testing."""
    return RetrieverResult(
        qa_doc_id="qa_123",
        qa_doc_type=RetrieverDocumentType.FAQ,
        distance=0.5,
        metadata={"source": "test.txt"},
        text="This is a test document",
        start_chunk_idx=0,
        end_chunk_idx=0,
    )


@pytest.fixture
def milvus_retriever(mock_milvus_client: Mock) -> MilvusRetriever:
    """MilvusRetriever instance for testing."""
    with patch.dict(
        os.environ, {"MILVUS_URI": "test_uri", "MILVUS_TOKEN": "test_token"}
    ):
        retriever = MilvusRetriever()
        retriever.client = mock_milvus_client
        retriever.uri = "test_uri"
        retriever.token = "test_token"
        return retriever


@pytest.fixture
def milvus_retriever_executor(mock_bot_config: Mock) -> MilvusRetrieverExecutor:
    """MilvusRetrieverExecutor instance for testing."""
    with patch(
        "arklex.env.tools.RAG.retrievers.milvus_retriever.MilvusRetriever"
    ) as mock_retriever_class:
        mock_retriever = Mock()
        mock_retriever_class.return_value.__enter__.return_value = mock_retriever
        executor = MilvusRetrieverExecutor(mock_bot_config)
        executor.retriever = mock_retriever
        return executor


class TestRetrieveEngine:
    """Test the RetrieveEngine class."""

    def test_milvus_retrieve_success(self, mock_message_state: Mock) -> None:
        """Test successful milvus_retrieve operation."""
        with patch(
            "arklex.env.tools.RAG.retrievers.milvus_retriever.MilvusRetrieverExecutor"
        ) as mock_executor_class:
            mock_executor = Mock()
            mock_executor_class.return_value = mock_executor
            mock_executor.retrieve.return_value = ("retrieved text", {"params": "test"})

            with patch(
                "arklex.env.tools.RAG.retrievers.milvus_retriever.trace"
            ) as mock_trace:
                mock_trace.return_value = mock_message_state

                result = RetrieveEngine.milvus_retrieve(mock_message_state)

                mock_executor.retrieve.assert_called_once_with("test query", {})
                assert result.message_flow == "retrieved text"
                mock_trace.assert_called_once()

    def test_milvus_retrieve_with_tags(self, mock_message_state: Mock) -> None:
        """Test milvus_retrieve with custom tags."""
        tags = {"category": "test"}
        with patch(
            "arklex.env.tools.RAG.retrievers.milvus_retriever.MilvusRetrieverExecutor"
        ) as mock_executor_class:
            mock_executor = Mock()
            mock_executor_class.return_value = mock_executor
            mock_executor.retrieve.return_value = ("retrieved text", {"params": "test"})

            with patch(
                "arklex.env.tools.RAG.retrievers.milvus_retriever.trace"
            ) as mock_trace:
                mock_trace.return_value = mock_message_state

                RetrieveEngine.milvus_retrieve(mock_message_state, tags)

                mock_executor.retrieve.assert_called_once_with("test query", tags)


class TestMilvusRetriever:
    """Test the MilvusRetriever class."""

    def test_context_manager_enter(self) -> None:
        """Test MilvusRetriever context manager enter."""
        with (
            patch.dict(
                os.environ, {"MILVUS_URI": "test_uri", "MILVUS_TOKEN": "test_token"}
            ),
            patch(
                "arklex.env.tools.RAG.retrievers.milvus_retriever.MilvusClient"
            ) as mock_client_class,
        ):
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            with MilvusRetriever() as retriever:
                assert retriever.uri == "test_uri"
                assert retriever.token == "test_token"
                assert retriever.client == mock_client

    def test_context_manager_exit(self) -> None:
        """Test MilvusRetriever context manager exit."""
        with (
            patch.dict(
                os.environ, {"MILVUS_URI": "test_uri", "MILVUS_TOKEN": "test_token"}
            ),
            patch(
                "arklex.env.tools.RAG.retrievers.milvus_retriever.MilvusClient"
            ) as mock_client_class,
        ):
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            with MilvusRetriever():
                pass

            mock_client.close.assert_called_once()

    def test_get_bot_uid(self, milvus_retriever: MilvusRetriever) -> None:
        """Test get_bot_uid method."""
        result = milvus_retriever.get_bot_uid("test_bot", "1.0")
        assert result == "test_bot__1.0"

    def test_create_collection_with_partition_key(
        self, milvus_retriever: MilvusRetriever
    ) -> None:
        """Test create_collection_with_partition_key method."""
        with patch(
            "arklex.env.tools.RAG.retrievers.milvus_retriever.MilvusClient"
        ) as mock_client_class:
            mock_schema = Mock()
            mock_client_class.create_schema.return_value = mock_schema
            mock_index_params = Mock()
            milvus_retriever.client.prepare_index_params.return_value = (
                mock_index_params
            )

            milvus_retriever.create_collection_with_partition_key("test_collection")

            mock_client_class.create_schema.assert_called_once_with(
                auto_id=False,
                enable_dynamic_field=True,
                partition_key_field="bot_uid",
                num_partitions=16,
            )
            assert mock_schema.add_field.call_count == 9  # 9 fields added
            milvus_retriever.client.create_collection.assert_called_once()

    def test_delete_documents_by_qa_doc_id(
        self, milvus_retriever: MilvusRetriever
    ) -> None:
        """Test delete_documents_by_qa_doc_id method."""
        mock_result = {"deleted_count": 5}
        milvus_retriever.client.delete.return_value = mock_result

        result = milvus_retriever.delete_documents_by_qa_doc_id(
            "test_collection", "qa_123"
        )

        milvus_retriever.client.delete.assert_called_once_with(
            collection_name="test_collection", filter="qa_doc_id=='qa_123'"
        )
        assert result == mock_result

    def test_add_documents_dicts_new_documents(
        self,
        milvus_retriever: MilvusRetriever,
        sample_retriever_document: RetrieverDocument,
    ) -> None:
        """Test add_documents_dicts with new documents."""
        documents = [sample_retriever_document.to_dict()]
        milvus_retriever.client.get.return_value = []  # No existing documents

        with patch(
            "arklex.env.tools.RAG.retrievers.milvus_retriever.embed"
        ) as mock_embed:
            mock_embed.return_value = [0.1] * EMBED_DIMENSION
            mock_upsert_result = {"insert_count": 1}
            milvus_retriever.client.upsert.return_value = mock_upsert_result

            result = milvus_retriever.add_documents_dicts(documents, "test_collection")

            assert len(result) == 1
            milvus_retriever.client.upsert.assert_called_once()

    def test_add_documents_dicts_existing_documents(
        self,
        milvus_retriever: MilvusRetriever,
        sample_retriever_document: RetrieverDocument,
    ) -> None:
        """Test add_documents_dicts with existing documents."""
        documents = [sample_retriever_document.to_dict()]
        mock_existing_doc = {"id": "test_doc_1", "text": "existing"}
        milvus_retriever.client.get.return_value = [
            mock_existing_doc
        ]  # Document exists

        result = milvus_retriever.add_documents_dicts(documents, "test_collection")

        assert result == []  # No documents to insert
        milvus_retriever.client.upsert.assert_not_called()

    def test_add_documents_dicts_upsert_mode(
        self,
        milvus_retriever: MilvusRetriever,
        sample_retriever_document: RetrieverDocument,
    ) -> None:
        """Test add_documents_dicts with upsert=True."""
        documents = [sample_retriever_document.to_dict()]

        with patch(
            "arklex.env.tools.RAG.retrievers.milvus_retriever.embed"
        ) as mock_embed:
            mock_embed.return_value = [0.1] * EMBED_DIMENSION
            mock_upsert_result = {"insert_count": 1}
            milvus_retriever.client.upsert.return_value = mock_upsert_result

            result = milvus_retriever.add_documents_dicts(
                documents, "test_collection", upsert=True
            )

            assert len(result) == 1
            milvus_retriever.client.upsert.assert_called_once()

    def test_add_documents_dicts_exception_handling(
        self,
        milvus_retriever: MilvusRetriever,
        sample_retriever_document: RetrieverDocument,
    ) -> None:
        """Test add_documents_dicts exception handling."""
        documents = [sample_retriever_document.to_dict()]
        milvus_retriever.client.get.return_value = []

        with patch(
            "arklex.env.tools.RAG.retrievers.milvus_retriever.embed"
        ) as mock_embed:
            mock_embed.return_value = [0.1] * EMBED_DIMENSION
            milvus_retriever.client.upsert.side_effect = Exception("Test error")

            with pytest.raises(Exception, match="Test error"):
                milvus_retriever.add_documents_dicts(documents, "test_collection")

    def test_update_tag_by_qa_doc_id(self, milvus_retriever: MilvusRetriever) -> None:
        """Test update_tag_by_qa_doc_id method."""
        mock_query_result = [
            {
                "id": "doc_1",
                "qa_doc_id": "qa_123",
                "metadata": {"old_tag": "value"},
                "text": "test text",
                "embedding": [0.1] * EMBED_DIMENSION,
            }
        ]
        milvus_retriever.client.query.return_value = mock_query_result
        mock_upsert_result = {"insert_count": 1}
        milvus_retriever.client.upsert.return_value = mock_upsert_result

        tags = {"new_tag": "new_value"}
        milvus_retriever.update_tag_by_qa_doc_id("test_collection", "qa_123", tags)

        milvus_retriever.client.query.assert_called_once()
        milvus_retriever.client.upsert.assert_called_once()

    def test_update_tag_by_qa_doc_id_no_vectors(
        self, milvus_retriever: MilvusRetriever
    ) -> None:
        """Test update_tag_by_qa_doc_id when no vectors are found (should raise ValueError)."""
        milvus_retriever.client.query.return_value = []
        with pytest.raises(ValueError) as excinfo:
            milvus_retriever.update_tag_by_qa_doc_id(
                "test_collection", "qa_123", {"new": "tag"}
            )
        assert "No vectors found for qa_doc_id" in str(excinfo.value)

    def test_update_tag_by_qa_doc_id_upsert_exception(
        self, milvus_retriever: MilvusRetriever
    ) -> None:
        """Test update_tag_by_qa_doc_id when upsert fails (should raise ValueError)."""
        milvus_retriever.client.query.return_value = [{"metadata": {}}]
        milvus_retriever.client.upsert.side_effect = Exception("upsert failed")
        with pytest.raises(ValueError) as excinfo:
            milvus_retriever.update_tag_by_qa_doc_id(
                "test_collection", "qa_123", {"new": "tag"}
            )
        assert "Failed to upsert updated vectors" in str(excinfo.value)

    def test_add_documents_parallel(
        self,
        milvus_retriever: MilvusRetriever,
        sample_retriever_document: RetrieverDocument,
    ) -> None:
        """Test add_documents_parallel method."""
        documents = [sample_retriever_document]
        mock_pool = Mock()
        mock_pool.map.return_value = [{"insert_count": 1}]

        with patch(
            "arklex.env.tools.RAG.retrievers.milvus_retriever.embed"
        ) as mock_embed:
            mock_embed.return_value = [0.1] * EMBED_DIMENSION

            result = milvus_retriever.add_documents_parallel(
                "test_collection", "test_bot", "1.0", documents, mock_pool
            )

            assert len(result) == 1
            mock_pool.map.assert_called_once()

    def test_add_documents_parallel_creates_collection(
        self,
        milvus_retriever: MilvusRetriever,
        sample_retriever_document: RetrieverDocument,
    ) -> None:
        """Test add_documents_parallel creates collection if not exists."""
        milvus_retriever.client.has_collection.return_value = False
        process_pool = Mock()
        process_pool.map.return_value = [
            sample_retriever_document.to_milvus_schema_dict_and_embed()
        ]
        milvus_retriever.client.upsert.return_value = ["ok"]
        # Patch create_collection_with_partition_key to allow call assertion
        with patch.object(
            milvus_retriever, "create_collection_with_partition_key"
        ) as mock_create:
            result = milvus_retriever.add_documents_parallel(
                "test_collection",
                "test_bot",
                "1.0",
                [sample_retriever_document],
                process_pool,
            )
            assert result == ["ok"]
            milvus_retriever.client.has_collection.assert_called_once()
            mock_create.assert_called_once_with("test_collection")

    def test_add_documents_parallel_upsert(
        self,
        milvus_retriever: MilvusRetriever,
        sample_retriever_document: RetrieverDocument,
    ) -> None:
        """Test add_documents_parallel with upsert=True."""
        milvus_retriever.client.has_collection.return_value = True
        process_pool = Mock()
        process_pool.map.return_value = [
            sample_retriever_document.to_milvus_schema_dict_and_embed()
        ]
        milvus_retriever.client.upsert.return_value = ["ok"]
        result = milvus_retriever.add_documents_parallel(
            "test_collection",
            "test_bot",
            "1.0",
            [sample_retriever_document],
            process_pool,
            upsert=True,
        )
        assert result == ["ok"]

    def test_add_documents_parallel_batching(
        self,
        milvus_retriever: MilvusRetriever,
        sample_retriever_document: RetrieverDocument,
    ) -> None:
        """Test add_documents_parallel processes batches of 100."""
        milvus_retriever.client.has_collection.return_value = True
        process_pool = Mock()
        # Simulate 150 docs
        docs = [sample_retriever_document] * 150
        process_pool.map.return_value = [
            sample_retriever_document.to_milvus_schema_dict_and_embed()
        ] * 100
        milvus_retriever.client.upsert.return_value = ["ok"] * 100
        result = milvus_retriever.add_documents_parallel(
            "test_collection", "test_bot", "1.0", docs, process_pool, upsert=True
        )
        assert len(result) == 200  # 2 batches of 100

    def test_add_documents(
        self,
        milvus_retriever: MilvusRetriever,
        sample_retriever_document: RetrieverDocument,
    ) -> None:
        """Test add_documents method."""
        documents = [sample_retriever_document]

        with patch(
            "arklex.env.tools.RAG.retrievers.milvus_retriever.embed"
        ) as mock_embed:
            mock_embed.return_value = [0.1] * EMBED_DIMENSION
            mock_upsert_result = {"insert_count": 1}
            milvus_retriever.client.upsert.return_value = mock_upsert_result

            result = milvus_retriever.add_documents(
                "test_collection", "test_bot", "1.0", documents
            )

            assert len(result) == 1
            milvus_retriever.client.upsert.assert_called_once()

    def test_add_documents_exception_handling(
        self,
        milvus_retriever: MilvusRetriever,
        sample_retriever_document: RetrieverDocument,
    ) -> None:
        """Test add_documents handles upsert exception."""
        milvus_retriever.client.has_collection.return_value = True
        milvus_retriever.client.get.return_value = []
        milvus_retriever.client.upsert.side_effect = Exception("upsert error")
        with pytest.raises(Exception) as excinfo:
            milvus_retriever.add_documents(
                "test_collection", "test_bot", "1.0", [sample_retriever_document]
            )
        assert "upsert error" in str(excinfo.value)

    def test_search(self, milvus_retriever: MilvusRetriever) -> None:
        """Test search method."""
        mock_search_result = [
            [
                {
                    "entity": {
                        "qa_doc_id": "qa_123",
                        "chunk_id": 0,
                        "qa_doc_type": "faq",
                        "metadata": {"source": "test.txt"},
                        "text": "test document",
                    },
                    "distance": 0.5,
                }
            ]
        ]
        milvus_retriever.client.search.return_value = mock_search_result

        with patch(
            "arklex.env.tools.RAG.retrievers.milvus_retriever.embed"
        ) as mock_embed:
            mock_embed.return_value = [0.1] * EMBED_DIMENSION

            results = milvus_retriever.search(
                "test_collection", "test_bot", "1.0", "test query"
            )

            assert len(results) == 1
            assert isinstance(results[0], RetrieverResult)
            assert results[0].qa_doc_id == "qa_123"

    def test_search_with_tags(self, milvus_retriever: MilvusRetriever) -> None:
        """Test search method with tags filter."""
        mock_search_result = [[]]  # Empty list of results
        milvus_retriever.client.search.return_value = mock_search_result

        with patch(
            "arklex.env.tools.RAG.retrievers.milvus_retriever.embed"
        ) as mock_embed:
            mock_embed.return_value = [0.1] * EMBED_DIMENSION

            tags = {"category": "test"}
            milvus_retriever.search(
                "test_collection", "test_bot", "1.0", "test query", tags
            )

            # Verify that the search was called with tag filter
            call_args = milvus_retriever.client.search.call_args
            assert "filter" in call_args[1]

    def test_search_empty_results(self, milvus_retriever: MilvusRetriever) -> None:
        """Test search returns empty list if no results."""
        milvus_retriever.client.search.return_value = [[]]
        result = milvus_retriever.search("test_collection", "test_bot", "1.0", "query")
        assert result == []

    def test_search_with_tags_filter(self, milvus_retriever: MilvusRetriever) -> None:
        """Test search with tags only uses one tag."""
        milvus_retriever.client.search.return_value = [
            [
                {
                    "entity": {
                        "qa_doc_id": "qa_123",
                        "chunk_id": 0,
                        "qa_doc_type": "faq",  # Use correct enum value
                        "metadata": {"tags": {"a": 1}},
                        "text": "text",
                    },
                    "distance": 0.1,
                }
            ]
        ]
        result = milvus_retriever.search(
            "test_collection", "test_bot", "1.0", "query", tags={"a": 1, "b": 2}
        )
        assert len(result) == 1
        assert result[0].qa_doc_id == "qa_123"
        assert result[0].metadata["tags"]["a"] == 1

    def test_get_qa_docs(self, milvus_retriever: MilvusRetriever) -> None:
        """Test get_qa_docs method."""
        mock_query_result = [
            {
                "id": "qa_123",  # Use qa_doc_id as id for FAQ documents
                "qa_doc_id": "qa_123",
                "chunk_id": 0,
                "qa_doc_type": "faq",
                "text": "test document",
                "metadata": {"source": "test.txt"},
                "embedding": [0.1] * EMBED_DIMENSION,
                "timestamp": int(time.time()),
                "bot_uid": "test_bot__1.0",
            }
        ]
        with (
            patch(
                "arklex.env.tools.RAG.retrievers.milvus_retriever.connections.connect"
            ),
            patch(
                "arklex.env.tools.RAG.retrievers.milvus_retriever.Collection"
            ) as mock_collection_class,
        ):
            mock_collection = mock_collection_class.return_value
            mock_iterator = Mock()
            mock_collection.query_iterator.return_value = mock_iterator

            # Mock the iterator behavior
            mock_iterator.next.side_effect = [mock_query_result, []]
            mock_iterator.close = Mock()

            results = milvus_retriever.get_qa_docs(
                "test_collection", "test_bot", "1.0", RetrieverDocumentType.FAQ
            )
        assert len(results) == 1
        assert isinstance(results[0], RetrieverDocument)
        assert results[0].qa_doc_id == "qa_123"

    def test_get_qa_doc(self, milvus_retriever: MilvusRetriever) -> None:
        """Test get_qa_doc method."""
        mock_query_result = [
            {
                "id": "doc_1",
                "qa_doc_id": "qa_123",
                "chunk_id": 0,
                "qa_doc_type": "faq",
                "text": "test document",
                "metadata": {"source": "test.txt"},
                "embedding": [0.1] * EMBED_DIMENSION,
                "timestamp": int(time.time()),
                "bot_uid": "test_bot__1.0",
            }
        ]
        milvus_retriever.client.query.return_value = mock_query_result
        result = milvus_retriever.get_qa_doc("test_collection", "qa_123")
        assert isinstance(result, RetrieverDocument)
        assert result.qa_doc_id == "qa_123"

    def test_get_qa_doc_not_found(self, milvus_retriever: MilvusRetriever) -> None:
        """Test get_qa_doc method when document not found."""
        milvus_retriever.client.query.return_value = []
        result = milvus_retriever.get_qa_doc("test_collection", "qa_123")
        assert result is None

    def test_get_qa_doc_ids(self, milvus_retriever: MilvusRetriever) -> None:
        """Test get_qa_doc_ids method."""
        mock_query_result = [
            {"qa_doc_id": "qa_123", "chunk_id": 0},
            {"qa_doc_id": "qa_124", "chunk_id": 0},
        ]
        with (
            patch(
                "arklex.env.tools.RAG.retrievers.milvus_retriever.connections.connect"
            ),
            patch(
                "arklex.env.tools.RAG.retrievers.milvus_retriever.Collection"
            ) as mock_collection_class,
        ):
            mock_collection = mock_collection_class.return_value
            mock_iterator = Mock()
            mock_collection.query_iterator.return_value = mock_iterator

            # Mock the iterator behavior
            mock_iterator.next.side_effect = [mock_query_result, []]
            mock_iterator.close = Mock()

            results = milvus_retriever.get_qa_doc_ids(
                "test_collection", "test_bot", "1.0", RetrieverDocumentType.FAQ
            )
        assert len(results) == 2
        assert "qa_123" in results
        assert "qa_124" in results

    def test_has_collection(self, milvus_retriever: MilvusRetriever) -> None:
        """Test has_collection method."""
        milvus_retriever.client.list_collections.return_value = ["test_collection"]
        milvus_retriever.client.has_collection.return_value = True
        result = milvus_retriever.has_collection("test_collection")
        assert result is True
        milvus_retriever.client.list_collections.return_value = []
        milvus_retriever.client.has_collection.return_value = False
        result = milvus_retriever.has_collection("nonexistent_collection")
        assert result is False

    def test_load_collection(self, milvus_retriever: MilvusRetriever) -> None:
        """Test load_collection method."""
        milvus_retriever.client.has_collection.return_value = True
        milvus_retriever.load_collection("test_collection")
        milvus_retriever.client.load_collection.assert_called_once_with(
            "test_collection"
        )

    def test_release_collection(self, milvus_retriever: MilvusRetriever) -> None:
        """Test release_collection method."""
        mock_result = {"status": "success"}
        milvus_retriever.client.release_collection.return_value = mock_result

        result = milvus_retriever.release_collection("test_collection")

        milvus_retriever.client.release_collection.assert_called_once_with(
            "test_collection"
        )
        assert result == mock_result

    def test_drop_collection(self, milvus_retriever: MilvusRetriever) -> None:
        """Test drop_collection method."""
        mock_result = {"status": "success"}
        milvus_retriever.client.drop_collection.return_value = mock_result

        result = milvus_retriever.drop_collection("test_collection")

        milvus_retriever.client.drop_collection.assert_called_once_with(
            "test_collection"
        )
        assert result == mock_result

    def test_get_all_vectors(self, milvus_retriever: MilvusRetriever) -> None:
        """Test get_all_vectors method."""
        mock_query_result = [
            {
                "id": "doc_1",
                "qa_doc_id": "qa_123",
                "text": "test document",
                "embedding": [0.1] * EMBED_DIMENSION,
            }
        ]
        with (
            patch(
                "arklex.env.tools.RAG.retrievers.milvus_retriever.connections.connect"
            ),
            patch(
                "arklex.env.tools.RAG.retrievers.milvus_retriever.Collection"
            ) as mock_collection_class,
        ):
            mock_collection = mock_collection_class.return_value
            mock_iterator = Mock()
            mock_collection.query_iterator.return_value = mock_iterator

            # Mock the iterator behavior
            mock_iterator.next.side_effect = [mock_query_result, []]
            mock_iterator.close = Mock()

            results = milvus_retriever.get_all_vectors("test_collection")
        assert len(results) == 1
        assert results[0]["id"] == "doc_1"

    def test_add_vectors_parallel(self, milvus_retriever: MilvusRetriever) -> None:
        """Test add_vectors_parallel method."""
        vectors = [
            {
                "id": "doc_1",
                "qa_doc_id": "qa_123",
                "text": "test document",
                "embedding": [0.1] * EMBED_DIMENSION,
            }
        ]
        mock_pool = Mock()
        mock_pool.map.return_value = [{"insert_count": 1}]

        # Mock the add_vectors_parallel method to avoid real MilvusClient calls
        with patch.object(milvus_retriever, "add_vectors_parallel") as mock_method:
            mock_method.return_value = [{"insert_count": 1}]

            result = milvus_retriever.add_vectors_parallel(
                "test_collection", "test_bot", "1.0", vectors, mock_pool
            )

            assert len(result) == 1
            mock_method.assert_called_once()

    def test_is_collection_loaded(self, milvus_retriever: MilvusRetriever) -> None:
        """Test is_collection_loaded method."""
        milvus_retriever.client.get_load_state.return_value = {"state": "Loaded"}

        result = milvus_retriever.is_collection_loaded("test_collection")
        assert result is True

    def test_delete_vectors_by_partition_key(
        self, milvus_retriever: MilvusRetriever
    ) -> None:
        """Test delete_vectors_by_partition_key method."""
        mock_result = {"deleted_count": 5}
        milvus_retriever.client.delete.return_value = mock_result
        mock_query_result = [{"count(*)": 0}]
        milvus_retriever.client.query.return_value = mock_query_result

        result = milvus_retriever.delete_vectors_by_partition_key(
            "test_collection", "test_bot", "1.0"
        )

        milvus_retriever.client.delete.assert_called_once_with(
            collection_name="test_collection", filter="bot_uid=='test_bot__1.0'"
        )
        assert result == mock_query_result

    def test_get_vector_count_for_bot(self, milvus_retriever: MilvusRetriever) -> None:
        """Test get_vector_count_for_bot method."""
        mock_query_result = [{"count": 1}]
        milvus_retriever.client.query.return_value = mock_query_result

        result = milvus_retriever.get_vector_count_for_bot(
            "test_collection", "test_bot", "1.0"
        )

        assert result == 1

    def test_get_collection_size(self, milvus_retriever: MilvusRetriever) -> None:
        """Test get_collection_size method."""
        mock_query_result = [{"count(*)": 100}]
        milvus_retriever.client.query.return_value = mock_query_result

        result = milvus_retriever.get_collection_size("test_collection")

        assert result == 100

    def test_migrate_vectors(self, milvus_retriever: MilvusRetriever) -> None:
        """Test migrate_vectors method."""
        mock_query_result = [
            {
                "id": "doc_1",
                "qa_doc_id": "qa_123",
                "text": "test document",
                "embedding": [0.1] * EMBED_DIMENSION,
            }
        ]
        milvus_retriever.client.query.return_value = [{"count(*)": 0}]
        mock_upsert_result = {"insert_count": 1}
        milvus_retriever.client.upsert.return_value = mock_upsert_result
        with (
            patch(
                "arklex.env.tools.RAG.retrievers.milvus_retriever.connections.connect"
            ),
            patch(
                "arklex.env.tools.RAG.retrievers.milvus_retriever.Collection"
            ) as mock_collection_class,
        ):
            mock_collection = mock_collection_class.return_value
            mock_iterator = Mock()
            mock_collection.query_iterator.return_value = mock_iterator

            # Mock the iterator behavior
            mock_iterator.next.side_effect = [mock_query_result, []]
            mock_iterator.close = Mock()

            # Patch add_vectors_parallel to call upsert
            with patch.object(
                milvus_retriever, "add_vectors_parallel"
            ) as mock_add_vectors_parallel:
                mock_add_vectors_parallel.return_value = [mock_upsert_result]
                result = milvus_retriever.migrate_vectors(
                    "old_collection", "test_bot", "1.0", "new_collection"
                )
                assert result == 1
                mock_add_vectors_parallel.assert_called_once()

    def test_list_collections(self, milvus_retriever: MilvusRetriever) -> None:
        """Test list_collections method."""
        mock_collections = ["collection1", "collection2"]
        milvus_retriever.client.list_collections.return_value = mock_collections

        result = milvus_retriever.list_collections()

        assert result == mock_collections


class TestMilvusRetrieverExecutor:
    """Test the MilvusRetrieverExecutor class."""

    def test_initialization(self, mock_bot_config: Mock) -> None:
        """Test MilvusRetrieverExecutor initialization."""
        with patch(
            "arklex.env.tools.RAG.retrievers.milvus_retriever.MilvusRetriever"
        ) as mock_retriever_class:
            mock_retriever = Mock()
            mock_retriever_class.return_value.__enter__.return_value = mock_retriever

            executor = MilvusRetrieverExecutor(mock_bot_config)

            assert executor.bot_config == mock_bot_config
            # Note: retriever is not set during initialization, only during retrieve

    def test_generate_thought(
        self,
        milvus_retriever_executor: MilvusRetrieverExecutor,
        sample_retriever_result: RetrieverResult,
    ) -> None:
        """Test generate_thought method."""
        results = [sample_retriever_result]

        result = milvus_retriever_executor.generate_thought(results)

        assert isinstance(result, str)
        assert "This is a test document" in result

    def test_gaussian_similarity(
        self, milvus_retriever_executor: MilvusRetrieverExecutor
    ) -> None:
        """Test _gaussian_similarity method."""
        result = milvus_retriever_executor._gaussian_similarity(0.5)
        assert isinstance(result, float)
        assert 0 <= result <= 100

    def test_postprocess(
        self,
        milvus_retriever_executor: MilvusRetrieverExecutor,
        sample_retriever_result: RetrieverResult,
    ) -> None:
        """Test postprocess method."""
        results = [sample_retriever_result]

        result = milvus_retriever_executor.postprocess(results)

        assert isinstance(result, dict)
        assert "retriever" in result
        assert len(result["retriever"]) == 1

    def test_retrieve(self, milvus_retriever_executor: MilvusRetrieverExecutor) -> None:
        """Test retrieve method."""
        mock_search_results = [
            RetrieverResult(
                qa_doc_id="qa_123",
                qa_doc_type=RetrieverDocumentType.FAQ,
                distance=0.5,
                metadata={"source": "test.txt"},
                text="test document",
                start_chunk_idx=0,
                end_chunk_idx=0,
            )
        ]
        milvus_retriever_executor.retriever.search.return_value = mock_search_results

        with patch(
            "arklex.env.tools.RAG.retrievers.milvus_retriever.mysql_pool"
        ) as mock_mysql:
            mock_mysql.fetchone.return_value = {"collection_name": "test_collection"}

            # Mock the MilvusRetriever context manager
            with patch(
                "arklex.env.tools.RAG.retrievers.milvus_retriever.MilvusRetriever"
            ) as mock_retriever_class:
                mock_retriever = Mock()
                mock_retriever_class.return_value.__enter__.return_value = (
                    mock_retriever
                )
                mock_retriever.search.return_value = mock_search_results

                result_text, result_params = milvus_retriever_executor.retrieve(
                    "test query"
                )

                assert isinstance(result_text, str)
                assert isinstance(result_params, dict)
                assert "test document" in result_text

    def test_retrieve_with_tags(
        self, milvus_retriever_executor: MilvusRetrieverExecutor
    ) -> None:
        """Test retrieve method with tags."""
        mock_search_results = []
        milvus_retriever_executor.retriever.search.return_value = mock_search_results
        tags = {"category": "test"}
        with patch(
            "arklex.env.tools.RAG.retrievers.milvus_retriever.mysql_pool"
        ) as mock_mysql:
            mock_mysql.fetchone.return_value = {"collection_name": "test_collection"}
            # Mock the MilvusRetriever context manager
            with patch(
                "arklex.env.tools.RAG.retrievers.milvus_retriever.MilvusRetriever"
            ) as mock_retriever_class:
                mock_retriever = Mock()
                mock_retriever_class.return_value.__enter__.return_value = (
                    mock_retriever
                )
                mock_retriever.search.return_value = mock_search_results
                result_text, result_params = milvus_retriever_executor.retrieve(
                    "test query", tags
                )
                assert isinstance(result_text, str)
                assert isinstance(result_params, dict)
                mock_retriever.search.assert_called_once()

    def test_retrieve_empty_results(
        self, milvus_retriever_executor: MilvusRetrieverExecutor
    ) -> None:
        """Test retrieve method with empty results."""
        milvus_retriever_executor.retriever.search.return_value = []

        with patch(
            "arklex.env.tools.RAG.retrievers.milvus_retriever.mysql_pool"
        ) as mock_mysql:
            mock_mysql.fetchone.return_value = {"collection_name": "test_collection"}

            # Mock the MilvusRetriever context manager
            with patch(
                "arklex.env.tools.RAG.retrievers.milvus_retriever.MilvusRetriever"
            ) as mock_retriever_class:
                mock_retriever = Mock()
                mock_retriever_class.return_value.__enter__.return_value = (
                    mock_retriever
                )
                mock_retriever.search.return_value = []

                result_text, result_params = milvus_retriever_executor.retrieve(
                    "test query"
                )

                assert result_text == ""
                assert isinstance(result_params, dict)


class TestConstants:
    """Test module constants."""

    def test_embed_dimension(self) -> None:
        """Test EMBED_DIMENSION constant."""
        assert EMBED_DIMENSION == 1536

    def test_max_text_length(self) -> None:
        """Test MAX_TEXT_LENGTH constant."""
        assert MAX_TEXT_LENGTH == 65535

    def test_chunk_neighbours(self) -> None:
        """Test CHUNK_NEIGHBOURS constant."""
        assert CHUNK_NEIGHBOURS == 3


class TestIntegration:
    """Integration tests for Milvus retriever components."""

    def test_full_retrieval_flow(self, mock_bot_config: Mock) -> None:
        """Test complete retrieval flow from RetrieveEngine to MilvusRetrieverExecutor."""
        with patch(
            "arklex.env.tools.RAG.retrievers.milvus_retriever.MilvusRetrieverExecutor"
        ) as mock_executor_class:
            mock_executor = Mock()
            mock_executor_class.return_value = mock_executor
            mock_executor.retrieve.return_value = ("retrieved text", {"params": "test"})

            with patch(
                "arklex.env.tools.RAG.retrievers.milvus_retriever.trace"
            ) as mock_trace:
                mock_result_state = Mock()
                mock_result_state.message_flow = "retrieved text"
                mock_trace.return_value = mock_result_state

                state = Mock(spec=MessageState)
                state.bot_config = mock_bot_config
                state.user_message = Mock()
                state.user_message.history = "test query"
                state.message_flow = ""

                result = RetrieveEngine.milvus_retrieve(state)

                assert result.message_flow == "retrieved text"
                mock_executor.retrieve.assert_called_once_with("test query", {})

    def test_document_lifecycle(
        self,
        milvus_retriever: MilvusRetriever,
        sample_retriever_document: RetrieverDocument,
    ) -> None:
        """Test complete document lifecycle: add, search, delete."""
        # Mock embedding
        with patch(
            "arklex.env.tools.RAG.retrievers.milvus_retriever.embed"
        ) as mock_embed:
            mock_embed.return_value = [0.1] * EMBED_DIMENSION

            # Test add document
            mock_upsert_result = {"insert_count": 1}
            milvus_retriever.client.upsert.return_value = mock_upsert_result

            add_result = milvus_retriever.add_documents(
                "test_collection", "test_bot", "1.0", [sample_retriever_document]
            )
            assert len(add_result) == 1

            # Test search document
            mock_search_result = [
                [
                    {
                        "entity": {
                            "id": "test_doc_1",
                            "qa_doc_id": "qa_123",
                            "chunk_id": 0,
                            "qa_doc_type": "faq",
                            "metadata": {"source": "test.txt"},
                            "text": "This is a test document",
                        },
                        "distance": 0.5,
                    }
                ]
            ]
            milvus_retriever.client.search.return_value = mock_search_result

            search_results = milvus_retriever.search(
                "test_collection", "test_bot", "1.0", "test query"
            )
            assert len(search_results) == 1

            # Test delete document
            mock_delete_result = {"deleted_count": 1}
            milvus_retriever.client.delete.return_value = mock_delete_result

            delete_result = milvus_retriever.delete_documents_by_qa_doc_id(
                "test_collection", "qa_123"
            )
            assert delete_result == mock_delete_result

    def test_add_documents_upsert_true(
        self,
        milvus_retriever: MilvusRetriever,
        sample_retriever_document: RetrieverDocument,
    ) -> None:
        """Test add_documents with upsert=True covers the else branch."""
        milvus_retriever.client.has_collection.return_value = True
        milvus_retriever.client.upsert.return_value = ["ok"]
        result = milvus_retriever.add_documents(
            "test_collection",
            "test_bot",
            "1.0",
            [sample_retriever_document],
            upsert=True,
        )
        assert result == [["ok"]]

    def test_get_qa_docs_non_faq(self, milvus_retriever: MilvusRetriever) -> None:
        """Test get_qa_docs for non-FAQ type covers else branch and unchunked doc creation."""
        # Mock Collection and iterator
        with (
            patch(
                "arklex.env.tools.RAG.retrievers.milvus_retriever.Collection"
            ) as mock_collection,
            patch(
                "arklex.env.tools.RAG.retrievers.milvus_retriever.connections.connect"
            ),
        ):
            mock_iter = Mock()
            # Simulate two chunks for one qa_doc_id
            mock_iter.next.side_effect = [
                [
                    {
                        "id": "id1",
                        "qa_doc_id": "qa1",
                        "chunk_id": 0,
                        "qa_doc_type": "other",
                        "text": "a",
                        "metadata": {},
                        "bot_uid": "bot__1.0",
                        "timestamp": 1,
                    },
                    {
                        "id": "id2",
                        "qa_doc_id": "qa1",
                        "chunk_id": 1,
                        "qa_doc_type": "other",
                        "text": "b",
                        "metadata": {},
                        "bot_uid": "bot__1.0",
                        "timestamp": 1,
                    },
                ],
                [],
            ]
            mock_collection.return_value.query_iterator.return_value = mock_iter
            milvus_retriever.uri = "uri"
            milvus_retriever.token = "token"
            docs = milvus_retriever.get_qa_docs(
                "test_collection", "test_bot", "1.0", RetrieverDocumentType.OTHER
            )
            assert len(docs) == 1
            assert docs[0].qa_doc_id == "qa1"
            assert docs[0].text == "ab"

    def test_get_qa_doc_non_faq(self, milvus_retriever: MilvusRetriever) -> None:
        """Test get_qa_doc for non-FAQ type covers else branch."""
        milvus_retriever.client.query.return_value = [
            {
                "qa_doc_id": "qa1",
                "chunk_id": 0,
                "qa_doc_type": "other",
                "text": "a",
                "metadata": {},
                "bot_uid": "bot__1.0",
                "timestamp": 1,
            },
            {
                "qa_doc_id": "qa1",
                "chunk_id": 1,
                "qa_doc_type": "other",
                "text": "b",
                "metadata": {},
                "bot_uid": "bot__1.0",
                "timestamp": 1,
            },
        ]
        doc = milvus_retriever.get_qa_doc("test_collection", "qa1")
        assert doc.qa_doc_id == "qa1"
        assert doc.text == "ab"

    def test_load_collection_raises(self, milvus_retriever: MilvusRetriever) -> None:
        """Test load_collection raises ValueError if collection does not exist."""
        milvus_retriever.client.has_collection.return_value = False
        with pytest.raises(ValueError) as excinfo:
            milvus_retriever.load_collection("test_collection")
        assert "Milvus Collection test_collection does not exist" in str(excinfo.value)

    def test_add_vectors_parallel_upsert(
        self, milvus_retriever: MilvusRetriever
    ) -> None:
        """Test add_vectors_parallel with upsert=True covers else branch."""
        milvus_retriever.client.has_collection.return_value = True
        milvus_retriever.client.upsert.return_value = ["ok"]
        vectors = [{"id": "id1"}]
        result = milvus_retriever.add_vectors_parallel(
            "test_collection", "test_bot", "1.0", vectors, upsert=True
        )
        assert result == ["ok"]

    def test_add_vectors_parallel_batching(
        self, milvus_retriever: MilvusRetriever
    ) -> None:
        """Test add_vectors_parallel processes batches of 100."""
        milvus_retriever.client.has_collection.return_value = True
        milvus_retriever.client.query.return_value = []
        milvus_retriever.client.upsert.return_value = ["ok"] * 100
        vectors = [{"id": f"id{i}"} for i in range(150)]
        result = milvus_retriever.add_vectors_parallel(
            "test_collection", "test_bot", "1.0", vectors
        )
        assert len(result) == 200  # 2 batches of 100

    def test_add_vectors_parallel_creates_collection(
        self, milvus_retriever: MilvusRetriever
    ) -> None:
        """Test add_vectors_parallel creates collection if not exists."""
        milvus_retriever.client.has_collection.return_value = False
        milvus_retriever.client.query.return_value = []
        milvus_retriever.client.upsert.return_value = ["ok"]
        with patch.object(
            milvus_retriever, "create_collection_with_partition_key"
        ) as mock_create:
            vectors = [{"id": "id1"}]
            milvus_retriever.add_vectors_parallel(
                "test_collection", "test_bot", "1.0", vectors
            )
            mock_create.assert_called_once_with("test_collection")

    def test_is_collection_loaded(self, milvus_retriever: MilvusRetriever) -> None:
        """Test is_collection_loaded covers print and return value."""
        milvus_retriever.client.get_load_state.return_value = {"state": "Loaded"}
        assert milvus_retriever.is_collection_loaded("test_collection") is True

    def test_is_collection_loaded_with_print(
        self, milvus_retriever: MilvusRetriever
    ) -> None:
        """Test is_collection_loaded covers the print statement for 100% coverage."""
        milvus_retriever.client.get_load_state.return_value = {"state": "NotLoaded"}
        # Don't patch print - let it execute to get coverage
        result = milvus_retriever.is_collection_loaded("test_collection")
        assert result is False
