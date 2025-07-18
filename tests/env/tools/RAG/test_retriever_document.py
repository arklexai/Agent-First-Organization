"""Comprehensive tests for the retriever_document module.

This module provides line-by-line test coverage for all classes and functions in
arklex.env.tools.RAG.retrievers.retriever_document, including edge cases and error conditions.
"""

import time
from unittest.mock import Mock, patch

import pytest

# Mock the entire mysql module to avoid connection issues
with patch.dict("sys.modules", {"arklex.utils.mysql": Mock()}):
    from arklex.env.tools.RAG.retrievers.retriever_document import (
        DEFAULT_CHUNK_ENCODING,
        RetrieverDocument,
        RetrieverDocumentType,
        RetrieverResult,
        embed,
        embed_retriever_document,
        get_bot_uid,
    )


class TestEmbedFunction:
    """Test the embed function."""

    @patch("arklex.env.tools.RAG.retrievers.retriever_document.OpenAI")
    @patch("arklex.env.tools.RAG.retrievers.retriever_document.redis_pool")
    def test_embed_success(
        self, mock_redis_pool: Mock, mock_openai_class: Mock
    ) -> None:
        """Test successful embedding generation."""
        # Mock Redis cache miss
        mock_redis_pool.get.return_value = None

        mock_client = Mock()
        mock_response = Mock()
        mock_data_item = Mock()
        mock_data_item.embedding = [0.1, 0.2, 0.3]
        mock_response.data = [mock_data_item]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        result = embed("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(
            input="test text", model="text-embedding-ada-002"
        )
        # Verify Redis cache was called
        mock_redis_pool.get.assert_called_once()
        mock_redis_pool.set.assert_called_once()

    @patch("arklex.env.tools.RAG.retrievers.retriever_document.redis_pool")
    @patch("arklex.env.tools.RAG.retrievers.retriever_document.log_context")
    def test_embed_cache_hit(
        self, mock_log_context: Mock, mock_redis_pool: Mock
    ) -> None:
        """Test embedding cache hit."""
        # Mock Redis cache hit
        cached_embedding = [0.1, 0.2, 0.3]
        mock_redis_pool.get.return_value = cached_embedding

        result = embed("test text")

        assert result == cached_embedding
        mock_redis_pool.get.assert_called_once()
        mock_log_context.info.assert_called_with("Cache hit for text of length 9")

    @patch("arklex.env.tools.RAG.retrievers.retriever_document.OpenAI")
    @patch("arklex.env.tools.RAG.retrievers.retriever_document.redis_pool")
    @patch("arklex.env.tools.RAG.retrievers.retriever_document.log_context")
    def test_embed_exception_handling(
        self, mock_log_context: Mock, mock_redis_pool: Mock, mock_openai_class: Mock
    ) -> None:
        """Test embedding exception handling."""
        # Mock Redis cache miss
        mock_redis_pool.get.return_value = None

        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client

        with pytest.raises(Exception, match="API Error"):
            embed("test text")

        mock_log_context.error.assert_called()
        mock_log_context.exception.assert_called()


class TestRetrieverDocumentType:
    """Test the RetrieverDocumentType enum."""

    def test_enum_values(self) -> None:
        """Test enum values are correct."""
        assert RetrieverDocumentType.WEBSITE.value == "website"
        assert RetrieverDocumentType.FAQ.value == "faq"
        assert RetrieverDocumentType.OTHER.value == "other"

    def test_enum_creation(self) -> None:
        """Test creating enum from string value."""
        assert RetrieverDocumentType("website") == RetrieverDocumentType.WEBSITE
        assert RetrieverDocumentType("faq") == RetrieverDocumentType.FAQ
        assert RetrieverDocumentType("other") == RetrieverDocumentType.OTHER


class TestRetrieverResult:
    """Test the RetrieverResult class."""

    def test_retriever_result_creation_with_dict_metadata(self) -> None:
        """Test RetrieverResult creation with dictionary metadata."""
        metadata = {"source": "test.txt", "tags": ["test"]}
        result = RetrieverResult(
            qa_doc_id="qa_123",
            qa_doc_type=RetrieverDocumentType.FAQ,
            distance=0.5,
            metadata=metadata,
            text="This is a test document",
            start_chunk_idx=0,
            end_chunk_idx=0,
        )

        assert result.qa_doc_id == "qa_123"
        assert result.qa_doc_type == RetrieverDocumentType.FAQ
        assert result.distance == 0.5
        assert result.metadata == metadata
        assert result.text == "This is a test document"
        assert result.start_chunk_idx == 0
        assert result.end_chunk_idx == 0

    def test_retriever_result_creation_with_string_metadata(self) -> None:
        """Test RetrieverResult creation with string metadata."""
        metadata_str = '{"source": "test.txt", "tags": ["test"]}'
        result = RetrieverResult(
            qa_doc_id="qa_123",
            qa_doc_type=RetrieverDocumentType.FAQ,
            distance=0.5,
            metadata=metadata_str,
            text="This is a test document",
            start_chunk_idx=0,
            end_chunk_idx=0,
        )

        assert result.metadata == {"source": "test.txt", "tags": ["test"]}


class TestRetrieverDocument:
    """Test the RetrieverDocument class."""

    def test_retriever_document_creation(self) -> None:
        """Test basic RetrieverDocument creation."""
        doc = RetrieverDocument(
            id="test_doc_1",
            qa_doc_id="qa_123",
            chunk_idx=0,
            qa_doc_type=RetrieverDocumentType.FAQ,
            text="This is a test document",
            metadata={"source": "test.txt"},
            is_chunked=False,
            bot_uid="test_bot__1.0",
            embedding=[0.1, 0.2, 0.3],
            timestamp=1234567890,
        )

        assert doc.id == "test_doc_1"
        assert doc.qa_doc_id == "qa_123"
        assert doc.chunk_idx == 0
        assert doc.qa_doc_type == RetrieverDocumentType.FAQ
        assert doc.text == "This is a test document"
        assert doc.metadata == {"source": "test.txt"}
        assert doc.is_chunked is False
        assert doc.bot_uid == "test_bot__1.0"
        assert doc.embedding == [0.1, 0.2, 0.3]
        assert doc.timestamp == 1234567890

    def test_retriever_document_creation_with_string_metadata(self) -> None:
        """Test RetrieverDocument creation with string metadata."""
        metadata_str = '{"source": "test.txt", "tags": ["test"]}'
        doc = RetrieverDocument(
            id="test_doc_1",
            qa_doc_id="qa_123",
            chunk_idx=0,
            qa_doc_type=RetrieverDocumentType.FAQ,
            text="This is a test document",
            metadata=metadata_str,
            is_chunked=False,
            bot_uid="test_bot__1.0",
            timestamp=1234567890,  # Add timestamp to avoid None error
        )

        assert doc.metadata == {"source": "test.txt", "tags": ["test"]}

    def test_retriever_document_creation_with_string_chunk_idx(self) -> None:
        """Test RetrieverDocument creation with string chunk_idx."""
        doc = RetrieverDocument(
            id="test_doc_1",
            qa_doc_id="qa_123",
            chunk_idx="0",  # String instead of int
            qa_doc_type=RetrieverDocumentType.FAQ,
            text="This is a test document",
            metadata={"source": "test.txt"},
            is_chunked=False,
            bot_uid="test_bot__1.0",
            timestamp=1234567890,  # Add timestamp to avoid None error
        )

        assert doc.chunk_idx == 0  # Should be converted to int

    def test_retriever_document_creation_with_string_timestamp(self) -> None:
        """Test RetrieverDocument creation with string timestamp."""
        doc = RetrieverDocument(
            id="test_doc_1",
            qa_doc_id="qa_123",
            chunk_idx=0,
            qa_doc_type=RetrieverDocumentType.FAQ,
            text="This is a test document",
            metadata={"source": "test.txt"},
            is_chunked=False,
            bot_uid="test_bot__1.0",
            timestamp="1234567890",  # String instead of int
        )

        assert doc.timestamp == 1234567890  # Should be converted to int

    def test_retriever_document_creation_with_none_timestamp(self) -> None:
        """Test RetrieverDocument creation with None timestamp."""
        # The actual implementation doesn't handle None timestamp properly
        # This test documents the current behavior
        with pytest.raises(
            TypeError,
            match="int\\(\\) argument must be a string, a bytes-like object or a real number, not 'NoneType'",
        ):
            RetrieverDocument(
                id="test_doc_1",
                qa_doc_id="qa_123",
                chunk_idx=0,
                qa_doc_type=RetrieverDocumentType.FAQ,
                text="This is a test document",
                metadata={"source": "test.txt"},
                is_chunked=False,
                bot_uid="test_bot__1.0",
                timestamp=None,
            )

    @patch(
        "arklex.env.tools.RAG.retrievers.retriever_document.RecursiveCharacterTextSplitter"
    )
    @patch("arklex.env.tools.RAG.retrievers.retriever_document.tiktoken.get_encoding")
    @patch("arklex.env.tools.RAG.retrievers.retriever_document.log_context")
    def test_chunk_success(
        self, mock_log_context: Mock, mock_get_encoding: Mock, mock_splitter_class: Mock
    ) -> None:
        """Test successful document chunking."""
        # Mock tiktoken encoding
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_get_encoding.return_value = mock_encoding

        # Mock text splitter
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = ["chunk 1", "chunk 2"]
        mock_splitter_class.from_tiktoken_encoder.return_value = mock_splitter

        doc = RetrieverDocument(
            id="test_doc_1",
            qa_doc_id="qa_123",
            chunk_idx=0,
            qa_doc_type=RetrieverDocumentType.WEBSITE,
            text="This is a long document that needs to be chunked",
            metadata={"source": "test.txt"},
            is_chunked=False,
            bot_uid="test_bot__1.0",
            timestamp=1234567890,  # Add timestamp to avoid None error
        )

        chunked_docs = doc.chunk()

        assert len(chunked_docs) == 2
        assert chunked_docs[0].id == "test_doc_1__0"
        assert chunked_docs[0].qa_doc_id == "qa_123"
        assert chunked_docs[0].chunk_idx == 0
        assert chunked_docs[0].text == "chunk 1"
        assert chunked_docs[0].is_chunked is True
        assert chunked_docs[0].bot_uid == "test_bot__1.0"

        assert chunked_docs[1].id == "test_doc_1__1"
        assert chunked_docs[1].chunk_idx == 1
        assert chunked_docs[1].text == "chunk 2"

        mock_log_context.info.assert_called()

    def test_chunk_already_chunked_error(self) -> None:
        """Test chunking an already chunked document raises error."""
        doc = RetrieverDocument(
            id="test_doc_1",
            qa_doc_id="qa_123",
            chunk_idx=0,
            qa_doc_type=RetrieverDocumentType.WEBSITE,
            text="This is a test document",
            metadata={"source": "test.txt"},
            is_chunked=True,  # Already chunked
            bot_uid="test_bot__1.0",
            timestamp=1234567890,  # Add timestamp to avoid None error
        )

        with pytest.raises(ValueError, match="Document is already chunked"):
            doc.chunk()

    def test_chunk_faq_document_error(self) -> None:
        """Test chunking a FAQ document raises error."""
        doc = RetrieverDocument(
            id="test_doc_1",
            qa_doc_id="qa_123",
            chunk_idx=0,
            qa_doc_type=RetrieverDocumentType.FAQ,
            text="This is a FAQ document",
            metadata={"source": "test.txt"},
            is_chunked=False,
            bot_uid="test_bot__1.0",
            timestamp=1234567890,  # Add timestamp to avoid None error
        )

        with pytest.raises(ValueError, match="Cannot chunk FAQ document"):
            doc.chunk()

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        doc = RetrieverDocument(
            id="test_doc_1",
            qa_doc_id="qa_123",
            chunk_idx=0,
            qa_doc_type=RetrieverDocumentType.FAQ,
            text="This is a test document",
            metadata={"source": "test.txt"},
            is_chunked=False,
            bot_uid="test_bot__1.0",
            embedding=[0.1, 0.2, 0.3],
            timestamp=1234567890,
        )

        result = doc.to_dict()

        expected = {
            "id": "test_doc_1",
            "qa_doc_id": "qa_123",
            "chunk_idx": 0,
            "qa_doc_type": "faq",
            "text": "This is a test document",
            "metadata": {"source": "test.txt"},
            "embedding": [0.1, 0.2, 0.3],
            "is_chunked": False,
            "timestamp": 1234567890,
            "bot_uid": "test_bot__1.0",
        }
        assert result == expected

    @patch("arklex.env.tools.RAG.retrievers.retriever_document.embed")
    def test_to_milvus_schema_dict_and_embed_success(self, mock_embed: Mock) -> None:
        """Test to_milvus_schema_dict_and_embed with valid data."""
        mock_embed.return_value = [0.1, 0.2, 0.3]

        doc = RetrieverDocument(
            id="test_doc_1",
            qa_doc_id="qa_123",
            chunk_idx=0,
            qa_doc_type=RetrieverDocumentType.FAQ,
            text="This is a test document",
            metadata={"source": "test.txt"},
            is_chunked=False,
            bot_uid="test_bot__1.0",
            timestamp=1234567890,
        )

        result = doc.to_milvus_schema_dict_and_embed()

        expected = {
            "id": "test_doc_1",
            "qa_doc_id": "qa_123",
            "chunk_id": 0,
            "qa_doc_type": "faq",
            "text": "This is a test document",
            "metadata": {"source": "test.txt"},
            "timestamp": 1234567890,
            "embedding": [0.1, 0.2, 0.3],
            "bot_uid": "test_bot__1.0",
        }
        assert result == expected
        mock_embed.assert_called_once_with("This is a test document")

    def test_to_milvus_schema_dict_and_embed_missing_values(self) -> None:
        """Test to_milvus_schema_dict_and_embed with missing values."""
        # Test with None id
        doc = RetrieverDocument(
            id=None,
            qa_doc_id="qa_123",
            chunk_idx=0,
            qa_doc_type=RetrieverDocumentType.FAQ,
            text="This is a test document",
            metadata={"source": "test.txt"},
            is_chunked=False,
            bot_uid="test_bot__1.0",
            timestamp=1234567890,
        )

        with pytest.raises(ValueError, match="Missing values"):
            doc.to_milvus_schema_dict_and_embed()

        # Test with None text
        doc = RetrieverDocument(
            id="test_doc_1",
            qa_doc_id="qa_123",
            chunk_idx=0,
            qa_doc_type=RetrieverDocumentType.FAQ,
            text=None,
            metadata={"source": "test.txt"},
            is_chunked=False,
            bot_uid="test_bot__1.0",
            timestamp=1234567890,
        )

        with pytest.raises(ValueError, match="Missing values"):
            doc.to_milvus_schema_dict_and_embed()

    def test_from_dict(self) -> None:
        """Test from_dict class method."""
        doc_dict = {
            "id": "test_doc_1",
            "qa_doc_id": "qa_123",
            "chunk_idx": 0,
            "qa_doc_type": "faq",
            "text": "This is a test document",
            "metadata": {"source": "test.txt"},
            "embedding": [0.1, 0.2, 0.3],
            "is_chunked": False,
            "timestamp": 1234567890,
            "bot_uid": "test_bot__1.0",
        }

        doc = RetrieverDocument.from_dict(doc_dict)

        assert doc.id == "test_doc_1"
        assert doc.qa_doc_id == "qa_123"
        assert doc.chunk_idx == 0
        assert doc.qa_doc_type == RetrieverDocumentType.FAQ
        assert doc.text == "This is a test document"
        assert doc.metadata == {"source": "test.txt"}
        assert doc.embedding == [0.1, 0.2, 0.3]
        assert doc.is_chunked is False
        assert doc.timestamp == 1234567890
        assert doc.bot_uid == "test_bot__1.0"

    def test_faq_retreiver_doc(self) -> None:
        """Test faq_retreiver_doc class method."""
        doc = RetrieverDocument.faq_retreiver_doc(
            id="faq_123",
            text="What is the FAQ?",
            metadata={"category": "general"},
            bot_uid="test_bot__1.0",
            timestamp=1234567890,
        )

        assert doc.id == "faq_123"
        assert doc.qa_doc_id == "faq_123"
        assert doc.chunk_idx == 0
        assert doc.qa_doc_type == RetrieverDocumentType.FAQ
        assert doc.text == "What is the FAQ?"
        assert doc.metadata == {"category": "general"}
        assert doc.is_chunked is False
        assert doc.embedding is None
        assert doc.timestamp == 1234567890
        assert doc.bot_uid == "test_bot__1.0"

    def test_faq_retreiver_doc_without_timestamp(self) -> None:
        """Test faq_retreiver_doc without timestamp."""
        # The actual implementation doesn't handle None timestamp properly
        # This test documents the current behavior
        with pytest.raises(
            TypeError,
            match="int\\(\\) argument must be a string, a bytes-like object or a real number, not 'NoneType'",
        ):
            RetrieverDocument.faq_retreiver_doc(
                id="faq_123",
                text="What is the FAQ?",
                metadata={"category": "general"},
                bot_uid="test_bot__1.0",
            )

    def test_unchunked_retreiver_doc(self) -> None:
        """Test unchunked_retreiver_doc class method."""
        doc = RetrieverDocument.unchunked_retreiver_doc(
            id="doc_123",
            qa_doc_type=RetrieverDocumentType.WEBSITE,
            text="This is a website document",
            metadata={"url": "https://example.com"},
            bot_uid="test_bot__1.0",
            timestamp=1234567890,
        )

        assert doc.id == "doc_123"
        assert doc.qa_doc_id == "doc_123"
        assert doc.chunk_idx == 0
        assert doc.qa_doc_type == RetrieverDocumentType.WEBSITE
        assert doc.text == "This is a website document"
        assert doc.metadata == {"url": "https://example.com"}
        assert doc.is_chunked is False
        assert doc.embedding is None
        assert doc.timestamp == 1234567890
        assert doc.bot_uid == "test_bot__1.0"

    def test_unchunked_retreiver_doc_without_timestamp(self) -> None:
        """Test unchunked_retreiver_doc without timestamp."""
        # The actual implementation doesn't handle None timestamp properly
        # This test documents the current behavior
        with pytest.raises(
            TypeError,
            match="int\\(\\) argument must be a string, a bytes-like object or a real number, not 'NoneType'",
        ):
            RetrieverDocument.unchunked_retreiver_doc(
                id="doc_123",
                qa_doc_type=RetrieverDocumentType.WEBSITE,
                text="This is a website document",
                metadata={"url": "https://example.com"},
                bot_uid="test_bot__1.0",
            )

    @patch(
        "arklex.env.tools.RAG.retrievers.retriever_document.RecursiveCharacterTextSplitter"
    )
    @patch("arklex.env.tools.RAG.retrievers.retriever_document.tiktoken.get_encoding")
    def test_chunked_retriever_docs_from_db_docs(
        self, mock_get_encoding: Mock, mock_splitter_class: Mock
    ) -> None:
        """Test chunked_retriever_docs_from_db_docs class method."""
        # Mock tiktoken encoding
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
        ]  # 10 tokens to trigger chunking
        mock_get_encoding.return_value = mock_encoding

        # Mock text splitter
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = ["chunk 1", "chunk 2"]
        mock_splitter_class.from_tiktoken_encoder.return_value = mock_splitter

        db_docs = [
            {
                "id": "doc_1",
                "content": "This is a long document that needs chunking",
                "metadata": {"source": "doc1.txt"},
                "timestamp": 1234567890,
            },
            {
                "id": "doc_2",
                "content": "Another document to chunk",
                "metadata": {"source": "doc2.txt"},
                "timestamp": 1234567891,
            },
        ]

        chunked_docs = RetrieverDocument.chunked_retriever_docs_from_db_docs(
            db_docs, RetrieverDocumentType.WEBSITE, "test_bot__1.0"
        )

        assert len(chunked_docs) == 4  # 2 docs * 2 chunks each
        assert chunked_docs[0].id == "doc_1__0"
        assert chunked_docs[0].qa_doc_id == "doc_1"
        assert chunked_docs[0].text == "chunk 1"
        assert chunked_docs[0].is_chunked is True

        assert chunked_docs[1].id == "doc_1__1"
        assert chunked_docs[1].text == "chunk 2"

        assert chunked_docs[2].id == "doc_2__0"
        assert chunked_docs[2].qa_doc_id == "doc_2"

    def test_chunked_retriever_docs_from_db_docs_with_none_content(self) -> None:
        """Test chunked_retriever_docs_from_db_docs with None content."""
        db_docs = [
            {
                "id": "doc_1",
                "content": None,  # None content should be skipped
                "metadata": {"source": "doc1.txt"},
                "timestamp": 1234567890,
            },
            {
                "id": "doc_2",
                "content": "Valid document",
                "metadata": {"source": "doc2.txt"},
                "timestamp": 1234567891,
            },
        ]

        chunked_docs = RetrieverDocument.chunked_retriever_docs_from_db_docs(
            db_docs, RetrieverDocumentType.WEBSITE, "test_bot__1.0"
        )

        # Should only process the second document
        assert len(chunked_docs) == 1
        assert chunked_docs[0].qa_doc_id == "doc_2"

    @patch("arklex.env.tools.RAG.retrievers.retriever_document.mysql_pool")
    @patch("arklex.env.tools.RAG.retrievers.retriever_document.get_bot_uid")
    @patch("arklex.env.tools.RAG.retrievers.retriever_document.log_context")
    def test_load_all_chunked_docs_from_mysql(
        self, mock_log_context: Mock, mock_get_bot_uid: Mock, mock_mysql_pool: Mock
    ) -> None:
        """Test load_all_chunked_docs_from_mysql class method."""
        mock_get_bot_uid.return_value = "test_bot__1.0"

        # Mock FAQ docs
        mock_mysql_pool.fetchall.side_effect = [
            [  # FAQ docs
                {
                    "id": "faq_1",
                    "content": "What is FAQ?",
                    "metadata": {"category": "general"},
                    "timestamp": 1234567890,
                }
            ],
            [  # Other docs
                {
                    "id": "other_1",
                    "content": "Other document content",
                    "metadata": {"source": "other.txt"},
                    "timestamp": 1234567891,
                }
            ],
            [  # Website docs
                {
                    "id": "website_1",
                    "content": "Website content",
                    "metadata": {"url": "https://example.com"},
                    "timestamp": 1234567892,
                }
            ],
        ]

        # Mock chunking for other and website docs
        with patch.object(
            RetrieverDocument, "chunked_retriever_docs_from_db_docs"
        ) as mock_chunk:
            mock_chunk.side_effect = [
                [Mock()],  # Other docs chunked
                [Mock(), Mock()],  # Website docs chunked
            ]

            docs = RetrieverDocument.load_all_chunked_docs_from_mysql("test_bot", "1.0")

            # Should return website + other + faq docs
            assert len(docs) == 4  # 2 website + 1 other + 1 faq

            mock_get_bot_uid.assert_called_with("test_bot", "1.0")
            assert mock_mysql_pool.fetchall.call_count == 3

            # Verify log messages
            mock_log_context.info.assert_called()


class TestEmbedRetrieverDocument:
    """Test the embed_retriever_document function."""

    @patch("arklex.env.tools.RAG.retrievers.retriever_document.embed")
    def test_embed_retriever_document(self, mock_embed: Mock) -> None:
        """Test embed_retriever_document function."""
        mock_embed.return_value = [0.1, 0.2, 0.3]

        doc = RetrieverDocument(
            id="test_doc_1",
            qa_doc_id="qa_123",
            chunk_idx=0,
            qa_doc_type=RetrieverDocumentType.FAQ,
            text="This is a test document",
            metadata={"source": "test.txt"},
            is_chunked=False,
            bot_uid="test_bot__1.0",
            timestamp=1234567890,
        )

        result = embed_retriever_document(doc)

        expected = {
            "id": "test_doc_1",
            "qa_doc_id": "qa_123",
            "chunk_id": 0,
            "qa_doc_type": "faq",
            "text": "This is a test document",
            "metadata": {"source": "test.txt"},
            "timestamp": 1234567890,
            "embedding": [0.1, 0.2, 0.3],
            "bot_uid": "test_bot__1.0",
        }
        assert result == expected


class TestGetBotUid:
    """Test the get_bot_uid function."""

    def test_get_bot_uid(self) -> None:
        """Test get_bot_uid function."""
        result = get_bot_uid("test_bot", "1.0")
        assert result == "test_bot__1.0"

        result = get_bot_uid("my_bot", "2.1")
        assert result == "my_bot__2.1"


class TestConstants:
    """Test module constants."""

    def test_default_chunk_encoding(self) -> None:
        """Test DEFAULT_CHUNK_ENCODING constant."""
        assert DEFAULT_CHUNK_ENCODING == "cl100k_base"


class TestIntegration:
    """Integration tests for the retriever_document module."""

    def test_full_document_lifecycle(self) -> None:
        """Test complete document lifecycle from creation to chunking."""
        # Create an unchunked document
        doc = RetrieverDocument.unchunked_retreiver_doc(
            id="test_doc",
            qa_doc_type=RetrieverDocumentType.WEBSITE,
            text="This is a long document that will be chunked into smaller pieces for better retrieval.",
            metadata={"source": "test.txt", "url": "https://example.com"},
            bot_uid="test_bot__1.0",
            timestamp=int(time.time()),
        )

        assert doc.is_chunked is False
        assert doc.chunk_idx == 0

        # Convert to dict and back
        doc_dict = doc.to_dict()
        doc_from_dict = RetrieverDocument.from_dict(doc_dict)

        assert doc_from_dict.id == doc.id
        assert doc_from_dict.text == doc.text
        assert doc_from_dict.metadata == doc.metadata

        # Test chunking (with mocked text splitter)
        with patch(
            "arklex.env.tools.RAG.retrievers.retriever_document.RecursiveCharacterTextSplitter"
        ) as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter.split_text.return_value = ["chunk 1", "chunk 2"]
            mock_splitter_class.from_tiktoken_encoder.return_value = mock_splitter

            with patch(
                "arklex.env.tools.RAG.retrievers.retriever_document.tiktoken.get_encoding"
            ) as mock_get_encoding:
                mock_encoding = Mock()
                mock_encoding.encode.return_value = [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                ]  # More tokens to trigger chunking
                mock_get_encoding.return_value = mock_encoding

                chunked_docs = doc.chunk()

                assert len(chunked_docs) == 2
                assert all(doc.is_chunked for doc in chunked_docs)
                assert chunked_docs[0].chunk_idx == 0
                assert chunked_docs[1].chunk_idx == 1

    def test_faq_document_special_handling(self) -> None:
        """Test that FAQ documents are handled specially (not chunked)."""
        # Create FAQ document with timestamp
        faq_doc = RetrieverDocument.faq_retreiver_doc(
            id="faq_1",
            text="What is the answer to this question?",
            metadata={"category": "general"},
            bot_uid="test_bot__1.0",
            timestamp=1234567890,
        )

        assert faq_doc.qa_doc_type == RetrieverDocumentType.FAQ
        assert faq_doc.is_chunked is False

        # Attempt to chunk FAQ document should fail
        with pytest.raises(ValueError, match="Cannot chunk FAQ document"):
            faq_doc.chunk()

    def test_metadata_handling_edge_cases(self) -> None:
        """Test metadata handling with various edge cases."""
        # Test with empty metadata
        doc = RetrieverDocument(
            id="test_doc",
            qa_doc_id="qa_123",
            chunk_idx=0,
            qa_doc_type=RetrieverDocumentType.FAQ,
            text="Test document",
            metadata={},
            is_chunked=False,
            bot_uid="test_bot__1.0",
            timestamp=1234567890,
        )

        assert doc.metadata == {}

        # Test with complex metadata
        complex_metadata = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "boolean": True,
            "null": None,
        }
        doc = RetrieverDocument(
            id="test_doc",
            qa_doc_id="qa_123",
            chunk_idx=0,
            qa_doc_type=RetrieverDocumentType.FAQ,
            text="Test document",
            metadata=complex_metadata,
            is_chunked=False,
            bot_uid="test_bot__1.0",
            timestamp=1234567890,
        )

        assert doc.metadata == complex_metadata

    def test_timestamp_handling(self) -> None:
        """Test timestamp handling with various inputs."""
        # Test with current timestamp
        current_time = int(time.time())
        doc = RetrieverDocument(
            id="test_doc",
            qa_doc_id="qa_123",
            chunk_idx=0,
            qa_doc_type=RetrieverDocumentType.FAQ,
            text="Test document",
            metadata={},
            is_chunked=False,
            bot_uid="test_bot__1.0",
            timestamp=current_time,
        )

        assert doc.timestamp == current_time

        # Test with None timestamp - this should fail with current implementation
        with pytest.raises(
            TypeError,
            match="int\\(\\) argument must be a string, a bytes-like object or a real number, not 'NoneType'",
        ):
            RetrieverDocument(
                id="test_doc",
                qa_doc_id="qa_123",
                chunk_idx=0,
                qa_doc_type=RetrieverDocumentType.FAQ,
                text="Test document",
                metadata={},
                is_chunked=False,
                bot_uid="test_bot__1.0",
                timestamp=None,
            )
