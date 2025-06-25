import pytest
from unittest.mock import patch, Mock
import pickle
from langchain_core.documents import Document
from arklex.env.tools.RAG.retrievers.faiss_retriever import (
    FaissRetrieverExecutor,
    RetrieveEngine,
)
from arklex.utils.graph_state import MessageState, LLMConfig


@pytest.fixture
def mock_documents():
    return [
        Document(
            page_content="Test document 1 content",
            metadata={"title": "Test Doc 1", "source": "test1.txt"},
        ),
        Document(
            page_content="Test document 2 content",
            metadata={"title": "Test Doc 2", "source": "test2.txt"},
        ),
    ]


@pytest.fixture
def mock_llm_config():
    config = Mock(spec=LLMConfig)
    config.llm_provider = "openai"
    config.model_type_or_path = "gpt-3.5-turbo"
    return config


@pytest.fixture
def faiss_retriever(mock_documents, mock_llm_config):
    with (
        patch(
            "arklex.env.tools.RAG.retrievers.faiss_retriever.PROVIDER_EMBEDDINGS"
        ) as mock_embeddings,
        patch(
            "arklex.env.tools.RAG.retrievers.faiss_retriever.PROVIDER_EMBEDDING_MODELS"
        ) as mock_embedding_models,
        patch(
            "arklex.env.tools.RAG.retrievers.faiss_retriever.PROVIDER_MAP"
        ) as mock_provider_map,
        patch("arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS") as mock_faiss,
    ):
        mock_embeddings.get.return_value = Mock()
        mock_embedding_models.get.return_value = "text-embedding-ada-002"
        mock_provider_map.get.return_value = Mock()

        mock_faiss_instance = Mock()
        mock_faiss.from_documents.return_value = mock_faiss_instance
        mock_faiss_instance.as_retriever.return_value = Mock()

        retriever = FaissRetrieverExecutor(
            texts=mock_documents, index_path="./test_index", llm_config=mock_llm_config
        )
        return retriever


class TestFaissRetrieverExecutor:
    """Test the FaissRetrieverExecutor class."""

    def test_initialization(self, mock_documents, mock_llm_config) -> None:
        """Test FaissRetrieverExecutor initialization."""
        with (
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.PROVIDER_EMBEDDINGS"
            ) as mock_embeddings,
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.PROVIDER_EMBEDDING_MODELS"
            ) as mock_embedding_models,
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.PROVIDER_MAP"
            ) as mock_provider_map,
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS"
            ) as mock_faiss,
        ):
            mock_embeddings.get.return_value = Mock()
            mock_embedding_models.get.return_value = "text-embedding-ada-002"
            mock_provider_map.get.return_value = Mock()

            mock_faiss_instance = Mock()
            mock_faiss.from_documents.return_value = mock_faiss_instance
            mock_faiss_instance.as_retriever.return_value = Mock()

            retriever = FaissRetrieverExecutor(
                texts=mock_documents,
                index_path="./test_index",
                llm_config=mock_llm_config,
            )

            assert retriever.texts == mock_documents
            assert retriever.index_path == "./test_index"
            assert retriever.embedding_model is not None
            assert retriever.llm is not None
            assert retriever.retriever is not None

    def test_initialization_anthropic_provider(self, mock_documents) -> None:
        """Test initialization with Anthropic provider."""
        config = Mock(spec=LLMConfig)
        config.llm_provider = "anthropic"
        config.model_type_or_path = "claude-3-sonnet"

        with (
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.PROVIDER_EMBEDDINGS"
            ) as mock_embeddings,
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.PROVIDER_EMBEDDING_MODELS"
            ) as mock_embedding_models,
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.PROVIDER_MAP"
            ) as mock_provider_map,
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS"
            ) as mock_faiss,
        ):
            mock_embeddings.get.return_value = Mock()
            mock_embedding_models.get.return_value = "text-embedding-ada-002"
            mock_provider_map.get.return_value = Mock()

            mock_faiss_instance = Mock()
            mock_faiss.from_documents.return_value = mock_faiss_instance
            mock_faiss_instance.as_retriever.return_value = Mock()

            retriever = FaissRetrieverExecutor(
                texts=mock_documents, index_path="./test_index", llm_config=config
            )

            assert retriever.embedding_model is not None

    def test_init_retriever(self, faiss_retriever) -> None:
        """Test _init_retriever method."""
        with patch(
            "arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS"
        ) as mock_faiss:
            mock_faiss_instance = Mock()
            mock_faiss.from_documents.return_value = mock_faiss_instance
            mock_faiss_instance.as_retriever.return_value = Mock()

            retriever = faiss_retriever._init_retriever(k=5)

            assert retriever is not None
            mock_faiss.from_documents.assert_called_once()
            mock_faiss_instance.as_retriever.assert_called_once_with(k=5)

    def test_retrieve_w_score_default_k(self, faiss_retriever) -> None:
        """Test retrieve_w_score with default k value."""
        mock_docs_and_scores = [
            (Document(page_content="doc1"), 0.8),
            (Document(page_content="doc2"), 0.6),
        ]

        faiss_retriever.retriever.search_kwargs = {}
        faiss_retriever.retriever.vectorstore.similarity_search_with_score.return_value = mock_docs_and_scores

        result = faiss_retriever.retrieve_w_score("test query")

        assert result == mock_docs_and_scores
        faiss_retriever.retriever.vectorstore.similarity_search_with_score.assert_called_once_with(
            "test query", k=4
        )

    def test_retrieve_w_score_custom_k(self, faiss_retriever) -> None:
        """Test retrieve_w_score with custom k value."""
        mock_docs_and_scores = [
            (Document(page_content="doc1"), 0.8),
            (Document(page_content="doc2"), 0.6),
        ]

        faiss_retriever.retriever.search_kwargs = {"k": 10}
        faiss_retriever.retriever.vectorstore.similarity_search_with_score.return_value = mock_docs_and_scores

        result = faiss_retriever.retrieve_w_score("test query")

        assert result == mock_docs_and_scores
        faiss_retriever.retriever.vectorstore.similarity_search_with_score.assert_called_once_with(
            "test query", k=10
        )

    def test_search(self, faiss_retriever):
        """Test search method."""
        mock_docs_and_scores = [
            (
                Document(
                    page_content="doc1 content",
                    metadata={"title": "Doc1", "source": "source1"},
                ),
                0.8,
            ),
            (
                Document(
                    page_content="doc2 content",
                    metadata={"title": "Doc2", "source": "source2"},
                ),
                0.6,
            ),
        ]

        faiss_retriever.retrieve_w_score = Mock(return_value=mock_docs_and_scores)

        with (
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.PromptTemplate"
            ) as mock_prompt_template,
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.StrOutputParser"
            ) as mock_str_parser,
        ):
            mock_prompt = Mock()
            mock_prompt_template.from_template.return_value = mock_prompt

            # Mock the chain operation
            mock_chain = Mock()
            mock_chain.invoke.return_value = "reformulated query"
            mock_prompt.__or__ = Mock(return_value=mock_chain)
            mock_chain.__or__ = Mock(return_value=mock_chain)

            retrieved_text, retriever_returns = faiss_retriever.search(
                "chat history", "contextualize prompt"
            )

            assert "doc1 content" in retrieved_text
            assert "doc2 content" in retrieved_text
            assert len(retriever_returns) == 2
            assert retriever_returns[0]["title"] == "Doc1"
            assert retriever_returns[0]["content"] == "doc1 content"
            assert retriever_returns[0]["source"] == "source1"
            assert retriever_returns[0]["confidence"] == 0.8

    def test_search_empty_results(self, faiss_retriever):
        """Test search method with empty results."""
        faiss_retriever.retrieve_w_score = Mock(return_value=[])

        with (
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.PromptTemplate"
            ) as mock_prompt_template,
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.StrOutputParser"
            ) as mock_str_parser,
        ):
            mock_prompt = Mock()
            mock_prompt_template.from_template.return_value = mock_prompt

            # Mock the chain operation
            mock_chain = Mock()
            mock_chain.invoke.return_value = "reformulated query"
            mock_prompt.__or__ = Mock(return_value=mock_chain)
            mock_chain.__or__ = Mock(return_value=mock_chain)

            retrieved_text, retriever_returns = faiss_retriever.search(
                "chat history", "contextualize prompt"
            )

            assert retrieved_text == ""
            assert retriever_returns == []

    @staticmethod
    def test_load_docs_success(mock_llm_config) -> None:
        """Test load_docs static method success."""
        mock_documents = [
            Document(page_content="doc1", metadata={"title": "Doc1"}),
            Document(page_content="doc2", metadata={"title": "Doc2"}),
        ]

        with (
            patch("builtins.open", create=True) as mock_open,
            patch("pickle.load", return_value=mock_documents),
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.FaissRetrieverExecutor"
            ) as mock_retriever_class,
        ):
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            mock_retriever_instance = Mock()
            mock_retriever_class.return_value = mock_retriever_instance

            result = FaissRetrieverExecutor.load_docs(
                database_path="/test/path",
                llm_config=mock_llm_config,
                index_path="./custom_index",
            )

            assert result == mock_retriever_instance
            mock_open.assert_called_once_with("/test/path/chunked_documents.pkl", "rb")
            pickle.load.assert_called_once_with(mock_file)
            mock_retriever_class.assert_called_once()

    def test_faiss_retriever_file_not_found(self) -> None:  # noqa: N802
        """Test load_docs with file not found."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                FaissRetrieverExecutor.load_docs(database_path="/tmp", llm_config=None)


class TestRetrieveEngine:
    """Test the RetrieveEngine class."""

    def test_faiss_retrieve(self):
        """Test faiss_retrieve static method."""
        mock_state = Mock(spec=MessageState)
        mock_state.user_message = Mock()
        mock_state.user_message.history = "test history"
        mock_state.bot_config = Mock()
        mock_state.bot_config.llm_config = Mock()

        mock_prompts = {"retrieve_contextualize_q_prompt": "test prompt"}
        mock_retrieved_text = "retrieved content"
        mock_retriever_returns = [{"title": "Doc1", "content": "content1"}]

        with (
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.load_prompts",
                return_value=mock_prompts,
            ),
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.FaissRetrieverExecutor"
            ) as mock_retriever_class,
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.trace"
            ) as mock_trace,
            patch.dict("os.environ", {"DATA_DIR": "/test/data"}),
        ):
            mock_retriever_instance = Mock()
            mock_retriever_class.load_docs.return_value = mock_retriever_instance
            mock_retriever_instance.search.return_value = (
                mock_retrieved_text,
                mock_retriever_returns,
            )
            mock_trace.return_value = mock_state

            result = RetrieveEngine.faiss_retrieve(mock_state)

            assert result == mock_state
            assert result.message_flow == mock_retrieved_text
            mock_retriever_class.load_docs.assert_called_once()
            mock_retriever_instance.search.assert_called_once_with(
                "test history", "test prompt"
            )
            mock_trace.assert_called_once_with(
                input=mock_retriever_returns, state=mock_state
            )
