from unittest.mock import Mock, mock_open, patch

import pytest
from langchain_core.documents import Document

from arklex.env.tools.RAG.retrievers.faiss_retriever import (
    FaissRetrieverExecutor,
    RetrieveEngine,
)
from arklex.orchestrator.entities.msg_state_entities import LLMConfig, MessageState


@pytest.fixture
def mock_documents() -> list[Document]:
    return [
        Document(
            page_content="Test document 1", metadata={"source": "test1.txt", "page": 1}
        ),
        Document(
            page_content="Test document 2", metadata={"source": "test2.txt", "page": 2}
        ),
    ]


@pytest.fixture
def mock_llm_config() -> Mock:
    config = Mock(spec=LLMConfig)
    config.llm_provider = "openai"
    config.model_type_or_path = "text-embedding-ada-002"
    return config


@pytest.fixture
def faiss_retriever(
    mock_documents: list[Document], mock_llm_config: Mock
) -> FaissRetrieverExecutor:
    with (
        patch("arklex.env.tools.RAG.retrievers.faiss_retriever.OpenAIEmbeddings"),
        patch("arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS"),
    ):
        retriever = FaissRetrieverExecutor(
            texts=mock_documents,
            index_path="/tmp/test_db/index",
            llm_config=mock_llm_config,
        )
        return retriever


class TestFaissRetrieverExecutor:
    """Test the FaissRetrieverExecutor class."""

    def test_initialization(
        self, mock_documents: list[Document], mock_llm_config: Mock
    ) -> None:
        """Test FaissRetrieverExecutor initialization."""
        with (
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.OpenAIEmbeddings"
            ) as mock_embeddings,
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS"
            ) as mock_faiss,
        ):
            mock_embeddings_instance = Mock()
            mock_embeddings.return_value = mock_embeddings_instance

            mock_faiss_instance = Mock()
            mock_faiss.from_documents.return_value = mock_faiss_instance
            mock_faiss_instance.as_retriever.return_value = Mock()

            retriever = FaissRetrieverExecutor(
                texts=mock_documents,
                index_path="/tmp/test_db/index",
                llm_config=mock_llm_config,
            )

            assert retriever.texts == mock_documents
            assert retriever.index_path == "/tmp/test_db/index"
            assert retriever.embedding_model is not None
            assert retriever.retriever is not None

    def test_initialization_anthropic_provider(
        self, mock_documents: list[Document]
    ) -> None:
        """Test initialization with Anthropic provider."""
        config = Mock(spec=LLMConfig)
        config.llm_provider = "anthropic"
        config.model_type_or_path = "claude-3-sonnet-20240229"

        with (
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.OpenAIEmbeddings"
            ) as mock_embeddings,
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS"
            ) as mock_faiss,
        ):
            mock_embeddings_instance = Mock()
            mock_embeddings.return_value = mock_embeddings_instance

            mock_faiss_instance = Mock()
            mock_faiss.from_documents.return_value = mock_faiss_instance
            mock_faiss_instance.as_retriever.return_value = Mock()

            retriever = FaissRetrieverExecutor(
                texts=mock_documents,
                index_path="/tmp/test_db/index",
                llm_config=config,
            )

            assert retriever.embedding_model is not None
            assert retriever.retriever is not None

    def test_init_retriever(self, faiss_retriever: FaissRetrieverExecutor) -> None:
        """Test _init_retriever method."""
        with patch(
            "arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS.from_documents"
        ) as mock_from_documents:
            mock_faiss_instance = Mock()
            mock_from_documents.return_value = mock_faiss_instance
            mock_faiss_instance.as_retriever.return_value = Mock()

            faiss_retriever._init_retriever()

            mock_from_documents.assert_called_once()
            mock_faiss_instance.as_retriever.assert_called_once()

    def test_retrieve_w_score_default_k(
        self, faiss_retriever: FaissRetrieverExecutor
    ) -> None:
        """Test retrieve_w_score with default k value."""
        mock_docs_and_scores = [
            (Document(page_content="doc1"), 0.8),
            (Document(page_content="doc2"), 0.6),
        ]
        faiss_retriever.retriever.vectorstore.similarity_search_with_score = Mock(
            return_value=mock_docs_and_scores
        )
        faiss_retriever.retriever.search_kwargs = {}

        result = faiss_retriever.retrieve_w_score("test query")

        assert result == mock_docs_and_scores
        faiss_retriever.retriever.vectorstore.similarity_search_with_score.assert_called_once_with(
            "test query", k=4
        )

    def test_retrieve_w_score_custom_k(
        self, faiss_retriever: FaissRetrieverExecutor
    ) -> None:
        """Test retrieve_w_score with custom k value."""
        mock_docs_and_scores = [
            (Document(page_content="doc1"), 0.8),
            (Document(page_content="doc2"), 0.6),
        ]
        faiss_retriever.retriever.vectorstore.similarity_search_with_score = Mock(
            return_value=mock_docs_and_scores
        )
        faiss_retriever.retriever.search_kwargs = {"k": 10}

        result = faiss_retriever.retrieve_w_score("test query")

        assert result == mock_docs_and_scores
        faiss_retriever.retriever.vectorstore.similarity_search_with_score.assert_called_once_with(
            "test query", k=10
        )

    def test_search(self, faiss_retriever: FaissRetrieverExecutor) -> None:
        """Test search method."""
        mock_docs_and_scores = [
            (Document(page_content="doc1", metadata={"source": "test1.txt"}), 0.8),
            (Document(page_content="doc2", metadata={"source": "test2.txt"}), 0.6),
        ]
        faiss_retriever.retrieve_w_score = Mock(return_value=mock_docs_and_scores)

        with (
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.PromptTemplate"
            ) as mock_prompt,
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.StrOutputParser"
            ) as mock_parser,
        ):
            mock_prompt_instance = Mock()
            mock_llm = Mock()
            mock_parser_instance = Mock()
            mock_chain = Mock()
            mock_chain.invoke.return_value = "reformulated query"

            # Patch | operator for the chain: PromptTemplate | llm | StrOutputParser
            mock_prompt.from_template.return_value = mock_prompt_instance
            mock_prompt_instance.__or__ = Mock(return_value=mock_llm)
            mock_llm.__or__ = Mock(return_value=mock_chain)
            mock_parser.return_value = mock_parser_instance

            result_text, result_returns = faiss_retriever.search(
                "test query", "test prompt"
            )

            assert result_text == "doc1 \ndoc2 \n"
            assert len(result_returns) == 2
            assert result_returns[0]["content"] == "doc1"
            assert result_returns[0]["source"] == "test1.txt"
            assert result_returns[0]["confidence"] == 0.8
            assert result_returns[1]["content"] == "doc2"
            assert result_returns[1]["source"] == "test2.txt"
            assert result_returns[1]["confidence"] == 0.6

    def test_search_empty_results(
        self, faiss_retriever: FaissRetrieverExecutor
    ) -> None:
        """Test search method with empty results."""
        faiss_retriever.retrieve_w_score = Mock(return_value=[])

        with (
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.PromptTemplate"
            ) as mock_prompt,
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.StrOutputParser"
            ) as mock_parser,
        ):
            mock_prompt_instance = Mock()
            mock_llm = Mock()
            mock_parser_instance = Mock()
            mock_chain = Mock()
            mock_chain.invoke.return_value = "reformulated query"

            # Patch | operator for the chain: PromptTemplate | llm | StrOutputParser
            mock_prompt.from_template.return_value = mock_prompt_instance
            mock_prompt_instance.__or__ = Mock(return_value=mock_llm)
            mock_llm.__or__ = Mock(return_value=mock_chain)
            mock_parser.return_value = mock_parser_instance

            result_text, result_returns = faiss_retriever.search(
                "test query", "test prompt"
            )

            assert result_text == ""
            assert result_returns == []

    @staticmethod
    def test_load_docs_success(mock_llm_config: Mock) -> None:
        """Test load_docs static method success."""
        mock_documents = [
            Document(page_content="doc1", metadata={"source": "test1.txt"}),
            Document(page_content="doc2", metadata={"source": "test2.txt"}),
        ]

        with (
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.pickle.load"
            ) as mock_load,
            patch("builtins.open", mock_open(read_data="test data")),
            patch("arklex.env.tools.RAG.retrievers.faiss_retriever.OpenAIEmbeddings"),
            patch("arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS"),
        ):
            mock_load.return_value = mock_documents

            result = FaissRetrieverExecutor.load_docs(
                database_path="/tmp/test_db", llm_config=mock_llm_config
            )

            assert isinstance(result, FaissRetrieverExecutor)
            assert result.texts == mock_documents

    @staticmethod
    def test_faiss_retriever_file_not_found() -> None:  # noqa: N802
        """Test load_docs with file not found."""
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            pytest.raises(FileNotFoundError),
        ):
            FaissRetrieverExecutor.load_docs(database_path="/tmp", llm_config=None)


class TestRetrieveEngine:
    """Test the RetrieveEngine class."""

    def test_faiss_retrieve(self) -> None:
        """Test faiss_retrieve static method."""
        mock_state = Mock(spec=MessageState)
        mock_state.user_message = Mock()
        mock_state.user_message.history = "test query"
        mock_state.bot_config = Mock()
        mock_state.bot_config.llm_config = Mock()

        mock_retriever = Mock()
        mock_retriever.search.return_value = (
            "retrieved text",
            [{"content": "doc1", "source": "test1.txt", "confidence": 0.8}],
        )

        with (
            patch("os.environ.get", return_value="/tmp/test_db"),
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.load_prompts",
                return_value={"retrieve_contextualize_q_prompt": "test prompt"},
            ),
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.FaissRetrieverExecutor.load_docs"
            ) as mock_load_docs,
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.trace"
            ) as mock_trace,
        ):
            mock_load_docs.return_value = mock_retriever
            mock_trace.return_value = mock_state

            result = RetrieveEngine.faiss_retrieve(mock_state)

            assert result == mock_state
            assert mock_state.message_flow == "retrieved text"
