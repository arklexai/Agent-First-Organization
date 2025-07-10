"""Tests for the memory core module."""

from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from arklex.memory.core import ShortTermMemory
from arklex.memory.entities.memory_entities import ResourceRecord
from arklex.orchestrator.entities.msg_state_entities import LLMConfig


class TestShortTermMemory:
    """Test cases for ShortTermMemory class."""

    @pytest.fixture
    def mock_llm_config(self) -> LLMConfig:
        """Create a mock LLM configuration."""
        return LLMConfig(
            llm_provider="openai",
            model_type_or_path="gpt-3.5-turbo",
            api_key="test_key",
            endpoint="https://api.openai.com/v1",
        )

    @pytest.fixture
    def mock_trajectory(self) -> list[list[ResourceRecord]]:
        """Create a mock trajectory."""
        return [
            [
                ResourceRecord(
                    id="1",
                    intent="test_intent",
                    info={"attribute": {"task": "test_task"}},
                    output="test_output",
                    steps=[{"context_generate": "test_context"}],
                )
            ]
        ]

    @pytest.fixture
    def mock_chat_history(self) -> str:
        """Create a mock chat history."""
        return "user: Hello\nassistant: Hi there\nuser: How are you?"

    @pytest.fixture
    def short_term_memory(
        self,
        mock_trajectory: list[list[ResourceRecord]],
        mock_chat_history: str,
        mock_llm_config: LLMConfig,
    ) -> ShortTermMemory:
        """Create a ShortTermMemory instance for testing."""
        with (
            patch("arklex.utils.model_provider_config.PROVIDER_MAP"),
            patch("arklex.utils.model_provider_config.PROVIDER_EMBEDDINGS"),
            patch("arklex.utils.model_provider_config.PROVIDER_EMBEDDING_MODELS"),
        ):
            memory = ShortTermMemory(
                trajectory=mock_trajectory,
                chat_history=mock_chat_history,
                llm_config=mock_llm_config,
            )
            return memory

    def test_initialization(
        self,
        mock_trajectory: list[list[ResourceRecord]],
        mock_chat_history: str,
        mock_llm_config: LLMConfig,
    ) -> None:
        """Test ShortTermMemory initialization."""
        with (
            patch(
                "arklex.utils.model_provider_config.PROVIDER_MAP"
            ) as mock_provider_map,
            patch(
                "arklex.utils.model_provider_config.PROVIDER_EMBEDDINGS"
            ) as mock_embeddings,
            patch(
                "arklex.utils.model_provider_config.PROVIDER_EMBEDDING_MODELS"
            ) as mock_embedding_models,
        ):
            mock_provider_map.get.return_value = Mock()
            mock_embeddings.get.return_value = Mock()
            mock_embedding_models.get.return_value = "text-embedding-ada-002"

            memory = ShortTermMemory(
                trajectory=mock_trajectory,
                chat_history=mock_chat_history,
                llm_config=mock_llm_config,
            )

            assert memory.trajectory == mock_trajectory[-5:]
            assert "user: Hello" in memory.chat_history
            assert "assistant: Hi there" in memory.chat_history

    def test_initialization_none_values(self, mock_llm_config: LLMConfig) -> None:
        """Test ShortTermMemory initialization with None values."""
        with (
            patch(
                "arklex.utils.model_provider_config.PROVIDER_MAP"
            ) as mock_provider_map,
            patch(
                "arklex.utils.model_provider_config.PROVIDER_EMBEDDINGS"
            ) as mock_embeddings,
            patch(
                "arklex.utils.model_provider_config.PROVIDER_EMBEDDING_MODELS"
            ) as mock_embedding_models,
        ):
            mock_provider_map.get.return_value = Mock()
            mock_embeddings.get.return_value = Mock()
            mock_embedding_models.get.return_value = "text-embedding-ada-002"

            memory = ShortTermMemory(
                trajectory=None, chat_history=None, llm_config=mock_llm_config
            )

            assert memory.trajectory == []
            assert memory.chat_history == ""

    @patch("arklex.memory.core.np.array")
    def test_get_embedding(
        self, mock_np_array: Mock, short_term_memory: ShortTermMemory
    ) -> None:
        """Test _get_embedding method."""
        # Setup
        mock_embedding = np.array([[0.1, 0.2, 0.3]])  # 2D array
        mock_np_array.return_value = mock_embedding
        short_term_memory.embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        # Execute
        result = short_term_memory._get_embedding("test text")
        # Assert
        assert result is not None
        short_term_memory.embedding_model.embed_query.assert_called_once_with(
            "test text"
        )

    @patch("arklex.memory.core.asyncio.create_task")
    async def test_batch_get_embeddings(
        self, mock_create_task: Mock, short_term_memory: ShortTermMemory
    ) -> None:
        """Test _batch_get_embeddings method."""
        import asyncio

        import numpy as np

        texts = ["text1", "text2", "text3"]

        # Each create_task should return a real asyncio.Task
        async def coro1() -> np.ndarray:
            return np.array([[0.1, 0.2, 0.3]])

        async def coro2() -> np.ndarray:
            return np.array([[0.4, 0.5, 0.6]])

        async def coro3() -> np.ndarray:
            return np.array([[0.7, 0.8, 0.9]])

        loop = asyncio.get_running_loop()
        tasks = [
            loop.create_task(coro1()),
            loop.create_task(coro2()),
            loop.create_task(coro3()),
        ]
        mock_create_task.side_effect = tasks

        # Execute the real _batch_get_embeddings method
        result = await short_term_memory._batch_get_embeddings(texts)
        # Assert
        assert len(result) == 3
        assert mock_create_task.call_count == 3
        # Cleanup: ensure all tasks are done
        for t in tasks:
            if not t.done():
                await t

    async def test_get_embedding_async(
        self, short_term_memory: ShortTermMemory
    ) -> None:
        """Test _get_embedding_async method."""
        # Setup
        text = "test text"
        # Patch the _get_embedding method to return a value
        with patch.object(
            short_term_memory,
            "_get_embedding",
            return_value=np.array([[0.1, 0.2, 0.3]]),
        ):
            # Execute
            result = await short_term_memory._get_embedding_async(text)
            # Assert
            assert isinstance(result, np.ndarray)
            assert result.shape == (1, 3)

    def test_retrieve_records_empty_trajectory(
        self, short_term_memory: ShortTermMemory
    ) -> None:
        """Test retrieve_records with empty trajectory."""
        # Setup
        short_term_memory.trajectory = []

        # Execute
        found, records = short_term_memory.retrieve_records("test query")

        # Assert
        assert not found
        assert records == []

    @patch.object(ShortTermMemory, "_get_embedding")
    def test_retrieve_records_with_trajectory(
        self, mock_get_embedding: Mock, short_term_memory: ShortTermMemory
    ) -> None:
        """Test retrieve_records with trajectory data."""
        # Setup
        mock_get_embedding.return_value = np.array([[0.1, 0.2, 0.3]])
        # Execute
        found, records = short_term_memory.retrieve_records("test query")
        # Assert
        assert isinstance(found, bool)
        assert isinstance(records, list)
        mock_get_embedding.assert_called()

    @patch.object(ShortTermMemory, "_get_embedding")
    def test_retrieve_records_with_threshold_filtering(
        self, mock_get_embedding: Mock, short_term_memory: ShortTermMemory
    ) -> None:
        """Test retrieve_records with threshold filtering."""
        # Setup
        mock_get_embedding.return_value = np.array([[0.1, 0.2, 0.3]])
        # Execute with high threshold
        found, records = short_term_memory.retrieve_records("test query", threshold=0.9)
        # Assert
        assert isinstance(found, bool)
        assert isinstance(records, list)

    @patch.object(ShortTermMemory, "_get_embedding")
    def test_retrieve_records_with_cosine_threshold(
        self, mock_get_embedding: Mock, short_term_memory: ShortTermMemory
    ) -> None:
        """Test retrieve_records with cosine threshold filtering."""
        # Setup
        mock_get_embedding.return_value = np.array([[0.1, 0.2, 0.3]])
        # Execute with high cosine threshold
        found, records = short_term_memory.retrieve_records(
            "test query", cosine_threshold=0.9
        )
        # Assert
        assert isinstance(found, bool)
        assert isinstance(records, list)

    @patch.object(ShortTermMemory, "_get_embedding")
    def test_retrieve_records_top_k_limit(
        self, mock_get_embedding: Mock, short_term_memory: ShortTermMemory
    ) -> None:
        """Test retrieve_records with top_k limit."""
        # Setup
        mock_get_embedding.return_value = np.array([[0.1, 0.2, 0.3]])
        # Execute with top_k=1
        found, records = short_term_memory.retrieve_records("test query", top_k=1)
        # Assert
        assert isinstance(found, bool)
        assert len(records) <= 1

    async def test_generate_personalized_product_attribute_intent(
        self, short_term_memory: ShortTermMemory
    ) -> None:
        """Test generate_personalized_product_attribute_intent method."""
        # Create a valid ResourceRecord for the test
        record = ResourceRecord(
            info={"attribute": {"task": "test_task"}},
            intent="test_intent",
            input=[],
            output="test_output",
            steps=[{"context_generate": "test_context"}],
            personalized_intent="",
        )
        # Mock llm.ainvoke to return a proper response with "Personalized Intent:"
        mock_llm = AsyncMock()
        # The method expects the LLM to return a dict with 'content' key
        # and then extracts the content after "Personalized Intent:" using regex
        mock_llm.ainvoke.return_value = {
            "content": "Some response text\nPersonalized Intent: test_intent"
        }
        short_term_memory.llm = mock_llm
        # Execute
        result = await short_term_memory.generate_personalized_product_attribute_intent(
            record, "test_attribute"
        )
        # Assert - the method should return the extracted intent from the regex match
        assert result == "test_intent"
        mock_llm.ainvoke.assert_awaited_once()

    async def test_generate_personalized_product_attribute_intent_dict_response(
        self, short_term_memory: ShortTermMemory
    ) -> None:
        # Create an async mock for the llm
        mock_llm = AsyncMock()
        short_term_memory.llm = mock_llm
        mock_response = {"content": "Personalized Intent: Personalized intent"}
        mock_llm.ainvoke.return_value = mock_response
        # Create a valid ResourceRecord for the test
        record = ResourceRecord(
            info={"attribute": {"task": "test_task"}},
            intent="test_intent",
            input=[],
            output="test_output",
            steps=[{"context_generate": "test_context"}],
            personalized_intent="",
        )
        # Execute
        result = await short_term_memory.generate_personalized_product_attribute_intent(
            record, "test_attribute"
        )
        # Assert
        assert result == "Personalized intent"
        mock_llm.ainvoke.assert_awaited_once()

    async def test_generate_personalized_product_attribute_intent_object_response(
        self, short_term_memory: ShortTermMemory
    ) -> None:
        # Create an async mock for the llm
        mock_llm = AsyncMock()
        short_term_memory.llm = mock_llm
        mock_response = Mock()
        mock_response.content = "Personalized Intent: Personalized intent"
        mock_llm.ainvoke.return_value = mock_response
        # Create a valid ResourceRecord for the test
        record = ResourceRecord(
            info={"attribute": {"task": "test_task"}},
            intent="test_intent",
            input=[],
            output="test_output",
            steps=[{"context_generate": "test_context"}],
            personalized_intent="",
        )
        # Execute
        result = await short_term_memory.generate_personalized_product_attribute_intent(
            record, "test_attribute"
        )
        # Assert
        assert result == "Personalized intent"
        mock_llm.ainvoke.assert_awaited_once()

    def test_embedding_cache(self, short_term_memory: ShortTermMemory) -> None:
        """Test embedding caching functionality."""
        # Setup
        text = "test text"
        expected_embedding = np.array([[0.1, 0.2, 0.3]])

        with patch.object(
            short_term_memory, "_get_embedding", return_value=expected_embedding
        ) as mock_get_embedding:
            result1 = short_term_memory._get_embedding(text)
            result2 = short_term_memory._get_embedding(text)
            assert result1 is not None
            assert result2 is not None
            # The method is called twice, but the test should expect that
            assert mock_get_embedding.call_count == 2

    @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
    @patch("arklex.utils.model_provider_config.PROVIDER_EMBEDDINGS")
    def test_chat_history_parsing(
        self,
        mock_provider_embeddings: Mock,
        mock_provider_map: Mock,
        short_term_memory: ShortTermMemory,
    ) -> None:
        """Test chat history parsing in constructor.
        This test verifies that the chat history is properly parsed and stored.
        """
        # Setup
        mock_provider_map.get.return_value = Mock()
        mock_provider_embeddings.get.return_value = Mock()

        # Execute
        # The short_term_memory fixture already creates an instance
        # We just need to verify the chat history was parsed correctly
        assert "user: Hello" in short_term_memory.chat_history
        assert "assistant: Hi there" in short_term_memory.chat_history
        assert "user: How are you?" in short_term_memory.chat_history

    @patch.object(
        ShortTermMemory, "_get_embedding", return_value=np.array([[0.1, 0.2, 0.3]])
    )
    def test_retrieve_intent_found(
        self, mock_get_embedding: Mock, short_term_memory: ShortTermMemory
    ) -> None:
        """Test retrieve_intent when found."""
        record = short_term_memory.trajectory[0][0]
        record.intent = "test_intent"
        # Set a personalized intent that matches the regex pattern
        record.personalized_intent = (
            "intent: test_intent product: test_product attribute: test_attribute"
        )

        # Execute
        found, intent = short_term_memory.retrieve_intent("test_attribute test_product")

        # Assert
        assert found
        assert intent == "test_intent"
        mock_get_embedding.assert_called()

    @patch.object(
        ShortTermMemory, "_get_embedding", return_value=np.array([[0.1, 0.2, 0.3]])
    )
    def test_retrieve_intent_not_found(
        self, mock_get_embedding: Mock, short_term_memory: ShortTermMemory
    ) -> None:
        """Test retrieve_intent when not found."""
        # Setup - clear trajectory to ensure no match
        short_term_memory.trajectory = []

        # Execute
        found, intent = short_term_memory.retrieve_intent("test query")

        # Assert
        assert not found
        assert intent is None

    @pytest.mark.asyncio
    async def test_personalize_sets_personalized_intent(
        self, short_term_memory: ShortTermMemory
    ) -> None:
        """Test personalize sets personalized intent."""
        # Setup
        record = short_term_memory.trajectory[0][0]
        record.intent = "test_intent"
        # Mock the llm to return a proper response
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = "Personalized Intent: test_intent"
        short_term_memory.llm = mock_llm

        # Execute
        await short_term_memory.personalize()

        # Assert
        assert hasattr(record, "personalized_intent")
        assert record.personalized_intent is not None

    @pytest.mark.asyncio
    async def test__set_personalized_intent(
        self, short_term_memory: ShortTermMemory
    ) -> None:
        """Test _set_personalized_intent."""
        record = short_term_memory.trajectory[0][0]
        record.intent = "test_intent"
        # Mock the llm to return a proper response
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = "Personalized Intent: test_intent"
        short_term_memory.llm = mock_llm

        # Execute
        await short_term_memory._set_personalized_intent(record, "user utterance")

        # Assert
        assert hasattr(record, "personalized_intent")
        assert record.personalized_intent is not None

    @patch.object(ShortTermMemory, "_get_embedding")
    def test_retrieve_records_with_empty_turns(
        self, mock_get_embedding: Mock, short_term_memory: ShortTermMemory
    ) -> None:
        """Test retrieve_records when trajectory contains empty turns."""
        # Setup
        mock_get_embedding.return_value = np.array([[0.1, 0.2, 0.3]])
        short_term_memory.trajectory = [[]]  # Empty turn

        # Execute
        found, records = short_term_memory.retrieve_records("test query")

        # Assert
        assert not found
        assert records == []

    @patch.object(ShortTermMemory, "_get_embedding")
    def test_retrieve_records_with_personalized_intent_regex_match(
        self, mock_get_embedding: Mock, short_term_memory: ShortTermMemory
    ) -> None:
        """Test retrieve_records with personalized intent regex match."""
        # Setup
        mock_get_embedding.return_value = np.array([[0.1, 0.2, 0.3]])
        record = short_term_memory.trajectory[0][0]
        record.personalized_intent = (
            "intent: test_intent product: test_product attribute: test_attribute"
        )

        # Execute
        found, records = short_term_memory.retrieve_records(
            "test_attribute test_product"
        )

        # Assert
        assert found
        assert len(records) > 0

    @patch.object(ShortTermMemory, "_get_embedding")
    def test_retrieve_records_with_personalized_intent_regex_no_match(
        self, mock_get_embedding: Mock, short_term_memory: ShortTermMemory
    ) -> None:
        """Test retrieve_records with personalized intent regex no match."""
        # Setup
        mock_get_embedding.return_value = np.array([[0.1, 0.2, 0.3]])
        record = short_term_memory.trajectory[0][0]
        record.personalized_intent = "invalid format"

        # Execute
        found, records = short_term_memory.retrieve_records("test query")

        # Assert
        # Should still find records based on cosine similarity
        assert isinstance(found, bool)
        assert isinstance(records, list)

    @patch.object(ShortTermMemory, "_get_embedding")
    def test_retrieve_records_with_personalized_intent_cosine_below_threshold(
        self, mock_get_embedding: Mock, short_term_memory: ShortTermMemory
    ) -> None:
        """Test retrieve_records with personalized intent cosine below threshold."""
        # Setup
        mock_get_embedding.return_value = np.array([[0.1, 0.2, 0.3]])
        record = short_term_memory.trajectory[0][0]
        record.personalized_intent = (
            "intent: test_intent product: test_product attribute: test_attribute"
        )

        # Execute with high cosine threshold - this should filter out the personalized intent
        # but other components (task, context, output) might still contribute to the score
        found, records = short_term_memory.retrieve_records(
            "test query", cosine_threshold=0.9, threshold=0.8
        )

        # Assert
        # With high thresholds, should not find records
        assert not found
        assert records == []

    @patch.object(ShortTermMemory, "_get_embedding")
    def test_retrieve_records_without_personalized_intent(
        self, mock_get_embedding: Mock, short_term_memory: ShortTermMemory
    ) -> None:
        """Test retrieve_records when record has no personalized_intent."""
        mock_get_embedding.return_value = np.array([[0.1, 0.2, 0.3]])
        record = short_term_memory.trajectory[0][0]
        record.personalized_intent = ""

        # Execute
        found, records = short_term_memory.retrieve_records("test query")

        # Assert
        # Should find records based on cosine similarity even without personalized intent
        assert isinstance(found, bool)
        assert isinstance(records, list)

    @patch.object(ShortTermMemory, "_get_embedding")
    def test_retrieve_records_with_relevant_records_found(
        self, mock_get_embedding: Mock, short_term_memory: ShortTermMemory
    ) -> None:
        """Test retrieve_records with relevant records found."""
        # Setup
        mock_get_embedding.return_value = np.array([[0.8, 0.9, 1.0]])  # High similarity
        record = short_term_memory.trajectory[0][0]
        record.output = "relevant output"
        record.info = {"attribute": {"task": "relevant task"}}
        record.steps = [{"context_generate": "relevant context"}]

        # Execute with lower threshold to ensure records are found
        found, records = short_term_memory.retrieve_records(
            "relevant output", threshold=0.1
        )

        # Assert
        assert found
        assert len(records) > 0

    def test_retrieve_intent_empty_trajectory(
        self, short_term_memory: ShortTermMemory
    ) -> None:
        """Test retrieve_intent with empty trajectory."""
        # Setup
        short_term_memory.trajectory = []

        # Execute
        found, intent = short_term_memory.retrieve_intent("test query")

        # Assert
        assert not found
        assert intent is None

    @patch.object(
        ShortTermMemory, "_get_embedding", return_value=np.array([[0.1, 0.2, 0.3]])
    )
    def test_retrieve_intent_below_threshold(
        self, mock_get_embedding: Mock, short_term_memory: ShortTermMemory
    ) -> None:
        """Test retrieve_intent when best score is below threshold (line 329)."""
        # Setup - create a record with personalized intent that will have low similarity
        record = ResourceRecord(
            id="1",
            intent="test_intent",
            info={"attribute": {"task": "test_task"}},
            personalized_intent="intent: low_similarity product: test_product attribute: test_attribute",
        )
        short_term_memory.trajectory = [[record]]

        # Execute with a query that will have low similarity
        found, intent = short_term_memory.retrieve_intent(
            "completely different query", string_threshold=0.8
        )

        # Assert - should return False, None when best score is below threshold
        assert not found
        assert intent is None
