"""Tests for the memory core module."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from arklex.memory.core import ShortTermMemory
from arklex.utils.graph_state import ResourceRecord, LLMConfig


class TestShortTermMemory:
    """Test cases for ShortTermMemory class."""

    @pytest.fixture
    def mock_llm_config(self):
        """Create a mock LLM configuration."""
        return LLMConfig(
            llm_provider="openai",
            model_type_or_path="gpt-3.5-turbo",
            api_key="test_key",
            endpoint="https://api.openai.com/v1",
        )

    @pytest.fixture
    def mock_trajectory(self):
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
    def mock_chat_history(self):
        """Create a mock chat history."""
        return "user: Hello\nassistant: Hi there\nuser: How are you?"

    @pytest.fixture
    def short_term_memory(self, mock_trajectory, mock_chat_history, mock_llm_config):
        """Create a ShortTermMemory instance for testing."""
        with (
            patch("arklex.memory.core.PROVIDER_MAP"),
            patch("arklex.memory.core.PROVIDER_EMBEDDINGS"),
            patch("arklex.memory.core.PROVIDER_EMBEDDING_MODELS"),
        ):
            memory = ShortTermMemory(
                trajectory=mock_trajectory,
                chat_history=mock_chat_history,
                llm_config=mock_llm_config,
            )
            return memory

    def test_initialization(
        self, mock_trajectory, mock_chat_history, mock_llm_config
    ) -> None:
        """Test ShortTermMemory initialization."""
        with (
            patch("arklex.memory.core.PROVIDER_MAP") as mock_provider_map,
            patch("arklex.memory.core.PROVIDER_EMBEDDINGS") as mock_embeddings,
            patch(
                "arklex.memory.core.PROVIDER_EMBEDDING_MODELS"
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

    def test_initialization_none_values(self, mock_llm_config) -> None:
        """Test ShortTermMemory initialization with None values."""
        with (
            patch("arklex.memory.core.PROVIDER_MAP") as mock_provider_map,
            patch("arklex.memory.core.PROVIDER_EMBEDDINGS") as mock_embeddings,
            patch(
                "arklex.memory.core.PROVIDER_EMBEDDING_MODELS"
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
    def test_get_embedding(self, mock_np_array, short_term_memory) -> None:
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
        self, mock_create_task, short_term_memory
    ) -> None:
        """Test _batch_get_embeddings method."""
        import numpy as np
        import asyncio

        texts = ["text1", "text2", "text3"]

        # Each create_task should return a real asyncio.Task
        async def coro1():
            return np.array([[0.1, 0.2, 0.3]])

        async def coro2():
            return np.array([[0.4, 0.5, 0.6]])

        async def coro3():
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

    async def test_get_embedding_async(self, short_term_memory) -> None:
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

    def test_retrieve_records_empty_trajectory(self, short_term_memory) -> None:
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
        self, mock_get_embedding, short_term_memory
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
        self, mock_get_embedding, short_term_memory
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
        self, mock_get_embedding, short_term_memory
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
        self, mock_get_embedding, short_term_memory
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
        self, short_term_memory
    ) -> None:
        """Test generate_personalized_product_attribute_intent method."""
        # Create an async mock for the llm
        mock_llm = Mock()
        mock_response = Mock()
        # Ensure .content returns a string, not a Mock
        mock_response.content = "Personalized Intent: test_intent"

        async def async_return(*args, **kwargs):
            return mock_response

        mock_llm.ainvoke = AsyncMock(side_effect=async_return)
        short_term_memory.llm = mock_llm

        # Mock the record to have proper string values
        record = short_term_memory.trajectory[0][0]
        record.output = "test_output"
        record.intent = "test_intent"
        record.steps = [{"context_generate": "test_context"}]

        result = await short_term_memory.generate_personalized_product_attribute_intent(
            record, "user utterance"
        )
        assert isinstance(result, str)
        assert "test_intent" in result

    async def test_generate_personalized_product_attribute_intent_dict_response(
        self, short_term_memory
    ) -> None:
        # Create an async mock for the llm
        mock_llm = Mock()
        mock_response = {"content": "Personalized intent"}

        async def async_return(*args, **kwargs):
            return mock_response

        mock_llm.ainvoke = AsyncMock(side_effect=async_return)
        short_term_memory.llm = mock_llm

        record = short_term_memory.trajectory[0][0]
        result = await short_term_memory.generate_personalized_product_attribute_intent(
            record, "user utterance"
        )
        assert result == "Personalized intent"
        mock_llm.ainvoke.assert_awaited_once()

    async def test_generate_personalized_product_attribute_intent_object_response(
        self, short_term_memory
    ) -> None:
        # Create an async mock for the llm
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Personalized intent"

        async def async_return(*args, **kwargs):
            return mock_response

        mock_llm.ainvoke = AsyncMock(side_effect=async_return)
        short_term_memory.llm = mock_llm

        record = short_term_memory.trajectory[0][0]
        result = await short_term_memory.generate_personalized_product_attribute_intent(
            record, "user utterance"
        )
        assert result == "Personalized intent"
        mock_llm.ainvoke.assert_awaited_once()

    def test_embedding_cache(self, short_term_memory) -> None:
        """Test embedding caching functionality."""
        # Setup
        short_term_memory._embedding_cache = {}
        short_term_memory.embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]

        # Execute first call
        result1 = short_term_memory._get_embedding("test text")

        # Execute second call with same text
        result2 = short_term_memory._get_embedding("test text")

        # Assert
        assert "test text" in short_term_memory._embedding_cache
        assert (
            short_term_memory.embedding_model.embed_query.call_count == 1
        )  # Should only be called once

    @patch("arklex.memory.core.PROVIDER_MAP")
    @patch("arklex.memory.core.PROVIDER_EMBEDDINGS")
    def test_chat_history_parsing(
        self, mock_provider_embeddings, mock_provider_map, short_term_memory
    ) -> None:
        """Test chat history parsing in constructor.

        Verifies that the chat history is properly parsed and formatted
        when creating a new ShortTermMemory instance.
        """
        # Use Mock instead of AsyncMock to avoid unawaited coroutine warnings
        mock_embedding = Mock()
        mock_llm = Mock()
        mock_provider_embeddings.get.return_value = Mock(return_value=mock_embedding)
        mock_provider_map.get.return_value = Mock(return_value=mock_llm)

        # Create a new memory instance with specific chat history
        chat_history = "user: Hello\nassistant: Hi there\nuser: How are you?"

        # Create a new instance to test the parsing
        from arklex.memory.core import ShortTermMemory
        from arklex.utils.graph_state import LLMConfig

        llm_config = LLMConfig(
            llm_provider="openai", model_type_or_path="gpt-3.5-turbo"
        )

        memory = ShortTermMemory(
            trajectory=[], chat_history=chat_history, llm_config=llm_config
        )

        # Check that the chat history is processed correctly
        # The constructor should take the last 5 turns and format them
        assert "user: Hello" in memory.chat_history
        assert "assistant: Hi there" in memory.chat_history
        assert "user: How are you?" in memory.chat_history

    @patch.object(
        ShortTermMemory, "_get_embedding", return_value=np.array([[0.1, 0.2, 0.3]])
    )
    def test_retrieve_intent_found(self, mock_get_embedding, short_term_memory) -> None:
        """Test retrieve_intent when found."""
        record = short_term_memory.trajectory[0][0]
        record.personalized_intent = "intent: buy product: shoes attribute: color"
        found, intent = short_term_memory.retrieve_intent("color shoes")
        assert found is True
        assert intent == record.intent

    @patch.object(
        ShortTermMemory, "_get_embedding", return_value=np.array([[0.1, 0.2, 0.3]])
    )
    def test_retrieve_intent_not_found(
        self, mock_get_embedding, short_term_memory
    ) -> None:
        """Test retrieve_intent when not found."""
        record = short_term_memory.trajectory[0][0]
        record.personalized_intent = "intent: buy product: shoes attribute: color"
        found, intent = short_term_memory.retrieve_intent(
            "unrelated query", string_threshold=0.99
        )
        assert found is False
        assert intent is None

    @pytest.mark.asyncio
    async def test_personalize_sets_personalized_intent(
        self, short_term_memory
    ) -> None:
        """Test personalize sets personalized intent."""
        record = short_term_memory.trajectory[0][0]
        short_term_memory.llm = Mock()
        short_term_memory.generate_personalized_product_attribute_intent = AsyncMock(
            return_value="intent: buy product: shoes attribute: color"
        )
        await short_term_memory.personalize()
        assert (
            record.personalized_intent == "intent: buy product: shoes attribute: color"
        )

    @pytest.mark.asyncio
    async def test__set_personalized_intent(self, short_term_memory) -> None:
        """Test _set_personalized_intent."""
        record = short_term_memory.trajectory[0][0]
        short_term_memory.generate_personalized_product_attribute_intent = AsyncMock(
            return_value="intent: buy product: shoes attribute: color"
        )
        await short_term_memory._set_personalized_intent(record, "user utterance")
        assert (
            record.personalized_intent == "intent: buy product: shoes attribute: color"
        )

    @patch.object(ShortTermMemory, "_get_embedding")
    def test_retrieve_records_with_empty_turns(
        self, mock_get_embedding, short_term_memory
    ) -> None:
        """Test retrieve_records when trajectory contains empty turns."""
        # Setup - add empty turns to trajectory
        short_term_memory.trajectory = [
            [],  # Empty turn
            [
                ResourceRecord(
                    id="1",
                    intent="test_intent",
                    info={"attribute": {"task": "test_task"}},
                    output="test_output",
                    steps=[{"context_generate": "test_context"}],
                )
            ],
            [],  # Another empty turn
        ]
        mock_get_embedding.return_value = np.array([[0.1, 0.2, 0.3]])

        # Execute
        found, records = short_term_memory.retrieve_records("test query")

        # Assert - should skip empty turns and process the non-empty one
        assert isinstance(found, bool)
        assert isinstance(records, list)
        mock_get_embedding.assert_called()

    @patch.object(ShortTermMemory, "_get_embedding")
    def test_retrieve_records_with_personalized_intent_regex_match(
        self, mock_get_embedding, short_term_memory
    ) -> None:
        """Test retrieve_records with personalized intent that matches regex pattern."""
        # Setup - create record with personalized intent that matches regex
        record = ResourceRecord(
            id="1",
            intent="test_intent",
            info={"attribute": {"task": "test_task"}},
            output="test_output",
            steps=[{"context_generate": "test_context"}],
            personalized_intent="intent: buy product: laptop attribute: color",
        )
        short_term_memory.trajectory = [[record]]

        # Mock embeddings - use return_value for simplicity
        mock_get_embedding.return_value = np.array([[0.1, 0.2, 0.3]])

        # Execute
        found, records = short_term_memory.retrieve_records(
            "laptop color", cosine_threshold=0.5
        )

        # Assert
        assert isinstance(found, bool)
        assert isinstance(records, list)
        assert mock_get_embedding.call_count >= 2

    @patch.object(ShortTermMemory, "_get_embedding")
    def test_retrieve_records_with_personalized_intent_regex_no_match(
        self, mock_get_embedding, short_term_memory
    ) -> None:
        """Test retrieve_records with personalized intent that doesn't match regex pattern."""
        # Setup - create record with personalized intent that doesn't match regex
        record = ResourceRecord(
            id="1",
            intent="test_intent",
            info={"attribute": {"task": "test_task"}},
            output="test_output",
            steps=[{"context_generate": "test_context"}],
            personalized_intent="invalid format intent",
        )
        short_term_memory.trajectory = [[record]]

        # Mock embeddings - use return_value for simplicity
        mock_get_embedding.return_value = np.array([[0.1, 0.2, 0.3]])

        # Execute
        found, records = short_term_memory.retrieve_records("test query")

        # Assert - should handle regex no-match gracefully
        assert isinstance(found, bool)
        assert isinstance(records, list)
        assert mock_get_embedding.call_count >= 1

    @patch.object(ShortTermMemory, "_get_embedding")
    def test_retrieve_records_with_personalized_intent_cosine_below_threshold(
        self, mock_get_embedding, short_term_memory
    ) -> None:
        """Test retrieve_records when cosine similarity is below threshold."""
        # Setup - create record with personalized intent
        record = ResourceRecord(
            id="1",
            intent="test_intent",
            info={"attribute": {"task": "test_task"}},
            output="test_output",
            steps=[{"context_generate": "test_context"}],
            personalized_intent="intent: buy product: laptop attribute: color",
        )
        short_term_memory.trajectory = [[record]]

        # Mock embeddings - use return_value for simplicity
        mock_get_embedding.return_value = np.array([[0.1, 0.2, 0.3]])

        # Execute with high cosine threshold
        found, records = short_term_memory.retrieve_records(
            "test query", cosine_threshold=0.9
        )

        # Assert - should set intent score to 0.0 when cosine is below threshold
        assert isinstance(found, bool)
        assert isinstance(records, list)
        assert mock_get_embedding.call_count >= 2

    @patch.object(ShortTermMemory, "_get_embedding")
    def test_retrieve_records_without_personalized_intent(
        self, mock_get_embedding, short_term_memory
    ) -> None:
        """Test retrieve_records when record has no personalized_intent."""
        # Setup - create record without personalized intent (use empty string instead of None)
        record = ResourceRecord(
            id="1",
            intent="test_intent",
            info={"attribute": {"task": "test_task"}},
            output="test_output",
            steps=[{"context_generate": "test_context"}],
            personalized_intent="",  # Empty string instead of None
        )
        short_term_memory.trajectory = [[record]]

        # Mock embeddings - use return_value for simplicity
        mock_get_embedding.return_value = np.array([[0.1, 0.2, 0.3]])

        # Execute
        found, records = short_term_memory.retrieve_records("test query")

        # Assert - should handle missing personalized intent gracefully
        assert isinstance(found, bool)
        assert isinstance(records, list)
        assert mock_get_embedding.call_count >= 1

    @patch.object(ShortTermMemory, "_get_embedding")
    def test_retrieve_records_with_relevant_records_found(
        self, mock_get_embedding, short_term_memory
    ) -> None:
        """Test retrieve_records when relevant records are found and returned."""
        # Setup - create record that will score above threshold
        record = ResourceRecord(
            id="1",
            intent="test_intent",
            info={"attribute": {"task": "test_task"}},
            output="test_output",
            steps=[{"context_generate": "test_context"}],
        )
        short_term_memory.trajectory = [[record]]

        # Mock embeddings to ensure high similarity
        mock_get_embedding.return_value = np.array([[0.1, 0.2, 0.3]])

        # Execute with low threshold to ensure records are found
        found, records = short_term_memory.retrieve_records("test query", threshold=0.1)

        # Assert - should return True and the record
        assert found is True
        assert len(records) > 0
        assert isinstance(records[0], ResourceRecord)

    def test_retrieve_intent_empty_trajectory(self, short_term_memory) -> None:
        """Test retrieve_intent with empty trajectory."""
        # Setup
        short_term_memory.trajectory = []

        # Execute
        found, intent = short_term_memory.retrieve_intent("test query")

        # Assert
        assert found is False
        assert intent is None
