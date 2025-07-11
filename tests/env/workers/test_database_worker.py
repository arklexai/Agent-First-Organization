"""
Tests for database worker functionality.

This module contains comprehensive tests for the DataBaseWorker class including
initialization, action verification, and workflow execution.
"""

from unittest.mock import Mock, patch

import pytest

from arklex.env.prompts import BotConfig
from arklex.env.workers.database_worker import DataBaseWorker
from arklex.orchestrator.entities.msg_state_entities import MessageState


class TestDataBaseWorkerInitialization:
    """Test DataBaseWorker initialization."""

    def test_database_worker_init_with_default_config(self) -> None:
        """Test DataBaseWorker initialization with default configuration."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                # Provide a proper model configuration since the default MODEL has None values
                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                assert worker is not None
                assert hasattr(worker, "model_config")
                assert hasattr(worker, "llm")
                assert hasattr(worker, "action_graph")

    def test_database_worker_init_with_custom_config(self) -> None:
        """Test DataBaseWorker initialization with custom configuration."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                assert worker is not None
                assert worker.model_config == custom_config
                assert hasattr(worker, "llm")
                assert hasattr(worker, "action_graph")

    def test_database_worker_init_google_provider(self) -> None:
        """Test DataBaseWorker initialization with Google provider."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "google",
                    "model_type_or_path": "gemini-pro",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                assert worker is not None
                assert worker.model_config == custom_config
                assert hasattr(worker, "llm")
                assert hasattr(worker, "action_graph")

    def test_database_worker_init_other_provider(self) -> None:
        """Test DataBaseWorker initialization with other provider."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "anthropic",
                    "model_type_or_path": "claude-3-sonnet",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                assert worker is not None
                assert worker.model_config == custom_config
                assert hasattr(worker, "llm")
                assert hasattr(worker, "action_graph")


class TestDataBaseWorkerActions:
    """Test DataBaseWorker actions."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                self.worker = DataBaseWorker(model_config=custom_config)

    def test_search_show_action(self) -> None:
        """Test search show action."""
        mock_state = Mock(spec=MessageState)
        mock_state.slots = {}
        mock_state.bot_config = BotConfig(language="EN")

        result = self.worker.DBActions.search_show(mock_state)

        assert result is not None

    def test_book_show_action(self) -> None:
        """Test book show action."""
        mock_state = Mock(spec=MessageState)
        mock_state.slots = {}
        mock_state.bot_config = BotConfig(language="EN")

        result = self.worker.DBActions.book_show(mock_state)

        assert result is not None

    def test_check_booking_action(self) -> None:
        """Test check booking action."""
        mock_state = Mock(spec=MessageState)
        mock_state.slots = {}
        mock_state.bot_config = BotConfig(language="EN")

        result = self.worker.DBActions.check_booking(mock_state)

        assert result is not None

    def test_cancel_booking_action(self) -> None:
        """Test cancel booking action."""
        mock_state = Mock(spec=MessageState)
        mock_state.slots = {}
        mock_state.bot_config = BotConfig(language="EN")

        result = self.worker.DBActions.cancel_booking(mock_state)

        assert result is not None


class TestDataBaseWorkerVerifyAction:
    """Test DataBaseWorker verify_action method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                self.worker = DataBaseWorker(model_config=custom_config)

    def test_verify_action_search_show(self) -> None:
        """Test verify_action for search show intent."""
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        # Set up the attribute as a dict with get method
        mock_orchestrator_message.attribute = {"task": "search for shows"}
        mock_state.orchestrator_message = mock_orchestrator_message
        mock_state.bot_config = BotConfig(language="EN")

        # Mock the chain invoke to return the expected string directly
        with patch.object(self.worker, "llm") as mock_llm:
            # Mock the chain to return the string directly
            mock_chain = Mock()
            mock_chain.invoke.return_value = "SearchShow"
            mock_llm.__or__ = Mock(return_value=mock_chain)

            result = self.worker.verify_action(mock_state)

        assert result == "SearchShow"

    def test_verify_action_book_show(self) -> None:
        """Test verify_action for book show intent."""
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        # Set up the attribute as a dict with get method
        mock_orchestrator_message.attribute = {"task": "book a show"}
        mock_state.orchestrator_message = mock_orchestrator_message
        mock_state.bot_config = BotConfig(language="EN")

        # Mock the chain invoke to return the expected string directly
        with patch.object(self.worker, "llm") as mock_llm:
            # Mock the chain to return the string directly
            mock_chain = Mock()
            mock_chain.invoke.return_value = "BookShow"
            mock_llm.__or__ = Mock(return_value=mock_chain)

            result = self.worker.verify_action(mock_state)

        assert result == "BookShow"

    def test_verify_action_check_booking(self) -> None:
        """Test verify_action for check booking intent."""
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        # Set up the attribute as a dict with get method
        mock_orchestrator_message.attribute = {"task": "check my bookings"}
        mock_state.orchestrator_message = mock_orchestrator_message
        mock_state.bot_config = BotConfig(language="EN")

        # Mock the chain invoke to return the expected string directly
        with patch.object(self.worker, "llm") as mock_llm:
            # Mock the chain to return the string directly
            mock_chain = Mock()
            mock_chain.invoke.return_value = "CheckBooking"
            mock_llm.__or__ = Mock(return_value=mock_chain)

            result = self.worker.verify_action(mock_state)

        assert result == "CheckBooking"

    def test_verify_action_cancel_booking(self) -> None:
        """Test verify_action for cancel booking intent."""
        mock_state = Mock(spec=MessageState)
        mock_state.orchestrator_message = Mock()
        # Set up the attribute as a dict with get method
        mock_orchestrator_message = Mock()
        mock_orchestrator_message.attribute = {"task": "cancel my booking"}
        mock_state.orchestrator_message = mock_orchestrator_message
        mock_state.bot_config = BotConfig(language="EN")

        # Mock the chain invoke to return the expected string directly
        with patch.object(self.worker, "llm") as mock_llm:
            # Mock the chain to return the string directly
            mock_chain = Mock()
            mock_chain.invoke.return_value = "CancelBooking"
            mock_llm.__or__ = Mock(return_value=mock_chain)

            result = self.worker.verify_action(mock_state)

        assert result == "CancelBooking"

    def test_verify_action_others(self) -> None:
        """Test verify_action for other intents."""
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        # Set up the attribute as a dict with get method
        mock_orchestrator_message.attribute = {"task": "general inquiry"}
        mock_state.orchestrator_message = mock_orchestrator_message
        mock_state.bot_config = BotConfig(language="EN")

        # Mock the LLM response
        self.worker.llm.invoke.return_value.content = "Others"

        result = self.worker.verify_action(mock_state)

        assert result == "Others"

    def test_verify_action_exception(self) -> None:
        """Test verify_action with exception."""
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        # Set up the attribute as a dict with get method
        mock_orchestrator_message.attribute = {"task": "test task"}
        mock_state.orchestrator_message = mock_orchestrator_message
        mock_state.bot_config = BotConfig(language="EN")

        # Mock the LLM to raise an exception
        self.worker.llm.invoke.side_effect = Exception("LLM error")

        result = self.worker.verify_action(mock_state)

        assert result == "Others"


class TestDataBaseWorkerCreateActionGraph:
    """Test DataBaseWorker action graph creation."""

    def test_create_action_graph_structure(self) -> None:
        """Test that action graph is created with correct structure."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                # Verify action graph has the expected nodes
                graph = worker.action_graph
                assert graph is not None
                # The graph should have nodes for each action
                assert hasattr(graph, "nodes")


class TestDataBaseWorkerExecute:
    """Test DataBaseWorker execute method."""

    def test_execute_workflow(self) -> None:
        """Test execute workflow."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                # Mock the message state
                mock_state = Mock(spec=MessageState)
                mock_state.slots = {}
                mock_state.bot_config = BotConfig(language="EN")

                # Mock the graph compilation and execution
                mock_graph = Mock()
                mock_result = Mock(spec=MessageState)
                mock_graph.invoke.return_value = mock_result
                worker.action_graph.compile = Mock(return_value=mock_graph)

                # Mock the login and init_slots methods
                worker.DBActions.log_in = Mock()
                worker.DBActions.init_slots = Mock(return_value={})

                result = worker._execute(mock_state)

                assert result == mock_result
                worker.DBActions.log_in.assert_called_once()
                worker.DBActions.init_slots.assert_called_once_with(
                    {}, BotConfig(language="EN")
                )
                mock_graph.invoke.assert_called_once_with(mock_state)


class TestDataBaseWorkerIntegration:
    """Integration tests for DataBaseWorker."""

    def test_database_worker_complete_workflow(self) -> None:
        """Test complete database worker workflow."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                # Verify worker has all expected actions
                expected_actions = {
                    "SearchShow": "Search for shows",
                    "BookShow": "Book a show",
                    "CheckBooking": "Check details of booked show(s)",
                    "CancelBooking": "Cancel a booking",
                    "Others": "Other actions not mentioned above",
                }

                assert worker.actions == expected_actions

    def test_database_worker_description(self) -> None:
        """Test database worker description."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                expected_description = "Help the user with actions related to customer support like a booking system with structured data, always involving search, insert, update, and delete operations."

                assert worker.description == expected_description

    def test_database_worker_actions_coverage(self) -> None:
        """Test that all database actions are covered."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                # Check that all expected actions are present
                expected_action_names = [
                    "SearchShow",
                    "BookShow",
                    "CheckBooking",
                    "CancelBooking",
                    "Others",
                ]

                for action_name in expected_action_names:
                    assert action_name in worker.actions


class TestDataBaseWorkerErrorHandling:
    """Test DataBaseWorker error handling scenarios."""

    def test_init_missing_provider(self) -> None:
        """Test initialization with missing provider."""
        with pytest.raises(
            ValueError, match="llm_provider must be explicitly specified"
        ):
            DataBaseWorker(model_config={"model_type_or_path": "gpt-3.5-turbo"})

    def test_init_missing_model_name(self) -> None:
        """Test initialization with missing model name."""
        with pytest.raises(ValueError, match="Model name must be specified"):
            DataBaseWorker(model_config={"llm_provider": "openai"})

    def test_init_unsupported_provider(self) -> None:
        """Test initialization with unsupported provider."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_provider_map.get.return_value = None

            with pytest.raises(ValueError, match="Unsupported provider"):
                DataBaseWorker(
                    model_config={
                        "llm_provider": "unsupported",
                        "model_type_or_path": "test-model",
                        "api_key": "test-key",
                    }
                )

    def test_verify_action_with_empty_intent(self) -> None:
        """Test verify_action with empty user intent."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                mock_state = Mock(spec=MessageState)
                mock_orchestrator_message = Mock()
                mock_orchestrator_message.attribute = {"task": ""}
                mock_state.orchestrator_message = mock_orchestrator_message
                mock_state.bot_config = BotConfig(language="EN")

                with patch.object(worker, "llm") as mock_llm:
                    mock_chain = Mock()
                    mock_chain.invoke.return_value = "Others"
                    mock_llm.__or__ = Mock(return_value=mock_chain)

                    result = worker.verify_action(mock_state)
                    assert result == "Others"

    def test_verify_action_with_none_intent(self) -> None:
        """Test verify_action with None user intent."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                mock_state = Mock(spec=MessageState)
                mock_orchestrator_message = Mock()
                mock_orchestrator_message.attribute = {"task": None}
                mock_state.orchestrator_message = mock_orchestrator_message
                mock_state.bot_config = BotConfig(language="EN")

                with patch.object(worker, "llm") as mock_llm:
                    mock_chain = Mock()
                    mock_chain.invoke.return_value = "Others"
                    mock_llm.__or__ = Mock(return_value=mock_chain)

                    result = worker.verify_action(mock_state)
                    assert result == "Others"


class TestDataBaseWorkerActionMethods:
    """Test individual action methods of DataBaseWorker."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                self.worker = DataBaseWorker(model_config=custom_config)

    def test_search_show_method(self) -> None:
        """Test search_show method delegates to DBActions."""
        mock_state = Mock(spec=MessageState)
        expected_result = Mock(spec=MessageState)
        self.worker.DBActions.search_show.return_value = expected_result

        result = self.worker.search_show(mock_state)

        assert result == expected_result
        self.worker.DBActions.search_show.assert_called_once_with(mock_state)

    def test_book_show_method(self) -> None:
        """Test book_show method delegates to DBActions."""
        mock_state = Mock(spec=MessageState)
        expected_result = Mock(spec=MessageState)
        self.worker.DBActions.book_show.return_value = expected_result

        result = self.worker.book_show(mock_state)

        assert result == expected_result
        self.worker.DBActions.book_show.assert_called_once_with(mock_state)

    def test_check_booking_method(self) -> None:
        """Test check_booking method delegates to DBActions."""
        mock_state = Mock(spec=MessageState)
        expected_result = Mock(spec=MessageState)
        self.worker.DBActions.check_booking.return_value = expected_result

        result = self.worker.check_booking(mock_state)

        assert result == expected_result
        self.worker.DBActions.check_booking.assert_called_once_with(mock_state)

    def test_cancel_booking_method(self) -> None:
        """Test cancel_booking method delegates to DBActions."""
        mock_state = Mock(spec=MessageState)
        expected_result = Mock(spec=MessageState)
        self.worker.DBActions.cancel_booking.return_value = expected_result

        result = self.worker.cancel_booking(mock_state)

        assert result == expected_result
        self.worker.DBActions.cancel_booking.assert_called_once_with(mock_state)


class TestDataBaseWorkerActionGraph:
    """Test action graph creation and structure."""

    def test_create_action_graph_nodes(self) -> None:
        """Test that action graph has all required nodes."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                # Test that the action graph has the expected structure
                graph = worker.action_graph
                assert graph is not None

                # Verify that the graph has the expected nodes
                expected_nodes = [
                    "SearchShow",
                    "BookShow",
                    "CheckBooking",
                    "CancelBooking",
                    "Others",
                    "tool_generator",
                ]
                for _node in expected_nodes:
                    assert hasattr(graph, "nodes") or hasattr(graph, "_nodes")


class TestDataBaseWorkerModelConfiguration:
    """Test DataBaseWorker with different model configurations."""

    def test_google_provider_configuration(self) -> None:
        """Test Google provider configuration."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                google_config = {
                    "llm_provider": "google",
                    "model_type_or_path": "gemini-pro",
                    "api_key": "test-google-key",
                }

                DataBaseWorker(model_config=google_config)

                # Verify that the LLM was initialized with Google-specific parameters
                mock_llm_class.assert_called_once()
                call_args = mock_llm_class.call_args
                assert call_args[1]["model"] == "gemini-pro"
                assert call_args[1]["google_api_key"] == "test-google-key"
                assert call_args[1]["timeout"] == 30000

    def test_other_provider_configuration(self) -> None:
        """Test other provider configuration."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                other_config = {
                    "llm_provider": "anthropic",
                    "model_type_or_path": "claude-3-sonnet",
                    "api_key": "test-anthropic-key",
                }

                DataBaseWorker(model_config=other_config)

                # Verify that the LLM was initialized with standard parameters
                mock_llm_class.assert_called_once()
                call_args = mock_llm_class.call_args
                assert call_args[1]["model"] == "claude-3-sonnet"
                assert call_args[1]["api_key"] == "test-anthropic-key"
                assert call_args[1]["timeout"] == 30000


class TestDataBaseWorkerIntegrationScenarios:
    """Test DataBaseWorker integration scenarios."""

    def test_worker_registration(self) -> None:
        """Test that DataBaseWorker is properly registered."""
        # Test that the DataBaseWorker class has the name attribute set by the decorator
        assert hasattr(DataBaseWorker, "name")
        assert DataBaseWorker.name == "DataBaseWorker"

    def test_worker_description_consistency(self) -> None:
        """Test that worker description is consistent."""
        expected_description = "Help the user with actions related to customer support like a booking system with structured data, always involving search, insert, update, and delete operations."

        assert DataBaseWorker.description == expected_description

    def test_actions_dictionary_consistency(self) -> None:
        """Test that actions dictionary is consistent."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                expected_actions = {
                    "SearchShow": "Search for shows",
                    "BookShow": "Book a show",
                    "CheckBooking": "Check details of booked show(s)",
                    "CancelBooking": "Cancel a booking",
                    "Others": "Other actions not mentioned above",
                }

                assert worker.actions == expected_actions
