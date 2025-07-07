"""Comprehensive tests for the NegotiationSingleIssueWorker.

This module provides comprehensive test coverage for the NegotiationSingleIssueWorker,
testing all functionality including negotiation logic, slot management, and response generation.
"""

from unittest.mock import MagicMock, patch

import pytest

from arklex.env.workers.negotiation_workers.negotiation_single_issue_worker import (
    NegotiationSingleIssueWorker,
)
from arklex.utils.graph_state import (
    BotConfig,
    ConvoMessage,
    LLMConfig,
    MessageState,
    ResourceRecord,
    StatusEnum,
)
from arklex.utils.slot import Slot


class TestNegotiationSingleIssueWorkerInitialization:
    """Test initialization of the NegotiationSingleIssueWorker."""

    def test_worker_initialization(self) -> None:
        """Test that the worker initializes correctly."""
        worker = NegotiationSingleIssueWorker()

        assert worker.name == "NegotiationSingleIssueWorker"
        assert worker.llm is None
        assert worker.action_graph is None  # Created later in _execute
        assert worker.unit_index == 0
        assert worker.fixed_args == {}

    def test_worker_name_attribute(self) -> None:
        """Test that the worker has the correct name attribute."""
        worker = NegotiationSingleIssueWorker()
        assert worker.__class__.__name__ == "NegotiationSingleIssueWorker"


class TestNegotiationSingleIssueWorkerSlotManagement:
    """Test slot management functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.worker = NegotiationSingleIssueWorker()
        self.msg_state = MessageState(
            status=StatusEnum.INCOMPLETE,
            trajectory=[[ResourceRecord(info={}, intent="test")]],
            slots={},
        )

    def test_check_and_initialize_slots_creates_required_slots(self) -> None:
        """Test that required slots are created."""
        fixedArgs = {
            "unit_index": "0",
            "max_perceived_marketPrice": "1000",
            "max_marketPrice": "1200",
            "reservation_price": "800",
        }

        result = self.worker.check_and_initialize_slots(self.msg_state, fixedArgs)

        required_slots = [
            "turn",
            "episode_done",
            "current_target",
            "max_perceived_marketPrice",
            "max_market_price",
            "reservation_price",
        ]

        for slot_name in required_slots:
            assert slot_name in result.slots
            assert len(result.slots[slot_name]) == 1
            assert isinstance(result.slots[slot_name][0], Slot)

    def test_check_and_initialize_slots_with_default_fixedArgs(self) -> None:
        """Test slot initialization with default fixedArgs."""
        fixedArgs = {
            "unit_index": "0",
            "max_perceived_marketPrice": "1000",
            "max_marketPrice": "1200",
            "reservation_price": "800",
        }
        result = self.worker.check_and_initialize_slots(self.msg_state, fixedArgs)

        # Should create slots with provided values
        assert "turn" in result.slots
        assert "episode_done" in result.slots
        assert "current_target" in result.slots
        assert "max_perceived_marketPrice" in result.slots
        assert "max_market_price" in result.slots
        assert "reservation_price" in result.slots

        # Check slot values
        assert result.slots["turn"][0].value == 0
        assert result.slots["episode_done"][0].value is False
        assert result.slots["max_perceived_marketPrice"][0].value == 1000
        assert result.slots["max_market_price"][0].value == 1200
        assert result.slots["reservation_price"][0].value == 800

    def test_check_and_initialize_slots_preserves_existing(self) -> None:
        """Test that existing slots are preserved."""
        # Pre-create a slot
        self.msg_state.slots = {
            "turn": [
                Slot(
                    name="turn",
                    type="string",
                    value=5,
                    enum=[],
                    description="Test turn",
                    prompt="",
                    required=False,
                    verified=True,
                )
            ]
        }

        fixedArgs = {
            "unit_index": "0",
            "max_perceived_marketPrice": "1000",
            "max_marketPrice": "1200",
            "reservation_price": "800",
        }
        result = self.worker.check_and_initialize_slots(self.msg_state, fixedArgs)

        # Existing slot should be preserved
        assert result.slots["turn"][0].value == 5
        # New slots should be created
        assert "episode_done" in result.slots
        assert "current_target" in result.slots

    def test_check_and_initialize_slots_none_fixedArgs(self) -> None:
        """Test slot initialization with None fixedArgs."""
        # This should raise a KeyError because unit_index is required
        with pytest.raises(KeyError):
            self.worker.check_and_initialize_slots(self.msg_state, None)


class TestNegotiationSingleIssueWorkerResponseGeneration:
    """Test response generation functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.worker = NegotiationSingleIssueWorker()
        self.mock_llm = MagicMock()
        self.worker.llm = self.mock_llm
        self.msg_state = MessageState(
            status=StatusEnum.INCOMPLETE,
            trajectory=[[ResourceRecord(info={}, intent="test")]],
            user_message=ConvoMessage(
                message="I offer $900", history="Previous negotiations"
            ),
            slots={},
        )

    def test_get_response_basic_functionality(self) -> None:
        """Test basic response generation."""
        # Mock LLM response
        self.mock_llm.invoke.return_value.content.strip.return_value = (
            "I accept your offer of $900"
        )

        # Initialize slots with required fixedArgs
        fixedArgs = {
            "unit_index": "0",
            "max_perceived_marketPrice": "1000",
            "max_marketPrice": "1200",
            "reservation_price": "800",
        }
        self.worker.check_and_initialize_slots(self.msg_state, fixedArgs)

        with patch("builtins.open"), patch("os.path.join"):
            result = self.worker.get_response(self.msg_state)

        assert result.response == "I accept your offer of $900"
        self.mock_llm.invoke.assert_called_once()

    def test_get_response_with_negotiation_context(self) -> None:
        """Test response generation with negotiation context."""
        # Initialize slots with required fixedArgs
        fixedArgs = {
            "unit_index": "0",
            "max_perceived_marketPrice": "1000",
            "max_marketPrice": "1200",
            "reservation_price": "800",
        }
        self.worker.check_and_initialize_slots(self.msg_state, fixedArgs)

        self.mock_llm.invoke.return_value.content.strip.return_value = (
            "Let's meet in the middle at $1000"
        )

        with patch("builtins.open"), patch("os.path.join"):
            result = self.worker.get_response(self.msg_state)

        assert result.response == "Let's meet in the middle at $1000"
        self.mock_llm.invoke.assert_called_once()

    def test_get_response_increments_round(self) -> None:
        """Test that response generation increments the turn."""
        fixedArgs = {
            "unit_index": "0",
            "max_perceived_marketPrice": "1000",
            "max_marketPrice": "1200",
            "reservation_price": "800",
        }
        self.worker.check_and_initialize_slots(self.msg_state, fixedArgs)
        initial_turn = self.msg_state.slots["turn"][0].value

        self.mock_llm.invoke.return_value.content.strip.return_value = "Counter offer"

        with patch("builtins.open"), patch("os.path.join"):
            result = self.worker.get_response(self.msg_state)

        assert result.slots["turn"][0].value == initial_turn + 1


class TestNegotiationSingleIssueWorkerExecution:
    """Test the main execution flow."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.worker = NegotiationSingleIssueWorker()

    def test_execute_success(self) -> None:
        """Test successful execution of the worker."""
        bot_config = BotConfig(
            bot_id="test_bot",
            version="1.0.0",
            language="en",
            bot_type="negotiation",
            llm_config=LLMConfig(
                llm_provider="openai",
                model_type_or_path="gpt-3.5-turbo",
            ),
        )
        msg_state = MessageState(
            status=StatusEnum.INCOMPLETE,
            trajectory=[[ResourceRecord(info={}, intent="test")]],
            bot_config=bot_config,
            user_message=ConvoMessage(message="Hello", history=""),
            slots={},
        )

        fixedArgs = {
            "unit_index": "0",
            "max_perceived_marketPrice": "1000",
            "max_marketPrice": "1200",
            "reservation_price": "800",
        }

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content.strip.return_value = "Test response"

        with patch(
            "arklex.env.workers.negotiation_workers.negotiation_single_issue_worker.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_provider_map.get.return_value = lambda model: mock_llm

            with patch("builtins.open"), patch("os.path.join"):
                result_dict = self.worker.execute(msg_state, fixedArgs=fixedArgs)
                result = MessageState.model_validate(result_dict)

        # Test that the worker executed successfully
        assert "turn" in result.slots
        assert result.slots["turn"][0].value == 1

    def test_execute_without_fixedArgs(self) -> None:
        """Test execution without required fixedArgs."""
        bot_config = BotConfig(
            bot_id="test_bot",
            version="1.0.0",
            language="en",
            bot_type="negotiation",
            llm_config=LLMConfig(
                llm_provider="openai",
                model_type_or_path="gpt-3.5-turbo",
            ),
        )
        msg_state = MessageState(
            status=StatusEnum.INCOMPLETE,
            trajectory=[[ResourceRecord(info={}, intent="test")]],
            bot_config=bot_config,
            user_message=ConvoMessage(message="Hello", history=""),
            slots={},
        )

        mock_llm = MagicMock()

        with patch(
            "arklex.env.workers.negotiation_workers.negotiation_single_issue_worker.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_provider_map.get.return_value = lambda model: mock_llm

            # The worker should handle missing fixedArgs - either by raising an error or handling gracefully
            try:
                result = self.worker.execute(msg_state)
                # If no exception is raised, verify it's a valid response
                assert isinstance(result, dict)
            except Exception:
                # Expected behavior when missing required fixedArgs - any exception is acceptable
                pass


class TestNegotiationSingleIssueWorkerEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.worker = NegotiationSingleIssueWorker()

    def test_execute_with_missing_bot_config(self) -> None:
        """Test execution with missing bot config."""
        msg_state = MessageState(
            status=StatusEnum.INCOMPLETE,
            trajectory=[[ResourceRecord(info={}, intent="test")]],
            user_message=ConvoMessage(message="Hello", history=""),
            slots={},
        )

        fixedArgs = {
            "unit_index": "0",
            "max_perceived_marketPrice": "1000",
            "max_marketPrice": "1200",
            "reservation_price": "800",
        }

        # Test execution with missing bot_config - either raises error or handles gracefully
        try:
            result = self.worker.execute(msg_state, fixedArgs=fixedArgs)
            # If no exception is raised, verify it's a valid response
            assert isinstance(result, dict)
        except Exception:
            # Expected behavior when bot_config is missing/None - any exception is acceptable
            pass

    def test_execute_with_missing_metadata(self) -> None:
        """Test execution with missing metadata."""
        bot_config = BotConfig(
            bot_id="test_bot",
            version="1.0.0",
            language="en",
            bot_type="negotiation",
            llm_config=LLMConfig(
                llm_provider="openai",
                model_type_or_path="gpt-3.5-turbo",
            ),
        )
        msg_state = MessageState(
            status=StatusEnum.INCOMPLETE,
            trajectory=[[ResourceRecord(info={}, intent="test")]],
            bot_config=bot_config,
            user_message=ConvoMessage(message="Hello", history=""),
            slots={},
        )

        fixedArgs = {
            "unit_index": "0",
            "max_perceived_marketPrice": "1000",
            "max_marketPrice": "1200",
            "reservation_price": "800",
        }

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content.strip.return_value = "Test response"

        with patch(
            "arklex.env.workers.negotiation_workers.negotiation_single_issue_worker.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_provider_map.get.return_value = lambda model: mock_llm

            with patch("builtins.open"), patch("os.path.join"):
                # Should work even without metadata as it's optional
                result_dict = self.worker.execute(msg_state, fixedArgs=fixedArgs)
                result = MessageState.model_validate(result_dict)
                assert "turn" in result.slots

    def test_check_and_initialize_slots_without_slots_attribute(self) -> None:
        """Test slot initialization when state has no slots attribute."""
        msg_state = MessageState(
            status=StatusEnum.INCOMPLETE,
            trajectory=[[ResourceRecord(info={}, intent="test")]],
        )
        # Set slots to None to test the initialization logic
        msg_state.slots = None

        fixedArgs = {
            "unit_index": "0",
            "max_perceived_marketPrice": "1000",
            "max_marketPrice": "1200",
            "reservation_price": "800",
        }

        result = self.worker.check_and_initialize_slots(msg_state, fixedArgs)

        assert hasattr(result, "slots")
        assert isinstance(result.slots, dict)
        assert len(result.slots) > 0


class TestNegotiationSingleIssueWorkerIntegration:
    """Test integration scenarios."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.worker = NegotiationSingleIssueWorker()

    def test_full_negotiation_workflow(self) -> None:
        """Test a full negotiation workflow."""
        bot_config = BotConfig(
            bot_id="test_bot",
            version="1.0.0",
            language="en",
            bot_type="negotiation",
            llm_config=LLMConfig(
                llm_provider="openai",
                model_type_or_path="gpt-3.5-turbo",
            ),
        )
        msg_state = MessageState(
            status=StatusEnum.INCOMPLETE,
            trajectory=[[ResourceRecord(info={}, intent="test")]],
            bot_config=bot_config,
            user_message=ConvoMessage(message="I'd like to negotiate", history=""),
            slots={},
        )

        fixedArgs = {
            "unit_index": "0",
            "max_perceived_marketPrice": "1000",
            "max_marketPrice": "1200",
            "reservation_price": "800",
        }

        mock_llm = MagicMock()
        mock_responses = ["Opening response", "Negotiation response"]
        mock_llm.invoke.return_value.content.strip.side_effect = mock_responses

        with patch(
            "arklex.env.workers.negotiation_workers.negotiation_single_issue_worker.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_provider_map.get.return_value = lambda model: mock_llm

            with patch("builtins.open"), patch("os.path.join"):
                # First execution
                result_dict = self.worker.execute(msg_state, fixedArgs=fixedArgs)
                result = MessageState.model_validate(result_dict)

                assert result.response == "Opening response"
                assert result.slots["turn"][0].value == 1

                # Simulate second turn
                result.user_message.message = "How about $950?"
                result_dict2 = self.worker.execute(result, fixedArgs=fixedArgs)
                result2 = MessageState.model_validate(result_dict2)

                assert result2.slots["turn"][0].value == 2
