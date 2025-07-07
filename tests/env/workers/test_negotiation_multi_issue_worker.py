"""Comprehensive tests for the NegotiationMultiIssueWorker.

This module provides comprehensive test coverage for the NegotiationMultiIssueWorker,
testing all functionality including negotiation logic, KDE analysis, slot management, and scoring.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from arklex.env.workers.negotiation_workers.negotiation_multi_issue_worker import (
    NegotiationMultiIssueWorker,
)
from arklex.utils.graph_state import (
    BotConfig,
    ConvoMessage,
    LLMConfig,
    MessageState,
    ResourceRecord,
    Slot,
    StatusEnum,
)


class TestNegotiationMultiIssueWorkerInitialization:
    """Test initialization of the NegotiationMultiIssueWorker."""

    def test_worker_initialization(self) -> None:
        """Test that the worker initializes correctly."""
        worker = NegotiationMultiIssueWorker()

        assert worker.name == "NegotiationMultiIssueWorker"
        assert worker.llm is None
        assert worker.action_graph is not None
        assert hasattr(worker, "buyer_utilities")
        assert hasattr(worker, "seller_utilities")
        assert hasattr(worker, "kde_models")
        assert hasattr(worker, "observations")
        assert worker.walk_away_point == 10000

    def test_utilities_initialization(self) -> None:
        """Test that utility dictionaries are properly initialized."""
        worker = NegotiationMultiIssueWorker()

        # Check buyer utilities structure
        assert "LOCATION" in worker.buyer_utilities
        assert "SALARY" in worker.buyer_utilities
        assert "HEALTHCARE" in worker.buyer_utilities
        assert "VACATION" in worker.buyer_utilities
        assert "MOVING EXPENSES" in worker.buyer_utilities
        assert "JOB ASSIGNMENT" in worker.buyer_utilities

        # Check seller utilities structure
        assert "LOCATION" in worker.seller_utilities
        assert "SALARY" in worker.seller_utilities
        assert "HEALTHCARE" in worker.seller_utilities
        assert "VACATION" in worker.seller_utilities
        assert "MOVING EXPENSES" in worker.seller_utilities
        assert "JOB ASSIGNMENT" in worker.seller_utilities

        # Check specific values
        assert worker.buyer_utilities["LOCATION"]["Boston"] == 3200
        assert worker.seller_utilities["SALARY"]["$85000"] == 4000

    def test_kde_models_initialization(self) -> None:
        """Test that KDE models are properly initialized."""
        worker = NegotiationMultiIssueWorker()

        # Check KDE models for numerical issues
        assert "SALARY" in worker.kde_models
        assert "HEALTHCARE" in worker.kde_models
        assert "MOVING EXPENSES" in worker.kde_models

        # Check observations tracking
        assert "SALARY" in worker.observations
        assert "HEALTHCARE" in worker.observations
        assert "MOVING EXPENSES" in worker.observations
        assert worker.observations["SALARY"] == []


class TestNegotiationMultiIssueWorkerUtilityMethods:
    """Test utility methods of the NegotiationMultiIssueWorker."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.worker = NegotiationMultiIssueWorker()

    def test_read_json(self) -> None:
        """Test reading JSON files."""
        # Create a temporary JSON file
        test_data = {"test": "data", "number": 123}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            result = self.worker.read_json(temp_path)
            assert result == test_data
        finally:
            os.unlink(temp_path)

    def test_calculate_scores(self) -> None:
        """Test calculating final utility scores."""
        final_outcomes = {
            "LOCATION": "Boston",
            "SALARY": "$100000",
            "HEALTHCARE": "Ajax POS",
            "VACATION": "15 days",
            "MOVING EXPENSES": "80% covered",
            "JOB ASSIGNMENT": "Technology",
        }

        buyer_score, seller_score = self.worker.calculate_scores(final_outcomes)

        # Calculate expected scores
        expected_buyer = 3200 + 3000 + 3000 + 750 + 750 + 1200  # 11900
        expected_seller = 3200 + 1000 + 500 + 500 + 500 + 400  # 6100

        assert buyer_score == expected_buyer
        assert seller_score == expected_seller

    def test_generate_combos(self) -> None:
        """Test generating valid combinations."""
        result = self.worker.generate_combos(sample_size=5)

        assert "Valid Combinations:" in result
        assert "Combination 1:" in result
        # Should generate at least one combination
        assert len(result.split("\n")) > 2

    def test_update_offers_and_kde_insufficient_data(self) -> None:
        """Test KDE update with insufficient data."""
        result = self.worker.update_offers_and_kde("SALARY", 100000, 95000)
        assert result == -1  # Should return -1 for insufficient data

    def test_update_offers_and_kde_sufficient_data(self) -> None:
        """Test KDE update with sufficient data."""
        # Add first observation
        self.worker.update_offers_and_kde("SALARY", 100000, 95000)
        # Add second observation
        result = self.worker.update_offers_and_kde("SALARY", 95000, 90000)

        assert result != -1  # Should return a peak value
        assert len(self.worker.observations["SALARY"]) == 2
        assert 5000 in self.worker.observations["SALARY"]


class TestNegotiationMultiIssueWorkerSlotManagement:
    """Test slot management functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.worker = NegotiationMultiIssueWorker()
        self.msg_state = MessageState(
            status=StatusEnum.INCOMPLETE,
            trajectory=[[ResourceRecord(info={}, intent="test")]],
            slots={},  # Initialize with empty dict instead of None
        )

    def test_check_and_initialize_slots_creates_all_slots(self) -> None:
        """Test that all required slots are created."""
        self.worker.check_and_initialize_slots(self.msg_state)

        required_slots = [
            "turn",
            "episode_done",
            "current_issue",
            "resolved_issues",
            "location",
            "salary",
            "healthcare",
            "vacation",
            "moving_expenses",
            "job_assignment",
            "current_target",
        ]

        for slot_name in required_slots:
            assert slot_name in self.msg_state.slots
            assert len(self.msg_state.slots[slot_name]) == 1
            assert isinstance(self.msg_state.slots[slot_name][0], Slot)

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
                    description="Test slot",
                    prompt="",
                    required=False,
                    verified=True,
                )
            ]
        }

        self.worker.check_and_initialize_slots(self.msg_state)

        # Existing slot should be preserved
        assert self.msg_state.slots["turn"][0].value == 5
        # Other slots should be created
        assert "episode_done" in self.msg_state.slots

    def test_slot_initial_values(self) -> None:
        """Test that slots have correct initial values."""
        self.worker.check_and_initialize_slots(self.msg_state)

        assert self.msg_state.slots["turn"][0].value == 0
        assert self.msg_state.slots["episode_done"][0].value is False
        assert self.msg_state.slots["current_issue"][0].value == ""
        assert self.msg_state.slots["resolved_issues"][0].value == []

        # Check JSON slots
        location_data = json.loads(self.msg_state.slots["location"][0].value)
        assert location_data == {"current": None, "previous": None}


class TestNegotiationMultiIssueWorkerLLMInteraction:
    """Test LLM interaction methods."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.worker = NegotiationMultiIssueWorker()
        self.mock_llm = MagicMock()
        self.worker.llm = self.mock_llm

    def test_common_sense_importance(self) -> None:
        """Test common sense importance analysis."""
        self.mock_llm.invoke.return_value.content.strip.return_value = (
            "SALARY is most important, followed by LOCATION..."
        )

        result = self.worker.common_sense_importance()

        assert "SALARY is most important" in result
        self.mock_llm.invoke.assert_called_once()

    def test_importance_estimation_kde(self) -> None:
        """Test KDE-based importance estimation."""
        conversation_history = "User: I want $100k salary. Agent: How about $95k?"
        self.mock_llm.invoke.return_value.content.strip.return_value = (
            "SALARY importance: 9/10"
        )

        result = self.worker.importance_estimation_kde(conversation_history)

        assert "SALARY importance: 9/10" in result
        self.mock_llm.invoke.assert_called_once()

    def test_determine_user_personality(self) -> None:
        """Test user personality determination."""
        conversation_history = "User: Take it or leave it!"
        self.mock_llm.invoke.return_value.content.strip.return_value = "AGGRESSIVE"

        result = self.worker.determine_user_personality(conversation_history)

        assert result == "AGGRESSIVE"
        self.mock_llm.invoke.assert_called_once()

    def test_extract_issue_and_offers(self) -> None:
        """Test extracting issues and offers from conversation."""
        conversation_history = "Discussing salary: $95k vs $100k"
        self.mock_llm.invoke.return_value.content.strip.return_value = (
            "ISSUE: SALARY\nCURRENT OFFER: 100000\nPREVIOUS OFFER: 95000"
        )

        result = self.worker.extract_issue_and_offers(conversation_history)

        assert "ISSUE: SALARY" in result
        assert "CURRENT OFFER: 100000" in result
        assert "PREVIOUS OFFER: 95000" in result


class TestNegotiationMultiIssueWorkerMonitoring:
    """Test monitoring and tracking functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.worker = NegotiationMultiIssueWorker()
        self.mock_llm = MagicMock()
        self.worker.llm = self.mock_llm
        self.msg_state = MessageState(
            status=StatusEnum.INCOMPLETE,
            trajectory=[[ResourceRecord(info={}, intent="test")]],
            user_message=ConvoMessage(message="test", history="negotiation history"),
            slots={},
        )

    def test_monitor_instance_incomplete_negotiation(self) -> None:
        """Test monitoring with incomplete negotiation."""
        self.mock_llm.invoke.return_value.content.strip.return_value = (
            "NUMBER OF RESOLVED ISSUES=3"
        )

        issues = [
            "LOCATION",
            "SALARY",
            "HEALTHCARE",
            "VACATION",
            "MOVING EXPENSES",
            "JOB ASSIGNMENT",
        ]
        result = self.worker.monitor_instance(self.msg_state, issues)

        assert result is None

    def test_monitor_instance_complete_negotiation(self) -> None:
        """Test monitoring with complete negotiation."""
        mock_response = """
        NUMBER OF RESOLVED ISSUES=6
        LOCATION=Boston
        SALARY=100000
        HEALTHCARE=Ajax POS
        VACATION=15 days
        MOVING EXPENSES=80% covered
        JOB ASSIGNMENT=Technology
        """
        self.mock_llm.invoke.return_value.content.strip.return_value = mock_response

        issues = [
            "LOCATION",
            "SALARY",
            "HEALTHCARE",
            "VACATION",
            "MOVING EXPENSES",
            "JOB ASSIGNMENT",
        ]
        result = self.worker.monitor_instance(self.msg_state, issues)

        assert result is not None
        assert result["LOCATION"] == "Boston"
        assert result["SALARY"] == "100000"
        assert result["HEALTHCARE"] == "Ajax POS"


class TestNegotiationMultiIssueWorkerResponse:
    """Test response generation functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.worker = NegotiationMultiIssueWorker()
        self.mock_llm = MagicMock()
        self.worker.llm = self.mock_llm
        self.msg_state = MessageState(
            status=StatusEnum.INCOMPLETE,
            trajectory=[[ResourceRecord(info={}, intent="test")]],
            user_message=ConvoMessage(message="Hello", history=""),
            slots={},
        )

    @patch("os.path.join")
    @patch("builtins.open")
    def test_get_response_first_turn(
        self, mock_open: MagicMock, mock_path_join: MagicMock
    ) -> None:
        """Test response generation for first turn."""
        # Mock file operations
        mock_path_join.return_value = "mock_path"
        mock_open.return_value.__enter__.return_value.read.return_value = (
            "System prompt content"
        )

        # Mock LLM responses
        self.mock_llm.invoke.return_value.content.strip.return_value = (
            "Welcome to negotiation!"
        )

        # Initialize slots first
        self.worker.check_and_initialize_slots(self.msg_state)

        result = self.worker.get_response(self.msg_state)

        assert result.slots["turn"][0].value == 1
        assert result.response == "Welcome to negotiation!"
        assert "System Prompt:" in result.message_flow

    def test_get_response_subsequent_turn(self) -> None:
        """Test response generation for subsequent turns."""
        # Initialize slots and set turn > 0
        self.worker.check_and_initialize_slots(self.msg_state)
        self.msg_state.slots["turn"][0].value = 2

        # Mock LLM responses
        self.mock_llm.invoke.return_value.content.strip.return_value = (
            "Let's continue negotiating"
        )

        with patch("os.path.join"), patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                "System prompt"
            )
            result = self.worker.get_response(self.msg_state)

        assert result.slots["turn"][0].value == 3
        assert result.response == "Let's continue negotiating"

    def test_episode_completion(self) -> None:
        """Test that episode ends after maximum turns."""
        self.worker.check_and_initialize_slots(self.msg_state)
        self.msg_state.slots["turn"][0].value = 7

        self.mock_llm.invoke.return_value.content.strip.return_value = "Final response"

        with patch("os.path.join"), patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                "System prompt"
            )
            result = self.worker.get_response(self.msg_state)

        assert result.slots["turn"][0].value == 8
        assert result.slots["episode_done"][0].value is True


class TestNegotiationMultiIssueWorkerExecution:
    """Test the main execution flow."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.worker = NegotiationMultiIssueWorker()

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

        mock_llm = MagicMock()
        mock_responses = [
            "Test response",  # For response generation
            "Common sense ranking",  # For common_sense_importance
        ]
        mock_llm.invoke.return_value.content.strip.side_effect = mock_responses

        with patch(
            "arklex.env.workers.negotiation_workers.negotiation_multi_issue_worker.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_provider_map.get.return_value = lambda model: mock_llm

            with patch("os.path.join"), patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = (
                    "System prompt"
                )
                result_dict = self.worker.execute(msg_state)
                result = MessageState.model_validate(result_dict)

        # Test that the worker executed successfully
        assert "turn" in result.slots
        assert result.slots["turn"][0].value == 1

    def test_execute_with_final_outcomes(self) -> None:
        """Test execution that results in final outcomes."""
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
            user_message=ConvoMessage(
                message="Final offer", history="complete negotiation"
            ),
            slots={},
        )

        # Mock complete negotiation response - needs exact format for monitor_instance
        mock_response = """NUMBER OF RESOLVED ISSUES=6
LOCATION=Boston
SALARY=$100000
HEALTHCARE=Ajax POS
VACATION=15 days
MOVING EXPENSES=80% covered
JOB ASSIGNMENT=Technology"""

        mock_llm = MagicMock()
        # Set up mock responses - need different responses for different LLM calls
        mock_responses = [
            "Test response",  # First call for response generation
            "Common sense ranking",  # For common_sense_importance
            mock_response,  # For monitor_instance
        ]
        mock_llm.invoke.return_value.content.strip.side_effect = mock_responses

        with patch(
            "arklex.env.workers.negotiation_workers.negotiation_multi_issue_worker.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_provider_map.get.return_value = lambda model: mock_llm

            with patch("os.path.join"), patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = (
                    "System prompt"
                )
                result_dict = self.worker.execute(msg_state)
                result = MessageState.model_validate(result_dict)

        assert result.status == StatusEnum.STAY
        assert "Buyer Score:" in result.message_flow
        assert "Seller Score:" in result.message_flow


class TestNegotiationMultiIssueWorkerEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.worker = NegotiationMultiIssueWorker()

    def test_update_kde_with_same_offers(self) -> None:
        """Test KDE update when offers are the same."""
        result = self.worker.update_offers_and_kde("SALARY", 100000, 100000)
        assert result == -1  # Should handle zero concession

    def test_calculate_scores_with_missing_issue(self) -> None:
        """Test score calculation with invalid issue."""
        final_outcomes = {
            "INVALID_ISSUE": "Some value"
        }  # Issue that doesn't exist in utilities

        with pytest.raises(KeyError):
            self.worker.calculate_scores(final_outcomes)

    def test_monitor_instance_malformed_response(self) -> None:
        """Test monitor with malformed LLM response."""
        self.worker.llm = MagicMock()
        self.worker.llm.invoke.return_value.content.strip.return_value = (
            "Invalid response format"
        )

        msg_state = MessageState(
            status=StatusEnum.INCOMPLETE,
            trajectory=[[ResourceRecord(info={}, intent="test")]],
            user_message=ConvoMessage(message="test", history="history"),
            slots={},
        )

        issues = ["LOCATION", "SALARY"]
        result = self.worker.monitor_instance(msg_state, issues)

        assert result is None

    def test_slots_without_attributes(self) -> None:
        """Test slot initialization when state has no slots attribute."""
        msg_state = MessageState(
            status=StatusEnum.INCOMPLETE,
            trajectory=[[ResourceRecord(info={}, intent="test")]],
        )
        # Set slots to None to test the initialization logic
        msg_state.slots = None

        self.worker.check_and_initialize_slots(msg_state)

        assert hasattr(msg_state, "slots")
        assert isinstance(msg_state.slots, dict)
        assert len(msg_state.slots) > 0
