"""Tests for the HITL worker module."""

from unittest.mock import Mock, patch

import pytest

from arklex.env.workers.hitl_worker import (
    HITLWorker,
    HITLWorkerChatFlag,
    HITLWorkerMCFlag,
    HITLWorkerTestChat,
    HITLWorkerTestMC,
)
from arklex.orchestrator.entities.msg_state_entities import (
    MessageState,
    Metadata,
    StatusEnum,
)
from arklex.orchestrator.NLU.core.slot import SlotFiller


class TestHITLWorker:
    """Test the base HITLWorker class."""

    def test_init(self) -> None:
        """Test HITLWorker initialization."""
        worker = HITLWorker(name="test_worker")
        assert worker.name == "test_worker"
        assert worker.action_graph is not None
        assert worker.slot_fill_api is None

    def test_verify_literal(self) -> None:
        """Test verify_literal method."""
        worker = HITLWorker(name="test_worker")
        state = MessageState()
        need_hitl, message = worker.verify_literal(state)
        assert need_hitl is True
        assert message == ""

    def test_verify_slots(self) -> None:
        """Test verify_slots method."""
        worker = HITLWorker(name="test_worker")
        message = {"slots": []}
        need_hitl, message = worker.verify_slots(message)
        assert need_hitl is True
        assert message == ""

    def test_verify_literal_returns_true(self) -> None:
        """Test verify method when verify_literal returns True."""
        worker = HITLWorker(name="test_worker")
        state = MessageState()
        need_hitl, message = worker.verify(state)
        assert need_hitl is True
        assert message == ""

    def test_verify_slots_returns_true(self) -> None:
        """Test verify method when verify_slots returns True."""
        worker = HITLWorker(name="test_worker")

        # Mock verify_literal to return False
        with patch.object(worker, "verify_literal", return_value=(False, "")):
            state = MessageState()
            need_hitl, message = worker.verify(state)
            assert need_hitl is True
            assert message == ""

    def test_verify_both_return_false(self) -> None:
        """Test verify method when both verify methods return False."""
        worker = HITLWorker(name="test_worker")

        # Mock both verify methods to return False
        with (
            patch.object(worker, "verify_literal", return_value=(False, "")),
            patch.object(worker, "verify_slots", return_value=(False, "")),
        ):
            state = MessageState()
            need_hitl, message = worker.verify(state)
            assert need_hitl is False
            assert message == ""

    def test_init_slot_filler(self) -> None:
        """Test slot filler initialization."""
        worker = HITLWorker(name="test_worker")
        slot_fill_api = Mock()
        worker.init_slot_filler(slot_fill_api)
        assert isinstance(worker.slot_fill_api, SlotFiller)

    def test_create_prompt(self) -> None:
        """Test prompt creation."""
        worker = HITLWorker(name="test_worker")
        worker.params = {
            "intro": "Test intro",
            "choices": {"A": "Choice A", "B": "Choice B"},
        }
        prompt = worker.create_prompt()
        expected = "Test intro\n(A) Choice A\n(B) Choice B"
        assert prompt == expected

    @patch("arklex.env.workers.hitl_worker.ChatClient")
    def test_chat(self, mock_chat_client_class: Mock) -> None:
        """Test chat method."""
        worker = HITLWorker(name="test_worker", server_ip="127.0.0.1", server_port=8080)
        mock_client = Mock()
        mock_chat_client_class.return_value = mock_client
        mock_client.sync_main.return_value = MessageState()

        state = MessageState()
        result = worker.chat(state)

        mock_chat_client_class.assert_called_once_with(
            "127.0.0.1", 8080, name="test_worker", mode="c"
        )
        mock_client.sync_main.assert_called_once()
        assert isinstance(result, MessageState)

    @patch("arklex.env.workers.hitl_worker.ChatClient")
    def test_multiple_choice(self, mock_chat_client_class: Mock) -> None:
        """Test multiple choice method."""
        worker = HITLWorker(name="test_worker", server_ip="127.0.0.1", server_port=8080)
        worker.params = {
            "intro": "Test intro",
            "choices": {"A": "Choice A", "B": "Choice B"},
        }

        mock_client = Mock()
        mock_chat_client_class.return_value = mock_client
        mock_client.sync_main.return_value = MessageState()

        state = MessageState()
        result = worker.multiple_choice(state)

        mock_chat_client_class.assert_called_once_with(
            "127.0.0.1", 8080, name="test_worker", mode="ro"
        )
        mock_client.sync_main.assert_called_once_with(message=worker.create_prompt())
        assert isinstance(result, MessageState)

    @patch.object(HITLWorker, "chat")
    def test_hitl_chat_mode(self, mock_chat: Mock) -> None:
        worker = HITLWorker(name="test_worker", mode="chat")
        worker.params = {"intro": "Test intro", "choices": {}}
        mock_chat.return_value = "test_worker: new message"
        state = MessageState()
        state.user_message = Mock()
        state.user_message.history = "existing history"
        state.user_message.message = "original message"
        result = worker.hitl(state)
        assert result.response == "Live Chat Completed"
        assert (
            state.user_message.history == "existing history\ntest_worker: new message"
        )
        assert state.user_message.message == "new message"

    @patch.object(HITLWorker, "multiple_choice")
    def test_hitl_mc_mode_success(self, mock_multiple_choice: Mock) -> None:
        worker = HITLWorker(name="test_worker", mode="mc")
        worker.params = {
            "intro": "Test intro",
            "max_retries": 3,
            "default": "Default choice",
            "choices": {"Y": "Yes choice", "N": "No choice"},
        }
        mock_multiple_choice.return_value = "Y"
        state = MessageState()
        result = worker.hitl(state)
        assert result.response == "Yes choice"

    @patch.object(HITLWorker, "multiple_choice")
    def test_hitl_mc_mode_max_retries_exceeded(
        self, mock_multiple_choice: Mock
    ) -> None:
        worker = HITLWorker(name="test_worker", mode="mc")
        worker.params = {
            "intro": "Test intro",
            "max_retries": 2,
            "default": "Default choice",
            "choices": {"Y": "Yes choice", "N": "No choice"},
        }
        mock_multiple_choice.return_value = "INVALID"
        state = MessageState()
        result = worker.hitl(state)
        assert result.response == "Default choice"

    def test_hitl_unknown_mode(self) -> None:
        worker = HITLWorker(name="test_worker", mode="unknown")
        state = MessageState()
        result = worker.hitl(state)
        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.INCOMPLETE

    def test_fallback(self) -> None:
        """Test fallback method."""
        worker = HITLWorker(name="test_worker")
        state = MessageState()

        result = worker.fallback(state)

        assert result == state
        assert state.message_flow == "The user don't need human help"
        assert state.status == StatusEnum.COMPLETE

    def test_create_action_graph(self) -> None:
        """Test action graph creation."""
        worker = HITLWorker(name="test_worker")
        graph = worker._create_action_graph()

        assert graph is not None
        # Check that the graph has the expected nodes
        assert "hitl" in graph.nodes

    @patch.object(HITLWorker, "verify")
    def test_execute_verify_fails(self, mock_verify: Mock) -> None:
        """Test execute method when verify fails."""
        worker = HITLWorker(name="test_worker")
        mock_verify.return_value = (False, "")
        state = MessageState()
        result = worker._execute(state)
        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.INCOMPLETE

    @patch.object(HITLWorker, "verify")
    def test_execute_verify_succeeds(self, mock_verify: Mock) -> None:
        """Test execute method when verify succeeds."""
        worker = HITLWorker(name="test_worker")
        mock_verify.return_value = (True, "")
        state = MessageState()

        # Mock the compiled graph
        mock_compiled_graph = Mock()
        mock_compiled_graph.invoke.return_value = state
        worker.action_graph.compile = Mock(return_value=mock_compiled_graph)

        result = worker._execute(state)
        assert isinstance(result, MessageState)

    def test_error_method_sets_status_incomplete(self) -> None:
        """Test error method sets status to incomplete."""
        worker = HITLWorker(name="test_worker")
        state = MessageState()
        result = worker.error(state)
        assert result == state
        assert state.status == StatusEnum.INCOMPLETE

    def test_execute_with_verify_failure_calls_error(self) -> None:
        """Test execute method when verify fails (covers line 196)."""
        worker = HITLWorker(name="test_worker")

        # Mock verify to return (False, message) - first element is False
        with (
            patch.object(worker, "verify", return_value=(False, "No HITL needed")),
            patch.object(worker, "error") as mock_error,
        ):
            error_state = MessageState()
            error_state.status = StatusEnum.INCOMPLETE
            mock_error.return_value = error_state

            state = MessageState()
            result = worker._execute(state)

            # Should call error method when verify returns (False, message)
            mock_error.assert_called_once_with(state)
            # The result should be a MessageState with INCOMPLETE status
            assert isinstance(result, MessageState)
            assert result.status == StatusEnum.INCOMPLETE

    def test_execute_with_verify_success_calls_action_graph(self) -> None:
        """Test execute method when verify succeeds."""
        worker = HITLWorker(name="test_worker")
        state = MessageState()
        state.metadata = Metadata()

        with patch.object(worker, "verify", return_value=(True, "Success")):
            # Mock the action_graph directly since it's created during initialization
            mock_graph = Mock()
            mock_compiled_graph = Mock()
            # Ensure the mock returns the actual state object, not a dictionary
            mock_compiled_graph.invoke.return_value = state
            mock_graph.compile.return_value = mock_compiled_graph
            worker.action_graph = mock_graph

            result = worker._execute(state)

            # The result should be the state object returned by the action graph
            # Check that the result has the same properties as the expected state
            assert isinstance(result, MessageState)
            assert result.status == state.status
            mock_graph.compile.assert_called_once()
            mock_compiled_graph.invoke.assert_called_once_with(state)

    def test_execute_verify_fails_returns_error_state(self) -> None:
        """Test execute method when verify fails returns error state."""
        worker = HITLWorker(name="test_worker")
        state = MessageState()
        state.metadata = Metadata()

        with patch.object(
            worker, "verify", return_value=(False, "Verification failed")
        ):
            result = worker._execute(state)

            # Should return the state object with INCOMPLETE status
            assert isinstance(result, MessageState)
            assert result.status == StatusEnum.INCOMPLETE


class TestHITLWorkerTestChat:
    """Test the HITLWorkerTestChat class."""

    def test_init_success(self) -> None:
        """Test successful initialization."""
        worker = HITLWorkerTestChat(
            name="test_chat", server_ip="127.0.0.1", server_port=8080
        )
        assert worker.name == "test_chat"
        assert worker.server_ip == "127.0.0.1"
        assert worker.server_port == 8080
        assert worker.mode == "chat"

    def test_init_missing_server_ip(self) -> None:
        """Test initialization with missing server IP."""
        with pytest.raises(ValueError, match="Server IP and Port are required"):
            HITLWorkerTestChat(name="test_chat", server_port=8080)

    def test_init_missing_server_port(self) -> None:
        """Test initialization with missing server port."""
        with pytest.raises(ValueError, match="Server IP and Port are required"):
            HITLWorkerTestChat(name="test_chat", server_ip="127.0.0.1")

    def test_verify_literal_with_chat(self) -> None:
        """Test verify_literal method with 'chat' in message."""
        worker = HITLWorkerTestChat(
            name="test_chat", server_ip="127.0.0.1", server_port=8080
        )
        result = worker.verify_literal("I want to chat with someone")
        assert result is True

    def test_verify_literal_without_chat(self) -> None:
        """Test verify_literal method without 'chat' in message."""
        worker = HITLWorkerTestChat(
            name="test_chat", server_ip="127.0.0.1", server_port=8080
        )
        result = worker.verify_literal("I want to buy something")
        assert result is False

    def test_HITLWorkerTestChat_init_missing_params(self) -> None:
        # Should raise ValueError if server_ip or server_port missing
        with pytest.raises(ValueError):
            HITLWorkerTestChat(name="test", server_port=1234)
        with pytest.raises(ValueError):
            HITLWorkerTestChat(name="test", server_ip="127.0.0.1")


class TestHITLWorkerTestMC:
    """Test the HITLWorkerTestMC class."""

    def test_init(self) -> None:
        """Test initialization."""
        worker = HITLWorkerTestMC(
            name="test_mc", server_ip="127.0.0.1", server_port=8080
        )
        assert worker.name == "test_mc"
        assert worker.server_ip == "127.0.0.1"
        assert worker.server_port == 8080
        assert worker.mode == "mc"
        assert (
            worker.params["intro"]
            == "Should the user continue with this purchase? (Y/N)"
        )

    def test_verify_literal_with_buy(self) -> None:
        """Test verify_literal method with 'buy' in message."""
        worker = HITLWorkerTestMC(
            name="test_mc", server_ip="127.0.0.1", server_port=8080
        )
        result = worker.verify_literal("I want to buy something")
        assert result is True

    def test_verify_literal_without_buy(self) -> None:
        """Test verify_literal method without 'buy' in message."""
        worker = HITLWorkerTestMC(
            name="test_mc", server_ip="127.0.0.1", server_port=8080
        )
        result = worker.verify_literal("I want to chat with someone")
        assert result is False

    def test_HITLWorkerTestMC_init_missing_params(self) -> None:
        # Should raise ValueError if server_ip or server_port missing
        with pytest.raises(ValueError):
            HITLWorkerTestMC(name="test", server_port=1234)
        with pytest.raises(ValueError):
            HITLWorkerTestMC(name="test", server_ip="127.0.0.1")


class TestHITLWorkerChatFlag:
    """Test the HITLWorkerChatFlag class."""

    def test_init(self) -> None:
        """Test initialization."""
        worker = HITLWorkerChatFlag(name="test_chat_flag")
        assert worker.name == "test_chat_flag"
        assert worker.mode == "chat"

    def test_verify_literal(self) -> None:
        """Test verify_literal method."""
        worker = HITLWorkerChatFlag(name="test_chat_flag")
        state = MessageState()
        need_hitl, message = worker.verify_literal(state)
        assert need_hitl is True
        assert message == "I'll connect you to a representative!"

    def test_execute_no_hitl_verify_fails(self) -> None:
        """Test execute method when no hitl and verify fails."""
        worker = HITLWorkerChatFlag(name="test_chat_flag")
        state = MessageState()
        state.metadata = Metadata()
        state.metadata.hitl = None

        with (
            patch.object(worker, "verify", return_value=(False, "")),
            patch.object(worker, "fallback", return_value=state) as mock_fallback,
        ):
            result = worker._execute(state)
            mock_fallback.assert_called_once_with(state)
            assert result == state

    def test_execute_no_hitl_verify_succeeds(self) -> None:
        """Test execute method when no hitl and verify succeeds."""
        worker = HITLWorkerChatFlag(name="test_chat_flag")
        state = MessageState()
        state.metadata = Metadata()
        state.metadata.hitl = None

        with patch.object(worker, "verify", return_value=(True, "Need help")):
            result = worker._execute(state)

            assert result == state
            assert state.message_flow == "Need help"
            assert state.metadata.hitl == "live"
            assert state.status == StatusEnum.STAY

    def test_execute_with_hitl(self) -> None:
        """Test execute method when hitl is already set."""
        worker = HITLWorkerChatFlag(name="test_chat_flag")
        state = MessageState()
        state.metadata = Metadata()
        state.metadata.hitl = "live"

        result = worker._execute(state)

        assert result == state
        assert state.message_flow == "Live chat completed"
        assert state.metadata.hitl is None
        assert state.status == StatusEnum.COMPLETE


class TestHITLWorkerMCFlag:
    """Test the HITLWorkerMCFlag class."""

    def test_init(self) -> None:
        """Test initialization."""
        worker = HITLWorkerMCFlag(name="test_mc_flag")
        assert worker.name == "test_mc_flag"
        assert worker.mode == "mc"
        assert (
            worker.params["intro"]
            == "Should the user continue with this purchase? (Y/N)"
        )

    def test_verify_literal_with_buy(self) -> None:
        """Test verify_literal method with 'buy' in message."""
        worker = HITLWorkerMCFlag(name="test_mc_flag")
        result = worker.verify_literal("I want to buy something")
        assert result is True

    def test_verify_literal_without_buy(self) -> None:
        """Test verify_literal method without 'buy' in message."""
        worker = HITLWorkerMCFlag(name="test_mc_flag")
        result = worker.verify_literal("I want to chat with someone")
        assert result is False

    def test_execute_no_hitl_verify_fails(self) -> None:
        """Test execute method when no hitl and verify fails."""
        worker = HITLWorkerMCFlag(name="test_mc_flag")
        state = MessageState()
        state.metadata = Metadata()
        state.metadata.hitl = None

        with (
            patch.object(worker, "verify", return_value=(False, "")),
            patch.object(worker, "fallback", return_value=state) as mock_fallback,
        ):
            result = worker._execute(state)
            mock_fallback.assert_called_once_with(state)
            assert result == state

    def test_execute_no_hitl_verify_succeeds(self) -> None:
        """Test execute method when no hitl and verify succeeds."""
        worker = HITLWorkerMCFlag(name="test_mc_flag")
        state = MessageState()
        state.metadata = Metadata()
        state.metadata.hitl = None

        with patch.object(worker, "verify", return_value=(True, "Need confirmation")):
            result = worker._execute(state)

            assert result == state
            assert (
                state.response
                == "[[sending confirmation : this should not show up for user]]"
            )
            assert state.metadata.hitl == "mc"
            assert state.metadata.attempts == 5
            assert state.status == StatusEnum.STAY

    def test_execute_with_hitl_valid_choice(self) -> None:
        """Test execute method when hitl is set and valid choice is made."""
        worker = HITLWorkerMCFlag(name="test_mc_flag")
        state = MessageState()
        state.metadata = Metadata()
        state.metadata.hitl = "mc"
        state.user_message = Mock()
        state.user_message.message = "Y"

        result = worker._execute(state)

        assert result == state
        assert state.response == "User is allowed to continue with the purchase"
        assert state.metadata.hitl is None
        assert state.status == StatusEnum.COMPLETE

    def test_execute_with_hitl_invalid_choice_attempts_remaining(self) -> None:
        """Test execute with HITL when invalid choice but attempts still remaining."""
        worker = HITLWorkerMCFlag(name="test_worker")
        state = MessageState()
        state.metadata = Metadata()
        state.metadata.hitl = "mc"
        state.metadata.attempts = 3
        state.user_message = Mock()
        state.user_message.message = "INVALID"

        result = worker._execute(state)

        assert result.metadata.hitl == "mc"
        assert result.status == StatusEnum.STAY
        assert result.metadata.attempts == 2
        assert (
            result.response
            == "[[sending confirmation : this should not show up for user]]"
        )

    def test_execute_with_hitl_invalid_choice_zero_attempts(self) -> None:
        """Test execute with HITL when invalid choice and attempts reach zero."""
        worker = HITLWorkerMCFlag(name="test_worker")
        state = MessageState()
        state.metadata = Metadata()
        state.metadata.hitl = "mc"
        state.metadata.attempts = 0
        state.user_message = Mock()
        state.user_message.message = "INVALID"

        result = worker._execute(state)

        assert result.metadata.hitl is None
        assert result.status == StatusEnum.INCOMPLETE
        assert result.response == "User is not allowed to continue with the purchase"

    def test_execute_with_hitl_invalid_choice_attempts_exhausted(self) -> None:
        """Test execute with HITL when invalid choice and attempts are exhausted."""
        worker = HITLWorkerMCFlag(name="test_worker")
        state = MessageState()
        state.metadata = Metadata()
        state.metadata.hitl = "mc"
        state.metadata.attempts = 1
        state.user_message = Mock()
        state.user_message.message = "INVALID"

        result = worker._execute(state)

        assert result.metadata.hitl is None
        assert result.status == StatusEnum.INCOMPLETE
        assert result.response == "User is not allowed to continue with the purchase"
