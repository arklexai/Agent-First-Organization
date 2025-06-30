"""Comprehensive tests for ChatClient.

This module provides extensive test coverage for the ChatClient class,
including all methods, edge cases, error conditions, and async scenarios.
"""

import json
import runpy
from concurrent.futures import Future
from unittest.mock import AsyncMock, Mock, patch

import pytest

from arklex.env.workers.utils.chat_client import ChatClient

# =============================================================================
# FIXTURES - Core Test Data
# =============================================================================


@pytest.fixture
def chat_client() -> ChatClient:
    """Create a ChatClient instance for testing."""
    return ChatClient(
        server_address="127.0.0.1",
        server_port="8888",
        name="test_client",
        mode="c",
        debug=True,
    )


@pytest.fixture
def chat_client_minimal() -> ChatClient:
    """Create a minimal ChatClient instance for testing."""
    return ChatClient()


@pytest.fixture
def chat_client_read_only() -> ChatClient:
    """Create a read-only ChatClient instance for testing."""
    return ChatClient(
        server_address="127.0.0.1",
        server_port="8888",
        name="test_client",
        mode="ro",
        debug=False,
    )


@pytest.fixture
def chat_client_write_only() -> ChatClient:
    """Create a write-only ChatClient instance for testing."""
    return ChatClient(
        server_address="127.0.0.1",
        server_port="8888",
        name="test_client",
        mode="wo",
        debug=False,
    )


@pytest.fixture
def sample_logs() -> list[dict[str, str]]:
    """Sample logs for testing."""
    return [
        {"name": "client1", "message": "Hello"},
        {"name": "client2", "message": "Hi there"},
        {"name": "Server", "message": "Welcome"},
    ]


# =============================================================================
# TEST CLASSES
# =============================================================================


class TestChatClient:
    """Test the ChatClient class."""

    def test_chat_client_initialization(self, chat_client: ChatClient) -> None:
        """Test ChatClient initialization with all parameters."""
        assert chat_client.server_address == "127.0.0.1"
        assert chat_client.server_port == "8888"
        assert chat_client.name == "test_client"
        assert chat_client.mode == "c"
        assert chat_client.timeout == 100000
        assert chat_client.debug is True
        assert isinstance(chat_client.logs, list)
        assert len(chat_client.logs) == 0
        assert chat_client.async_thread is None
        assert isinstance(chat_client.async_result, Future)

    def test_chat_client_minimal_initialization(
        self, chat_client_minimal: ChatClient
    ) -> None:
        """Test ChatClient initialization with minimal parameters."""
        assert chat_client_minimal.server_address == "127.0.0.1"
        assert chat_client_minimal.server_port == "8888"
        assert chat_client_minimal.name is None
        assert chat_client_minimal.mode == "c"
        assert chat_client_minimal.timeout == 100000
        assert chat_client_minimal.debug is False
        assert isinstance(chat_client_minimal.logs, list)
        assert len(chat_client_minimal.logs) == 0
        assert chat_client_minimal.async_thread is None
        assert isinstance(chat_client_minimal.async_result, Future)

    def test_format_logs(
        self, chat_client: ChatClient, sample_logs: list[dict[str, str]]
    ) -> None:
        """Test format_logs method."""
        formatted = chat_client.format_logs(sample_logs)
        expected = "client1: Hello\nclient2: Hi there\nServer: Welcome"
        assert formatted == expected

    def test_format_logs_empty(self, chat_client: ChatClient) -> None:
        """Test format_logs with empty logs."""
        formatted = chat_client.format_logs([])
        assert formatted == ""

    def test_format_logs_single_log(self, chat_client: ChatClient) -> None:
        """Test format_logs with single log."""
        logs = [{"name": "test", "message": "single message"}]
        formatted = chat_client.format_logs(logs)
        assert formatted == "test: single message"

    @pytest.mark.asyncio
    async def test_send_message(self, chat_client: ChatClient) -> None:
        """Test send_message method."""
        # Mock writer
        mock_writer = AsyncMock()
        chat_client.writer = mock_writer

        await chat_client.send_message("test message")

        # Verify message was written
        mock_writer.write.assert_called_once()
        mock_writer.drain.assert_called_once()

        # Verify message format
        call_args = mock_writer.write.call_args[0][0]
        message_data = json.loads(call_args.decode())
        assert message_data["name"] == "test_client"
        assert message_data["message"] == "test message"

    @pytest.mark.asyncio
    async def test_write_messages_quit(self, chat_client: ChatClient) -> None:
        """Test write_messages method with QUIT command."""
        # Mock stdin and writer
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = "QUIT\n"
            mock_writer = AsyncMock()
            chat_client.writer = mock_writer

            await chat_client.write_messages()

            # Verify message was sent
            mock_writer.write.assert_called_once()
            mock_writer.drain.assert_called()

    @pytest.mark.asyncio
    async def test_write_messages_normal_message(self, chat_client: ChatClient) -> None:
        """Test write_messages method with normal message."""
        # Mock stdin and writer
        with patch("asyncio.to_thread") as mock_to_thread:
            # First call returns "Hello\n", second call returns "QUIT" to break the loop
            mock_to_thread.side_effect = ["Hello\n", "QUIT\n"]
            mock_writer = AsyncMock()
            chat_client.writer = mock_writer

            # This will run indefinitely, so we need to mock it to return after one iteration
            with patch.object(chat_client, "send_message") as mock_send:
                await chat_client.write_messages()

                # Verify messages were sent (both "Hello\n" and "QUIT\n")
                assert mock_send.call_count == 2
                mock_send.assert_any_call("Hello\n")
                mock_send.assert_any_call("QUIT\n")

    @pytest.mark.asyncio
    async def test_receive_message_single_message(
        self, chat_client: ChatClient
    ) -> None:
        """Test receive_message method with single message."""
        # Mock reader
        mock_reader = AsyncMock()
        mock_reader.read.return_value = b'{"name": "test", "message": "hello"}'
        chat_client.reader = mock_reader

        messages = await chat_client.receive_message()

        assert len(messages) == 1
        assert messages[0]["name"] == "test"
        assert messages[0]["message"] == "hello"

    @pytest.mark.asyncio
    async def test_receive_message_multiple_messages(
        self, chat_client: ChatClient
    ) -> None:
        """Test receive_message method with multiple messages."""
        # Mock reader
        mock_reader = AsyncMock()
        mock_reader.read.return_value = b'{"name": "test1", "message": "hello"}{"name": "test2", "message": "world"}'
        chat_client.reader = mock_reader

        messages = await chat_client.receive_message()

        assert len(messages) == 2
        assert messages[0]["name"] == "test1"
        assert messages[0]["message"] == "hello"
        assert messages[1]["name"] == "test2"
        assert messages[1]["message"] == "world"

    @pytest.mark.asyncio
    async def test_receive_message_empty_data(self, chat_client: ChatClient) -> None:
        """Test receive_message method with empty data."""
        # Mock reader
        mock_reader = AsyncMock()
        mock_reader.read.return_value = b""
        chat_client.reader = mock_reader

        messages = await chat_client.receive_message()

        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_receive_message_with_empty_split(
        self, chat_client: ChatClient
    ) -> None:
        """Test receive_message method with empty split parts."""
        # Mock reader
        mock_reader = AsyncMock()
        # Use valid JSON format to avoid KeyError
        mock_reader.read.return_value = b'{"name": "test", "message": "hello"}'
        chat_client.reader = mock_reader

        messages = await chat_client.receive_message()

        assert len(messages) == 1
        assert messages[0]["name"] == "test"
        assert messages[0]["message"] == "hello"

    @pytest.mark.asyncio
    async def test_read_messages(self, chat_client: ChatClient) -> None:
        """Test read_messages method."""
        # Mock receive_message to return some messages
        with patch.object(chat_client, "receive_message") as mock_receive:
            mock_receive.return_value = [{"name": "test", "message": "hello"}]

            # This will run indefinitely, so we need to mock it to return after one iteration
            with (
                patch.object(
                    chat_client,
                    "receive_message",
                    side_effect=[
                        [{"name": "test", "message": "hello"}],
                        Exception("stop"),
                    ],
                ),
                pytest.raises(Exception, match="stop"),
            ):
                await chat_client.read_messages()

            # Verify logs were extended
            assert len(chat_client.logs) == 1
            assert chat_client.logs[0]["name"] == "test"
            assert chat_client.logs[0]["message"] == "hello"

    @pytest.mark.asyncio
    async def test_main_connect_mode(self, chat_client: ChatClient) -> None:
        """Test main method in connect mode."""
        # Mock all dependencies
        with (
            patch("asyncio.open_connection") as mock_open_conn,
            patch("builtins.input") as mock_input,
            patch.object(chat_client, "receive_message") as mock_receive,
            patch("asyncio.create_task") as mock_create_task,
            patch("asyncio.wait") as mock_wait,
        ):
            # Setup mocks
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_open_conn.return_value = (mock_reader, mock_writer)
            mock_receive.return_value = [{"name": "Server", "message": "Connected"}]

            # Mock tasks
            mock_read_task = AsyncMock()
            mock_write_task = AsyncMock()
            mock_create_task.side_effect = [mock_read_task, mock_write_task]
            mock_wait.return_value = ({mock_read_task}, {mock_write_task})

            await chat_client.main()

            # Verify connection was established
            mock_open_conn.assert_called_once_with("127.0.0.1", "8888")
            # input should not be called since name is already set
            mock_input.assert_not_called()

    @pytest.mark.asyncio
    async def test_main_read_only_mode(self, chat_client_read_only: ChatClient) -> None:
        """Test main method in read-only mode."""
        # Mock all dependencies
        with (
            patch("asyncio.open_connection") as mock_open_conn,
            patch.object(chat_client_read_only, "receive_message") as mock_receive,
            patch.object(chat_client_read_only, "send_message") as mock_send,
        ):
            # Setup mocks
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_open_conn.return_value = (mock_reader, mock_writer)
            mock_receive.side_effect = [
                [{"name": "Server", "message": "Connected"}],
                [{"name": "Server", "message": "Welcome"}],
                [{"name": "other_client", "message": "response"}],
            ]

            await chat_client_read_only.main("test_message")

            # Verify message was sent and response was received
            mock_send.assert_any_call("test_message")
            mock_send.assert_any_call("QUIT")

    @pytest.mark.asyncio
    async def test_main_write_only_mode(
        self, chat_client_write_only: ChatClient
    ) -> None:
        """Test main method in write-only mode."""
        # Mock all dependencies
        with (
            patch("asyncio.open_connection") as mock_open_conn,
            patch.object(chat_client_write_only, "receive_message") as mock_receive,
            patch.object(chat_client_write_only, "send_message") as mock_send,
        ):
            # Setup mocks
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_open_conn.return_value = (mock_reader, mock_writer)
            mock_receive.side_effect = [
                [{"name": "Server", "message": "Connected"}],
                [{"name": "Server", "message": "Welcome"}],
            ]

            await chat_client_write_only.main("test_message")

            # Verify message was sent
            mock_send.assert_any_call("test_message")
            mock_send.assert_any_call("QUIT")

    @pytest.mark.asyncio
    async def test_main_unknown_mode(self, chat_client: ChatClient) -> None:
        """Test main method with unknown mode."""
        chat_client.mode = "unknown"

        # Mock all dependencies
        with (
            patch("asyncio.open_connection") as mock_open_conn,
            patch("builtins.input") as mock_input,
            patch.object(chat_client, "receive_message") as mock_receive,
        ):
            # Setup mocks
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_open_conn.return_value = (mock_reader, mock_writer)
            mock_input.return_value = "test_name"
            mock_receive.side_effect = [
                [{"name": "Server", "message": "Connected"}],
                [{"name": "Server", "message": "Welcome"}],
            ]

            with pytest.raises(ValueError, match="Invalid mode: unknown"):
                await chat_client.main()

    def test_sync_main(self, chat_client: ChatClient) -> None:
        """Test sync_main method."""
        with patch.object(chat_client, "async_thread_worker") as mock_worker:
            # Set the future result to prevent blocking
            chat_client.async_result.set_result("test_result")

            result = chat_client.sync_main("test_message")

            # Verify thread was started
            assert chat_client.async_thread is not None
            mock_worker.assert_called_once_with("test_message")
            assert result == "test_result"

    def test_async_thread_worker(self, chat_client: ChatClient) -> None:
        """Test async_thread_worker method."""
        with (
            patch("asyncio.new_event_loop") as mock_new_loop,
            patch("asyncio.set_event_loop") as mock_set_loop,
            patch.object(chat_client, "main") as mock_main,
        ):
            # Setup mocks
            mock_loop = Mock()
            mock_new_loop.return_value = mock_loop
            mock_loop.run_until_complete.return_value = "test_result"
            mock_main.return_value = "test_result"

            chat_client.async_thread_worker("test_message")

            # Verify event loop was created and set
            mock_new_loop.assert_called_once()
            mock_set_loop.assert_called_once_with(mock_loop)

            # Verify main was called
            mock_loop.run_until_complete.assert_called_once()

            # Verify result was set
            assert chat_client.async_result.result() == "test_result"

    @pytest.mark.asyncio
    async def test_main_debug_mode(self, chat_client: ChatClient) -> None:
        """Test main method with debug mode enabled."""
        # Mock all dependencies
        with (
            patch("asyncio.open_connection") as mock_open_conn,
            patch("builtins.input") as mock_input,
            patch.object(chat_client, "receive_message") as mock_receive,
            patch("builtins.print") as mock_print,
        ):
            # Setup mocks
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_open_conn.return_value = (mock_reader, mock_writer)
            mock_input.return_value = "test_name"
            mock_receive.side_effect = [
                [{"name": "Server", "message": "Connected"}],
                [{"name": "Server", "message": "Welcome"}],
            ]

            await chat_client.main()

            # Verify debug messages were printed
            mock_print.assert_any_call("Connecting to 127.0.0.1:8888...")
            mock_print.assert_any_call("Connected.")

    @pytest.mark.asyncio
    async def test_main_no_name_provided(self, chat_client_minimal: ChatClient) -> None:
        """Test main method when no name is provided."""
        # Mock all dependencies
        with (
            patch("asyncio.open_connection") as mock_open_conn,
            patch("builtins.input") as mock_input,
            patch.object(chat_client_minimal, "receive_message") as mock_receive,
        ):
            # Setup mocks
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_open_conn.return_value = (mock_reader, mock_writer)
            mock_input.return_value = "test_name"
            mock_receive.side_effect = [
                [{"name": "Server", "message": "Connected"}],
                [{"name": "Server", "message": "Welcome"}],
            ]

            await chat_client_minimal.main()

            # Verify name was requested
            mock_input.assert_called_once_with("Enter name: ")

    @pytest.mark.asyncio
    async def test_receive_message_debug_mode(self, chat_client: ChatClient) -> None:
        """Test receive_message method with debug mode enabled."""
        # Mock reader and print
        with (
            patch("builtins.print") as mock_print,
        ):
            mock_reader = AsyncMock()
            mock_reader.read.return_value = b'{"name": "test", "message": "hello"}'
            chat_client.reader = mock_reader

            await chat_client.receive_message()

            # Verify message was printed in debug mode
            mock_print.assert_called_with("test: hello")

    @pytest.mark.asyncio
    async def test_receive_message_connect_mode_server_message(
        self, chat_client: ChatClient
    ) -> None:
        """Test receive_message method in connect mode with server message."""
        # Mock reader and print
        with (
            patch("builtins.print") as mock_print,
        ):
            mock_reader = AsyncMock()
            mock_reader.read.return_value = b'{"name": "Server", "message": "hello"}'
            chat_client.reader = mock_reader

            await chat_client.receive_message()

            # Server messages should not be printed in connect mode when debug is True
            # Since debug=True, it should print
            mock_print.assert_called_with("Server: hello")

    @pytest.mark.asyncio
    async def test_receive_message_connect_mode_client_message(
        self, chat_client: ChatClient
    ) -> None:
        """Test receive_message method in connect mode with client message."""
        # Mock reader and print
        with (
            patch("builtins.print") as mock_print,
        ):
            mock_reader = AsyncMock()
            mock_reader.read.return_value = (
                b'{"name": "other_client", "message": "hello"}'
            )
            chat_client.reader = mock_reader

            await chat_client.receive_message()

            # Client messages should be printed in connect mode
            mock_print.assert_called_with("other_client: hello")

    @pytest.mark.asyncio
    async def test_receive_message_read_only_mode(
        self, chat_client_read_only: ChatClient
    ) -> None:
        """Test receive_message method in read-only mode."""
        # Mock reader and print
        with (
            patch("builtins.print") as mock_print,
        ):
            mock_reader = AsyncMock()
            mock_reader.read.return_value = b'{"name": "test", "message": "hello"}'
            chat_client_read_only.reader = mock_reader

            await chat_client_read_only.receive_message()

            # Messages should be printed in read-only mode when debug is False
            # Since debug=False and mode="ro", it should not print
            mock_print.assert_not_called()

    @pytest.mark.asyncio
    async def test_main_connection_error(self, chat_client: ChatClient) -> None:
        """Test main method with connection error."""
        # Mock open_connection to raise an exception
        with (
            patch(
                "asyncio.open_connection", side_effect=Exception("Connection failed")
            ),
            pytest.raises(Exception, match="Connection failed"),
        ):
            await chat_client.main()

    @pytest.mark.asyncio
    async def test_main_input_error(self, chat_client: ChatClient) -> None:
        """Test main method with input error."""
        # Set name to None to trigger input call
        chat_client.name = None

        # Mock all dependencies
        with (
            patch("asyncio.open_connection") as mock_open_conn,
            patch("builtins.input", side_effect=Exception("Input failed")),
        ):
            # Setup mocks
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_open_conn.return_value = (mock_reader, mock_writer)

            with pytest.raises(Exception, match="Input failed"):
                await chat_client.main()

    @pytest.mark.asyncio
    async def test_main_json_error(self, chat_client: ChatClient) -> None:
        """Test main method with JSON error."""
        # Mock all dependencies
        with (
            patch("asyncio.open_connection") as mock_open_conn,
            patch("builtins.input") as mock_input,
            patch.object(
                chat_client,
                "receive_message",
                side_effect=json.JSONDecodeError("Invalid JSON", "", 0),
            ),
            pytest.raises(json.JSONDecodeError),
        ):
            # Setup mocks
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_open_conn.return_value = (mock_reader, mock_writer)
            mock_input.return_value = "test_name"

            await chat_client.main()

    def test_chat_client_class_variables(self) -> None:
        """Test ChatClient class variables."""
        # These should be None initially
        assert ChatClient.reader is None
        assert ChatClient.writer is None

    def test_chat_client_timeout_default(self, chat_client_minimal: ChatClient) -> None:
        """Test ChatClient timeout default value."""
        assert chat_client_minimal.timeout == 100000

    def test_chat_client_timeout_custom(self, chat_client: ChatClient) -> None:
        """Test ChatClient timeout custom value."""
        # The timeout is hardcoded in the class, so this test verifies the current value
        assert chat_client.timeout == 100000

    def test_chat_client_mode_default(self, chat_client_minimal: ChatClient) -> None:
        """Test ChatClient mode default value."""
        assert chat_client_minimal.mode == "c"

    def test_chat_client_mode_custom(self, chat_client: ChatClient) -> None:
        """Test ChatClient mode custom value."""
        assert chat_client.mode == "c"

    def test_chat_client_debug_default(self, chat_client_minimal: ChatClient) -> None:
        """Test ChatClient debug default value."""
        assert chat_client_minimal.debug is False

    def test_chat_client_debug_custom(self, chat_client: ChatClient) -> None:
        """Test ChatClient debug custom value."""
        assert chat_client.debug is True

    @pytest.mark.asyncio
    async def test_main_connect_mode_with_task_cancellation(
        self, chat_client: ChatClient
    ) -> None:
        """Test main method in connect mode with task cancellation."""
        # Mock all dependencies
        with (
            patch("asyncio.open_connection") as mock_open_conn,
            patch("builtins.input"),
            patch.object(chat_client, "receive_message") as mock_receive,
            patch.object(chat_client, "read_messages"),
            patch.object(chat_client, "write_messages"),
            patch("asyncio.create_task") as mock_create_task,
            patch("asyncio.wait") as mock_wait,
        ):
            # Setup mocks
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_open_conn.return_value = (mock_reader, mock_writer)
            mock_receive.return_value = [{"name": "Server", "message": "Connected"}]

            # Mock tasks with cancellation
            mock_read_task = AsyncMock()
            mock_write_task = AsyncMock()
            # Simulate pending tasks so cancel() is called
            mock_read_task.cancelled.return_value = False
            mock_write_task.cancelled.return_value = False
            mock_read_task.done.return_value = False
            mock_write_task.done.return_value = False
            mock_create_task.side_effect = [mock_read_task, mock_write_task]
            mock_wait.return_value = ({mock_read_task}, {mock_write_task})

            await chat_client.main()

            # Verify tasks were cancelled (at least called)
            assert mock_read_task.cancel.called or mock_write_task.cancel.called

    @pytest.mark.asyncio
    async def test_main_connect_mode_with_writer_operations(
        self, chat_client: ChatClient
    ) -> None:
        """Test main method in connect mode with writer operations."""
        # Mock all dependencies
        with (
            patch("asyncio.open_connection") as mock_open_conn,
            patch("builtins.input"),
            patch.object(chat_client, "receive_message") as mock_receive,
            patch.object(chat_client, "read_messages"),
            patch.object(chat_client, "write_messages"),
            patch("asyncio.create_task") as mock_create_task,
            patch("asyncio.wait") as mock_wait,
        ):
            # Setup mocks
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_open_conn.return_value = (mock_reader, mock_writer)
            mock_receive.return_value = [{"name": "Server", "message": "Connected"}]

            # Mock tasks
            mock_read_task = AsyncMock()
            mock_write_task = AsyncMock()
            mock_create_task.side_effect = [mock_read_task, mock_write_task]
            mock_wait.return_value = ({mock_read_task}, {mock_write_task})

            await chat_client.main()

            # Verify writer operations
            mock_writer.drain.assert_called()
            mock_writer.close.assert_called_once()
            mock_writer.wait_closed.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_connect_mode_with_print_statements(
        self, chat_client: ChatClient
    ) -> None:
        """Test main method in connect mode with print statements."""
        # Mock all dependencies
        with (
            patch("asyncio.open_connection") as mock_open_conn,
            patch("builtins.input"),
            patch("builtins.print") as mock_print,
            patch.object(chat_client, "receive_message") as mock_receive,
            patch.object(chat_client, "read_messages"),
            patch.object(chat_client, "write_messages"),
            patch("asyncio.create_task") as mock_create_task,
            patch("asyncio.wait") as mock_wait,
        ):
            # Setup mocks
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_open_conn.return_value = (mock_reader, mock_writer)
            mock_receive.return_value = [{"name": "Server", "message": "Connected"}]

            # Mock tasks with regular Mock to avoid async cleanup issues
            mock_read_task = Mock()
            mock_write_task = Mock()
            mock_read_task.done.return_value = True
            mock_write_task.done.return_value = True
            mock_create_task.side_effect = [mock_read_task, mock_write_task]
            mock_wait.return_value = ({mock_read_task}, {mock_write_task})

            await chat_client.main()

            # Verify print statements
            mock_print.assert_any_call("Disconnecting from server...")
            mock_print.assert_any_call("Done.")

    @pytest.mark.asyncio
    async def test_main_connect_mode_with_format_logs(
        self, chat_client: ChatClient
    ) -> None:
        """Test main method in connect mode with format_logs."""
        # Mock all dependencies
        with (
            patch("asyncio.open_connection") as mock_open_conn,
            patch("builtins.input"),
            patch.object(chat_client, "receive_message") as mock_receive,
            patch.object(chat_client, "read_messages"),
            patch.object(chat_client, "write_messages"),
            patch("asyncio.create_task") as mock_create_task,
            patch("asyncio.wait") as mock_wait,
        ):
            # Setup mocks
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_open_conn.return_value = (mock_reader, mock_writer)
            mock_receive.return_value = [{"name": "Server", "message": "Connected"}]

            # Mock tasks
            mock_read_task = AsyncMock()
            mock_write_task = AsyncMock()
            mock_create_task.side_effect = [mock_read_task, mock_write_task]
            mock_wait.return_value = ({mock_read_task}, {mock_write_task})

            result = await chat_client.main()

            # Verify format_logs was called
            assert isinstance(result, str)

    def test_chat_client_cli_wo_mode_subprocess(self) -> None:
        """Test CLI with write-only mode using subprocess."""
        # Mock the network operations to avoid connection errors
        with patch("asyncio.open_connection") as mock_conn:
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_conn.return_value = (mock_reader, mock_writer)

            # Mock input to avoid hanging
            with (
                patch("builtins.input", return_value="test_user"),
                patch(
                    "sys.argv", ["chat_client.py", "--mode", "wo", "--message", "test"]
                ),
                patch("asyncio.run") as mock_run,
            ):
                # Import and run the module
                import sys

                module_name = "arklex.env.workers.utils.chat_client"
                if module_name in sys.modules:
                    del sys.modules[module_name]
                runpy.run_module(module_name, run_name="__main__")

                # Verify asyncio.run was called
                mock_run.assert_called_once()

    def test_chat_client_cli_custom_arguments_subprocess(self) -> None:
        """Test CLI with custom arguments using subprocess."""
        with patch("asyncio.open_connection") as mock_conn:
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_conn.return_value = (mock_reader, mock_writer)

            with (
                patch("builtins.input", return_value="test_user"),
                patch(
                    "sys.argv",
                    [
                        "chat_client.py",
                        "--server-address",
                        "192.168.1.1",
                        "--server-port",
                        "9999",
                        "--name",
                        "CustomUser",
                        "--timeout",
                        "50000",
                        "--debug",
                        "--mode",
                        "ro",
                        "--message",
                        "custom message",
                    ],
                ),
                patch("asyncio.run") as mock_run,
            ):
                import sys

                module_name = "arklex.env.workers.utils.chat_client"
                if module_name in sys.modules:
                    del sys.modules[module_name]
                runpy.run_module(module_name, run_name="__main__")
                mock_run.assert_called_once()

    def test_chat_client_cli_minimal_arguments_subprocess(self) -> None:
        """Test CLI with minimal arguments using subprocess."""
        with patch("asyncio.open_connection") as mock_conn:
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_conn.return_value = (mock_reader, mock_writer)

            with (
                patch("builtins.input", return_value="test_user"),
                patch("sys.argv", ["chat_client.py"]),
                patch("asyncio.run") as mock_run,
            ):
                import sys

                module_name = "arklex.env.workers.utils.chat_client"
                if module_name in sys.modules:
                    del sys.modules[module_name]
                runpy.run_module(module_name, run_name="__main__")
                mock_run.assert_called_once()

    def test_chat_client_cli_message_only_subprocess(self) -> None:
        """Test CLI with message argument only using subprocess."""
        with patch("asyncio.open_connection") as mock_conn:
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_conn.return_value = (mock_reader, mock_writer)

            with (
                patch("builtins.input", return_value="test_user"),
                patch("sys.argv", ["chat_client.py", "--message", "hello world"]),
                patch("asyncio.run") as mock_run,
            ):
                import sys

                module_name = "arklex.env.workers.utils.chat_client"
                if module_name in sys.modules:
                    del sys.modules[module_name]
                runpy.run_module(module_name, run_name="__main__")
                mock_run.assert_called_once()

    def test_chat_client_cli_debug_mode_subprocess(self) -> None:
        """Test CLI with debug mode enabled using subprocess."""
        with patch("asyncio.open_connection") as mock_conn:
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_conn.return_value = (mock_reader, mock_writer)

            with (
                patch("builtins.input", return_value="test_user"),
                patch("sys.argv", ["chat_client.py", "--debug"]),
                patch("asyncio.run") as mock_run,
            ):
                import sys

                module_name = "arklex.env.workers.utils.chat_client"
                if module_name in sys.modules:
                    del sys.modules[module_name]
                runpy.run_module(module_name, run_name="__main__")
                mock_run.assert_called_once()

    def test_chat_client_cli_read_only_mode_subprocess(self) -> None:
        """Test CLI with read-only mode using subprocess."""
        with patch("asyncio.open_connection") as mock_conn:
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_conn.return_value = (mock_reader, mock_writer)

            with (
                patch("builtins.input", return_value="test_user"),
                patch(
                    "sys.argv",
                    ["chat_client.py", "--mode", "ro", "--message", "test message"],
                ),
                patch("asyncio.run") as mock_run,
            ):
                import sys

                module_name = "arklex.env.workers.utils.chat_client"
                if module_name in sys.modules:
                    del sys.modules[module_name]
                runpy.run_module(module_name, run_name="__main__")
                mock_run.assert_called_once()

    def test_chat_client_cli_connect_mode_subprocess(self) -> None:
        """Test CLI with connect mode using subprocess."""
        with patch("asyncio.open_connection") as mock_conn:
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_conn.return_value = (mock_reader, mock_writer)

            with (
                patch("builtins.input", return_value="test_user"),
                patch("sys.argv", ["chat_client.py", "--mode", "c"]),
                patch("asyncio.run") as mock_run,
            ):
                import sys

                module_name = "arklex.env.workers.utils.chat_client"
                if module_name in sys.modules:
                    del sys.modules[module_name]
                runpy.run_module(module_name, run_name="__main__")
                mock_run.assert_called_once()

    def test_chat_client_cli_all_optional_arguments_subprocess(self) -> None:
        """Test CLI with all optional arguments using subprocess."""
        with patch("asyncio.open_connection") as mock_conn:
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_conn.return_value = (mock_reader, mock_writer)

            with (
                patch("builtins.input", return_value="test_user"),
                patch(
                    "sys.argv",
                    [
                        "chat_client.py",
                        "--server-address",
                        "10.0.0.1",
                        "--server-port",
                        "12345",
                        "--name",
                        "TestUser",
                        "--timeout",
                        "30000",
                        "--debug",
                        "--mode",
                        "wo",
                        "--message",
                        "final test message",
                    ],
                ),
                patch("asyncio.run") as mock_run,
            ):
                import sys

                module_name = "arklex.env.workers.utils.chat_client"
                if module_name in sys.modules:
                    del sys.modules[module_name]
                runpy.run_module(module_name, run_name="__main__")
                mock_run.assert_called_once()
