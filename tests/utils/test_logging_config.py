"""Tests for logging configuration."""

import json
import logging
import os
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from arklex.utils.logging_config import (
    DEFAULT_LOG_LEVEL,
    LOG_LEVELS,
    MODULE_LOG_LEVELS,
    ContextFilter,
    JSONFormatter,
    RequestIdFilter,
    setup_logging,
)

log_context = logging.getLogger(__name__)


@pytest.fixture
def temp_log_dir(tmp_path: Path) -> Generator[str, None, None]:
    """Create a temporary directory for log files.

    Args:
        tmp_path: Pytest fixture providing a temporary directory.

    Yields:
        Path to the temporary directory.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    yield str(log_dir)


def test_get_log_context() -> None:
    """Test getting a log_context instance."""
    log_context = logging.getLogger("test_log_context")
    assert log_context.name == "test_log_context"


def test_get_log_context_with_level() -> None:
    """Test getting a log_context with a specific level."""
    log_context = logging.getLogger("test_level")
    log_context.setLevel(logging.DEBUG)
    assert log_context.level == logging.DEBUG


def test_get_log_context_with_format() -> None:
    """Test getting a log_context with a custom format."""
    custom_format = "%(levelname)s - %(message)s"
    formatter = logging.Formatter(custom_format)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    log_context = logging.getLogger("test_format")
    log_context.addHandler(handler)
    assert log_context.handlers[0].formatter._fmt == custom_format


def test_setup_logging(temp_log_dir: str) -> None:
    """Test setting up logging configuration.

    Args:
        temp_log_dir: Path to temporary directory for log files.
    """
    setup_logging(log_level="DEBUG", log_dir=temp_log_dir)
    root_log_context = logging.getLogger()
    assert root_log_context.level == logging.DEBUG
    assert len(root_log_context.handlers) == 2  # Console and file handlers


def test_setup_logging_without_log_dir() -> None:
    """Test setup_logging without log directory (console only)."""
    setup_logging(log_level="INFO")
    root_log_context = logging.getLogger()
    assert root_log_context.level == logging.INFO
    assert len(root_log_context.handlers) == 1  # Console handler only


def test_setup_logging_with_json_format() -> None:
    """Test setup_logging with JSON formatting."""
    setup_logging(use_json=True)
    root_log_context = logging.getLogger()
    assert isinstance(root_log_context.handlers[0].formatter, JSONFormatter)


def test_setup_logging_with_custom_app_name(temp_log_dir: str) -> None:
    """Test setup_logging with custom app name."""
    setup_logging(log_dir=temp_log_dir, app_name="test_app")
    log_files = list(Path(temp_log_dir).glob("*.log"))
    assert len(log_files) == 1
    assert log_files[0].name == "test_app.log"


def test_setup_logging_with_custom_max_bytes(temp_log_dir: str) -> None:
    """Test setup_logging with custom max bytes."""
    custom_max_bytes = 1024
    setup_logging(log_dir=temp_log_dir, max_bytes=custom_max_bytes)
    root_log_context = logging.getLogger()
    file_handler = next(
        h
        for h in root_log_context.handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
    )
    assert file_handler.maxBytes == custom_max_bytes


def test_setup_logging_with_invalid_log_level() -> None:
    """Test setup_logging with invalid log level string."""
    setup_logging(log_level="INVALID_LEVEL")
    root_log_context = logging.getLogger()
    assert root_log_context.level == DEFAULT_LOG_LEVEL


def test_setup_logging_with_integer_log_level() -> None:
    """Test setup_logging with integer log level."""
    setup_logging(log_level=logging.WARNING)
    root_log_context = logging.getLogger()
    assert root_log_context.level == logging.WARNING


def test_setup_logging_without_hostname() -> None:
    """Test setup_logging with include_hostname=False."""
    setup_logging(use_json=True, include_hostname=False)
    root_log_context = logging.getLogger()
    formatter = root_log_context.handlers[0].formatter
    assert isinstance(formatter, JSONFormatter)
    assert not formatter.include_hostname


def test_request_id_filter() -> None:
    """Test request ID filter."""
    filter_obj = RequestIdFilter("test-123")
    record = logging.LogRecord(
        "test", logging.INFO, "test.py", 1, "Test message", (), None
    )
    assert filter_obj.filter(record)
    assert record.request_id == "test-123"


def test_request_id_filter_default() -> None:
    """Test request ID filter with default request_id."""
    filter_obj = RequestIdFilter()
    record = logging.LogRecord(
        "test", logging.INFO, "test.py", 1, "Test message", (), None
    )
    assert filter_obj.filter(record)
    assert record.request_id == "N/A"


def test_context_filter() -> None:
    """Test context filter."""
    context = {"user_id": "123", "action": "test"}
    filter_obj = ContextFilter(context)
    record = logging.LogRecord(
        "test", logging.INFO, "test.py", 1, "Test message", (), None
    )
    assert filter_obj.filter(record)
    assert record.context == context


def test_context_filter_no_context() -> None:
    """Test context filter with no context."""
    filter_obj = ContextFilter()
    record = logging.LogRecord(
        "test", logging.INFO, "test.py", 1, "Test message", (), None
    )
    assert filter_obj.filter(record)
    assert record.context == {}


def test_context_filter_none_context() -> None:
    """Test context filter with None context."""
    filter_obj = ContextFilter(None)
    record = logging.LogRecord(
        "test", logging.INFO, "test.py", 1, "Test message", (), None
    )
    assert filter_obj.filter(record)
    assert record.context == {}


def test_json_formatter_basic() -> None:
    """Test JSONFormatter basic functionality."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        "test", logging.INFO, "test.py", 1, "Test message", (), None
    )
    record.funcName = "test_json_formatter_basic"  # Set funcName explicitly
    result = formatter.format(record)
    log_data = json.loads(result)

    assert log_data["level"] == "INFO"
    assert log_data["name"] == "test"
    assert log_data["message"] == "Test message"
    assert log_data["module"] == "test"
    assert log_data["function"] == "test_json_formatter_basic"
    assert log_data["line"] == 1
    assert "hostname" in log_data
    assert "platform" in log_data
    assert "python_version" in log_data


def test_json_formatter_without_hostname() -> None:
    """Test JSONFormatter without hostname."""
    formatter = JSONFormatter(include_hostname=False)
    record = logging.LogRecord(
        "test", logging.INFO, "test.py", 1, "Test message", (), None
    )
    result = formatter.format(record)
    log_data = json.loads(result)

    assert "hostname" not in log_data
    assert "platform" not in log_data
    assert "python_version" not in log_data


def test_json_formatter_with_request_id() -> None:
    """Test JSONFormatter with request_id in record."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        "test", logging.INFO, "test.py", 1, "Test message", (), None
    )
    record.request_id = "req-123"
    result = formatter.format(record)
    log_data = json.loads(result)

    assert log_data["request_id"] == "req-123"


def test_json_formatter_with_context() -> None:
    """Test JSONFormatter with context in record."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        "test", logging.INFO, "test.py", 1, "Test message", (), None
    )
    record.context = {"user_id": "123", "action": "test"}
    result = formatter.format(record)
    log_data = json.loads(result)

    assert log_data["context"] == {"user_id": "123", "action": "test"}


def test_json_formatter_with_exception() -> None:
    """Test JSONFormatter with exception information."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        "test", logging.ERROR, "test.py", 1, "Test error", (), None
    )

    # Create an exception
    try:
        raise ValueError("Test exception")
    except ValueError:
        record.exc_info = (ValueError, ValueError("Test exception"), None)

    result = formatter.format(record)
    log_data = json.loads(result)

    assert "exception" in log_data
    assert log_data["exception"]["type"] == "ValueError"
    assert log_data["exception"]["message"] == "Test exception"
    assert "traceback" in log_data["exception"]


def test_json_formatter_without_exception() -> None:
    """Test JSONFormatter without exception information."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        "test", logging.INFO, "test.py", 1, "Test message", (), None
    )
    record.exc_info = None
    result = formatter.format(record)
    log_data = json.loads(result)

    assert "exception" not in log_data


def test_json_formatter_socket_error() -> None:
    """Test JSONFormatter when socket.gethostname() fails."""
    with patch(
        "arklex.utils.logging_config.socket.gethostname",
        side_effect=OSError("Network error"),
    ):
        formatter = JSONFormatter()
        assert formatter.hostname is None

        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 1, "Test message", (), None
        )
        result = formatter.format(record)
        log_data = json.loads(result)

        assert "hostname" in log_data
        assert log_data["hostname"] is None


def test_log_levels_constants() -> None:
    """Test LOG_LEVELS constant."""
    assert LOG_LEVELS["CRITICAL"] == logging.CRITICAL
    assert LOG_LEVELS["ERROR"] == logging.ERROR
    assert LOG_LEVELS["WARNING"] == logging.WARNING
    assert LOG_LEVELS["INFO"] == logging.INFO
    assert LOG_LEVELS["DEBUG"] == logging.DEBUG


def test_module_log_levels_constants() -> None:
    """Test MODULE_LOG_LEVELS constant."""
    assert "urllib3" in MODULE_LOG_LEVELS
    assert "requests" in MODULE_LOG_LEVELS
    assert "arklex.api" in MODULE_LOG_LEVELS
    assert "arklex.utils" in MODULE_LOG_LEVELS


def test_setup_logging_module_specific_levels() -> None:
    """Test that module-specific log levels are set correctly."""
    setup_logging()

    # Check that module-specific loggers have correct levels
    for module_name, expected_level in MODULE_LOG_LEVELS.items():
        module_logger = logging.getLogger(module_name)
        assert module_logger.level == expected_level


def test_setup_logging_removes_existing_handlers() -> None:
    """Test that setup_logging removes existing handlers."""
    # Add a handler first
    root_logger = logging.getLogger()
    original_handler = logging.StreamHandler()
    root_logger.addHandler(original_handler)

    # Setup logging
    setup_logging()

    # Check that original handler was removed
    assert len(root_logger.handlers) == 1  # Only console handler
    assert original_handler not in root_logger.handlers


def test_setup_logging_console_handler_has_filters() -> None:
    """Test that console handler has RequestIdFilter."""
    setup_logging()
    root_logger = logging.getLogger()
    console_handler = root_logger.handlers[0]

    assert any(isinstance(f, RequestIdFilter) for f in console_handler.filters)


def test_setup_logging_file_handler_has_filters(temp_log_dir: str) -> None:
    """Test that file handler has RequestIdFilter."""
    setup_logging(log_dir=temp_log_dir)
    root_logger = logging.getLogger()
    file_handler = next(
        h
        for h in root_logger.handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
    )

    assert any(isinstance(f, RequestIdFilter) for f in file_handler.filters)


def test_setup_logging_creates_log_directory(temp_log_dir: str) -> None:
    """Test that setup_logging creates log directory if it doesn't exist."""
    non_existent_dir = Path(temp_log_dir) / "new_logs"
    setup_logging(log_dir=str(non_existent_dir))

    assert non_existent_dir.exists()
    assert non_existent_dir.is_dir()


def test_log_rotation(temp_log_dir: str) -> None:
    """Test log file rotation.

    Args:
        temp_log_dir: Path to temporary directory for log files.
    """
    # Use a smaller max_bytes for testing to ensure rotation occurs
    test_max_bytes = 512
    setup_logging(log_dir=temp_log_dir, max_bytes=test_max_bytes)
    log_context = (
        logging.getLogger()
    )  # Use root log_context to ensure file handler is attached

    # Ensure all handlers are at INFO level
    for handler in log_context.handlers:
        handler.setLevel(logging.INFO)

    # Write logs until rotation occurs (with a safety cap)
    max_attempts = 1000
    for i in range(max_attempts):
        log_context.info("Test log message %d %s", i, "X" * 100)
        for handler in log_context.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                handler.flush()
                if hasattr(handler, "stream") and hasattr(handler.stream, "fileno"):
                    os.fsync(handler.stream.fileno())
                if hasattr(handler, "baseFilename"):
                    file_size = Path(handler.baseFilename).stat().st_size
                    if file_size > test_max_bytes:
                        handler.doRollover()
        log_files = list(Path(temp_log_dir).glob("*.log*"))
        if len(log_files) > 1:
            break
    else:
        raise AssertionError(
            f"Log rotation did not occur after {max_attempts} attempts"
        )

    # Verify that rotation occurred
    log_files = list(Path(temp_log_dir).glob("*.log*"))
    assert len(log_files) > 1, "Expected multiple log files after rotation"

    # Clean up handlers
    for handler in log_context.handlers:
        handler.flush()
        if hasattr(handler, "close"):
            handler.close()


def test_log_levels() -> None:
    """Test different log levels."""
    log_context = logging.getLogger("test_levels")
    log_context.setLevel(logging.INFO)

    # Test each log level
    log_context.debug("Debug message")
    log_context.info("Info message")
    log_context.warning("Warning message")
    log_context.error("Error message")
    log_context.critical("Critical message")

    # Verify that the log_context has the correct level
    assert log_context.level == logging.INFO


def test_log_format() -> None:
    """Test custom log format."""
    custom_format = "%(levelname)s - %(message)s"
    formatter = logging.Formatter(custom_format)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    log_context = logging.getLogger("test_format")
    log_context.addHandler(handler)
    log_context.info("Test message")

    # Verify that the format was applied
    assert log_context.handlers[0].formatter._fmt == custom_format
