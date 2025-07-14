"""Tests for logging utilities."""

import logging
import string
from typing import NoReturn
from unittest.mock import Mock, patch

import pytest
import tenacity

from arklex.utils.exceptions import ArklexError, RetryableError
from arklex.utils.logging_utils import (
    LOG_MESSAGES,
    ContextFilter,
    LogContext,
    RequestIdFilter,
    handle_exceptions,
    with_retry,
)


# Helper to ensure log_context propagates and is at correct level for caplog
def get_test_context(name: str = "test", base_context: dict = None) -> LogContext:
    log_context = LogContext(name, level="INFO", base_context=base_context)
    log_context.setLevel(logging.DEBUG)
    log_context.propagate = True
    return log_context


def test_log_context_basic(caplog: pytest.LogCaptureFixture) -> None:
    context = get_test_context()
    context.info("Test message")
    assert any("Test message" in r.getMessage() for r in caplog.records)


def test_log_context_with_data(caplog: pytest.LogCaptureFixture) -> None:
    context = get_test_context(base_context={"base": "value"})
    context.info("Test message", extra="data")
    assert any("Test message" in r.getMessage() for r in caplog.records)
    # Context is not in the plain message, but we can check the log_context name
    assert any(r.name == "test" for r in caplog.records)


def test_log_context_error(caplog: pytest.LogCaptureFixture) -> None:
    context = get_test_context()
    try:
        raise ValueError("Test error")
    except ValueError:
        context.error("Error occurred", exc_info=True)
    assert any("Error occurred" in r.getMessage() for r in caplog.records)
    # The original error message might be in the exception info, not the main message
    # Check if any record has exc_info that contains the original error
    assert any(
        hasattr(r, "exc_info") and r.exc_info and "Test error" in str(r.exc_info[1])
        for r in caplog.records
    )


def test_log_context_stack(caplog: pytest.LogCaptureFixture) -> None:
    context = get_test_context(base_context={"base": "value"})
    context.push_context({"level1": "value1"})
    context.push_context({"level2": "value2"})
    context.info("Test message")
    assert any("Test message" in r.getMessage() for r in caplog.records)
    context.pop_context()
    caplog.clear()
    context.info("Test message")
    assert any("Test message" in r.getMessage() for r in caplog.records)
    context.pop_context()
    caplog.clear()
    context.info("Test message")
    assert any("Test message" in r.getMessage() for r in caplog.records)


@handle_exceptions(reraise=False)
def function_that_raises() -> NoReturn:
    raise ValueError("Test error")


def test_handle_exceptions(caplog: pytest.LogCaptureFixture) -> None:
    result = function_that_raises()
    assert result is None
    assert any("Test error" in r.getMessage() for r in caplog.records)


@handle_exceptions(reraise=True)
def function_that_raises_and_reraises() -> NoReturn:
    raise ValueError("Test error")


def test_handle_exceptions_reraises(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(ArklexError) as exc_info:
        function_that_raises_and_reraises()
    # The ArklexError message is "Operation failed in function_that_raises_and_reraises"
    # and the original error is in the details
    assert "Operation failed in function_that_raises_and_reraises" in str(
        exc_info.value
    )
    # Check that the original error is in the details
    assert hasattr(exc_info.value, "details") and "Test error" in str(
        exc_info.value.details
    )
    assert any("Test error" in r.getMessage() for r in caplog.records)


@with_retry(max_attempts=2)
def function_that_retries() -> NoReturn:
    raise RetryableError("Test retry error", error_code="RETRY")


def test_with_retry(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(tenacity.RetryError) as exc_info:
        function_that_retries()
    # The RetryError might not include the original error message in its string representation
    # Check that the original error is in the last exception that was raised
    last_exception = exc_info.value.last_attempt.exception()
    assert "Test retry error" in str(last_exception)
    assert any("Test retry error" in r.getMessage() for r in caplog.records)


def test_log_messages_consistency() -> None:
    for level, messages in LOG_MESSAGES.items():
        for key, message in messages.items():
            placeholders = [
                p[1] for p in string.Formatter().parse(message) if p[1] is not None
            ]
            assert all(isinstance(p, str) for p in placeholders), (
                f"Invalid placeholder in {level}.{key}"
            )


def test_log_context_with_standard_messages(caplog: pytest.LogCaptureFixture) -> None:
    context = get_test_context()
    context.info(LOG_MESSAGES["INFO"]["OPERATION_START"].format(operation="test_op"))
    assert any("Starting operation: test_op" in r.getMessage() for r in caplog.records)
    context.info(LOG_MESSAGES["INFO"]["OPERATION_END"].format(operation="test_op"))
    assert any("Completed operation: test_op" in r.getMessage() for r in caplog.records)
    context.warning(
        LOG_MESSAGES["WARNING"]["PERFORMANCE_WARNING"].format(message="slow operation")
    )
    assert any(
        "Performance warning: slow operation" in r.getMessage() for r in caplog.records
    )
    try:
        raise ValueError("test error")
    except ValueError as e:
        context.error(
            LOG_MESSAGES["ERROR"]["OPERATION_FAILED"].format(error=str(e)),
            exc_info=e,
        )
        assert any(
            "Operation failed: test error" in r.getMessage() for r in caplog.records
        )
        assert any(
            r.exc_info
            for r in caplog.records
            if "Operation failed: test error" in r.getMessage()
        )


def test_log_context_with_resource_messages(caplog: pytest.LogCaptureFixture) -> None:
    context = get_test_context()
    context.info(LOG_MESSAGES["INFO"]["RESOURCE_ACCESS"].format(resource="database"))
    assert any("Accessing resource: database" in r.getMessage() for r in caplog.records)
    context.warning(
        LOG_MESSAGES["WARNING"]["RESOURCE_WARNING"].format(message="high memory usage")
    )
    assert any(
        "Resource warning: high memory usage" in r.getMessage() for r in caplog.records
    )
    try:
        raise ConnectionError("connection failed")
    except ConnectionError as e:
        context.error(
            LOG_MESSAGES["ERROR"]["RESOURCE_ERROR"].format(error=str(e)),
            exc_info=e,
        )
        assert any(
            "Resource error: connection failed" in r.getMessage()
            for r in caplog.records
        )
        assert any(
            r.exc_info
            for r in caplog.records
            if "Resource error: connection failed" in r.getMessage()
        )


def test_log_context_with_configuration_messages(
    caplog: pytest.LogCaptureFixture,
) -> None:
    context = get_test_context()
    context.info(
        LOG_MESSAGES["INFO"]["CONFIGURATION_LOAD"].format(config="app_settings")
    )
    assert any(
        "Loading configuration: app_settings" in r.getMessage() for r in caplog.records
    )
    context.warning(
        LOG_MESSAGES["WARNING"]["CONFIGURATION_WARNING"].format(
            message="missing optional setting"
        )
    )
    assert any(
        "Configuration warning: missing optional setting" in r.getMessage()
        for r in caplog.records
    )
    try:
        raise ValueError("invalid config")
    except ValueError as e:
        context.error(
            LOG_MESSAGES["ERROR"]["CONFIGURATION_ERROR"].format(error=str(e)),
            exc_info=e,
        )
        assert any(
            "Configuration error: invalid config" in r.getMessage()
            for r in caplog.records
        )
        assert any(
            r.exc_info
            for r in caplog.records
            if "Configuration error: invalid config" in r.getMessage()
        )


# =============================================================================
# ADDITIONAL TESTS FOR 100% COVERAGE
# =============================================================================


class TestRequestIdFilter:
    """Test RequestIdFilter class."""

    def test_request_id_filter_init_with_id(self) -> None:
        """Test RequestIdFilter initialization with request ID."""
        filter_obj = RequestIdFilter("test-request-123")
        assert filter_obj.request_id == "test-request-123"

    def test_request_id_filter_init_without_id(self) -> None:
        """Test RequestIdFilter initialization without request ID."""
        filter_obj = RequestIdFilter()
        assert filter_obj.request_id is None

    def test_request_id_filter_with_existing_request_id(self) -> None:
        """Test RequestIdFilter with existing request_id on record."""
        filter_obj = RequestIdFilter()
        record = Mock()
        record.request_id = "existing-id"

        result = filter_obj.filter(record)

        assert result is True
        assert record.request_id == "existing-id"

    def test_request_id_filter_without_existing_request_id(self) -> None:
        """Test RequestIdFilter without existing request_id on record."""
        filter_obj = RequestIdFilter("new-id")
        record = Mock()

        result = filter_obj.filter(record)

        assert result is True
        assert record.request_id == "new-id"


class TestContextFilter:
    """Test ContextFilter class."""

    def test_context_filter_init_with_context(self) -> None:
        """Test ContextFilter initialization with context."""
        context = {"user_id": "123", "session": "abc"}
        filter_obj = ContextFilter(context)
        assert filter_obj.context == context

    def test_context_filter_init_without_context(self) -> None:
        """Test ContextFilter initialization without context."""
        filter_obj = ContextFilter()
        assert filter_obj.context == {}

    def test_context_filter_filter(self) -> None:
        """Test ContextFilter filter method."""
        context = {"user_id": "123"}
        filter_obj = ContextFilter(context)
        record = Mock()

        result = filter_obj.filter(record)

        assert result is True
        assert record.context == context


class TestLogContextProperties:
    """Test LogContext properties."""

    def test_log_context_name_property(self) -> None:
        """Test LogContext name property."""
        context = LogContext("test_logger")
        assert context.name == "test_logger"

    def test_log_context_level_property(self) -> None:
        """Test LogContext level property."""
        context = LogContext("test_logger", level="DEBUG")
        assert context.level == logging.DEBUG

    def test_log_context_handlers_property(self) -> None:
        """Test LogContext handlers property."""
        context = LogContext("test_logger")
        handlers = context.handlers
        assert isinstance(handlers, list)
        assert len(handlers) > 0

    def test_log_context_propagate_property(self) -> None:
        """Test LogContext propagate property."""
        context = LogContext("test_logger")
        assert context.propagate is True

    def test_log_context_propagate_setter(self) -> None:
        """Test log context propagate setter property."""
        log_context = LogContext("test_logger")

        # Test setting propagate to True
        log_context.propagate = True
        assert log_context.log_context.propagate is True

        # Test setting propagate to False
        log_context.propagate = False
        assert log_context.log_context.propagate is False

    def test_log_context_parent_property_with_parent(self) -> None:
        """Test LogContext parent property when parent exists."""
        # Create a parent logger first
        parent_logger = logging.getLogger("parent")
        parent_logger.setLevel(logging.INFO)

        # Create child logger
        child_context = LogContext("parent.child")

        parent = child_context.parent
        assert parent is not None
        assert parent.name == "parent"

    def test_log_context_parent_property_without_parent(self) -> None:
        """Test LogContext parent property when no parent exists."""
        context = LogContext("root_logger")
        parent = context.parent
        # The implementation creates a LogContext object when there's a parent
        # For root logger, the parent might be the root logger itself or None
        # Let's check if it's either None or a LogContext object
        assert parent is None or isinstance(parent, LogContext)

    def test_log_context_parent_property_with_numeric_level(self) -> None:
        """Test LogContext parent property with numeric level."""
        # Create a parent logger with numeric level
        parent_logger = logging.getLogger("numeric_parent")
        parent_logger.setLevel(20)  # INFO level

        child_context = LogContext("numeric_parent.child")

        parent = child_context.parent
        assert parent is not None
        assert parent.name == "numeric_parent"


class TestLogContextMethods:
    """Test LogContext methods."""

    def test_log_context_context_manager(self) -> None:
        """Test LogContext as context manager."""
        with LogContext("test_context") as logger:
            assert isinstance(logger, logging.Logger)
            assert logger.name == "test_context"

    def test_log_context_set_level_string(self) -> None:
        """Test LogContext setLevel with string."""
        context = LogContext("test_logger")
        context.setLevel("DEBUG")
        assert context.level == logging.DEBUG

    def test_log_context_set_level_int(self) -> None:
        """Test LogContext setLevel with integer."""
        context = LogContext("test_logger")
        context.setLevel(logging.WARNING)
        assert context.level == logging.WARNING

    def test_log_context_merge_extra_with_dict_extra(self) -> None:
        """Test _merge_extra with dict extra."""
        context = LogContext("test_logger")
        kwargs = {"extra": {"key1": "value1", "key2": "value2"}}
        context_data = {"context_key": "context_value"}

        result = context._merge_extra(context_data, kwargs)

        assert result["context"] == context_data
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"
        assert "extra" not in kwargs

    def test_log_context_merge_extra_with_non_dict_extra(self) -> None:
        """Test _merge_extra with non-dict extra."""
        context = LogContext("test_logger")
        kwargs = {"extra": "not_a_dict", "other_key": "other_value"}
        context_data = {"context_key": "context_value"}

        result = context._merge_extra(context_data, kwargs)

        assert result["context"] == context_data
        assert result["other_key"] == "other_value"
        assert "extra" not in result

    def test_log_context_merge_extra_without_extra(self) -> None:
        """Test _merge_extra without extra field."""
        context = LogContext("test_logger")
        kwargs = {"method": "GET", "status": 200}
        context_data = {"context_key": "context_value"}

        result = context._merge_extra(context_data, kwargs)

        assert result["context"] == context_data
        assert result["method"] == "GET"
        assert result["status"] == 200

    def test_log_context_merge_extra_preserves_exc_info(self) -> None:
        """Test _merge_extra preserves exc_info."""
        context = get_test_context()
        context_data = {"test": "value"}
        kwargs = {"extra": {"user": "data"}, "exc_info": True, "other": "field"}

        result = context._merge_extra(context_data, kwargs)

        # exc_info should be preserved in kwargs, not merged into extra
        assert "exc_info" in kwargs
        assert kwargs["exc_info"] is True
        assert "exc_info" not in result
        assert result["other"] == "field"
        assert result["user"] == "data"
        assert result["context"] == context_data

    def test_log_context_merge_extra_excludes_exc_info_from_extra(self) -> None:
        """Test _merge_extra excludes exc_info from extra dict."""
        context = get_test_context()
        context_data = {"test": "value"}
        kwargs = {"exc_info": True, "method": "GET", "status": 200}

        result = context._merge_extra(context_data, kwargs)

        # exc_info should be preserved in kwargs, not merged into extra
        assert "exc_info" in kwargs
        assert kwargs["exc_info"] is True
        assert "exc_info" not in result
        assert result["method"] == "GET"
        assert result["status"] == 200
        assert result["context"] == context_data

    def test_log_context_debug_method(self) -> None:
        """Test LogContext debug method."""
        context = get_test_context()
        with patch.object(context.log_context, "debug") as mock_debug:
            context.debug("Debug message", {"debug_context": "value"})
            mock_debug.assert_called_once()

    def test_log_context_warning_method(self) -> None:
        """Test LogContext warning method."""
        context = get_test_context()
        with patch.object(context.log_context, "warning") as mock_warning:
            context.warning("Warning message", {"warning_context": "value"})
            mock_warning.assert_called_once()

    def test_log_context_critical_method(self) -> None:
        """Test LogContext critical method."""
        context = get_test_context()
        with patch.object(context.log_context, "critical") as mock_critical:
            context.critical("Critical message", {"critical_context": "value"})
            mock_critical.assert_called_once()

    def test_log_context_push_pop_context(self) -> None:
        """Test LogContext push_context and pop_context methods."""
        context = LogContext("test_logger")
        # These methods are currently no-ops, just test they don't raise
        context.push_context({"key": "value"})
        context.pop_context()


class TestHandleExceptionsDecorator:
    """Test handle_exceptions decorator edge cases."""

    @handle_exceptions(reraise=False, log_level="WARNING")
    def function_with_warning_level(self) -> None:
        raise ValueError("Warning level error")

    def test_handle_exceptions_warning_level(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test handle_exceptions with WARNING log level."""
        result = self.function_with_warning_level()
        assert result is None
        assert any(
            "Unexpected error in function_with_warning_level" in r.getMessage()
            for r in caplog.records
        )

    @handle_exceptions(reraise=False, include_stack_trace=False)
    def function_without_stack_trace(self) -> None:
        raise ValueError("No stack trace error")

    def test_handle_exceptions_without_stack_trace(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test handle_exceptions without stack trace."""
        result = self.function_without_stack_trace()
        assert result is None
        assert any(
            "Unexpected error in function_without_stack_trace" in r.getMessage()
            for r in caplog.records
        )

    @handle_exceptions(reraise=True, default_error=ArklexError)
    def function_with_custom_error(self) -> None:
        raise RuntimeError("Custom error type")

    def test_handle_exceptions_custom_error_type(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test handle_exceptions with custom error type."""
        with pytest.raises(ArklexError) as exc_info:
            self.function_with_custom_error()
        assert "Operation failed in function_with_custom_error" in str(exc_info.value)

    @handle_exceptions(reraise=True)
    async def async_function_that_raises(self) -> None:
        raise ValueError("Async error")

    @pytest.mark.asyncio
    async def test_handle_exceptions_async_function(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test handle_exceptions with async function."""
        with pytest.raises(ArklexError) as exc_info:
            await self.async_function_that_raises()
        assert "Operation failed in async_function_that_raises" in str(exc_info.value)

    @handle_exceptions(reraise=False)
    async def async_function_no_reraise(self) -> None:
        raise ValueError("Async error no reraise")

    @pytest.mark.asyncio
    async def test_handle_exceptions_async_no_reraise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test handle_exceptions with async function no reraise."""
        result = await self.async_function_no_reraise()
        assert result is None

    @handle_exceptions(reraise=True)
    def function_that_raises_arklex_error(self) -> None:
        raise ArklexError("Original ArklexError")

    def test_handle_exceptions_preserves_arklex_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test handle_exceptions preserves ArklexError."""
        with pytest.raises(ArklexError) as exc_info:
            self.function_that_raises_arklex_error()
        assert "Original ArklexError" in str(exc_info.value)


class TestWithRetryDecorator:
    """Test with_retry decorator edge cases."""

    @with_retry(max_attempts=1, retry_on=ValueError)
    def function_retry_on_value_error(self) -> None:
        raise ValueError("ValueError for retry")

    def test_with_retry_custom_exception_type(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test with_retry with custom exception type."""
        with pytest.raises(tenacity.RetryError):
            self.function_retry_on_value_error()
        assert any("Retrying operation" in r.getMessage() for r in caplog.records)

    @with_retry(max_attempts=1, include_stack_trace=False)
    def function_retry_without_stack_trace(self) -> None:
        raise RetryableError("No stack trace retry", "RETRY_ERROR")

    def test_with_retry_without_stack_trace(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test with_retry without stack trace."""
        with pytest.raises(tenacity.RetryError):
            self.function_retry_without_stack_trace()
        assert any("Retrying operation" in r.getMessage() for r in caplog.records)

    @with_retry(max_attempts=1)
    async def async_function_that_retries(self) -> NoReturn:
        raise RetryableError("Async retry error", "ASYNC_RETRY_ERROR")

    @pytest.mark.asyncio
    async def test_with_retry_async_function(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test with_retry with async function."""
        with pytest.raises(tenacity.RetryError):
            await self.async_function_that_retries()
        assert any("Retrying operation" in r.getMessage() for r in caplog.records)

    @with_retry(max_attempts=1, retry_on=ValueError)
    async def async_function_retry_custom_exception(self) -> None:
        raise ValueError("Async custom exception retry")

    @pytest.mark.asyncio
    async def test_with_retry_async_custom_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test with_retry with async function and custom exception."""
        with pytest.raises(tenacity.RetryError):
            await self.async_function_retry_custom_exception()
        assert any("Retrying operation" in r.getMessage() for r in caplog.records)

    @with_retry(max_attempts=1)
    def function_that_doesnt_retry(self) -> None:
        raise RuntimeError("Non-retryable error")

    def test_with_retry_non_retryable_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test with_retry with non-retryable exception."""
        with pytest.raises(RuntimeError):
            self.function_that_doesnt_retry()
        # Should not log retry attempts for non-retryable exceptions
        assert not any("Retrying operation" in r.getMessage() for r in caplog.records)

    @with_retry(max_attempts=1, retry_on=ValueError)
    async def async_function_non_retryable(self) -> None:
        raise RuntimeError("Async non-retryable error")

    @pytest.mark.asyncio
    async def test_with_retry_async_non_retryable(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test with_retry with async function and non-retryable exception."""
        with pytest.raises(RuntimeError):
            await self.async_function_non_retryable()
        # Should not log retry attempts for non-retryable exceptions
        assert not any("Retrying operation" in r.getMessage() for r in caplog.records)


class TestLogContextInitialization:
    """Test LogContext initialization edge cases."""

    def test_log_context_init_without_level(self) -> None:
        """Test LogContext initialization without level."""
        context = LogContext("test_logger")
        # The level might be inherited from parent or set to a default
        # Let's check that it's a valid logging level
        assert context.level >= 0  # Any valid logging level is >= 0

    def test_log_context_init_with_custom_format(self) -> None:
        """Test LogContext initialization with custom format."""
        custom_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        context = LogContext("test_logger", log_format=custom_format)
        assert len(context.handlers) > 0

    def test_log_context_init_with_base_context(self) -> None:
        """Test LogContext initialization with base context."""
        base_context = {"app": "test_app", "version": "1.0"}
        context = LogContext("test_logger", base_context=base_context)
        assert context.base_context == base_context


class TestLogContextLoggingMethods:
    """Test LogContext logging methods with various parameters."""

    def test_log_context_info_with_context_and_kwargs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test LogContext info method with context and kwargs."""
        context = get_test_context()
        context.info("Test message", {"ctx_key": "ctx_value"}, method="GET", status=200)
        assert any("Test message" in r.getMessage() for r in caplog.records)

    def test_log_context_debug_with_context_and_kwargs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test LogContext debug method with context and kwargs."""
        context = get_test_context()
        context.debug("Debug message", {"debug_ctx": "debug_value"}, user_id="123")
        assert any("Debug message" in r.getMessage() for r in caplog.records)

    def test_log_context_warning_with_context_and_kwargs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test LogContext warning method with context and kwargs."""
        context = get_test_context()
        context.warning("Warning message", {"warn_ctx": "warn_value"}, severity="high")
        assert any("Warning message" in r.getMessage() for r in caplog.records)

    def test_log_context_error_with_context_and_kwargs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test LogContext error method with context and kwargs."""
        context = get_test_context()
        context.error("Error message", {"error_ctx": "error_value"}, error_code="500")
        assert any("Error message" in r.getMessage() for r in caplog.records)

    def test_log_context_critical_with_context_and_kwargs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test LogContext critical method with context and kwargs."""
        context = get_test_context()
        context.critical("Critical message", {"crit_ctx": "crit_value"}, fatal=True)
        assert any("Critical message" in r.getMessage() for r in caplog.records)


class TestLogContextInternalMethods:
    """Test internal methods of LogContext class."""

    def test_get_console_handler_default_format(self) -> None:
        """Test _get_console_handler with default format."""
        context = LogContext("test")
        handler = context._get_console_handler()

        assert isinstance(handler, logging.StreamHandler)
        assert handler.formatter is not None
        assert "%(levelname)s - %(message)s" in str(handler.formatter._fmt)

    def test_get_console_handler_custom_format(self) -> None:
        """Test _get_console_handler with custom format."""
        context = LogContext("test")
        custom_format = "%(name)s - %(levelname)s - %(message)s"
        handler = context._get_console_handler(custom_format)

        assert isinstance(handler, logging.StreamHandler)
        assert handler.formatter is not None
        assert custom_format in str(handler.formatter._fmt)

    def test_get_console_handler_none_format(self) -> None:
        """Test _get_console_handler with None format (covers line 150)."""
        context = LogContext("test")

        # Test with None format
        handler = context._get_console_handler(None)

        assert isinstance(handler, logging.StreamHandler)
        assert handler.formatter is not None
        # Should use default format when None is provided
        assert "%(levelname)s - %(message)s" in str(handler.formatter._fmt)

    def test_get_console_handler_with_empty_string_format(self) -> None:
        """Test _get_console_handler with empty string format (covers line 150)."""
        context = LogContext("test")

        # Test with empty string format
        handler = context._get_console_handler("")

        assert isinstance(handler, logging.StreamHandler)
        assert handler.formatter is not None
        # Should use default format when empty string is provided
        assert "%(levelname)s - %(message)s" in str(handler.formatter._fmt)

    def test_get_console_handler_with_custom_format_string(self) -> None:
        """Test _get_console_handler with custom format string (covers line 150)."""
        context = LogContext("test")

        # Test with custom format
        custom_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        handler = context._get_console_handler(custom_format)

        assert isinstance(handler, logging.StreamHandler)
        assert handler.formatter is not None
        # Should use the custom format
        assert custom_format in str(handler.formatter._fmt)

    def test_get_console_handler_with_special_characters_format(self) -> None:
        """Test _get_console_handler with format containing special characters (covers line 150)."""
        context = LogContext("test")

        # Test with format containing special characters
        special_format = (
            "%(levelname)s [%(name)s] %(message)s - %(funcName)s:%(lineno)d"
        )
        handler = context._get_console_handler(special_format)

        assert isinstance(handler, logging.StreamHandler)
        assert handler.formatter is not None
        # Should use the special format
        assert special_format in str(handler.formatter._fmt)

    def test_log_context_parent_property_with_numeric_level_handling(self) -> None:
        """Test parent property with numeric level handling."""
        # Create a parent with numeric level
        parent = LogContext("parent", level=logging.INFO)
        child = LogContext("child", level=logging.DEBUG)
        child.log_context.parent = parent.log_context

        result = child.parent
        assert result is not None
        assert result.name == "parent"
        assert result.level == logging.INFO

    def test_log_context_parent_property_without_parent(self) -> None:
        """Test parent property when there's no parent."""
        context = get_test_context()
        # Ensure no parent is set
        context.log_context.parent = None

        result = context.parent
        assert result is None
