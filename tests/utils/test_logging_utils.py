"""Tests for logging utilities."""

import pytest
import logging
import string
from arklex.utils.logging_utils import (
    LogContext,
    handle_exceptions,
    with_retry,
    LOG_MESSAGES,
)
from arklex.utils.exceptions import ArklexError, RetryableError
import tenacity


# Helper to ensure log_context propagates and is at correct level for caplog
def get_test_context(name="test", base_context=None):
    log_context = LogContext(name, level="INFO", base_context=base_context)
    log_context.setLevel(logging.DEBUG)
    log_context.propagate = True
    return log_context


def test_log_context_basic(caplog):
    context = get_test_context()
    context.info("Test message")
    assert any("Test message" in r.getMessage() for r in caplog.records)


def test_log_context_with_data(caplog):
    context = get_test_context(base_context={"base": "value"})
    context.info("Test message", extra="data")
    assert any("Test message" in r.getMessage() for r in caplog.records)
    # Context is not in the plain message, but we can check the log_context name
    assert any(r.name == "test" for r in caplog.records)


def test_log_context_error(caplog):
    context = get_test_context()
    try:
        raise ValueError("Test error")
    except ValueError as e:
        context.error("Error occurred", exc_info=e)
        assert any("Error occurred" in r.getMessage() for r in caplog.records)
        # Exception type is not in the plain message, but should be in exc_info
        assert any(
            r.exc_info for r in caplog.records if "Error occurred" in r.getMessage()
        )


def test_log_context_stack(caplog):
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
def function_that_raises():
    raise ValueError("Test error")


def test_handle_exceptions(caplog):
    result = function_that_raises()
    assert result is None
    assert any(
        "Unexpected error in function_that_raises" in r.getMessage()
        for r in caplog.records
    )
    assert any(
        r.exc_info
        for r in caplog.records
        if "Unexpected error in function_that_raises" in r.getMessage()
    )


@handle_exceptions(reraise=True)
def function_that_raises_and_reraises():
    raise ValueError("Test error")


def test_handle_exceptions_reraises(caplog):
    with pytest.raises(ArklexError) as exc_info:
        function_that_raises_and_reraises()
    assert "Operation failed in function_that_raises_and_reraises" in str(
        exc_info.value
    )
    assert "UNKNOWN_ERROR" in str(exc_info.value)
    assert any(
        "Unexpected error in function_that_raises_and_reraises" in r.getMessage()
        for r in caplog.records
    )
    assert any(
        r.exc_info
        for r in caplog.records
        if "Unexpected error in function_that_raises_and_reraises" in r.getMessage()
    )


@with_retry(max_attempts=2)
def function_that_retries():
    raise RetryableError("Test retry error", error_code="RETRY")


def test_with_retry(caplog):
    with pytest.raises(tenacity.RetryError) as exc_info:
        function_that_retries()
    # Check that the last exception is a RetryableError
    assert isinstance(exc_info.value.last_attempt.exception(), RetryableError)
    assert any("Retrying operation" in r.getMessage() for r in caplog.records)


def test_log_messages_consistency():
    for level, messages in LOG_MESSAGES.items():
        for key, message in messages.items():
            placeholders = [
                p[1] for p in string.Formatter().parse(message) if p[1] is not None
            ]
            assert all(isinstance(p, str) for p in placeholders), (
                f"Invalid placeholder in {level}.{key}"
            )


def test_log_context_with_standard_messages(caplog):
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


def test_log_context_with_resource_messages(caplog):
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


def test_log_context_with_configuration_messages(caplog):
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
