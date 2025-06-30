"""Tests for the custom exceptions module."""

from typing import Any

import pytest

from arklex.utils.exceptions import (
    APIError,
    ArklexError,
    AuthenticationError,
    ConfigurationError,
    DatabaseError,
    EnvironmentError,
    ModelError,
    NetworkError,
    OrchestratorError,
    PlannerError,
    RateLimitError,
    ResourceNotFoundError,
    RetryableError,
    SearchError,
    ServiceUnavailableError,
    TaskGraphError,
    TimeoutError,
    ToolError,
    ToolExecutionError,
    UserFacingError,
    ValidationError,
)


def test_arklex_error_creation() -> None:
    """Test creation of ArklexError."""
    error = ArklexError("Test error")
    assert str(error) == "Test error (UNKNOWN_ERROR)"
    assert error.code is None
    assert error.status_code == 500
    assert error.details is None


def test_arklex_error_with_code() -> None:
    """Test creation of ArklexError with error code."""
    error = ArklexError("Test error", code="TEST_ERROR")
    assert str(error) == "Test error (TEST_ERROR)"
    assert error.code == "TEST_ERROR"
    assert error.status_code == 500
    assert error.details is None


def test_arklex_error_with_details() -> None:
    """Test creation of ArklexError with details."""
    details: dict[str, Any] = {"field": "value"}
    error = ArklexError("Test error", details=details)
    assert str(error) == "Test error (UNKNOWN_ERROR)"
    assert error.code is None
    assert error.status_code == 500
    assert error.details == details


def test_arklex_error_empty_message() -> None:
    """Test ArklexError with empty message to cover line 182."""
    error = ArklexError("")
    assert str(error) == " (UNKNOWN_ERROR)"
    assert error.code is None
    assert error.status_code == 500


def test_authentication_error() -> None:
    """Test creation of AuthenticationError."""
    error = AuthenticationError("Invalid credentials")
    assert str(error) == "Invalid credentials (AUTHENTICATION_ERROR)"
    assert error.code == "AUTHENTICATION_ERROR"
    assert error.status_code == 401
    assert error.details is None


def test_validation_error() -> None:
    """Test creation of ValidationError."""
    error = ValidationError("Invalid input")
    assert str(error) == "Invalid input (VALIDATION_ERROR)"
    assert error.code == "VALIDATION_ERROR"
    assert error.status_code == 400
    assert error.details is None


def test_validation_error_with_details() -> None:
    """Test creation of ValidationError with details."""
    details: dict[str, Any] = {"field": "value"}
    error = ValidationError("Invalid input", details=details)
    assert str(error) == "Invalid input (VALIDATION_ERROR)"
    assert error.code == "VALIDATION_ERROR"
    assert error.status_code == 400
    assert error.details == details


def test_api_error() -> None:
    """Test APIError."""
    error = APIError("API call failed")
    assert str(error) == "[API_ERROR] API call failed"
    assert error.error_code == "API_ERROR"


def test_model_error() -> None:
    """Test ModelError."""
    error = ModelError("Model failed")
    assert str(error) == "[MODEL_ERROR] Model failed"
    assert error.error_code == "MODEL_ERROR"


def test_configuration_error() -> None:
    """Test ConfigurationError."""
    error = ConfigurationError("Config invalid")
    assert str(error) == "[CONFIG_ERROR] Config invalid"
    assert error.error_code == "CONFIG_ERROR"


def test_database_error() -> None:
    """Test DatabaseError."""
    error = DatabaseError("DB operation failed")
    assert str(error) == "[DB_ERROR] DB operation failed"
    assert error.error_code == "DB_ERROR"


def test_resource_not_found_error() -> None:
    """Test ResourceNotFoundError."""
    error = ResourceNotFoundError("Resource not found")
    assert str(error) == "[NOT_FOUND] Resource not found"
    assert error.error_code == "NOT_FOUND"


def test_rate_limit_error() -> None:
    """Test RateLimitError."""
    error = RateLimitError("Rate limit exceeded")
    assert str(error) == "[RATE_LIMIT] Rate limit exceeded"
    assert error.error_code == "RATE_LIMIT"


def test_planner_error() -> None:
    """Test PlannerError."""
    error = PlannerError("Planning failed")
    assert str(error) == "Planning failed (PLANNER_ERROR)"
    assert error.error_code == "PLANNER_ERROR"


def test_tool_execution_error() -> None:
    """Test ToolExecutionError."""
    error = ToolExecutionError("test_tool", "Tool execution failed")
    assert (
        str(error)
        == "Tool test_tool execution failed: Tool execution failed (TOOL_ERROR)"
    )
    assert error.error_code == "TOOL_ERROR"
    assert error.extra_message is None


def test_tool_execution_error_with_extra_message() -> None:
    """Test ToolExecutionError with extra message."""
    error = ToolExecutionError(
        "test_tool", "Tool execution failed", extra_message="Try again"
    )
    assert (
        str(error)
        == "Tool test_tool execution failed: Tool execution failed (TOOL_ERROR)"
    )
    assert error.error_code == "TOOL_ERROR"
    assert error.extra_message == "Try again"


def test_tool_execution_error_with_details() -> None:
    """Test ToolExecutionError with details."""
    details = {"tool_id": "123", "status": "failed"}
    error = ToolExecutionError("test_tool", "Tool execution failed", details=details)
    assert (
        str(error)
        == "Tool test_tool execution failed: Tool execution failed (TOOL_ERROR)"
    )
    assert error.error_code == "TOOL_ERROR"
    assert error.details == details


def test_user_facing_error() -> None:
    """Test UserFacingError."""
    error = UserFacingError("User friendly error", "USER_ERROR")
    assert str(error) == "User friendly error (USER_ERROR)"
    assert error.error_code == "USER_ERROR"


def test_user_facing_error_with_details() -> None:
    """Test UserFacingError with details."""
    details = {"user_id": "123", "action": "login"}
    error = UserFacingError("User friendly error", "USER_ERROR", details=details)
    assert str(error) == "User friendly error (USER_ERROR)"
    assert error.error_code == "USER_ERROR"
    assert error.details == details


def test_retryable_error() -> None:
    """Test RetryableError."""
    error = RetryableError("Retryable error", "RETRY_ERROR")
    assert str(error) == "Retryable error (RETRY_ERROR)"
    assert error.error_code == "RETRY_ERROR"
    assert error.max_retries == 3


def test_retryable_error_with_custom_retries() -> None:
    """Test RetryableError with custom max_retries."""
    error = RetryableError("Retryable error", "RETRY_ERROR", max_retries=5)
    assert str(error) == "Retryable error (RETRY_ERROR)"
    assert error.error_code == "RETRY_ERROR"
    assert error.max_retries == 5


def test_retryable_error_with_details() -> None:
    """Test RetryableError with details."""
    details = {"attempt": 1, "max_attempts": 3}
    error = RetryableError("Retryable error", "RETRY_ERROR", details=details)
    assert str(error) == "Retryable error (RETRY_ERROR)"
    assert error.error_code == "RETRY_ERROR"
    assert error.details == details


def test_network_error() -> None:
    """Test NetworkError."""
    error = NetworkError("Network connection failed")
    assert str(error) == "Network connection failed (NETWORK_ERROR)"
    assert error.error_code == "NETWORK_ERROR"
    assert error.max_retries == 3


def test_network_error_with_details() -> None:
    """Test NetworkError with details."""
    details = {"host": "example.com", "port": 443}
    error = NetworkError("Network connection failed", details=details)
    assert str(error) == "Network connection failed (NETWORK_ERROR)"
    assert error.error_code == "NETWORK_ERROR"
    assert error.details == details


def test_timeout_error() -> None:
    """Test TimeoutError."""
    error = TimeoutError("Operation timed out")
    assert str(error) == "Operation timed out (TIMEOUT_ERROR)"
    assert error.error_code == "TIMEOUT_ERROR"
    assert error.max_retries == 3


def test_timeout_error_with_details() -> None:
    """Test TimeoutError with details."""
    details = {"timeout": 30, "operation": "api_call"}
    error = TimeoutError("Operation timed out", details=details)
    assert str(error) == "Operation timed out (TIMEOUT_ERROR)"
    assert error.error_code == "TIMEOUT_ERROR"
    assert error.details == details


def test_service_unavailable_error() -> None:
    """Test ServiceUnavailableError."""
    error = ServiceUnavailableError("Service temporarily unavailable")
    assert str(error) == "Service temporarily unavailable (SERVICE_UNAVAILABLE)"
    assert error.error_code == "SERVICE_UNAVAILABLE"
    assert error.max_retries == 3


def test_service_unavailable_error_with_details() -> None:
    """Test ServiceUnavailableError with details."""
    details = {"service": "api", "retry_after": 60}
    error = ServiceUnavailableError("Service temporarily unavailable", details=details)
    assert str(error) == "Service temporarily unavailable (SERVICE_UNAVAILABLE)"
    assert error.error_code == "SERVICE_UNAVAILABLE"
    assert error.details == details


def test_environment_error() -> None:
    """Test EnvironmentError."""
    error = EnvironmentError("Environment configuration failed")
    assert str(error) == "Environment configuration failed (ENVIRONMENT_ERROR)"
    assert error.error_code == "ENVIRONMENT_ERROR"


def test_environment_error_with_details() -> None:
    """Test EnvironmentError with details."""
    details = {"env_var": "API_KEY", "status": "missing"}
    error = EnvironmentError("Environment configuration failed", details=details)
    assert str(error) == "Environment configuration failed (ENVIRONMENT_ERROR)"
    assert error.error_code == "ENVIRONMENT_ERROR"
    assert error.details == details


def test_task_graph_error() -> None:
    """Test TaskGraphError."""
    error = TaskGraphError("Task graph operation failed")
    assert str(error) == "Task graph operation failed (TASK_GRAPH_ERROR)"
    assert error.error_code == "TASK_GRAPH_ERROR"


def test_task_graph_error_with_details() -> None:
    """Test TaskGraphError with details."""
    details = {"graph_id": "123", "operation": "create"}
    error = TaskGraphError("Task graph operation failed", details=details)
    assert str(error) == "Task graph operation failed (TASK_GRAPH_ERROR)"
    assert error.error_code == "TASK_GRAPH_ERROR"
    assert error.details == details


def test_tool_error() -> None:
    """Test ToolError."""
    error = ToolError("General tool error")
    assert str(error) == "General tool error (TOOL_ERROR)"
    assert error.error_code == "TOOL_ERROR"


def test_tool_error_with_details() -> None:
    """Test ToolError with details."""
    details = {"tool_name": "calculator", "operation": "divide"}
    error = ToolError("General tool error", details=details)
    assert str(error) == "General tool error (TOOL_ERROR)"
    assert error.error_code == "TOOL_ERROR"
    assert error.details == details


def test_orchestrator_error() -> None:
    """Test OrchestratorError."""
    error = OrchestratorError("Orchestrator operation failed")
    assert str(error) == "Orchestrator operation failed (ORCHESTRATOR_ERROR)"
    assert error.error_code == "ORCHESTRATOR_ERROR"


def test_orchestrator_error_with_details() -> None:
    """Test OrchestratorError with details."""
    details = {"orchestrator_id": "456", "operation": "execute"}
    error = OrchestratorError("Orchestrator operation failed", details=details)
    assert str(error) == "Orchestrator operation failed (ORCHESTRATOR_ERROR)"
    assert error.error_code == "ORCHESTRATOR_ERROR"
    assert error.details == details


def test_search_error() -> None:
    """Test SearchError."""
    error = SearchError("Search operation failed")
    assert str(error) == "Search operation failed (SEARCH_ERROR)"
    assert error.error_code == "SEARCH_ERROR"


def test_search_error_with_details() -> None:
    """Test SearchError with details."""
    details = {"query": "test", "index": "documents"}
    error = SearchError("Search operation failed", details=details)
    assert str(error) == "Search operation failed (SEARCH_ERROR)"
    assert error.error_code == "SEARCH_ERROR"
    assert error.details == details


def test_error_inheritance() -> None:
    """Test that custom errors inherit from ArklexError."""
    auth_error = AuthenticationError("Auth error")
    validation_error = ValidationError("Validation error")

    assert isinstance(auth_error, ArklexError)
    assert isinstance(validation_error, ArklexError)


def test_error_details_immutability() -> None:
    """Test that error details are immutable."""
    details = {"field": "test"}
    error = ArklexError("Test error", details=details)

    # Attempt to modify details should raise TypeError
    with pytest.raises(TypeError):
        error.details["field"] = "modified"

    # Original details should remain unchanged
    assert error.details["field"] == "test"


def test_error_message_formatting() -> None:
    """Test error message formatting with different inputs."""
    # Test with empty message
    error = ArklexError("")
    assert str(error) == " (UNKNOWN_ERROR)"

    # Test with message only
    error = ArklexError("Simple error")
    assert str(error) == "Simple error (UNKNOWN_ERROR)"

    # Test with message and code
    error = ArklexError("Complex error", code="COMPLEX_ERROR")
    assert str(error) == "Complex error (COMPLEX_ERROR)"
