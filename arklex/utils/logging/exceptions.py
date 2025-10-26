"""Exception definitions for the Arklex framework.

This module defines custom exceptions used throughout the Arklex framework,
including authentication errors, validation errors, and user-facing errors.
It provides a base class for all exceptions and implements specific error types
for different scenarios.
"""

import copy
from types import MappingProxyType
from typing import Any


class ExceptionPrompt:
    """Base class for exception prompts."""

    pass


class ArklexError(Exception):
    """Base exception class for all Arklex errors.

    Attributes:
        message: A human-readable error message
        code: An error code for programmatic handling
        status_code: HTTP status code for API responses
        details: Additional error details
    """

    def __init__(
        self,
        message: str,
        code: str | None = None,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: A human-readable error message
            code: An error code for programmatic handling
            status_code: HTTP status code for API responses
            details: Additional error details
        """
        self.message = message
        self.code = code
        self.status_code = status_code
        self._original_details = copy.deepcopy(details) if details is not None else None
        if details is not None:
            self.details = MappingProxyType(copy.deepcopy(details))
        else:
            self.details = None
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return a string representation of the error.

        Returns:
            str: The error message with error code in parentheses
        """
        bracket_codes = [
            "API_ERROR",
            "MODEL_ERROR",
            "CONFIG_ERROR",
            "DB_ERROR",
            "NOT_FOUND",
            "RATE_LIMIT",
        ]
        if not self.message:
            return " (UNKNOWN_ERROR)"
        if self.code in bracket_codes:
            return f"[{self.code}] {self.message}"
        if self.code:
            return f"{self.message} ({self.code})"
        return f"{self.message} (UNKNOWN_ERROR)"

    @property
    def error_code(self) -> str:
        """Get the error code.

        Returns:
            str: The error code
        """
        return self.code or "UNKNOWN_ERROR"


class AuthenticationError(ArklexError):
    """Exception raised for authentication errors."""

    def __init__(
        self,
        message: str = "Authentication failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: A human-readable error message
            details: Additional error details
        """
        super().__init__(
            message=message,
            code="AUTHENTICATION_ERROR",
            status_code=401,
            details=details,
        )


class ValidationError(ArklexError):
    """Exception raised for validation errors."""

    def __init__(
        self,
        message: str = "Validation failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: A human-readable error message
            details: Additional error details
        """
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=400,
            details=details,
        )


class APIError(ArklexError):
    """Raised when API calls fail.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the APIError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "API_ERROR", 500, details)


class ModelError(ArklexError):
    """Raised when model operations fail.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the ModelError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "MODEL_ERROR", 500, details)


class PlannerError(ArklexError):
    """Raised when planning operations fail.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the PlannerError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "PLANNER_ERROR", 500, details)


class ConfigurationError(ArklexError):
    """Raised when there are configuration issues.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the ConfigurationError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "CONFIG_ERROR", 500, details)


class DatabaseError(ArklexError):
    """Raised when database operations fail.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the DatabaseError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "DB_ERROR", 500, details)


class ResourceNotFoundError(ArklexError):
    """Raised when a requested resource is not found.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the ResourceNotFoundError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "NOT_FOUND", 404, details)


class RateLimitError(ArklexError):
    """Raised when rate limits are exceeded.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the RateLimitError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "RATE_LIMIT", 429, details)


class ToolExecutionError(ArklexError):
    """Raised when tool execution fails.

    Args:
        tool_name: Name of the tool that failed.
        message: The error message.
        details: Optional dictionary with additional error details.
        extra_message: Optional additional message for user guidance.
    """

    def __init__(
        self,
        tool_name: str,
        message: str | None = None,
        details: dict[str, Any] | None = None,
        extra_message: str | None = None,
    ) -> None:
        """Initialize the ToolExecutionError.

        Args:
            tool_name: Name of the tool that failed.
            message: The error message.
            details: Optional dictionary with additional error details.
            extra_message: Optional additional message for user guidance.
        """
        super().__init__(
            f"Tool {tool_name} execution failed: {message}",
            "TOOL_ERROR",
            500,
            details,
        )
        self.extra_message = extra_message


class UserFacingError(ArklexError):
    """Base class for user-facing errors with additional guidance.

    Args:
        message: The error message.
        error_code: Error code for categorization.
        details: Optional dictionary with additional error details.
        extra_message: Optional additional message for user guidance.
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        details: dict[str, Any] | None = None,
        extra_message: str | None = None,
    ) -> None:
        """Initialize the UserFacingError.

        Args:
            message: The error message.
            error_code: Error code for categorization.
            details: Optional dictionary with additional error details.
            extra_message: Optional additional message for user guidance.
        """
        super().__init__(message, error_code, 500, details)


class RetryableError(ArklexError):
    """Base class for errors that can be retried.

    Args:
        message: The error message.
        error_code: Error code for categorization.
        details: Optional dictionary with additional error details.
        max_retries: Maximum number of retry attempts.
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        details: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> None:
        """Initialize the RetryableError.

        Args:
            message: The error message.
            error_code: Error code for categorization.
            details: Optional dictionary with additional error details.
            max_retries: Maximum number of retry attempts.
        """
        super().__init__(message, error_code, 500, details)
        self.max_retries = max_retries


class NetworkError(RetryableError):
    """Raised when network operations fail.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the NetworkError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "NETWORK_ERROR", details)


class TimeoutError(RetryableError):
    """Raised when operations timeout.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the TimeoutError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "TIMEOUT_ERROR", details)


class ServiceUnavailableError(RetryableError):
    """Raised when a service is temporarily unavailable.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the ServiceUnavailableError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "SERVICE_UNAVAILABLE", details)


class EnvironmentError(ArklexError):
    """Raised when environment-related operations fail.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the EnvironmentError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "ENVIRONMENT_ERROR", 500, details)


class TaskGraphError(ArklexError):
    """Raised when task graph operations fail.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the TaskGraphError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "TASK_GRAPH_ERROR", 500, details)


class ToolError(ArklexError):
    """Raised when a general tool error occurs.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the ToolError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "TOOL_ERROR", 500, details)


class OrchestratorError(ArklexError):
    """Raised when orchestrator operations fail.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the OrchestratorError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "ORCHESTRATOR_ERROR", 500, details)


class SearchError(ArklexError):
    """Raised when there is an error with search operations."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the SearchError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "SEARCH_ERROR", 500, details)


class ShopifyError(ArklexError):
    """Raised when Shopify operations fail."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the ShopifyError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "SHOPIFY_ERROR", 500, details)
