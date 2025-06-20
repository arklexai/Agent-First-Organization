"""Tests for the custom exceptions module."""

import pytest
from typing import Dict, Any
from arklex.utils.exceptions import (
    ArklexError,
    AuthenticationError,
    ValidationError,
    APIError,
    ModelError,
    ConfigurationError,
    DatabaseError,
    ResourceNotFoundError,
    RateLimitError,
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
    details: Dict[str, Any] = {"field": "value"}
    error = ArklexError("Test error", details=details)
    assert str(error) == "Test error (UNKNOWN_ERROR)"
    assert error.code is None
    assert error.status_code == 500
    assert error.details == details


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
    details: Dict[str, Any] = {"field": "value"}
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
