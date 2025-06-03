"""Exception definitions for the Arklex framework.

This module defines custom exceptions used throughout the Arklex framework,
including authentication errors, tool execution errors, and user-facing errors.
It provides a base class for tool-specific exception prompts and implements
error handling mechanisms for various system components.

Key Components:
1. AuthenticationError: Handles authentication and authorization failures
2. UserFacingError: Base class for user-visible error messages
3. ToolExecutionError: Specific errors during tool execution
4. ExceptionPrompt: Base class for tool-specific error messages

Usage:
    try:
        # Tool execution code
        raise ToolExecutionError("search", "Invalid search parameters")
    except ToolExecutionError as e:
        print(f"Error: {e}")
        print(f"Guidance: {e.extra_message}")
"""

__all__ = [
    "AuthenticationError",
    "ToolExecutionError",
    "ExceptionPrompt",
]


class AuthenticationError(Exception):
    """Exception raised when authentication fails.

    This exception is used to handle authentication and authorization failures
    throughout the system. It provides a standardized way to report authentication
    issues with clear error messages.

    Attributes:
        message (str): The error message describing the authentication failure.

    Example:
        raise AuthenticationError("Invalid API key")
    """

    def __init__(self, message: str):
        self.message = f"Authentication failed: {message}"
        super().__init__(self.message)


class UserFacingError(Exception):
    """Exception raised to guide the user to update their query.

    This is a base class for exceptions that should be shown to the user,
    providing both an error message and additional guidance on how to
    resolve the issue.

    Attributes:
        message (str): The main error message.
        extra_message (str): Additional guidance for the user.

    Example:
        raise UserFacingError("Invalid input", "Please provide a valid email address")
    """

    def __init__(self, message: str, extra_message: str):
        super().__init__(message)
        # Store the additional message in a custom attribute, which will be used to guide the user to update their query.
        self.extra_message = extra_message


class ToolExecutionError(UserFacingError):
    """Exception raised when a tool execution fails.

    This exception is used to handle failures during tool execution,
    providing both the error details and guidance on how to fix the issue.

    Attributes:
        message (str): The error message describing the tool execution failure.
        extra_message (str): Additional guidance for resolving the error.

    Example:
        raise ToolExecutionError("search", "Please provide valid search parameters")
    """

    def __init__(self, message: str, extra_message: str):
        self.message = f"Tool {message} execution failed"
        super().__init__(self.message, extra_message)


class ExceptionPrompt:
    """Base class for tool-specific exception prompts.

    This class serves as a parent class for tool collections (like Shopify, HubSpot)
    to define their own exception prompts as class attributes. It provides a standardized
    way to define and access error messages across different tool collections.

    Key Features:
    - Centralized error message management
    - Tool-specific error definitions
    - Consistent error message format
    - Easy extension for new tool collections

    Example:
        class ShopifyExceptionPrompt(ExceptionPrompt):
            ORDER_NOT_FOUND = "Order could not be found."
            PRODUCT_NOT_AVAILABLE = "Product is not available."

    Each tool collection should create their own _exception_prompt.py file
    that inherits from this base class to define tool-specific error messages.
    """

    # Common exception prompts shared across tool collections
    pass
