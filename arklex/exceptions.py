"""Exception definitions for the Arklex framework.

This module defines custom exceptions used throughout the Arklex framework,
including authentication errors, tool execution errors, and user-facing errors.
It provides a base class for tool-specific exception prompts and implements
error handling mechanisms for various system components.
"""

__all__ = [
    "AuthenticationError",
    "ToolExecutionError",
    "ExceptionPrompt",
]


class AuthenticationError(Exception):
    """
    Exception raised when authentication fails.
    """

    def __init__(self, message: str):
        self.message = f"Authentication failed: {message}"
        super().__init__(self.message)


class UserFacingError(Exception):
    """
    Exception raised to guide the user to update their query.
    """

    def __init__(self, message: str, extra_message: str):
        super().__init__(message)
        # Store the additional message in a custom attribute, which will be used to guide the user to update their query.
        self.extra_message = extra_message


class ToolExecutionError(UserFacingError):
    """
    Exception raised when a tool execution fails.
    """

    def __init__(self, message: str, extra_message: str):
        self.message = f"Tool {message} execution failed"
        super().__init__(self.message, extra_message)


class ExceptionPrompt:
    """
    Base class for tool-specific exception prompts.

    This class serves as a parent class for tool collections (like Shopify, HubSpot)
    to define their own exception prompts as class attributes.

    Example:
        class ShopifyExceptionPrompt(ExceptionPrompt):
            ORDER_NOT_FOUND = "Order could not be found."
            PRODUCT_NOT_AVAILABLE = "Product is not available."

    Each tool collection should create their own _exception_prompt.py file
    that inherits from this base class.
    """

    # Common exception prompts shared across tool collections
    pass
