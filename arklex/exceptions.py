from typing import Final

__all__: Final[list[str]] = [
    "AuthenticationError",
    "ToolExecutionError",
    "ExceptionPrompt",
]


class AuthenticationError(Exception):
    """
    Exception raised when authentication fails.

    Attributes:
        message (str): The error message describing the authentication failure.
    """

    def __init__(self, message: str) -> None:
        self.message: str = f"Authentication failed: {message}"
        super().__init__(self.message)


class UserFacingError(Exception):
    """
    Exception raised to guide the user to update their query.

    Attributes:
        message (str): The main error message.
        extra_message (str): Additional message to guide the user in updating their query.
    """

    def __init__(self, message: str, extra_message: str) -> None:
        super().__init__(message)
        # Store the additional message in a custom attribute, which will be used to guide the user to update their query.
        self.extra_message: str = extra_message


class ToolExecutionError(UserFacingError):
    """
    Exception raised when a tool execution fails.

    Attributes:
        message (str): The error message describing the tool execution failure.
        extra_message (str): Additional message to guide the user in resolving the tool execution error.
    """

    def __init__(self, message: str, extra_message: str) -> None:
        self.message: str = f"Tool {message} execution failed"
        super().__init__(self.message, extra_message)


class ExceptionPrompt:
    """
    Base class for tool-specific exception prompts.

    This class serves as a parent class for tool collections (like Shopify, HubSpot)
    to define their own exception prompts as class attributes.

    Example:
        class ShopifyExceptionPrompt(ExceptionPrompt):
            ORDER_NOT_FOUND: str = "Order could not be found."
            PRODUCT_NOT_AVAILABLE: str = "Product is not available."

    Each tool collection should create their own _exception_prompt.py file
    that inherits from this base class.
    """

    # Common exception prompts shared across tool collections
    pass
