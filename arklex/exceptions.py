__all__ = [
    "AuthenticationError",
    "ToolExecutionError",
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
