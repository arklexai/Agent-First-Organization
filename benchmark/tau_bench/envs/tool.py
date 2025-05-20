import abc
from typing import Any, Dict, TypeVar, Generic

T = TypeVar("T")


class Tool(abc.ABC, Generic[T]):
    """
    Abstract base class for tools in the benchmark environment.

    This class defines the interface that all tool implementations must follow.
    Tools are used to perform specific actions in the environment.
    """

    @staticmethod
    def invoke(data: T, **kwargs: Any) -> str:
        """
        Invoke the tool with the given data and arguments.

        Args:
            data (T): The data to use.
            **kwargs: Additional arguments for the tool.

        Returns:
            str: The result of the tool invocation.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def get_info() -> Dict[str, Any]:
        """
        Get information about the tool.

        Returns:
            Dict[str, Any]: A dictionary containing tool information.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError
