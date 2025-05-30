"""Base worker implementation for the Arklex framework.

This module provides the base abstract class for workers in the framework. It defines the
interface for worker execution and includes functionality for worker registration, error
handling, and state management. The module serves as a foundation for implementing specific
workers that handle different types of tasks and operations within the system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar
from arklex.utils.graph_state import MessageState, StatusEnum
import logging
import traceback

logger = logging.getLogger(__name__)

T = TypeVar("T")


def register_worker(cls: Type[T]) -> Type[T]:
    """Register a worker class with the Arklex framework.

    This decorator registers a worker class and automatically sets its name
    to the class name. It is used to mark classes as workers in the system.

    Args:
        cls (Type[T]): The worker class to register.

    Returns:
        Type[T]: The registered worker class.
    """
    cls.name = cls.__name__  # Automatically set name to the class name
    return cls


class BaseWorker(ABC):
    """Base abstract class for workers in the Arklex framework.

    This class defines the interface for worker execution and provides common
    functionality for all workers. It includes methods for string representation,
    execution handling, and error management.

    Attributes:
        description (Optional[str]): Description of the worker's functionality.
    """

    description: Optional[str] = None

    def __str__(self) -> str:
        """Get a string representation of the worker.

        Returns:
            str: The name of the worker class.
        """
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        """Get a detailed string representation of the worker.

        Returns:
            str: The name of the worker class.
        """
        return f"{self.__class__.__name__}"

    @abstractmethod
    def _execute(self, msg_state: MessageState, **kwargs: Any) -> Dict[str, Any]:
        """Execute the worker's core functionality.

        This abstract method must be implemented by concrete worker classes to
        define their specific execution logic.

        Args:
            msg_state (MessageState): The current message state.
            **kwargs (Any): Additional keyword arguments for the execution.

        Returns:
            Dict[str, Any]: The execution results as a dictionary.
        """
        pass

    def execute(self, msg_state: MessageState, **kwargs: Any) -> MessageState:
        """Execute the worker with error handling and state management.

        This method wraps the worker's execution with error handling and state
        management. It processes the execution results and updates the message state.

        Args:
            msg_state (MessageState): The current message state.
            **kwargs (Any): Additional keyword arguments for the execution.

        Returns:
            MessageState: The updated message state after execution.
        """
        try:
            response_return: Dict[str, Any] = self._execute(msg_state, **kwargs)
            response_state: MessageState = MessageState.model_validate(response_return)
            response_state.trajectory[-1][-1].output = (
                response_state.response
                if response_state.response
                else response_state.message_flow
            )
            if response_state.status == StatusEnum.INCOMPLETE:
                response_state.status = StatusEnum.COMPLETE
            return response_state
        except Exception as e:
            logger.error(traceback.format_exc())
            msg_state.status = StatusEnum.INCOMPLETE
            return msg_state
