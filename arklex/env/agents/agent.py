import traceback
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from arklex.orchestrator.entities.msg_state_entities import MessageState
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)

T = TypeVar("T")


def register_agent(cls: type[T]) -> type[T]:
    """Register an agent class with the Arklex framework.

    This decorator registers an agent class and automatically sets its name
    to the class name. It is used to mark classes as agents in the system.

    Args:
        cls (Type[T]): The agent class to register.

    Returns:
        Type[T]: The registered agent class.
    """
    cls.name = cls.__name__  # Automatically set name to the class name
    return cls


class BaseAgent(ABC):
    """Base abstract class for agents in the Arklex framework.

    This class defines the interface for agent execution and provides common
    functionality for all agents. It includes methods for string representation,
    execution handling, and error management.

    Attributes:
        description (Optional[str]): Description of the agent's functionality.
        name (str): The name of the agent class.
    """

    description: str | None = None
    name: str

    def __str__(self) -> str:
        """Get a string representation of the agent.

        Returns:
            str: The name of the agent class.
        """
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        """Get a detailed string representation of the agent.

        Returns:
            str: The name of the agent class.
        """
        return f"{self.__class__.__name__}"

    @abstractmethod
    def _execute(self, msg_state: MessageState, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Execute the agent's core functionality.

        This abstract method must be implemented by concrete agent classes to
        define their specific execution logic.

        Args:
            msg_state (MessageState): The current message state.
            **kwargs (Any): Additional keyword arguments for the execution.

        Returns:
            Dict[str, Any]: The execution results as a dictionary.
        """
        pass

    def execute(self, msg_state: MessageState, **kwargs: Any) -> MessageState:  # noqa: ANN401
        """Execute the agent with error handling and state management.

        This method wraps the agent's execution with error handling and state
        management. It processes the execution results and updates the message state.

        Args:
            msg_state (MessageState): The current message state.
            **kwargs (Any): Additional keyword arguments for the execution.

        Returns:
            MessageState: The updated message state after execution.
        """
        try:
            response_return: dict[str, Any] = self._execute(msg_state, **kwargs)
            response_state: MessageState = MessageState.model_validate(response_return)
            if response_state.trajectory and response_state.trajectory[-1]:
                response_state.trajectory[-1][-1].output = (
                    response_state.response
                    if response_state.response
                    else response_state.message_flow
                )
            return response_state
        except Exception:
            log_context.error(traceback.format_exc())
            return msg_state
