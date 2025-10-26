from typing import TypeVar

from arklex.utils.logging.logging_utils import LogContext

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
    cls.__name__ = cls.__name__
    return cls


class BaseAgent:
    pass
