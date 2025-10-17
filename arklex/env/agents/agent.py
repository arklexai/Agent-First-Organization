from typing import TypeVar

from pydantic import BaseModel

from arklex.orchestrator.entities.orchestrator_state_entities import (
    StatusEnum,
)
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)

T = TypeVar("T")


class AgentOutput(BaseModel):
    """Output for the agent."""

    response: str
    status: StatusEnum


def register_agent(cls: type[T]) -> type[T]:
    """Register an agent class with the Arklex framework.

    This decorator registers an agent class and automatically sets its name
    to the class name. It is used to mark classes as agents in the system.

    Args:
        cls (Type[T]): The agent class to register.

    Returns:
        Type[T]: The registered agent class.
    """
    cls.name = cls.__name__
    return cls


class BaseAgent:
    pass
