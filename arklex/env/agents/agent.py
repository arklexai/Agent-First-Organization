import traceback
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

from arklex.env.tools.tools import Tool
from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
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
    tools: list[Tool] = []

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
    def init_agent_data(
        self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
    ) -> None:
        """Initialize the agent data.

        Args:
            orch_state (OrchestratorState): The current orchestrator state.
            node_specific_data (dict[str, Any]): Additional keyword arguments for the execution.
        """

    @abstractmethod
    def _execute(self) -> tuple[OrchestratorState, AgentOutput]:
        """Execute the agent's core functionality.

        This abstract method must be implemented by concrete agent classes to
        define their specific execution logic.
        """

    def is_async(self) -> bool:
        """Indicate whether this agent needs async execution."""
        return False

    def execute(
        self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
    ) -> tuple[OrchestratorState, Any]:  # noqa: ANN401
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
            self.init_agent_data(orch_state, node_specific_data)
            response_state, output = self._execute()
            if response_state.trajectory and response_state.trajectory[-1]:
                response_state.trajectory[-1][-1].output = output.response
            agent_output: AgentOutput = AgentOutput(
                response=output.response,
                status=StatusEnum.INCOMPLETE,
            )
        except Exception:
            log_context.error(traceback.format_exc())
            agent_output: AgentOutput = AgentOutput(
                response="",
                status=StatusEnum.INCOMPLETE,
            )
        return orch_state, agent_output

    async def _async_execute(
        self,
        orch_state: OrchestratorState,
        **kwargs: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        """Async version of _execute. Override in async agents."""
        raise NotImplementedError("This agent does not support async execution")

    async def async_execute(
        self,
        orch_state: OrchestratorState,
        **kwargs: Any,  # noqa: ANN401
    ) -> OrchestratorState:
        """Public async method with error handling."""
        try:
            response_return = await self._async_execute(orch_state, **kwargs)
            response_state = OrchestratorState.model_validate(response_return)

            if response_state.trajectory and response_state.trajectory[-1]:
                response_state.trajectory[-1][-1].output = (
                    response_state.response or response_state.message_flow
                )
            return response_state
        except Exception:
            log_context.error(traceback.format_exc())
            return orch_state
