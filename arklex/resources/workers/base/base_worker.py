"""Base worker implementation for the Arklex framework.

This module provides the base abstract class for workers in the framework. It defines the
interface for worker execution and includes functionality for worker registration, error
handling, and state management. The module serves as a foundation for implementing specific
workers that handle different types of tasks and operations within the system.
"""

from abc import ABC, abstractmethod
from typing import Any

from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
)
from arklex.resources.workers.base.entities import WorkerOutput
from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)


class BaseWorker(ABC):
    """Base abstract class for workers in the Arklex framework.

    This class defines the interface for worker execution and provides common
    functionality for all workers. It includes methods for string representation,
    execution handling, and error management.

    Attributes:
        description (Optional[str]): Description of the worker's functionality.
    """

    description: str | None = None

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
    def init_worker_data(
        self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
    ) -> None:
        """Initialize the worker data.

        Args:
            orch_state (OrchestratorState): The current orchestrator state.
            node_specific_data (dict[str, Any]): Additional keyword arguments for the execution.
        """

    @abstractmethod
    def _execute(self) -> WorkerOutput:
        """Execute the worker's core functionality.

        This abstract method must be implemented by concrete worker classes to
        define their specific execution logic.

        Returns:
            WorkerOutput: The execution results.
        """

    def execute(
        self,
        orch_state: OrchestratorState,
        node_specific_data: dict[str, Any],
    ) -> tuple[OrchestratorState, WorkerOutput]:
        """Execute the worker with error handling and state management.

        This method wraps the worker's execution with error handling and state
        management. It processes the execution results and updates the message state.

        Args:
            orch_state (OrchestratorState): The current orchestrator state.
            node_specific_data (dict[str, Any]): Additional keyword arguments for the execution.

        Returns:
            MessageState: The updated message state after execution.
        """
        self.init_worker_data(orch_state, node_specific_data)
        worker_output: WorkerOutput = self._execute()

        return self.orch_state, worker_output
