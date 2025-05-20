# Copyright Sierra

import abc
from typing import Optional
from benchmark.tau_bench.envs.base import Env
from benchmark.tau_bench.tau_types import SolveResult


class Agent(abc.ABC):
    """
    Abstract base class for all agents in the benchmark.

    This class defines the interface that all agent implementations must follow.
    """

    @abc.abstractmethod
    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        """
        Solve a task in the given environment.

        Args:
            env (Env): The environment to solve the task in.
            task_index (Optional[int]): Index of the task to solve. If None, a random task is chosen.
            max_num_steps (int): Maximum number of steps to take before giving up.

        Returns:
            SolveResult: The result of solving the task.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError
