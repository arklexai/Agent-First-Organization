# Copyright Sierra

from benchmark.tau_bench.envs.airline.data import load_data
from benchmark.tau_bench.envs.airline.rules import RULES
from benchmark.tau_bench.envs.airline.tools import ALL_TOOLS
from benchmark.tau_bench.envs.airline.wiki import WIKI
from benchmark.tau_bench.envs.base import Env
from typing import Optional, Union, List
from benchmark.tau_bench.envs.user import UserStrategy


class MockAirlineDomainEnv(Env):
    """
    Environment for simulating airline domain interactions.

    This environment provides a mock airline domain for testing and training
    conversational agents in an airline context.
    """

    terminate_tools: List[str]

    def __init__(
        self,
        user_strategy: Union[str, UserStrategy] = UserStrategy.LLM,
        user_model: str = "gpt-4o",
        user_provider: Optional[str] = None,
        task_split: str = "test",
        task_index: Optional[int] = None,
    ) -> None:
        """
        Initialize the airline domain environment.

        Args:
            user_strategy (Union[str, UserStrategy]): Strategy for user simulation.
            user_model (str): Model to use for user simulation.
            user_provider (Optional[str]): Provider for the user model.
            task_split (str): Which task split to use ('test').
            task_index (Optional[int]): Index of the task to use.

        Raises:
            ValueError: If an unknown task split is provided.
        """
        match task_split:
            case "test":
                from benchmark.tau_bench.envs.airline.tasks_test import TASKS as tasks
            case _:
                raise ValueError(f"Unknown task split: {task_split}")
        super().__init__(
            data_load_func=load_data,
            tools=ALL_TOOLS,
            tasks=tasks,
            wiki=WIKI,
            rules=RULES,
            user_strategy=user_strategy,
            user_model=user_model,
            user_provider=user_provider,
            task_index=task_index,
        )
        self.terminate_tools: List[str] = ["transfer_to_human_agents"]
