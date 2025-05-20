# Copyright Sierra

import random
from hashlib import sha256
from benchmark.tau_bench.envs.tool import Tool
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Type,
    Optional,
    Set,
    Union,
    Tuple,
)

from benchmark.tau_bench.envs.user import load_user, UserStrategy
from benchmark.tau_bench.tau_types import (
    Action,
    Task,
    EnvInfo,
    EnvResetResponse,
    EnvResponse,
    RewardResult,
    RewardOutputInfo,
    RewardActionInfo,
    RESPOND_ACTION_NAME,
)

ToHashable = Union[
    str, int, float, Dict[str, "ToHashable"], List["ToHashable"], Set["ToHashable"]
]
Hashable = Union[str, int, float, Tuple["Hashable"], Tuple[Tuple[str, "Hashable"]]]


def to_hashable(item: ToHashable) -> Hashable:
    """
    Convert an item to a hashable type.

    Args:
        item (ToHashable): The item to convert.

    Returns:
        Hashable: The converted hashable item.
    """
    if isinstance(item, dict):
        return tuple((key, to_hashable(value)) for key, value in sorted(item.items()))
    elif isinstance(item, list):
        return tuple(to_hashable(element) for element in item)
    elif isinstance(item, set):
        return tuple(sorted(to_hashable(element) for element in item))
    else:
        return item


def consistent_hash(value: Hashable) -> str:
    """
    Generate a consistent hash for a value.

    Args:
        value (Hashable): The value to hash.

    Returns:
        str: The hexadecimal hash of the value.
    """
    return sha256(str(value).encode("utf-8")).hexdigest()


class Env:
    """
    Base environment class for the benchmark.

    This class provides the core functionality for interacting with tasks,
    tools, and users in the benchmark environment.
    """

    def __init__(
        self,
        data_load_func: Callable[[], Dict[str, Any]],
        tools: List[Type[Tool]],
        tasks: List[Task],
        wiki: str,
        rules: List[str],
        user_strategy: Union[str, UserStrategy],
        user_model: str,
        user_provider: Optional[str] = None,
        task_index: Optional[int] = None,
    ) -> None:
        """
        Initialize the environment.

        Args:
            data_load_func (Callable[[], Dict[str, Any]]): Function to load data.
            tools (List[Type[Tool]]): List of tool classes.
            tasks (List[Task]): List of tasks.
            wiki (str): Wiki text for context.
            rules (List[str]): List of rules.
            user_strategy (Union[str, UserStrategy]): User strategy to use.
            user_model (str): Model to use for user simulation.
            user_provider (Optional[str]): Provider for the user model.
            task_index (Optional[int]): Index of the task to use.
        """
        super().__init__()
        self.data_load_func: Callable[[], Dict[str, Any]] = data_load_func
        self.data: Dict[str, Any] = data_load_func()
        self.tools_map: Dict[str, Type[Tool]] = {
            tool.get_info()["function"]["name"]: tool for tool in tools
        }
        self.tools_info: List[Dict[str, Any]] = [tool.get_info() for tool in tools]
        self.terminate_tools: List[str] = []
        self.tasks: List[Task] = tasks
        if task_index is not None:
            self.task_index: int = task_index
        else:
            self.task_index: int = random.randint(0, len(tasks))
        self.task: Task = tasks[self.task_index]
        self.wiki: str = wiki
        self.rules: List[str] = rules
        self.user = load_user(
            user_strategy=user_strategy, model=user_model, provider=user_provider
        )
        self.actions: List[Action] = []

    def reset(self, task_index: Optional[int] = None) -> EnvResetResponse:
        """
        Reset the environment to a new task.

        Args:
            task_index (Optional[int]): Index of the task to use.

        Returns:
            EnvResetResponse: The reset response containing initial observation and info.
        """
        if task_index is None:
            task_index = random.randint(0, len(self.tasks))
        self.task_index = task_index
        self.data = self.data_load_func()
        self.task = self.tasks[task_index]
        self.actions = []
        initial_observation: str = self.user.reset(instruction=self.task.instruction)
        return EnvResetResponse(
            observation=initial_observation, info=EnvInfo(task=self.task, source="user")
        )

    def step(self, action: Action) -> EnvResponse:
        """
        Take a step in the environment.

        Args:
            action (Action): The action to take.

        Returns:
            EnvResponse: The response containing observation, reward, done flag, and info.
        """
        self.actions.append(action)

        info: EnvInfo = EnvInfo(task=self.task)
        reward: float = 0
        done: bool = False
        observation: str

        if action.name == RESPOND_ACTION_NAME:
            observation = self.user.step(action.kwargs["content"])
            info.source = "user"
            done = "###STOP###" in observation
        elif action.name in self.tools_map:
            try:
                observation = self.tools_map[action.name].invoke(
                    data=self.data, **action.kwargs
                )
            except Exception as e:
                observation = f"Error: {e}"
            info.source = action.name
            if action.name in self.terminate_tools:
                done = True
        else:
            observation = f"Unknown action {action.name}"
            info.source = action.name

        if done:
            reward_res: RewardResult = self.calculate_reward()
            reward = reward_res.reward
            info.reward_info = reward_res
            info.user_cost = self.user.get_total_cost()
        return EnvResponse(observation=observation, reward=reward, done=done, info=info)

    def get_data_hash(self) -> str:
        """
        Get the hash of the current data state.

        Returns:
            str: The hash of the data.
        """
        return consistent_hash(to_hashable(self.data))

    def calculate_reward(self) -> RewardResult:
        """
        Calculate the reward for the current state.

        Returns:
            RewardResult: The reward result containing reward value and info.
        """
        data_hash: str = self.get_data_hash()
        reward: float = 1.0
        actions: List[Action] = [
            action for action in self.task.actions if action.name != RESPOND_ACTION_NAME
        ]

        # Check if the database changes are correct. If they are not correct, then we set the reward to 0.
        # TODO: cache gt_data_hash in tasks.py (low priority)
        self.data = self.data_load_func()
        for action in self.task.actions:
            if action.name not in self.terminate_tools:
                self.step(action)
        gt_data_hash: str = self.get_data_hash()
        info: RewardActionInfo = RewardActionInfo(
            r_actions=data_hash == gt_data_hash, gt_data_hash=gt_data_hash
        )
        if not info.r_actions:
            reward = 0.0

        if len(self.task.outputs) > 0:
            # check outputs
            r_outputs: float = 1.0
            outputs: Dict[str, bool] = {}
            for output in self.task.outputs:
                found: bool = False
                for action in self.actions:
                    if (
                        action.name == RESPOND_ACTION_NAME
                        and output.lower()
                        in action.kwargs["content"].lower().replace(",", "")
                    ):
                        found = True
                        break
                outputs[output] = found
                if not found:
                    r_outputs = 0.0
                    reward = 0.0
            info = RewardOutputInfo(r_outputs=r_outputs, outputs=outputs)

        return RewardResult(reward=reward, info=info, actions=actions)
