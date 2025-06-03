"""Type definitions for the TAU benchmark.

This module defines the core data types used throughout the TAU benchmark,
including actions, tasks, rewards, and environment responses. These types
are used to represent the state and results of benchmark evaluations.
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union

RESPOND_ACTION_NAME = "respond"
RESPOND_ACTION_FIELD_NAME = "content"


class Action(BaseModel):
    """Represents an action in the benchmark environment.

    This class defines the structure of an action, including its name and
    keyword arguments.

    Attributes:
        name (str): The name of the action.
        kwargs (Dict[str, Any]): The keyword arguments for the action.
    """

    name: str
    kwargs: Dict[str, Any]


class Task(BaseModel):
    """Represents a task in the benchmark environment.

    This class defines the structure of a task, including user ID, required
    actions, instruction, and expected outputs.

    Attributes:
        user_id (str): The ID of the user performing the task.
        actions (List[Action]): The list of actions required to complete the task.
        instruction (str): The instruction describing the task.
        outputs (List[str]): The expected outputs from completing the task.
    """

    user_id: str
    actions: List[Action]
    instruction: str
    outputs: List[str]


class RewardOutputInfo(BaseModel):
    """Information about reward based on outputs.

    This class contains information about the reward calculated based on
    output matching, including the reward value and output match status.

    Attributes:
        r_outputs (float): The reward value based on outputs.
        outputs (Dict[str, bool]): Dictionary mapping outputs to their match status.
    """

    r_outputs: float
    outputs: Dict[str, bool]


class RewardActionInfo(BaseModel):
    """Information about reward based on actions.

    This class contains information about the reward calculated based on
    action matching, including the reward value and ground truth data hash.

    Attributes:
        r_actions (float): The reward value based on actions.
        gt_data_hash (str): Hash of the ground truth data.
    """

    r_actions: float
    gt_data_hash: str


class RewardResult(BaseModel):
    """Result of reward calculation.

    This class represents the result of calculating a reward, including
    the reward value, reward information, and actions taken.

    Attributes:
        reward (float): The calculated reward value.
        info (Union[RewardOutputInfo, RewardActionInfo]): Information about the reward calculation.
        actions (List[Action]): The actions that led to this reward.
    """

    reward: float
    info: Union[RewardOutputInfo, RewardActionInfo]
    actions: List[Action]


class SolveResult(BaseModel):
    """Result of solving a task.

    This class represents the result of attempting to solve a task, including
    the reward, messages exchanged, information about the solution, and total cost.

    Attributes:
        reward (float): The reward received for the solution.
        messages (List[Dict[str, Any]]): The messages exchanged during the solution.
        info (Dict[str, Any]): Additional information about the solution.
        total_cost (Optional[float]): The total cost of the solution attempt.
    """

    reward: float
    messages: List[Dict[str, Any]]
    info: Dict[str, Any]
    total_cost: Optional[float] = None


class EnvInfo(BaseModel):
    """Information about the environment state.

    This class contains information about the current state of the environment,
    including the current task, source, user cost, and reward information.

    Attributes:
        task (Task): The current task.
        source (Optional[str]): The source of the task.
        user_cost (Optional[float]): The cost incurred by the user.
        reward_info (Optional[RewardResult]): Information about the reward.
    """

    task: Task
    source: Optional[str] = None
    user_cost: Optional[float] = None
    reward_info: Optional[RewardResult] = None


class EnvResponse(BaseModel):
    """Response from the environment.

    This class represents a response from the environment after an action,
    including the observation, reward, completion status, and environment info.

    Attributes:
        observation (str): The observation from the environment.
        reward (float): The reward received.
        done (bool): Whether the task is complete.
        info (EnvInfo): Information about the environment state.
    """

    observation: str
    reward: float
    done: bool
    info: EnvInfo


class EnvResetResponse(BaseModel):
    """Response from environment reset.

    This class represents the response from resetting the environment,
    including the initial observation and environment info.

    Attributes:
        observation (str): The initial observation.
        info (EnvInfo): Information about the environment state.
    """

    observation: str
    info: EnvInfo


class EnvRunResult(BaseModel):
    """Result of running an environment.

    This class represents the result of running an environment for a task,
    including the task ID, reward, information, trajectory, and trial number.

    Attributes:
        task_id (int): The ID of the task.
        reward (float): The reward received.
        info (Dict[str, Any]): Additional information about the run.
        traj (List[Dict[str, Any]]): The trajectory of the run.
        trial (int): The trial number.
    """

    task_id: int
    reward: float
    info: Dict[str, Any]
    traj: List[Dict[str, Any]]
    trial: int


class RunConfig(BaseModel):
    """Configuration for running the benchmark.

    This class defines the configuration parameters for running the benchmark,
    including model settings, task settings, and execution parameters.

    Attributes:
        user_model_provider (str): The provider of the user model.
        user_model (str): The user model to use.
        num_trials (int): Number of trials to run.
        env (str): The environment to use.
        task_split (str): The task split to use.
        start_index (int): The starting index for tasks.
        end_index (int): The ending index for tasks.
        task_ids (Optional[List[int]]): Specific task IDs to run.
        log_dir (str): Directory for logs.
        max_concurrency (int): Maximum number of concurrent tasks.
        seed (int): Random seed for reproducibility.
        shuffle (int): Whether to shuffle tasks.
        user_strategy (str): The user strategy to use.
        output_dir (str): Directory for output.
        taskgraph_dir (str): Directory containing task graphs.
    """

    user_model_provider: str
    user_model: str = "gpt-4o"
    num_trials: int = 1
    env: str = "retail"
    task_split: str = "test"
    start_index: int = 0
    end_index: int = -1
    task_ids: Optional[List[int]] = None
    log_dir: str = "results"
    max_concurrency: int = 1
    seed: int = 10
    shuffle: int = 0
    user_strategy: str = "llm"
    output_dir: str
    taskgraph_dir: str
