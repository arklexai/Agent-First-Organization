"""Automatic error identification for TAU benchmark results.

This module provides functionality for analyzing and categorizing errors in TAU benchmark
results. It includes tools for fault assignment (identifying responsible entities) and
fault type classification, with support for different grading strategies and concurrent
analysis of multiple results.

The module includes:
- Command line argument parsing for error identification
- Data models for results and fault analysis
- Fault assignment and classification
- Context formatting and display utilities
- Concurrent processing of multiple results
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any

from pydantic import BaseModel

from benchmark.tau_bench.envs.airline.tasks_test import TASKS as AIRLINE_TASKS
from benchmark.tau_bench.envs.retail.tasks_test import TASKS_TEST as RETAIL_TASKS
from benchmark.tau_bench.model_utils import API, default_api_from_args
from benchmark.tau_bench.model_utils.args import api_parser
from benchmark.tau_bench.tau_types import Action, Task


def get_args() -> argparse.Namespace:
    """Parse command line arguments for error identification.

    This function sets up and parses command line arguments for the error
    identification process, including environment selection, results path,
    concurrency settings, and output directory.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - env: The environment (airline or retail)
            - results-path: Path to the results file
            - max-concurrency: Maximum number of concurrent API calls
            - output-dir: Path to the output directory
            - max-num-failed-results: Maximum number of failed results to analyze
    """
    parser: argparse.ArgumentParser = api_parser()
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=["airline", "retail"],
        help="The environment that the original trajectories are from (used to fetch the user instructions)",
    )
    parser.add_argument(
        "--results-path", type=str, required=True, help="Path to the results file"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent API calls",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to the output file"
    )
    parser.add_argument(
        "--max-num-failed-results",
        "-n",
        type=int,
        help="Maximum number of failed results to analyze",
    )
    return parser.parse_args()


class OriginalResult(BaseModel):
    """Model for original benchmark results.

    This class represents the original results from a benchmark run, including
    task ID, user instruction, trajectory, and ground truth information.

    Attributes:
        task_id (int): The ID of the task.
        user_instruction (str): The instruction given to the user.
        traj (List[Dict[str, Any]]): The sequence of messages in the trajectory.
        ground_truth_actions (List[Action]): The expected actions for the task.
        ground_truth_outputs (List[str]): The expected outputs for the task.
    """

    task_id: int
    user_instruction: str
    traj: list[dict[str, Any]]
    ground_truth_actions: list[Action]
    ground_truth_outputs: list[str]


class FaultAuthor(Enum):
    """Enumeration of possible fault authors.

    This enum defines the possible entities that could be responsible for a fault:
    the user, the agent, or the environment.

    Values:
        USER: The user is responsible for the fault.
        AGENT: The agent is responsible for the fault.
        ENVIRONMENT: The environment is responsible for the fault.
    """

    USER = "user"
    AGENT = "agent"
    ENVIRONMENT = "environment"


class FaultAssignmentResult(BaseModel):
    """Model for fault assignment results.

    This class represents the result of assigning responsibility for a fault,
    including the task ID, responsible entity, and a description of the fault.

    Attributes:
        task_id (int): The ID of the task.
        author (FaultAuthor): The entity responsible for the fault.
        description (str): A description of the fault.

    Methods:
        model_dump: Convert the result to a dictionary format.
    """

    task_id: int
    author: FaultAuthor
    description: str

    def model_dump(self) -> dict[str, Any]:
        """Convert the fault assignment result to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the fault assignment result.
        """
        return {
            "task_id": self.task_id,
            "author": self.author.value,
            "description": self.description,
        }


class FaultType(Enum):
    """Enumeration of possible fault types.

    This enum defines the different types of faults that can occur in a benchmark
    run, such as calling the wrong tool or using incorrect arguments.

    Values:
        CALLED_WRONG_TOOL: The agent called an incorrect tool.
        USED_WRONG_TOOL_ARGUMENT: The agent used incorrect arguments for a tool.
        GOAL_PARTIALLY_COMPLETED: The agent partially completed the goal.
        OTHER: Any other type of fault.
    """

    CALLED_WRONG_TOOL = "called_wrong_tool"
    USED_WRONG_TOOL_ARGUMENT = "used_wrong_tool_argument"
    GOAL_PARTIALLY_COMPLETED = "goal_partially_completed"
    OTHER = "other"


class FaultTypeResult(BaseModel):
    """Model for fault type classification results.

    This class represents the result of classifying a fault's type, including
    the task ID, fault type, and a description of the fault.

    Attributes:
        task_id (int): The ID of the task.
        fault_type (FaultType): The type of fault that occurred.
        description (str): A description of the fault.

    Methods:
        model_dump: Convert the result to a dictionary format.
    """

    task_id: int
    fault_type: FaultType
    description: str

    def model_dump(self) -> dict[str, Any]:
        """Convert the fault type result to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the fault type result.
        """
        return {
            "task_id": self.task_id,
            "fault_type": self.fault_type.value,
            "description": self.description,
        }


class GradingStrategy(Enum):
    """Enumeration of grading strategies.

    This enum defines the different strategies for grading benchmark results,
    either based on actions or outputs.

    Values:
        ACTIONS: Grade based on the actions taken.
        OUTPUTS: Grade based on the outputs produced.
    """

    ACTIONS = "actions"
    OUTPUTS = "outputs"


def context_description(grading_strategy: GradingStrategy) -> str:
    """Generate a description of the context for error analysis.

    This function creates a description of the context that will be provided
    to the error analysis model, based on the chosen grading strategy. The
    description includes information about the user instruction, ground truth
    data, and trajectory.

    Args:
        grading_strategy (GradingStrategy): The strategy to use for grading.

    Returns:
        str: A description of the context for error analysis.
    """
    if grading_strategy == GradingStrategy.ACTIONS:
        return """You will be given a user instruction, the ground truth action sequence, and a trajectory.
- The user instruction is the instruction given to the simulated user.
- The ground truth action sequence is one example of a valid sequence of actions that lead to the goal state (the sequence of actions could be empty, meaning that no action should have been taken).
- The trajectory is the sequence of messages between the user and the agent.
- The trajectory has been determined to have a fault."""
    return """You will be given a user instruction, the set of required agent response outputs, and a trajectory.
- The user instruction is the instruction given to the simulated user.
- The required agent response outputs are the set of outputs that the agent is expected to communicate to the user.
- The trajectory is the sequence of messages between the user and the agent.
- The trajectory has been determined to have a fault."""


def display_traj(traj: list[dict[str, Any]]) -> str:
    """Format a trajectory for display.

    This function formats a trajectory (sequence of messages) into a readable
    string format, excluding system messages. Each message is formatted as
    "Role: Content" on a new line.

    Args:
        traj (List[Dict[str, Any]]): The trajectory to format.

    Returns:
        str: A formatted string representation of the trajectory.

    Raises:
        ValueError: If the trajectory is empty.
    """
    if len(traj) == 0:
        raise ValueError("Trajectory is empty")
    stripped_traj: list[dict[str, Any]] = [
        item for item in traj if item["role"] != "system"
    ]
    return "\n".join(
        [f"{item['role'].capitalize()}: {item['content']}" for item in stripped_traj]
    )


def display_actions(actions: list[Action]) -> str:
    """Format a list of actions for display.

    This function formats a list of actions into a JSON string for display,
    with each action's properties properly indented.

    Args:
        actions (List[Action]): The actions to format.

    Returns:
        str: A JSON string representation of the actions.
    """
    return json.dumps([action.model_dump() for action in actions], indent=4)


def display_context(
    user_instruction: str,
    ground_truth_actions: list[Action],
    ground_truth_outputs: list[str],
    trajectory: list[dict[str, Any]],
) -> str:
    """Format the complete context for error analysis.

    This function combines user instruction, ground truth information, and
    trajectory into a formatted context string for error analysis. The context
    includes clear section markers and appropriate formatting for each component.

    Args:
        user_instruction (str): The user's instruction.
        ground_truth_actions (List[Action]): The expected actions.
        ground_truth_outputs (List[str]): The expected outputs.
        trajectory (List[Dict[str, Any]]): The actual trajectory.

    Returns:
        str: A formatted string containing all context information.
    """
    traj_display: str = display_traj(trajectory)
    context: str = f"""----- start user instruction -----
{user_instruction}
----- end user instruction -----"""
    if len(ground_truth_outputs) > 0:
        context += f"""

----- start required outputs -----
{ground_truth_outputs}
----- end required outputs -----"""
    else:
        context += f"""

----- start ground truth action sequence -----
{display_actions(ground_truth_actions)}
----- end ground truth action sequence -----

----- start trajectory -----
{traj_display}
----- end trajectory -----\n"""
    return context


def fault_assignment_analysis(
    api: API, results: list[OriginalResult], max_concurrency: int
) -> list[FaultAssignmentResult]:
    """Analyze and assign responsibility for faults.

    This function analyzes a list of benchmark results to determine which entity
    (user, agent, or environment) is responsible for each fault.

    Args:
        api (API): The API to use for analysis.
        results (List[OriginalResult]): The results to analyze.
        max_concurrency (int): Maximum number of concurrent API calls.

    Returns:
        List[FaultAssignmentResult]: List of fault assignment results.
    """

    def assign_fault(
        task_id: int,
        user_instruction: str,
        traj: list[dict[str, Any]],
        ground_truth_actions: list[Action],
        ground_truth_outputs: list[str],
    ) -> FaultAssignmentResult:
        idx_to_author: dict[int, FaultAuthor] = {
            0: FaultAuthor.USER,
            1: FaultAuthor.AGENT,
            2: FaultAuthor.ENVIRONMENT,
        }
        grading_strategy: GradingStrategy = (
            GradingStrategy.OUTPUTS
            if len(ground_truth_outputs) > 0
            else GradingStrategy.ACTIONS
        )
        ctx_desc: str = context_description(grading_strategy)
        context: str = display_context(
            user_instruction, ground_truth_actions, ground_truth_outputs, traj
        )
        res: int = api.classify(
            instruction=f"{ctx_desc}\n\nDetermine the entity that is responsible for the fault. The user is responsible for the fault if they caused an action that was not grounded in the user instruction. The agent is responsible for the fault if they took an action that was not correct (or took the action with the wrong arguments). The environment is responsible for all other faults.",
            text=context,
            options=[
                "The user",
                "The agent",
                "The environment (neither user nor agent)",
            ],
        )
        author: FaultAuthor = idx_to_author[res]
        description: str = api.generate(
            instruction=f"{ctx_desc}\n\nDescribe the reason why {author.value} is responsible for the fault in the trajectory. Be concise and only focus on the functional differences between the ground truth and the trajectory.",
            text=context,
        )
        return FaultAssignmentResult(
            task_id=task_id, author=author, description=description
        )

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        task_ids: list[int] = [r.task_id for r in results]
        user_instructions: list[str] = [r.user_instruction for r in results]
        trajs: list[list[dict[str, Any]]] = [r.traj for r in results]
        ground_truth_actions: list[list[Action]] = [
            r.ground_truth_actions for r in results
        ]
        ground_truth_outputs: list[list[str]] = [
            r.ground_truth_outputs for r in results
        ]
        results: list[FaultAssignmentResult] = list(
            executor.map(
                assign_fault,
                task_ids,
                user_instructions,
                trajs,
                ground_truth_actions,
                ground_truth_outputs,
            )
        )
    return results


def fault_type_analysis(
    api: API, results: list[OriginalResult], max_concurrency: int
) -> list[FaultTypeResult]:
    """Analyze and classify fault types.

    This function analyzes a list of benchmark results to determine the type
    of each fault that occurred.

    Args:
        api (API): The API to use for analysis.
        results (List[OriginalResult]): The results to analyze.
        max_concurrency (int): Maximum number of concurrent API calls.

    Returns:
        List[FaultTypeResult]: List of fault type classification results.
    """

    def get_fault_type(
        task_id: int,
        user_instruction: str,
        traj: list[dict[str, Any]],
        ground_truth_actions: list[Action],
        ground_truth_outputs: list[str],
    ) -> FaultTypeResult:
        idx_to_fault_type: dict[int, FaultType] = {
            0: FaultType.CALLED_WRONG_TOOL,
            1: FaultType.USED_WRONG_TOOL_ARGUMENT,
            2: FaultType.GOAL_PARTIALLY_COMPLETED,
            3: FaultType.OTHER,
        }
        grading_strategy: GradingStrategy = (
            GradingStrategy.OUTPUTS
            if len(ground_truth_outputs) > 0
            else GradingStrategy.ACTIONS
        )
        ctx_desc: str = context_description(grading_strategy)
        context: str = display_context(
            user_instruction, ground_truth_actions, ground_truth_outputs, traj
        )
        res: int = api.classify(
            instruction=f"{ctx_desc}\n\nDetermine the type of fault of the first instance of the fault.",
            text=context,
            options=[
                "The user called the wrong tool",
                "The user used the correct tool with a wrong argument",
                "The goal was only partially completed",
                "Other",
            ],
        )
        fault_type: FaultType = idx_to_fault_type[res]
        description: str = api.generate(
            instruction=f"{ctx_desc}\n\nDescribe the reason why this is a {fault_type.value} fault in the trajectory. Be concise and only focus on the functional differences between the ground truth and the trajectory.",
            text=context,
        )
        return FaultTypeResult(
            task_id=task_id, fault_type=fault_type, description=description
        )

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        task_ids: list[int] = [r.task_id for r in results]
        user_instructions: list[str] = [r.user_instruction for r in results]
        trajs: list[list[dict[str, Any]]] = [r.traj for r in results]
        ground_truth_actions: list[list[Action]] = [
            r.ground_truth_actions for r in results
        ]
        ground_truth_outputs: list[list[str]] = [
            r.ground_truth_outputs for r in results
        ]
        results: list[FaultTypeResult] = list(
            executor.map(
                get_fault_type,
                task_ids,
                user_instructions,
                trajs,
                ground_truth_actions,
                ground_truth_outputs,
            )
        )
    return results


def run_error_identification(args: argparse.Namespace) -> None:
    api: API = default_api_from_args(args)
    with open(args.results_path) as f:
        results: list[dict[str, Any]] = json.load(f)
    failed_results: list[dict[str, Any]] = [r for r in results if r["reward"] < 1]
    if args.max_num_failed_results is not None:
        failed_results = failed_results[: args.max_num_failed_results]
    original_results: list[OriginalResult] = []
    for r in failed_results:
        task_id: int = r["task_id"]
        if args.env == "airline":
            task: Task = AIRLINE_TASKS[task_id]
        else:
            task: Task = RETAIL_TASKS[task_id]
        original_results.append(
            OriginalResult(
                task_id=task_id,
                user_instruction=task.instruction,
                traj=r["traj"],
                ground_truth_actions=task.actions,
                ground_truth_outputs=task.outputs,
            )
        )
    fault_assignment_results: list[FaultAssignmentResult] = fault_assignment_analysis(
        api, original_results, args.max_concurrency
    )
    fault_type_results: list[FaultTypeResult] = fault_type_analysis(
        api, original_results, args.max_concurrency
    )
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "fault_assignment.json"), "w") as f:
        json.dump([r.model_dump() for r in fault_assignment_results], f, indent=4)
    with open(os.path.join(args.output_dir, "fault_type.json"), "w") as f:
        json.dump([r.model_dump() for r in fault_type_results], f, indent=4)
