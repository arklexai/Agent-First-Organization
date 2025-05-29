# Copyright Sierra
import os
import json
import argparse
from enum import Enum
from pydantic import BaseModel
from benchmark.tau_bench.model_utils import default_api_from_args, API
from benchmark.tau_bench.envs.airline.tasks_test import TASKS as AIRLINE_TASKS
from benchmark.tau_bench.envs.retail.tasks_test import TASKS_TEST as RETAIL_TASKS
from benchmark.tau_bench.model_utils.args import api_parser
from benchmark.tau_bench.tau_types import Task, Action
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor


def get_args() -> argparse.Namespace:
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
    task_id: int
    user_instruction: str
    traj: List[Dict[str, Any]]
    ground_truth_actions: List[Action]
    ground_truth_outputs: List[str]


class FaultAuthor(Enum):
    USER = "user"
    AGENT = "agent"
    ENVIRONMENT = "environment"


class FaultAssignmentResult(BaseModel):
    task_id: int
    author: FaultAuthor
    description: str

    def model_dump(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "author": self.author.value,
            "description": self.description,
        }


class FaultType(Enum):
    CALLED_WRONG_TOOL = "called_wrong_tool"
    USED_WRONG_TOOL_ARGUMENT = "used_wrong_tool_argument"
    GOAL_PARTIALLY_COMPLETED = "goal_partially_completed"
    OTHER = "other"


class FaultTypeResult(BaseModel):
    task_id: int
    fault_type: FaultType
    description: str

    def model_dump(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "fault_type": self.fault_type.value,
            "description": self.description,
        }


class GradingStrategy(Enum):
    ACTIONS = "actions"
    OUTPUTS = "outputs"


def context_description(grading_strategy: GradingStrategy) -> str:
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


def display_traj(traj: List[Dict[str, Any]]) -> str:
    if len(traj) == 0:
        raise ValueError("Trajectory is empty")
    stripped_traj: List[Dict[str, Any]] = [
        item for item in traj if item["role"] != "system"
    ]
    return "\n".join(
        [f"{item['role'].capitalize()}: {item['content']}" for item in stripped_traj]
    )


def display_actions(actions: List[Action]) -> str:
    return json.dumps([action.model_dump() for action in actions], indent=4)


def display_context(
    user_instruction: str,
    ground_truth_actions: List[Action],
    ground_truth_outputs: List[str],
    trajectory: List[Dict[str, Any]],
) -> str:
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
    api: API, results: List[OriginalResult], max_concurrency: int
) -> List[FaultAssignmentResult]:
    def assign_fault(
        task_id: int,
        user_instruction: str,
        traj: List[Dict[str, Any]],
        ground_truth_actions: List[Action],
        ground_truth_outputs: List[str],
    ) -> FaultAssignmentResult:
        idx_to_author: Dict[int, FaultAuthor] = {
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
        task_ids: List[int] = [r.task_id for r in results]
        user_instructions: List[str] = [r.user_instruction for r in results]
        trajs: List[List[Dict[str, Any]]] = [r.traj for r in results]
        ground_truth_actions: List[List[Action]] = [
            r.ground_truth_actions for r in results
        ]
        ground_truth_outputs: List[List[str]] = [
            r.ground_truth_outputs for r in results
        ]
        results: List[FaultAssignmentResult] = list(
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
    api: API, results: List[OriginalResult], max_concurrency: int
) -> List[FaultTypeResult]:
    def get_fault_type(
        task_id: int,
        user_instruction: str,
        traj: List[Dict[str, Any]],
        ground_truth_actions: List[Action],
        ground_truth_outputs: List[str],
    ) -> FaultTypeResult:
        idx_to_fault_type: Dict[int, FaultType] = {
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
        task_ids: List[int] = [r.task_id for r in results]
        user_instructions: List[str] = [r.user_instruction for r in results]
        trajs: List[List[Dict[str, Any]]] = [r.traj for r in results]
        ground_truth_actions: List[List[Action]] = [
            r.ground_truth_actions for r in results
        ]
        ground_truth_outputs: List[List[str]] = [
            r.ground_truth_outputs for r in results
        ]
        results: List[FaultTypeResult] = list(
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
        results: List[Dict[str, Any]] = json.load(f)
    failed_results: List[Dict[str, Any]] = [r for r in results if r["reward"] < 1]
    if args.max_num_failed_results is not None:
        failed_results = failed_results[: args.max_num_failed_results]
    original_results: List[OriginalResult] = []
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
    fault_assignment_results: List[FaultAssignmentResult] = fault_assignment_analysis(
        api, original_results, args.max_concurrency
    )
    fault_type_results: List[FaultTypeResult] = fault_type_analysis(
        api, original_results, args.max_concurrency
    )
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "fault_assignment.json"), "w") as f:
        json.dump([r.model_dump() for r in fault_assignment_results], f, indent=4)
    with open(os.path.join(args.output_dir, "fault_type.json"), "w") as f:
        json.dump([r.model_dump() for r in fault_type_results], f, indent=4)
