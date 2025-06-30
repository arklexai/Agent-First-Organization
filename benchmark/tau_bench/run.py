"""Benchmark runner for the TAU benchmark.

This module provides functionality for running and evaluating agents on the TAU benchmark,
including support for different environments (retail, airline), user strategies, and
model providers. It handles task execution, result collection, and metric calculation
for evaluating agent performance.

The module includes:
- Task execution and result collection
- Performance metric calculation (average reward, pass^k)
- Concurrent task execution with configurable concurrency
- Checkpointing and result persistence
- Environment and agent management
"""

import json
import multiprocessing
import os
import random
import traceback
from concurrent.futures import ThreadPoolExecutor
from math import comb
from typing import Any

from litellm import provider_list

from benchmark.tau_bench.agents.base import Agent
from benchmark.tau_bench.envs import get_env
from benchmark.tau_bench.envs.user import UserStrategy
from benchmark.tau_bench.tau_types import EnvRunResult, RunConfig


def run(config: RunConfig) -> list[EnvRunResult]:
    """Run the TAU benchmark with the specified configuration.

    This function executes the benchmark tasks according to the provided configuration,
    including environment setup, task execution, and result collection. It supports
    multiple trials, task shuffling, and concurrent execution. Results are saved to
    a checkpoint file for persistence and analysis.

    Args:
        config (RunConfig): Configuration for the benchmark run, including environment,
            user strategy, model provider, and task settings.

    Returns:
        List[EnvRunResult]: List of results from all task executions.

    Raises:
        AssertionError: If the environment, user model provider, task split, or
            user strategy is invalid.
    """
    assert config.env in ["retail", "airline"], (
        "Only retail and airline envs are supported"
    )
    assert config.user_model_provider in provider_list, "Invalid user model provider"
    assert config.task_split in ["train", "test", "dev"], "Invalid task split"
    assert config.user_strategy in [item.value for item in UserStrategy], (
        "Invalid user strategy"
    )

    random.seed(config.seed)
    ckpt_path: str = f"{config.output_dir}/tau_bench_evaluation.json"
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    print(f"Loading user with strategy: {config.user_strategy}")
    env: Any = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
    )
    agent: Agent = agent_factory(config=config)
    end_index: int = (
        len(env.tasks)
        if config.end_index == -1
        else min(config.end_index, len(env.tasks))
    )
    results: list[EnvRunResult] = []
    lock: multiprocessing.Lock = multiprocessing.Lock()
    if config.task_ids and len(config.task_ids) > 0:
        print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
    else:
        print(
            f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})"
        )
    for i in range(config.num_trials):
        if config.task_ids and len(config.task_ids) > 0:
            idxs: list[int] = config.task_ids
        else:
            idxs: list[int] = list(range(config.start_index, end_index))
        if config.shuffle:
            random.shuffle(idxs)

        def _run(idx: int, trial_num: int = i) -> EnvRunResult:
            isolated_env: Any = get_env(
                config.env,
                user_strategy=config.user_strategy,
                user_model=config.user_model,
                task_split=config.task_split,
                user_provider=config.user_model_provider,
                task_index=idx,
            )

            print(f"Running task {idx}")
            try:
                res: Any = agent.solve(
                    env=isolated_env,
                    task_index=idx,
                )
                result: EnvRunResult = EnvRunResult(
                    task_id=idx,
                    reward=res.reward,
                    info=res.info,
                    traj=res.messages,
                    trial=trial_num,
                )
            except Exception as e:
                result: EnvRunResult = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=trial_num,
                )
            print(
                "✅" if result.reward == 1 else "❌",
                f"task_id={idx}",
                result.info,
            )
            print("-----")
            with lock:
                data: list[Any] = []
                with open(ckpt_path, "w") as f:
                    json.dump(data + [result.model_dump()], f, indent=2)
            return result

        with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
            res: list[EnvRunResult] = list(executor.map(_run, idxs))
            results.extend(res)

    avg_reward: float
    pass_hat_ks: dict[int, float]
    avg_reward, pass_hat_ks = get_metrics(results)
    display_metrics(avg_reward, pass_hat_ks)

    tasks_res: list[dict[str, Any]] = [result.model_dump() for result in results]
    with open(ckpt_path, "w") as f:
        tau_bench_evaluation: dict[str, Any] = {
            "average_reward": avg_reward,
            "pass_k": pass_hat_ks,
            "task_results": tasks_res,
        }
        json.dump(tau_bench_evaluation, f, indent=2)
        print(f"\n📄 Results saved to {ckpt_path}\n")
    return results


def agent_factory(config: RunConfig) -> Agent:
    """Create an agent instance based on the configuration.

    This function creates and returns an appropriate agent instance for the benchmark
    based on the provided configuration. Currently, it creates an AgentFirstOrg
    instance with the specified taskgraph directory.

    Args:
        config (RunConfig): Configuration for the agent, including taskgraph directory.

    Returns:
        Agent: An instance of the appropriate agent class.
    """
    from benchmark.tau_bench.agents.agent_first_org import AgentFirstOrg

    return AgentFirstOrg(taskgraph_dir=config.taskgraph_dir)


def get_metrics(results: list[EnvRunResult]) -> tuple[float, dict[int, float]]:
    """Calculate performance metrics from benchmark results.

    This function calculates the average reward and pass^k metrics from the benchmark
    results. The pass^k metric represents the probability of passing k trials, which
    is calculated using the binomial coefficient formula from the TAU benchmark paper.

    Args:
        results (List[EnvRunResult]): List of results from task executions.

    Returns:
        Tuple[float, Dict[int, float]]: A tuple containing:
            - float: The average reward across all tasks
            - Dict[int, float]: A dictionary mapping k to pass^k values
    """

    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials: int = len({r.trial for r in results})
    rewards: list[float] = [r.reward for r in results]
    avg_reward: float = sum(rewards) / len(rewards)
    # c from https://arxiv.org/pdf/2406.12045
    c_per_task_id: dict[int, int] = {}
    for result in results:
        if result.task_id not in c_per_task_id:
            c_per_task_id[result.task_id] = 1 if is_successful(result.reward) else 0
        else:
            c_per_task_id[result.task_id] += 1 if is_successful(result.reward) else 0
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k: float = 0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    return avg_reward, pass_hat_ks


def display_metrics(avg_reward: float, pass_hat_ks: dict[int, float]) -> None:
    """Display benchmark performance metrics.

    This function prints the average reward and pass^k metrics in a formatted way,
    providing a clear overview of the benchmark performance.

    Args:
        avg_reward (float): The average reward across all tasks.
        pass_hat_ks (Dict[int, float]): Dictionary mapping k to pass^k values.
    """
    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")
