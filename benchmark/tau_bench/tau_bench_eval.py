"""TAU benchmark evaluation module.

This module provides functionality for evaluating agents on the TAU benchmark,
including configuration generation, task graph creation, and evaluation execution.
It supports different environments, user strategies, and model providers, with
configurable parameters for trials, task selection, and concurrency.

The module includes:
- Tool mapping and initialization
- Configuration generation for the retail environment
- Task graph generation and management
- Evaluation execution with customizable parameters
"""

import argparse
import json
import logging
import os
import sys
import uuid
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from arklex.env.env import DefaultResourceInitializer
from arklex.env.tools.tools import Tool
from arklex.orchestrator.generator.generator import Generator
from arklex.utils.logging_utils import LogContext
from arklex.utils.model_config import MODEL
from benchmark.tau_bench.envs.retail.data import load_data
from benchmark.tau_bench.envs.retail.tools import ALL_TOOLS
from benchmark.tau_bench.run import run
from benchmark.tau_bench.tau_types import RunConfig

root_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
load_dotenv()

NLUAPI_ADDR: str = ""
SLOTFILLAPI_ADDR: str = ""

tool_name_class_map: dict[str, Any] = {}

log_context = LogContext(__name__)


def get_tool_name_class_map() -> dict[str, Any]:
    """Create a mapping of tool names to their class implementations.

    This function creates a dictionary that maps tool names to their corresponding
    class implementations from the ALL_TOOLS list. This mapping is used for tool
    initialization and execution during the benchmark evaluation.

    Returns:
        Dict[str, Any]: A dictionary mapping tool names to their class implementations.
    """
    tool_map: dict[str, Any] = {}
    for tool in ALL_TOOLS:
        name: str = tool.get_info()["function"]["name"]
        tool_map[name] = tool
    return tool_map


class TauBenchResourceInitializer(DefaultResourceInitializer):
    """Resource initializer for TAU benchmark tools.

    This class extends the default resource initializer to handle TAU benchmark
    specific tool initialization, including parameter mapping and slot creation.
    It manages the creation and configuration of tools used in the benchmark
    evaluation process.
    """

    @staticmethod
    def init_tools(tools: dict[str, Any]) -> dict[str, Any]:
        """Initialize tools for the TAU benchmark.

        This method creates tool instances with appropriate parameters, slots,
        and descriptions based on the provided tool information. It handles the
        creation of tool functions, parameter slots, and output configurations.

        Args:
            tools (Dict[str, Any]): Dictionary of tool information to initialize.

        Returns:
            Dict[str, Any]: Dictionary of initialized tools with their configurations.
        """
        tool_name_class_map: dict[str, Any] = get_tool_name_class_map()
        tool_registry: dict[str, Any] = {}

        def tool_lambda(val: object) -> object:
            return lambda: val

        for tool_id, tool_info in tools.items():
            tool_name: str = tool_info["name"]
            tool_original_class: Any = tool_name_class_map[tool_name]
            tool_func: Any = tool_original_class.invoke
            tool_key: str = tool_name
            tool_desc: str = tool_info["description"]
            params: dict[str, Any] = tool_original_class.get_info()["function"][
                "parameters"
            ]
            tool_slots: list[dict[str, Any]] = []

            for param_name, param_info in params["properties"].items():
                slot: dict[str, Any] = {}
                slot["name"] = param_name
                slot["type"] = param_info["type"]
                slot["items"] = param_info.get("items", {})
                slot["description"] = param_info["description"]
                prompt_param_name: str = param_name.replace("_", " ")
                slot["prompt"] = (
                    f"In order to proceed, please provide the {prompt_param_name}"
                )
                slot["required"] = param_name in params["required"]
                tool_slots.append(slot)
            tool_output: list[Any] = []

            tool: Any = tool_lambda(
                Tool(
                    tool_func,
                    tool_key,
                    tool_desc,
                    tool_slots,
                    tool_output,
                    isResponse=False,
                )
            )

            tool_registry[tool_id] = {
                "name": tool_name,
                "description": tool_desc,
                "execute": tool,
                "fixed_args": {"data": load_data()},
            }
        return tool_registry


def generate_tau_bench_config(output_dir: str) -> None:
    """Generate TAU benchmark configuration file.

    This function creates a configuration file for the TAU benchmark with
    predefined settings for the retail environment. The configuration includes
    tools, workers, objectives, and other necessary parameters for running
    the benchmark evaluation.

    Args:
        output_dir (str): Directory where the configuration file will be saved.
    """
    retain_tools: list[Any] = ALL_TOOLS
    tools: dict[str, Any] = {}
    for tool in retain_tools:
        tool_id: str = str(uuid.uuid1())
        tools[tool_id] = {}
        tools[tool_id]["name"] = tool.get_info()["function"]["name"]
        tools[tool_id]["description"] = tool.get_info()["function"]["description"]
    retail_config: dict[str, Any] = {
        "role": "Retail Agent",
        "user_objective": "The core goal of the agent is to assist a single, authenticated user per conversation in managing their retail orders—resolving any questions, cancellations, modifications, exchanges, or returns—while strictly following the rules and confirmation steps set by the retail policy.",
        "builder_objective": "Users want a convenient, reliable way to manage their orders—whether that means updating their shipping address, switching payment methods, or returning/exchanging items they've received. They come to the Retail Agent because they need to quickly resolve questions about their orders, get real-time updates on shipping statuses, and handle any necessary cancellations or modifications with confidence that every action is confirmed and secure.",
        "domain": "retail",
        "intro": "Welcome to the Retail Agent service. By confirming your identity, I can help you with detailed information on your orders, profile, and products. If you need to cancel or modify any pending orders, change your shipping address, payment method, or exchange/return delivered items, I can guide you through it step by step. I will always ask you to confirm before making any changes to ensure accuracy and security.",
        "task_docs": [
            {
                "source": "https://raw.githubusercontent.com/sierra-research/tau-bench/refs/heads/main/tau_bench/envs/retail/wiki.md",
                "num": 20,
            }
        ],
        "rag_docs": [],
        "tasks": [],
        "workers": [
            {
                "id": "26bb6634-3bee-417d-ad75-23269ac17bc3",
                "name": "MessageWorker",
                "path": "message_worker.py",
            },
        ],
        "tools": tools,
        "tool_initialization": False,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(retail_config, f, indent=4)


def generate_taskgraph(config_file: str, output_dir: str) -> None:
    """Generate task graph for the TAU benchmark.

    This function creates a task graph using the provided configuration and
    saves it to the specified output directory. It also updates the task graph
    with API URLs for NLU and slot filling services.

    Args:
        config_file (str): Path to the configuration file.
        output_dir (str): Directory where the task graph will be saved.
    """
    model: ChatOpenAI = ChatOpenAI(model=MODEL["model_type_or_path"], timeout=30000)
    resource_initializer: TauBenchResourceInitializer = TauBenchResourceInitializer()
    with open(config_file) as f:
        config: dict[str, Any] = json.load(f)
    generator: Generator = Generator(config, model, output_dir, resource_initializer)
    taskgraph: dict[str, Any] = generator.generate()
    taskgraph_filepath: str = generator.save_task_graph(taskgraph)
    # Update the task graph with the API URLs
    with open(os.path.join(root_dir, taskgraph_filepath)) as f:
        task_graph: dict[str, Any] = json.load(f)
    task_graph["nluapi"] = NLUAPI_ADDR
    task_graph["slotfillapi"] = SLOTFILLAPI_ADDR
    with open(taskgraph_filepath, "w") as f:
        json.dump(task_graph, f, indent=4)


def run_tau_bench_eval(
    taskgraph_dir: str,
    output_dir: str,
    num_trials: int,
    task_ids: list[int],
    env: str,
    task_split: str = "test",
    user_strategy: str = "llm",
    max_concurrency: int = 10,
) -> None:
    """Run TAU benchmark evaluation.

    This function executes the TAU benchmark evaluation with the specified
    parameters. It handles the configuration and execution of the evaluation
    process, including task selection, trial management, and result collection.

    Args:
        taskgraph_dir (str): Directory containing the task graph.
        output_dir (str): Directory for evaluation output.
        num_trials (int): Number of trials to run.
        task_ids (List[int]): List of task IDs to evaluate.
        env (str): Environment to use (e.g., "retail").
        task_split (str, optional): Task split to use. Defaults to "test".
        user_strategy (str, optional): User strategy to use. Defaults to "llm".
        max_concurrency (int, optional): Maximum number of concurrent tasks.
            Defaults to 10.
    """
    start_index: int = 0
    end_index: int = -1
    seed: int = 10
    shuffle: int = 0

    config: RunConfig = RunConfig(
        user_model_provider="openai",
        user_model="gpt-4o",
        num_trials=num_trials,
        env=env,
        task_split=task_split,
        start_index=start_index,
        end_index=end_index,
        task_ids=task_ids,
        output_dir=output_dir,
        max_concurrency=max_concurrency,
        seed=seed,
        shuffle=shuffle,
        user_strategy=user_strategy,
        taskgraph_dir=taskgraph_dir,
    )
    run(config)


if __name__ == "__main__":
    """
        Provide --output-dir
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./examples/tau_bench")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--env", type=str, default="retail", choices=["retail"])

    import random

    random.seed(42)
    random_list: list[int] = random.sample(range(118), 10)
    print(f"Running Tau Bench on tasks {random_list}")
    parser.add_argument("--task-ids", type=list, default=random_list)
    # parser.add_argument('--task-ids', type=list, default=[1,2,3,4,5,6,7,8,9,10])

    parser.add_argument(
        "--model_api", type=str, default="http://127.0.0.1:8000/eval/chat"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", help="Model to use for evaluation"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args: argparse.Namespace = parser.parse_args()

    assert args.output_dir is not None, "Output dir must be provided"

    os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "temp"), exist_ok=True)

    temp_output_dir: str = os.path.join(args.output_dir, "temp")
    eval_output_dir: str = os.path.join(args.output_dir, "eval")

    MODEL["model_type_or_path"] = args.model
    log_level: int = getattr(logging, args.log_level.upper(), logging.INFO)
    logger_instance: logging.Logger = log_context.get_logger(
        log_level=log_level,
        filename=os.path.join(root_dir, "logs", "tau_bench_eval.log"),
    )

    generate_tau_bench_config(temp_output_dir)
    config_file: str = os.path.join(temp_output_dir, "config.json")
    generate_taskgraph(config_file, temp_output_dir)
    print("taskgraph done")
    run_tau_bench_eval(
        taskgraph_dir=temp_output_dir,
        output_dir=eval_output_dir,
        num_trials=args.num_trials,
        env=args.env,
        task_ids=args.task_ids,
    )
