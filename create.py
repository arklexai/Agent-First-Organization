"""Task graph generation and worker initialization for the Arklex framework.

This module provides functionality for generating task graphs and initializing
workers in the Arklex framework. It includes utilities for creating task graphs
based on configuration files, setting up workers like FaissRAGWorker and
DataBaseWorker, and managing the overall system initialization process.
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Set

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from arklex.utils.utils import init_logger
from arklex.orchestrator.generator.generator import Generator
from arklex.env.tools.RAG.build_rag import build_rag
from arklex.env.tools.database.build_database import build_database
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import LLM_PROVIDERS, PROVIDER_MAP

logger = init_logger(
    log_level=logging.INFO,
    filename=os.path.join(os.path.dirname(__file__), "logs", "arklex.log"),
)
load_dotenv()


def generate_taskgraph(args: argparse.Namespace) -> None:
    """Generate a task graph based on the provided configuration.

    This function initializes a language model, loads the configuration, and uses the Generator
    to create a task graph. It then saves the task graph to a file and updates it with API URLs.

    Args:
        args (argparse.Namespace): Command-line arguments containing configuration and output settings.
    """
    model = PROVIDER_MAP.get(MODEL["llm_provider"], ChatOpenAI)(
        model=MODEL["model_type_or_path"], timeout=30000
    )
    config: Dict[str, Any] = json.load(open(args.config))
    generator = Generator(config, model, args.output_dir)
    taskgraph = generator.generate()
    taskgraph_filepath: str = generator.save_task_graph(taskgraph)
    # Update the task graph with the API URLs
    task_graph: Dict[str, Any] = json.load(
        open(os.path.join(os.path.dirname(__file__), taskgraph_filepath))
    )
    task_graph["nluapi"] = ""
    task_graph["slotfillapi"] = ""
    with open(taskgraph_filepath, "w") as f:
        json.dump(task_graph, f, indent=4)


def init_worker(args: argparse.Namespace) -> None:
    """Initialize workers based on the provided configuration.

    This function initializes and sets up various workers required by the Arklex framework
    based on the configuration file. It supports different types of workers including:
    - FaissRAGWorker: For RAG (Retrieval-Augmented Generation) functionality
    - DataBaseWorker: For database operations including search, booking, and cancellation

    The function reads the worker configurations from the provided config file and
    initializes the appropriate workers based on their names. Each worker type
    requires specific setup procedures and dependencies.

    Args:
        args (argparse.Namespace): Command-line arguments containing:
            - config: Path to the configuration file
            - output_dir: Directory where worker data will be stored
    """
    # Load configuration from the specified file
    config: Dict[str, Any] = json.load(open(args.config))
    workers: List[Dict[str, Any]] = config["workers"]
    worker_names: Set[str] = set([worker["name"] for worker in workers])

    # Initialize FaissRAGWorker if specified in configuration
    if "FaissRAGWorker" in worker_names:
        logger.info("Initializing FaissRAGWorker...")
        build_rag(args.output_dir, config["rag_docs"])

    # Initialize DataBaseWorker and related workers if specified
    elif any(
        node in worker_names
        for node in (
            "DataBaseWorker",
            "search_show",
            "book_show",
            "check_booking",
            "cancel_booking",
        )
    ):
        logger.info("Initializing DataBaseWorker...")
        build_database(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./arklex/orchestrator/examples/customer_service_config.json",
    )
    parser.add_argument("--output-dir", type=str, default="./examples/test")
    parser.add_argument("--model", type=str, default=MODEL["model_type_or_path"])
    parser.add_argument(
        "--llm-provider", type=str, default=MODEL["llm_provider"], choices=LLM_PROVIDERS
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--task", type=str, choices=["gen_taskgraph", "init", "all"], default="all"
    )
    args = parser.parse_args()
    MODEL["model_type_or_path"] = args.model
    MODEL["llm_provider"] = args.llm_provider
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = init_logger(
        log_level=log_level,
        filename=os.path.join(os.path.dirname(__file__), "logs", "arklex.log"),
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if args.task == "all":
        generate_taskgraph(args)
        init_worker(args)
    elif args.task == "gen_taskgraph":
        generate_taskgraph(args)
    elif args.task == "init":
        init_worker(args)
