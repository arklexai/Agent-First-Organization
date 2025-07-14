"""Evaluation utilities for the Arklex framework.

This module provides functionality for evaluating the performance of the Arklex
framework through conversation simulation and metrics extraction. It includes
utilities for simulating first-pass and second-pass conversations, extracting
task completion metrics, and generating labeled conversation data for analysis.
"""

import argparse
import json
import os
import re
from typing import Any
from urllib.parse import urlparse

import requests

from arklex.evaluation.chatgpt_utils import create_client
from arklex.evaluation.extract_conversation_info import extract_task_completion_metrics
from arklex.evaluation.simulate_first_pass_convos import simulate_conversations
from arklex.evaluation.simulate_second_pass_convos import get_labeled_convos
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import LLM_PROVIDERS
from arklex.utils.provider_utils import (
    get_api_key_for_provider,
    get_endpoint_for_provider,
)


def validate_model_api(model_api: str) -> None:
    """Validate the model_api URL parameter.

    Args:
        model_api (str): The model API URL to validate.

    Raises:
        ValueError: If the model_api is not a valid URL or contains placeholder values.
    """
    if not model_api:
        raise ValueError("model_api parameter is required")

    # Check for common placeholder values
    placeholder_patterns = [
        r"your-api-endpoint",
        r"example\.com",
        r"placeholder",
        r"api\.example",
    ]

    for pattern in placeholder_patterns:
        if re.search(pattern, model_api, re.IGNORECASE):
            raise ValueError(
                f"Invalid model_api URL: '{model_api}'. "
                "Please provide a valid API endpoint URL.\n\n"
                "To set up the evaluation:\n"
                "1. First, start the model API server:\n"
                "   python model_api.py --input-dir ./examples/customer_service\n"
                "2. Then run the evaluation with the correct API URL:\n"
                "   python eval.py --model_api http://127.0.0.1:8000/eval/chat ...\n\n"
                "If you're running the model API locally, use:\n"
                "'http://127.0.0.1:8000/eval/chat' or 'http://localhost:8000/eval/chat'"
            )

    # Basic URL validation
    try:
        parsed = urlparse(model_api)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("URL must have a valid scheme and host")
    except Exception as e:
        raise ValueError(f"Invalid URL format: {model_api}. Error: {str(e)}") from e

    # Try to connect to the API endpoint to verify it's reachable
    # Since this is a POST-only endpoint, we'll test with a minimal POST request
    try:
        # Test with a minimal POST request to check connectivity
        test_data = {
            "history": [{"role": "user", "content": "test"}],
            "parameters": {},
            "workers": [],
            "tools": [],
        }
        response = requests.post(model_api, json=test_data, timeout=5)
        if response.status_code == 200:
            # Success - endpoint is working
            pass
        elif response.status_code >= 400:
            print(f"Warning: API endpoint returned status code {response.status_code}")
    except requests.exceptions.ConnectionError as e:
        raise ValueError(
            f"Cannot connect to API endpoint: {model_api}\n"
            "Please make sure the model API server is running.\n"
            "Start it with: python model_api.py --input-dir ./examples/customer_service"
        ) from e
    except requests.exceptions.Timeout as e:
        raise ValueError(
            f"Timeout connecting to API endpoint: {model_api}\n"
            "Please check if the server is running and accessible."
        ) from e
    except Exception as e:
        print(f"Warning: Could not verify API endpoint connectivity: {str(e)}")


def evaluate(
    config: dict[str, Any],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]
]:
    """Evaluate the performance of the Arklex framework based on the provided configuration.

    This function simulates conversations and extracts metrics based on the task specified in the config.
    It supports first-pass and second-pass evaluations, and returns the results of the evaluation.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing task, model API, model parameters,
                                 synthetic data parameters, and other settings.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
            A tuple containing the first-pass data, labeled conversations, goal metrics, and goals.
    """
    task: str = config["task"]
    model_api: str = config["model_api"]
    model_params: dict[str, Any] = config["model_params"]
    synthetic_data_params: dict[str, Any] = config["synthetic_data_params"]
    bot_goal: str | None = config.get("builder_objective")
    bot_goal = None if bot_goal == "" else bot_goal

    print(f"Starting evaluation with task: {task}")
    print(f"Model API: {model_api}")
    print(f"Synthetic data params: {synthetic_data_params}")

    # Always perform first pass simulation
    print("Starting first pass simulation...")
    first_pass_data, goals = simulate_conversations(
        model_api, model_params, synthetic_data_params, config
    )
    print(
        f"First pass simulation completed. Generated {len(first_pass_data)} conversations."
    )

    print("Extracting goal completion metrics...")
    goal_metrics = extract_task_completion_metrics(
        first_pass_data, config["client"], bot_goal
    )
    print("Goal metrics extraction completed.")

    # Perform second pass only if task is "all"
    if task == "all":
        print("Starting second pass simulation...")
        labeled_convos = get_labeled_convos(
            first_pass_data, model_api, synthetic_data_params, model_params, config
        )
        print(
            f"Second pass simulation completed. Generated {len(labeled_convos)} labeled conversations."
        )
    else:
        labeled_convos = []
        print("Skipping second pass simulation (task != 'all')")

    print("Evaluation completed successfully!")
    return first_pass_data, labeled_convos, goal_metrics, goals


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Evaluate the performance of the Arklex framework through conversation simulation.",
        epilog="""
Example usage:
1. Start the model API server:
   python model_api.py --input-dir ./examples/customer_service

2. Run the evaluation:
   python eval.py \\
     --model_api http://127.0.0.1:8000/eval/chat \\
     --config examples/customer_service/customer_service_config.json \\
     --documents_dir examples/customer_service \\
     --task first_pass
        """,
    )
    parser.add_argument(
        "--model_api",
        type=str,
        help="URL of the API endpoint for the dialogue model (e.g., http://127.0.0.1:8000/eval/chat)",
    )
    parser.add_argument(
        "--model_params",
        type=str,
        default="{}",
        help="JSON string for model parameters",
    )
    parser.add_argument("--num_convos", type=int, default=5)
    parser.add_argument("--num_goals", type=int, default=5)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--documents_dir", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", help="Model to use for evaluation"
    )
    parser.add_argument(
        "--llm_provider",
        type=str,
        default="openai",
        choices=LLM_PROVIDERS,
        help="LLM provider to use",
    )
    parser.add_argument(
        "--customer_type", type=str, default=None, choices=["b2b", "b2c"]
    )
    parser.add_argument(
        "--task",
        type=str,
        default="first_pass",
        choices=["first_pass", "simulate_conv_only", "all"],
    )
    parser.add_argument(
        "--user_attributes", type=str, default="arklex/evaluation/user_attributes.json"
    )
    parser.add_argument("--custom_profile", action="store_true")
    parser.add_argument("--system_inputs", action="store_true")
    parser.add_argument("--data_file", type=str, default=None)
    args: argparse.Namespace = parser.parse_args()

    # Update model configuration with proper provider settings
    MODEL["model_type_or_path"] = args.model
    MODEL["llm_provider"] = args.llm_provider
    MODEL["api_key"] = get_api_key_for_provider(args.llm_provider)
    MODEL["endpoint"] = get_endpoint_for_provider(args.llm_provider)

    client: Any = create_client()

    assert args.model_api is not None, "Model api must be provided"
    assert args.config is not None, "Config file must be provided"
    assert args.user_attributes is not None, "User attribute file must be provided"
    assert args.documents_dir is not None, "Documents directory must be provided"

    # Validate that required files exist first (to avoid unnecessary network calls)
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not os.path.exists(args.user_attributes):
        raise FileNotFoundError(
            f"User attributes file not found: {args.user_attributes}"
        )
    if not os.path.exists(args.documents_dir):
        raise FileNotFoundError(f"Documents directory not found: {args.documents_dir}")

    # Validate the model_api URL
    validate_model_api(args.model_api)

    if not args.output_dir:
        args.output_dir = os.path.join(args.documents_dir, "eval")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        config: dict[str, Any] = json.load(f)
    with open(args.user_attributes) as f:
        user_attributes: dict[str, Any] = json.load(f)
    # if args.testset:
    #     testset = json.load(open(args.testset))
    # else:
    #     testset = {}

    # Parse model_params from JSON string
    try:
        model_params = json.loads(args.model_params) if args.model_params else {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in --model_params: {args.model_params}")
        model_params = {}

    config["model_api"] = args.model_api
    config["documents_dir"] = args.documents_dir
    config["output_dir"] = args.output_dir
    config["model_params"] = model_params
    config["synthetic_data_params"] = {
        "num_convos": args.num_convos,
        "num_goals": args.num_goals,
        "max_turns": args.max_turns,
        "customer_type": args.customer_type,
        "data_file": args.data_file,
    }
    config["task"] = args.task
    config["user_attributes"] = user_attributes
    config["custom_profile"] = args.custom_profile
    config["system_inputs"] = args.system_inputs
    config["client"] = client

    first_pass_data: list[dict[str, Any]]
    final_convos: list[dict[str, Any]]
    goal_metrics: dict[str, Any]
    goals: list[dict[str, Any]]
    first_pass_data, final_convos, goal_metrics, goals = evaluate(config)

    with open(os.path.join(args.output_dir, "goals.json"), "w") as f:
        json.dump(goals, f, indent=4)

    with open(os.path.join(args.output_dir, "simulate_data.json"), "w") as f:
        json.dump(first_pass_data, f, indent=4)

    with open(os.path.join(args.output_dir, "labeled_data.json"), "w") as f:
        json.dump(final_convos, f, indent=4)

    with open(os.path.join(args.output_dir, "goal_completion.json"), "w") as f:
        json.dump(goal_metrics, f, indent=4)
