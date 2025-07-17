"""Command-line interface for running the Arklex framework.

This module provides a command-line interface for running the Arklex framework,
allowing users to interact with the system through a text-based interface. It
handles user input, processes it through the orchestrator, and displays the
responses. The module also manages configuration loading, environment setup,
and logging.
"""

import argparse
import json
import os
import time
from pprint import pprint
from typing import Any

from dotenv import load_dotenv

from arklex.env.env import Environment
from arklex.orchestrator.NLU.services.model_service import ModelService
from arklex.orchestrator.orchestrator import AgentOrg
from arklex.utils.logging_utils import LogContext
from arklex.utils.model_provider_config import LLM_PROVIDERS
from arklex.utils.provider_utils import get_provider_config

load_dotenv()

log_context = LogContext(__name__)


def pprint_with_color(
    data: object, color_code: str = "\033[34m"
) -> None:  # Default to blue
    """Print data with a specified color.

    This function prints the provided data with the specified color code.

    Args:
        data (object): The data to be printed.
        color_code (str, optional): The color code to use for printing. Defaults to blue.
    """
    print(color_code, end="")  # Set the color
    pprint(data)
    print("\033[0m", end="")


def get_api_bot_response(
    config: dict[str, Any],
    history: list[dict[str, str]],
    user_text: str,
    parameters: dict[str, Any],
    env: Environment,
) -> tuple[str, dict[str, Any], bool]:
    """Get a response from the bot based on the provided input.

    This function processes the user input and chat history through the orchestrator
    to generate a response from the bot.

    Args:
        config (Dict[str, Any]): Configuration dictionary for the orchestrator.
        history (List[Dict[str, str]]): List of previous chat messages.
        user_text (str): The current user message.
        parameters (Dict[str, Any]): Additional parameters for the bot response.
        env (Env): Environment object containing tools and workers.

    Returns:
        Tuple[str, Dict[str, Any], bool]: A tuple containing the bot's response, updated parameters, and a boolean indicating if human intervention is required.
    """
    data: dict[str, Any] = {
        "text": user_text,
        "chat_history": history,
        "parameters": parameters,
    }
    orchestrator = AgentOrg(config=config, env=env)
    result: dict[str, Any] = orchestrator.get_response(data)

    return result["answer"], result["parameters"], result["human_in_the_loop"]


if __name__ == "__main__":
    # Set up command line argument parsing for configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="./examples/test")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use (e.g., gpt-4o, claude-3-5-haiku-20241022, gemini-1.5-flash)",
    )
    parser.add_argument(
        "--llm_provider",
        type=str,
        default="openai",
        choices=LLM_PROVIDERS,
        help="LLM provider to use (openai, anthropic, google, huggingface)",
    )
    args = parser.parse_args()

    # Set up environment variables and model configuration
    # Use absolute path to ensure RAG files can be found
    input_dir_abs = os.path.abspath(args.input_dir)
    os.environ["DATA_DIR"] = input_dir_abs

    # Get complete provider configuration
    model = get_provider_config(args.llm_provider, args.model)

    # Load task graph configuration and initialize environment
    with open(os.path.join(input_dir_abs, "taskgraph.json")) as f:
        config: dict[str, Any] = json.load(f)
    config["model"] = model

    # Initialize model service
    model_service = ModelService(model)

    # Initialize environment with model service
    env = Environment(
        tools=config.get("tools", []),
        workers=config.get("workers", []),
        agents=config.get("agents", []),
        slot_fill_api=config["slotfillapi"],
        planner_enabled=True,
        model_service=model_service,  # Pass model service to environment
    )

    # Initialize chat history and parameters
    history: list[dict[str, str]] = []
    params: dict[str, Any] = {}
    user_prefix: str = "user"
    worker_prefix: str = "assistant"

    # Find and display the initial message from the start node
    for node in config["nodes"]:
        if node[1].get("type", "") == "start":
            start_message: str = node[1]["attribute"]["value"]
            break
    history.append({"role": worker_prefix, "content": start_message})
    pprint_with_color(f"Bot: {start_message}")

    # Main conversation loop
    while True:
        user_text: str = input("You: ")
        if user_text.lower() == "quit":
            break
        start_time: float = time.time()
        output, params, hitl = get_api_bot_response(
            config, history, user_text, params, env
        )
        history.append({"role": user_prefix, "content": user_text})
        history.append({"role": worker_prefix, "content": output})
        print(f"getAPIBotResponse Time: {time.time() - start_time}")
        pprint_with_color(f"Bot: {output}")
