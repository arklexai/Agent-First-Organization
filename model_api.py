"""FastAPI server for the Arklex framework.

This module provides a FastAPI server implementation for the Arklex framework,
exposing endpoints for model evaluation and chat interactions. It handles
HTTP requests, processes them through the orchestrator, and returns the
responses. The module also manages server configuration, logging, and
environment setup.
"""

import argparse
import logging
import os
from typing import Any

import uvicorn
from fastapi import FastAPI

from arklex.env.env import Environment
from arklex.orchestrator.orchestrator import AgentOrg
from arklex.utils.logging_utils import LogContext
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import LLM_PROVIDERS
from arklex.utils.provider_utils import (
    get_api_key_for_provider,
    get_endpoint_for_provider,
)

log_context = LogContext(__name__)
app: FastAPI = FastAPI()

# Global variable to store command line arguments
args: argparse.Namespace = None


def get_api_bot_response(
    args: argparse.Namespace,
    history: list[dict[str, str]],
    user_text: str,
    parameters: dict[str, Any],
    env: Environment,
) -> tuple[str, dict[str, Any]]:
    """Get a response from the bot based on the provided input.

    This function processes the user input and chat history through the orchestrator
    to generate a response from the bot.

    Args:
        args (argparse.Namespace): Command-line arguments containing configuration settings.
        history (List[Dict[str, str]]): List of previous chat messages.
        user_text (str): The current user message.
        parameters (Dict[str, Any]): Additional parameters for the bot response.
        env (Environment): Environment object containing tools and workers.

    Returns:
        Tuple[str, Dict[str, Any]]: A tuple containing the bot's response and updated parameters.
    """
    data: dict[str, Any] = {
        "text": user_text,
        "chat_history": history,
        "parameters": parameters,
    }
    orchestrator: AgentOrg = AgentOrg(
        config=os.path.join(args.input_dir, "taskgraph.json"), env=env
    )
    result: dict[str, Any] = orchestrator.get_response(data)

    return result["answer"], result["parameters"]


@app.post("/eval/chat")
def predict(data: dict[str, Any]) -> dict[str, Any]:
    """Predict a response based on the provided chat data.

    This endpoint processes chat interactions through the Arklex framework. It takes a dictionary
    containing the chat history, parameters, and worker/tool configurations, and returns the
    bot's response along with updated parameters.

    The endpoint expects the following structure in the input data:
    {
        "history": List[Dict[str, str]],  # List of previous chat messages
        "parameters": Dict[str, Any],     # Current conversation parameters
        "workers": List[Dict[str, Any]],  # Worker configurations
        "tools": List[Dict[str, Any]]     # Tool configurations
    }

    Args:
        data (Dict[str, Any]): Dictionary containing:
            - history: List of previous chat messages
            - parameters: Current conversation parameters
            - workers: List of worker configurations
            - tools: List of tool configurations

    Returns:
        Dict[str, Any]: A dictionary containing:
            - answer: The bot's response text
            - parameters: Updated conversation parameters
    """
    global args

    # Extract conversation components from input data
    history: list[dict[str, str]] = data["history"]
    params: dict[str, Any] = data["parameters"]
    workers: list[dict[str, Any]] = data["workers"]
    tools: list[dict[str, Any]] = data["tools"]
    user_text: str = history[-1]["content"]

    # Initialize environment with provided workers and tools
    env: Environment = Environment(
        tools=tools, workers=workers, agents=[], slotsfillapi=""
    )

    # Get bot response using the orchestrator
    answer: str
    params: dict[str, Any]
    answer, params = get_api_bot_response(args, history[:-1], user_text, params, env)
    return {"answer": answer, "parameters": params}


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Start FastAPI with custom config."
    )
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
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the FastAPI app"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    args = parser.parse_args()
    os.environ["DATA_DIR"] = args.input_dir

    # Update model configuration with proper provider settings
    MODEL["model_type_or_path"] = args.model
    MODEL["llm_provider"] = args.llm_provider
    MODEL["api_key"] = get_api_key_for_provider(args.llm_provider)
    MODEL["endpoint"] = get_endpoint_for_provider(args.llm_provider)

    log_level: int = getattr(logging, args.log_level.upper(), logging.WARNING)

    # run server
    uvicorn.run(app, host="0.0.0.0", port=args.port)
