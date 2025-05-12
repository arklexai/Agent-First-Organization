import os
import json
import argparse
import time
import logging
import importlib
from dotenv import load_dotenv
from pprint import pprint

from arklex.utils.utils import init_logger
from arklex.orchestrator.orchestrator import AgentOrg
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import LLM_PROVIDERS
from arklex.env.env import Env
from arklex.env.types import ResourceType

load_dotenv()


def pprint_with_color(data, color_code="\033[34m"):  # Default to blue
    print(color_code, end="")  # Set the color
    pprint(data)
    print("\033[0m", end="")


def get_api_bot_response(config, history, user_text, parameters, env):
    data = {"text": user_text, 'chat_history': history, 'parameters': parameters}
    orchestrator = AgentOrg(config=config, env=env)
    result = orchestrator.get_response(data)

    return result['answer'], result['parameters'], result['human_in_the_loop']


def _load_resources(platform, resource_type):
    try:
        resources = getattr(importlib.import_module(
            f"arklex.env.{platform}.resources"), "RESOURCES", [])
        return [
            r for r in resources if r.type == resource_type
        ]
    except (ImportError, AttributeError) as e:
        print(
            f"Error loading {resource_type.value} resources for '{platform}': {e}")
        return []


def load_workers(platform):
    workers = _load_resources(platform, ResourceType.WORKER)
    return [worker.to_dict() for worker in workers]


def load_tools(platform, fixed_arguments):
    if fixed_arguments is None:
        print(f"Error: fixed_arguments not provided for {platform}")
        return []
    tools = _load_resources(platform, ResourceType.TOOL)
    return [{**tool.to_dict(), 'fixed_args': fixed_arguments} for tool in tools]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default="./examples/test")
    parser.add_argument('--model', type=str,
                        default=MODEL["model_type_or_path"])
    parser.add_argument('--llm-provider', type=str,
                        default=MODEL["llm_provider"], choices=LLM_PROVIDERS)
    parser.add_argument('--log-level', type=str, default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()
    os.environ["DATA_DIR"] = args.input_dir
    model = {
        "model_type_or_path": args.model,
        "llm_provider": args.llm_provider
    }
    log_level = getattr(logging, args.log_level.upper(), logging.WARNING)
    logger = init_logger(log_level=log_level, filename=os.path.join(
        os.path.dirname(__file__), "logs", "arklex.log"))

    # Initialize env
    config = json.load(open(os.path.join(args.input_dir, "taskgraph.json")))
    config["model"] = model

    platform = config.get("platform")
    tools = load_tools(platform, config.get("fixed_args")
                       ) if platform else config.get("tools", [])
    workers = load_workers(platform) if platform else config.get("workers", [])

    env = Env(
        tools=tools,
        workers=workers,
        slotsfillapi=config["slotfillapi"]
    )

    history = []
    params = {}
    user_prefix = "user"
    worker_prefix = "assistant"
    for node in config['nodes']:
        if node[1].get("type", "") == 'start':
            start_message = node[1]['attribute']["value"]
            break
    history.append({"role": worker_prefix, "content": start_message})
    pprint_with_color(f"Bot: {start_message}")

    while True:
        user_text = input("You: ")
        if user_text.lower() == "quit":
            break
        start_time = time.time()
        output, params, hitl = get_api_bot_response(
            config, history, user_text, params, env)
        history.append({"role": user_prefix, "content": user_text})
        history.append({"role": worker_prefix, "content": output})
        print(f"getAPIBotResponse Time: {time.time() - start_time}")
        pprint_with_color(f"Bot: {output}")
