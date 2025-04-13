import os
import json
import argparse
import time
import logging
from dotenv import load_dotenv
from pprint import pprint

from arklex.utils.utils import init_logger
from arklex.orchestrator.orchestrator import AgentOrg
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import LLM_PROVIDERS
from arklex.env.env import Env

import agentops
from agentops.sdk.decorators import session, operation

load_dotenv()

def pprint_with_color(data, color_code="\033[34m"):  # Default to blue
    print(color_code, end="")  # Set the color
    pprint(data)
    print("\033[0m", end="")  

@session(name="Api Bot Response")
@operation(name="Fetch AgentOrg Response")
def get_api_bot_response(args, history, user_text, parameters, env):
    data = {"text": user_text, 'chat_history': history, 'parameters': parameters}
    orchestrator = AgentOrg(config=os.path.join(args.input_dir, "taskgraph.json"), env=env)
    result = orchestrator.get_response(data)

    return result['answer'], result['parameters'], result['human_in_the_loop']


if __name__ == "__main__":
    # AgentOps Session
    AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")
    agentops.init(api_key=AGENTOPS_API_KEY, tags=['run_script'])
    agentops.start_session(tags=['run_script'])

    # Track Agent Event
    agentops.track_agent("Agent Execution", metadata={"task": "Agent Execution"})
    agentops.logging.logger.info("run_script - run.py")

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default="./examples/test")
    parser.add_argument('--model', type=str, default=MODEL["model_type_or_path"])
    parser.add_argument( '--llm-provider',type=str,default=MODEL["llm_provider"],choices=LLM_PROVIDERS)
    parser.add_argument('--log-level', type=str, default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()
    os.environ["DATA_DIR"] = args.input_dir
    MODEL["model_type_or_path"] = args.model
    MODEL["llm_provider"] = args.llm_provider
    log_level = getattr(logging, args.log_level.upper(), logging.WARNING)
    logger = init_logger(log_level=log_level, filename=os.path.join(os.path.dirname(__file__), "logs", "arklex.log"))

    # Initialize env
    config = json.load(open(os.path.join(args.input_dir, "taskgraph.json")))
    env = Env(
        tools = config.get("tools", []),
        workers = config.get("workers", []),
        slotsfillapi = config["slotfillapi"]
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
        output, params, hitl = get_api_bot_response(args, history, user_text, params, env)
        history.append({"role": user_prefix, "content": user_text})
        history.append({"role": worker_prefix, "content": output})
        print(f"getAPIBotResponse Time: {time.time() - start_time}")
        pprint_with_color(f"Bot: {output}")