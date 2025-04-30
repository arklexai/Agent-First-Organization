import os
import logging
from typing import Dict
import argparse
import uvicorn

from fastapi import FastAPI

from arklex.utils.utils import init_logger
from arklex.env.env import Env
from arklex.orchestrator.orchestrator import AgentOrg
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import LLM_PROVIDERS


logger = logging.getLogger(__name__)
app = FastAPI()


def get_api_bot_response(args, history, user_text, parameters, env):
    data = {"text": user_text, 'chat_history': history, 'parameters': parameters}
    orchestrator = AgentOrg(config=os.path.join(args.input_dir, "taskgraph.json"), env=env)
    result = orchestrator.get_response(data)

    return result['answer'], result['parameters']


@app.post("/eval/chat")
def predict(data: Dict):
    history = data['history']
    params = data['parameters']
    workers = data['workers']
    tools = data['tools']
    user_text = history[-1]['content']

    env = Env(
        tools = tools,
        workers = workers,
        slotsfillapi = ""
    )
    answer, params = get_api_bot_response(args, history[:-1], user_text, params, env)
    return {"answer": answer, "parameters": params}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start FastAPI with custom config.")
    parser.add_argument('--input-dir', type=str, default="./examples/test")
    parser.add_argument('--model', type=str, default=MODEL["model_type_or_path"])
    parser.add_argument( '--llm-provider',type=str,default=MODEL["llm_provider"],choices=LLM_PROVIDERS)
    parser.add_argument('--port', type=int, default=8000, help="Port to run the FastAPI app")
    parser.add_argument('--log-level', type=str, default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    
    args = parser.parse_args()
    os.environ["DATA_DIR"] = args.input_dir
    MODEL["model_type_or_path"] = args.model
    MODEL["llm_provider"] = args.llm_provider

    log_level = getattr(logging, args.log_level.upper(), logging.WARNING)
    logger = init_logger(log_level=log_level, filename=os.path.join(os.path.dirname(__file__), "logs", "arklex.log"))

    #run server
    uvicorn.run(app, host="0.0.0.0", port=args.port)