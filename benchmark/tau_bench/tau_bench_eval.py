import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
import json
import argparse
import logging
import subprocess
import subprocess
import uuid
import atexit
from dotenv import load_dotenv


from langchain_openai import ChatOpenAI

from arklex.orchestrator.orchestrator import AgentOrg
from arklex.utils.utils import init_logger
from arklex.utils.model_config import MODEL
from arklex.orchestrator.generator.generator import Generator
from arklex.env.env import DefaulResourceInitializer
from arklex.env.tools.tools import Tool

from benchmark.tau_bench.envs.retail.tools import ALL_TOOLS
from benchmark.tau_bench.envs import get_env
from benchmark.tau_bench.types import RunConfig
from benchmark.tau_bench.run import run
from benchmark.tau_bench.envs.retail.data import load_data

load_dotenv()

API_PORT = "55135"
NLUAPI_ADDR = f"http://localhost:{API_PORT}/nlu"
SLOTFILLAPI_ADDR = f"http://localhost:{API_PORT}/slotfill"

tool_name_class_map = {}

def get_tool_name_class_map():
    tool_map = {}
    for tool in ALL_TOOLS:
        name = tool.get_info()["function"]["name"]
        tool_map[name] = tool
    return tool_map

class TauBenchResourceInitializer(DefaulResourceInitializer):
    @staticmethod
    def init_tools(tools):
        tool_name_class_map = get_tool_name_class_map()
        tool_registry = {}
        def tool_lambda(val): return lambda: val
        for tool_id, tool_info in tools.items():
            tool_name = tool_info["name"]
            tool_original_class = tool_name_class_map[tool_name]
            tool_func = tool_original_class.invoke
            tool_key = tool_name
            tool_desc = tool_info["description"]
            params = tool_original_class.get_info()["function"]["parameters"]
            tool_slots = []
   
            for param_name, param_info in params["properties"].items():
                slot = {}
                slot["name"] = param_name
                slot["type"] = param_info["type"]
                slot["items"] = param_info.get("items", {})
                slot["description"] = param_info["description"]
                prompt_param_name = param_name.replace("_", " ")
                slot["prompt"] = f"In order to proceed, please provide the {prompt_param_name}"
                slot["required"] = param_name in params["required"]
                tool_slots.append(slot)
            tool_output = []
            
            tool = tool_lambda(Tool(tool_func, tool_key, tool_desc, tool_slots, tool_output))

            tool_registry[tool_id] = {
                "name": tool_name,
                "description": tool_desc,
                "execute": tool,
                "fixed_args": {"data": load_data()},
            }
        return tool_registry

def generate_tau_bench_config(output_dir):
    retain_tools = ALL_TOOLS
    tools = {}
    for tool in retain_tools:
        tool_id = str(uuid.uuid1())
        tools[tool_id] = {}
        tools[tool_id]["name"] = tool.get_info()["function"]["name"]
        tools[tool_id]["description"] = tool.get_info()["function"]["description"]
    retail_config = {
        "role": "Retail Agent",
        "user_objective": "The core goal of the agent is to assist a single, authenticated user per conversation in managing their retail orders—resolving any questions, cancellations, modifications, exchanges, or returns—while strictly following the rules and confirmation steps set by the retail policy.",
        "builder_objective": "Users want a convenient, reliable way to manage their orders—whether that means updating their shipping address, switching payment methods, or returning/exchanging items they've received. They come to the Retail Agent because they need to quickly resolve questions about their orders, get real-time updates on shipping statuses, and handle any necessary cancellations or modifications with confidence that every action is confirmed and secure.",
        "domain": "retail",
        "intro": "Welcome to the Retail Agent service. By confirming your identity, I can help you with detailed information on your orders, profile, and products. If you need to cancel or modify any pending orders, change your shipping address, payment method, or exchange/return delivered items, I can guide you through it step by step. I will always ask you to confirm before making any changes to ensure accuracy and security.",
        "task_docs": [{
            "source": "https://raw.githubusercontent.com/sierra-research/tau-bench/refs/heads/main/tau_bench/envs/retail/wiki.md",
            "num": 20
        }],
        "rag_docs": [],
        "tasks": [],
        "workers": [
            {"id": "26bb6634-3bee-417d-ad75-23269ac17bc3", "name": "MessageWorker", "path": "message_worker.py"},
        ],
        "tools": tools,
        "tool_initialization": False
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(retail_config, f, indent=4)

def generate_taskgraph(config_file, output_dir):
    model = ChatOpenAI(model=MODEL["model_type_or_path"], timeout=30000)
    resource_initializer = TauBenchResourceInitializer()
    generator = Generator(args, config_file, model, output_dir, resource_initializer)
    taskgraph_filepath = generator.generate()
    # Update the task graph with the API URLs
    task_graph = json.load(open(os.path.join(root_dir, taskgraph_filepath)))
    task_graph["nluapi"] = NLUAPI_ADDR
    task_graph["slotfillapi"] = SLOTFILLAPI_ADDR
    with open(taskgraph_filepath, "w") as f:
        json.dump(task_graph, f, indent=4)



nlu_process = None
def terminate_subprocess():
    """Terminate the FastAPI subprocess."""
    global nlu_process
    if nlu_process and nlu_process.poll() is None:  # Check if process is running
        logger.info(f"Terminating FastAPI process with PID: {nlu_process.pid}")
        nlu_process.terminate()  # Send SIGTERM
        nlu_process.wait()  # Ensure it stops
        logger.info("FastAPI process terminated.")


atexit.register(terminate_subprocess)

def start_apis():
    """Start the FastAPI subprocess and update task graph API URLs."""
    global nlu_process
    command = [
        "uvicorn",
        "arklex.orchestrator.NLU.api:app",  # Replace with proper import path
        "--port", API_PORT,
        "--host", "0.0.0.0",
        "--log-level", "warning"
    ]

    # Redirect FastAPI logs to a file
    with open(os.path.join(root_dir, "logs", "api.log"), "w") as log_file:
        nlu_process = subprocess.Popen(
            command,
            stdout=log_file,  # Redirect stdout to a log file
            stderr=subprocess.STDOUT,  # Redirect stderr to the same file
            start_new_session=True  # Run in a separate process group
        )
    logger.info(f"Started FastAPI process with PID: {nlu_process.pid}")

def run_tau_bench_eval(
        taskgraph_dir,
        output_dir,
        num_trials,
        task_ids,
        env,
        task_split="test",
        user_strategy="llm",
        max_concurrency=10,
):
 
    start_index = 0
    end_index = -1
    seed=10
    shuffle=0
    
    
    config = RunConfig(
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
        taskgraph_dir=taskgraph_dir
    )
    run(config)
    



if __name__ == "__main__":
    '''
        Provide --output-dir
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default="./examples/tau_bench")
    parser.add_argument('--num-trials', type=int, default=1)
    parser.add_argument('--env', type=str, default="retail", choices=["retail"])
    parser.add_argument('--task-ids', type=list, default=None)

    parser.add_argument('--model_api', type=str, default="http://127.0.0.1:8000/eval/chat")
    parser.add_argument('--model', type=str, default=MODEL["model_type_or_path"])
    parser.add_argument('--log-level', type=str, default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()
    
    
    assert args.output_dir is not None, "Output dir must be provided"

    os.makedirs(os.path.join(args.output_dir, 'eval'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'temp'), exist_ok=True)

    temp_output_dir = os.path.join(args.output_dir, 'temp')
    eval_output_dir = os.path.join(args.output_dir, 'eval')

    MODEL["model_type_or_path"] = args.model
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = init_logger(log_level=log_level, filename=os.path.join(root_dir, "logs", "tau_bench_eval.log"))
    
    # generate_tau_bench_config(temp_output_dir)
    # config_file = os.path.join(temp_output_dir, 'config.json')
    # generate_taskgraph(config_file, temp_output_dir)

    start_apis()
    run_tau_bench_eval(
        taskgraph_dir=temp_output_dir,
        output_dir=eval_output_dir,
        num_trials=args.num_trials,
        env=args.env,
        task_ids=args.task_ids
    )
    