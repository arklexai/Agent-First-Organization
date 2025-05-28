# Copyright Sierra
import os
import json
from copy import deepcopy
from typing import Optional, Dict, Any, List, Tuple

from arklex.orchestrator.orchestrator import AgentOrg
from arklex.env.env import Env

from benchmark.tau_bench.agents.base import Agent
from benchmark.tau_bench.tau_types import (
    SolveResult,
    Action,
    RESPOND_ACTION_NAME,
    EnvResetResponse,
    EnvResponse,
)


class AgentFirstOrg(Agent):
    def __init__(self, taskgraph_dir: str) -> None:
        self.taskgraph_dir: str = taskgraph_dir
        self.taskgraph_path: str = os.path.join(self.taskgraph_dir, "taskgraph.json")
        from benchmark.tau_bench.tau_bench_eval import TauBenchResourceInitializer

        with open(self.taskgraph_path) as taskgraph:
            taskgraph: Dict[str, Any] = json.load(taskgraph)
            tau_bench_resource_initializer: TauBenchResourceInitializer = (
                TauBenchResourceInitializer()
            )
            self.env: Env = Env(
                tools=taskgraph.get("tools", []),
                workers=taskgraph.get("workers", []),
                slotsfillapi=taskgraph["slotfillapi"],
                resource_inizializer=tau_bench_resource_initializer,
            )

            self.start_message: Optional[str] = None
            for node in taskgraph["nodes"]:
                if node[1].get("type", "") == "start":
                    self.start_message = node[1]["attribute"]["value"]
                    break

    def get_api_bot_response(
        self, history: List[Dict[str, Any]], user_text: str, parameters: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        data: Dict[str, Any] = {
            "text": user_text,
            "chat_history": history,
            "parameters": parameters,
        }
        orchestrator: AgentOrg = AgentOrg(config=self.taskgraph_path, env=self.env)
        result: Dict[str, Any] = orchestrator.get_response(data)
        return result["answer"], result["parameters"]

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost: float = 0.0
        env_reset_res: EnvResetResponse = env.reset(task_index=task_index)
        obs: str = env_reset_res.observation
        info: Dict[str, Any] = env_reset_res.info.model_dump()
        reward: float = 0.0
        history: List[Dict[str, Any]] = [
            {"role": "assistant", "content": self.start_message}
        ]
        messages: List[Dict[str, Any]] = [
            {"role": "assistant", "content": self.start_message}
        ]
        params: Dict[str, Any] = {}
        user_text: str = obs
        message_index: int = 1

        for _ in range(max_num_steps):
            new_messages: List[Dict[str, Any]] = []
            output: str
            params: Dict[str, Any]
            output, params = self.get_api_bot_response(
                deepcopy(history), user_text, params
            )

            user_message: Dict[str, str] = {"role": "user", "content": user_text}
            assistant_message: Dict[str, str] = {"role": "assistant", "content": output}
            assistant_message_metadata: Dict[str, Any] = {
                "role": "assistant",
                "content": output,
                "curr_node": deepcopy(params["taskgraph"]["curr_node"]),
                "intent": deepcopy(params["taskgraph"]["intent"]),
                "metadata": deepcopy(params["memory"]["trajectory"][-1]),
            }
            history.append(user_message)
            history.append(assistant_message)

            print("=============trajectory============")
            trajectory: List[Dict[str, Any]] = params["memory"][
                "function_calling_trajectory"
            ]
            print(trajectory)

            while message_index < len(trajectory):
                msg: Dict[str, Any] = trajectory[message_index]

                if not is_message_worker(msg):
                    if (
                        is_assistant_with_tool_calls(msg)
                        or is_user(msg)
                        or is_tool(msg)
                    ):
                        new_messages.append(msg)

                    if is_assistant_with_tool_calls(msg):
                        action: Action = message_to_action(msg)
                        env_response: EnvResponse = env.step(action)
                        reward = env_response.reward
                        info = {**info, **env_response.info.model_dump()}

                message_index += 1

            # total_cost += res._hidden_params["response_cost"]
            new_messages.append(assistant_message_metadata)
            action: Action = message_to_action(assistant_message)
            env_response: EnvResponse = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            user_text = env_response.observation

            if env_response.done:
                user_message = {"role": "user", "content": user_text}
                new_messages.append(user_message)
            messages.extend(new_messages)
            if env_response.done:
                break
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )


def is_user(message: Dict[str, Any]) -> bool:
    if message.get("role") == "user":
        return True
    return False


def is_tool(message: Dict[str, Any]) -> bool:
    if message.get("role") == "tool":
        return True
    return False


def is_assistant_with_tool_calls(message: Dict[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False
    if "tool_calls" not in message:
        return False
    if message["tool_calls"] is None:
        return False
    if len(message["tool_calls"]) == 0:
        return False
    if "function" not in message["tool_calls"][0]:
        return False
    if message["tool_calls"][0]["function"] is None:
        return False
    return True


def is_message_worker(message: Dict[str, Any]) -> bool:
    if message.get("name") == "MessageWorker":
        return True
    if "tool_calls" not in message:
        return False
    if message["tool_calls"] is None:
        return False
    if len(message["tool_calls"]) == 0:
        return False
    if "function" not in message["tool_calls"][0]:
        return False
    if message["tool_calls"][0]["function"] is None:
        return False
    return message["tool_calls"][0]["function"].get("name") == "MessageWorker"


def message_to_action(
    message: Dict[str, Any],
) -> Action:
    if (
        "tool_calls" in message
        and message["tool_calls"] is not None
        and len(message["tool_calls"]) > 0
        and message["tool_calls"][0]["function"] is not None
    ):
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})
