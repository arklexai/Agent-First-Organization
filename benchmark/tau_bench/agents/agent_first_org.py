import json
import os
from copy import deepcopy
from typing import Any

from arklex.env.env import Env
from arklex.orchestrator.orchestrator import AgentOrg
from benchmark.tau_bench.agents.base import Agent
from benchmark.tau_bench.tau_types import (
    RESPOND_ACTION_NAME,
    Action,
    EnvResetResponse,
    EnvResponse,
    SolveResult,
)


class AgentFirstOrg(Agent):
    def __init__(self, taskgraph_dir: str) -> None:
        self.taskgraph_dir: str = taskgraph_dir
        self.taskgraph_path: str = os.path.join(self.taskgraph_dir, "taskgraph.json")
        from benchmark.tau_bench.tau_bench_eval import TauBenchResourceInitializer

        with open(self.taskgraph_path) as taskgraph:
            taskgraph: dict[str, Any] = json.load(taskgraph)
            tau_bench_resource_initializer: TauBenchResourceInitializer = (
                TauBenchResourceInitializer()
            )
            self.env: Env = Env(
                tools=taskgraph.get("tools", []),
                workers=taskgraph.get("workers", []),
                slotsfillapi=taskgraph["slotfillapi"],
                resource_inizializer=tau_bench_resource_initializer,
            )

            self.start_message: str | None = None
            for node in taskgraph["nodes"]:
                if node[1].get("type", "") == "start":
                    self.start_message = node[1]["attribute"]["value"]
                    break

    def get_api_bot_response(
        self, history: list[dict[str, Any]], user_text: str, parameters: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        data: dict[str, Any] = {
            "text": user_text,
            "chat_history": history,
            "parameters": parameters,
        }
        orchestrator: AgentOrg = AgentOrg(config=self.taskgraph_path, env=self.env)
        result: dict[str, Any] = orchestrator.get_response(data)
        return result["answer"], result["parameters"]

    def solve(
        self, env: Env, task_index: int | None = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost: float = 0.0
        env_reset_res: EnvResetResponse = env.reset(task_index=task_index)
        obs: str = env_reset_res.observation
        info: dict[str, Any] = env_reset_res.info.model_dump()
        reward: float = 0.0
        history: list[dict[str, Any]] = [
            {"role": "assistant", "content": self.start_message}
        ]
        messages: list[dict[str, Any]] = [
            {"role": "assistant", "content": self.start_message}
        ]
        params: dict[str, Any] = {}
        user_text: str = obs
        message_index: int = 1

        for _ in range(max_num_steps):
            new_messages: list[dict[str, Any]] = []
            output: str
            params: dict[str, Any]
            output, params = self.get_api_bot_response(
                deepcopy(history), user_text, params
            )

            user_message: dict[str, str] = {"role": "user", "content": user_text}
            assistant_message: dict[str, str] = {"role": "assistant", "content": output}
            assistant_message_metadata: dict[str, Any] = {
                "role": "assistant",
                "content": output,
                "curr_node": deepcopy(params["taskgraph"]["curr_node"]),
                "intent": deepcopy(params["taskgraph"]["intent"]),
                "metadata": deepcopy(params["memory"]["trajectory"][-1]),
            }
            history.append(user_message)
            history.append(assistant_message)

            print("=============trajectory============")
            trajectory: list[dict[str, Any]] = params["memory"][
                "function_calling_trajectory"
            ]
            print(trajectory)

            while message_index < len(trajectory):
                msg: dict[str, Any] = trajectory[message_index]

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


def is_user(message: dict[str, Any]) -> bool:
    return message.get("role") == "user"


def is_tool(message: dict[str, Any]) -> bool:
    return message.get("role") == "tool"


def is_assistant_with_tool_calls(message: dict[str, Any]) -> bool:
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
    return message["tool_calls"][0]["function"] is not None


def is_message_worker(message: dict[str, Any]) -> bool:
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
    message: dict[str, Any],
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
