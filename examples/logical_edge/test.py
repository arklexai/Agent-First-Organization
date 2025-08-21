import json
import os
import unittest
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from arklex.env.env import Environment
from arklex.orchestrator.NLU.services.model_service import ModelService
from arklex.orchestrator.orchestrator import AgentOrg
from arklex.utils.logging_utils import LogContext
from arklex.utils.provider_utils import get_provider_config

log_context = LogContext(__name__)
current_dir = Path(__file__).parent


class LogicEdge_Test(unittest.TestCase):
    file_path: str = str(current_dir / "test_cases.json")
    with open(file_path, encoding="UTF-8") as f:
        TEST_CASES: list[dict[str, Any]] = json.load(f)

    @classmethod
    def setUpClass(cls) -> None:
        load_dotenv("/Users/yuju/arklex-intern/arklex-main/.env")
        cls.user_prefix: str = "user"
        cls.worker_prefix: str = "assistant"
        file_path: str = str(current_dir / "taskgraph.json")
        with open(file_path, encoding="UTF-8") as f:
            cls.config: dict[str, Any] = json.load(f)
        llm_provider = "openai"
        model_name = "gpt-4o-mini"
        model = get_provider_config(llm_provider, model_name)
        cls.config["model"] = model

        os.environ["DATA_DIR"] = (
            "/Users/yuju/arklex-intern/arklex-main/examples/logical_edge"
        )

        # Init ModelService
        model_service = ModelService(model)
        cls.env = Environment(
            tools=cls.config.get("tools", []),
            workers=cls.config.get("workers", []),
            agents=cls.config.get("agents", []),
            nodes=cls.config.get("nodes", []),
            slot_fill_api=cls.config.get("slotfillapi", ""),
            planner_enabled=True,
            model_service=model_service,
        )
        cls.config["bot_config"] = {
            "bot_id": "test-bot",
            "version": "v1.0",
            "language": "en",
            "llm_config": {
                "model_name": cls.config["model"].get(
                    "model_type_or_path", "gpt-4o-mini"
                ),
                "model_type_or_path": model.get("model_type_or_path", "gpt-4o-mini"),
                "llm_provider": cls.config["model"].get("llm_provider", "openai"),
            },
        }

    def _get_api_bot_response(
        self, user_text: str, history: list[dict[str, str]], params: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        data: dict[str, Any] = {
            "text": user_text,
            "chat_history": history,
            "parameters": params,
        }
        orchestrator = AgentOrg(config=self.config, env=self.env)
        result: dict[str, Any] = orchestrator.get_response(data)
        return result["answer"], result["parameters"]

    def test_LogicEdgeTrajectory1(self) -> None:
        print("\n============= Logic Edge Trajectory Test =============")
        print(f"Task description: {self.TEST_CASES[0]['description']}")
        history: list[dict[str, str]] = []
        params: dict[str, Any] = {}
        nodes: list[str] = []

        # get start message
        start_message = None
        for node in self.config["nodes"]:
            if node[1].get("attribute", {}).get("start", False):
                start_message = node[1].get("data", {}).get("message", "Hello!")
                break
        if start_message is None:
            raise ValueError("No start node found in config['nodes']")
        history.append({"role": self.worker_prefix, "content": start_message})

        # for node in self.config["nodes"]:
        #     if node[1].get("type", "") == "start":
        #         start_message: str = node[1]["attribute"]["value"]
        #         break
        # history.append({"role": self.worker_prefix, "content": start_message})

        for user_text in self.TEST_CASES[0]["user_utterance"]:
            print(f"User: {user_text}")
            output, params = self._get_api_bot_response(user_text, history, params)
            print(f"Bot: {output}")
            curr_node = params.get("taskgraph", {}).get("curr_node")
            print(f"Reached Node: {curr_node}")
            nodes.append(curr_node)
            history.append({"role": self.user_prefix, "content": user_text})
            history.append({"role": self.worker_prefix, "content": output})

        print(f"Expected trajectory: {self.TEST_CASES[0]['trajectory']}")
        print(f"Observed trajectory: {nodes}")
        self.assertEqual(nodes, self.TEST_CASES[0]["trajectory"])

    def test_LogicEdgeTrajectory2(self) -> None:
        print("\n============= Logic Edge Trajectory Test =============")
        print(f"Task description: {self.TEST_CASES[1]['description']}")
        history: list[dict[str, str]] = []
        params: dict[str, Any] = {}
        nodes: list[str] = []

        start_message = None
        for node in self.config["nodes"]:
            if node[1].get("attribute", {}).get("start", False):
                start_message = node[1].get("data", {}).get("message", "Hello!")
                break
        if start_message is None:
            raise ValueError("No start node found in config['nodes']")
        history.append({"role": self.worker_prefix, "content": start_message})

        # for node in self.config["nodes"]:
        #     if node[1].get("type", "") == "start":
        #         start_message: str = node[1]["attribute"]["value"]
        #         break
        # history.append({"role": self.worker_prefix, "content": start_message})

        for user_text in self.TEST_CASES[1]["user_utterance"]:
            print(f"User: {user_text}")
            output, params = self._get_api_bot_response(user_text, history, params)
            print(f"Bot: {output}")
            curr_node = params.get("taskgraph", {}).get("curr_node")
            print(f"Reached Node: {curr_node}")
            nodes.append(curr_node)
            history.append({"role": self.user_prefix, "content": user_text})
            history.append({"role": self.worker_prefix, "content": output})

        print(f"Expected trajectory: {self.TEST_CASES[1]['trajectory']}")
        print(f"Observed trajectory: {nodes}")
        self.assertEqual(nodes, self.TEST_CASES[1]["trajectory"])


if __name__ == "__main__":
    unittest.main()
