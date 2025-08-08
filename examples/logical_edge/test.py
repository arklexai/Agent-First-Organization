import json
import unittest
from typing import Any, Dict, List, Tuple

from arklex.orchestrator.orchestrator import AgentOrg
from arklex.env.env import Environment
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class LogicEdge_Test(unittest.TestCase):
    file_path: str = "./examples/logical_edge/test_cases.json"
    with open(file_path, "r", encoding="UTF-8") as f:
        TEST_CASES: List[Dict[str, Any]] = json.load(f)

    @classmethod
    def setUpClass(cls) -> None:
        """Prepare the test fixture. Run BEFORE test methods."""
        cls.user_prefix: str = "user"
        cls.worker_prefix: str = "assistant"
        file_path: str = "./examples/logical_edge/taskgraph.json"
        with open(file_path, "r", encoding="UTF-8") as f:
            cls.config: Dict[str, Any] = json.load(f)
        cls.env: Environment = Environment(
            tools=cls.config.get("tools", []),
            workers=cls.config.get("workers", []),
            slotsfillapi=cls.config["slotfillapi"],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down the test fixture."""

    def _get_api_bot_response(
        self, user_text: str, history: List[Dict[str, str]], params: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Helper method to get bot response."""
        data: Dict[str, Any] = {
            "text": user_text,
            "chat_history": history,
            "parameters": params,
        }
        orchestrator = AgentOrg(config=self.config, env=self.env)
        result: Dict[str, Any] = orchestrator.get_response(data)
        return result["answer"], result["parameters"]

    def test_LogicEdgeTrajectory1(self) -> None:
        print("\n============= Logic Edge Trajectory Test =============")
        print(f"Task description: {self.TEST_CASES[0]['description']}")
        history: List[Dict[str, str]] = []
        params: Dict[str, Any] = {}
        nodes: List[str] = []

        # get start message
        for node in self.config["nodes"]:
            if node[1].get("type", "") == "start":
                start_message: str = node[1]["attribute"]["value"]
                break
        history.append({"role": self.worker_prefix, "content": start_message})

        # iterate over user utterances
        for user_text in self.TEST_CASES[0]["user_utterance"]:
            print(f"User: {user_text}")
            output, params = self._get_api_bot_response(user_text, history, params)
            print(f"Bot: {output}")
            curr_node = params.get("taskgraph").get('curr_node')
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
        history: List[Dict[str, str]] = []
        params: Dict[str, Any] = {}
        nodes: List[str] = []

        # get start message
        for node in self.config["nodes"]:
            if node[1].get("type", "") == "start":
                start_message: str = node[1]["attribute"]["value"]
                break
        history.append({"role": self.worker_prefix, "content": start_message})

        # iterate over user utterances
        for user_text in self.TEST_CASES[1]["user_utterance"]:
            print(f"User: {user_text}")
            output, params = self._get_api_bot_response(user_text, history, params)
            print(f"Bot: {output}")
            curr_node = params.get("taskgraph").get('curr_node')
            print(f"Reached Node: {curr_node}")
            nodes.append(curr_node)

            history.append({"role": self.user_prefix, "content": user_text})
            history.append({"role": self.worker_prefix, "content": output})

        print(f"Expected trajectory: {self.TEST_CASES[1]['trajectory']}")
        print(f"Observed trajectory: {nodes}")
        self.assertEqual(nodes, self.TEST_CASES[1]["trajectory"])

if __name__ == "__main__":
    unittest.main()
