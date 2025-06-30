# Install required packages in the root directory: pip install -e .
# Go to the parent folder of this file (multiple_choice), then Run python -m unittest test.py to test the code in this file.
import json
import unittest
from typing import Any

from arklex.env.env import Env
from arklex.orchestrator.orchestrator import AgentOrg


class Logic_Test(unittest.TestCase):
    file_path: str = "test_cases.json"
    with open(file_path, encoding="UTF-8") as f:
        TEST_CASES: list[dict[str, Any]] = json.load(f)

    @classmethod
    def setUpClass(cls) -> None:
        """Method to prepare the test fixture. Run BEFORE the test methods."""
        cls.user_prefix: str = "user"
        cls.worker_prefix: str = "assistant"
        file_path: str = "taskgraph.json"
        with open(file_path, encoding="UTF-8") as f:
            cls.config: dict[str, Any] = json.load(f)
        cls.env: Env = Env(
            tools=cls.config.get("tools", []),
            workers=cls.config.get("workers", []),
            slotsfillapi=cls.config["slotfillapi"],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Method to tear down the test fixture. Run AFTER the test methods."""

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

    def test_Unittest0(self) -> None:
        print("\n=============Unit Test 0=============")
        print(f"Task descrption: {self.TEST_CASES[0]['description']}")
        history: list[dict[str, str]] = []
        params: dict[str, Any] = {}
        nodes: list[str] = []
        for node in self.config["nodes"]:
            if node[1].get("type", "") == "start":
                start_message: str = node[1]["attribute"]["value"]
                break
        history.append({"role": self.worker_prefix, "content": start_message})

        assistant_messages: list[str] = []
        for user_text in self.TEST_CASES[0]["user_utterance"]:
            print(f"User: {user_text}")
            output, params = self._get_api_bot_response(user_text, history, params)
            print(f"Bot: {output}")
            nodes.append(params.get("curr_node"))
            history.append({"role": self.user_prefix, "content": user_text})
            history.append({"role": self.worker_prefix, "content": output})
            assistant_messages.append(output)
        print(f"Success criteria: {self.TEST_CASES[0]['criteria']}")
        self.assertEqual(assistant_messages, self.TEST_CASES[0]["assistant_messages"])


if __name__ == "__main__":
    unittest.main()
