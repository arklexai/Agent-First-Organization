# Go to the parent folder of this file (calendar), then Run python -m unittest test.py to test the code in this file.
import json
import os
import sys
import unittest
from typing import Any

from arklex.env.env import Env
from arklex.orchestrator.orchestrator import AgentOrg

# May not need after pip install agentorg
sys.path.insert(0, os.path.abspath("../../"))
print(sys.path)


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
        cls.config["input_dir"] = "./"
        os.environ["DATA_DIR"] = cls.config["input_dir"]

    @classmethod
    def tearDownClass(cls) -> None:
        """Method to tear down the test fixture. Run AFTER the test methods."""

    def _get_api_bot_response(
        self, user_text: str, history: list[dict[str, str]], params: dict[str, Any]
    ) -> tuple[str, dict[str, Any], str | None]:
        data: dict[str, Any] = {
            "text": user_text,
            "chat_history": history,
            "parameters": params,
        }
        orchestrator = AgentOrg(config=self.config, env=self.env)
        result: dict[str, Any] = orchestrator.get_response(data)

        return result["answer"], result["parameters"], result["human_in_the_loop"]

    def test_Unittest0(self) -> None:
        print("\n=============Unit Test 0=============")
        print(f"{self.TEST_CASES[0]['description']}")
        history: list[dict[str, str]] = []
        params: dict[str, Any] = {}
        nodes: list[str] = []
        final_state: list[str] = []
        for node in self.config["nodes"]:
            if node[1].get("type", "") == "start":
                start_message: str = node[1]["attribute"]["value"]
                break
        history.append({"role": self.worker_prefix, "content": start_message})

        for user_text in self.TEST_CASES[0]["user_utterance"]:
            print(f"User: {user_text}")
            output, params, hitl = self._get_api_bot_response(
                user_text, history, params
            )
            print(f"Bot: {output}")
            nodes.append(params["taskgraph"]["curr_node"])
            final_state.append(hitl if hitl else "")
            history.append({"role": self.user_prefix, "content": user_text})
            history.append({"role": self.worker_prefix, "content": output})
        print(f"{self.TEST_CASES[0]['criteria']}")
        # print(json.dumps(history, indent=4))
        self.assertEqual(nodes, self.TEST_CASES[0]["trajectory"])
        self.assertEqual(final_state, self.TEST_CASES[0]["final_state"])

    def test_Unittest1(self) -> None:
        print("\n=============Unit Test 1=============")
        print(f"{self.TEST_CASES[1]['description']}")
        history: list[dict[str, str]] = []
        params: dict[str, Any] = {}
        nodes: list[str] = []
        final_state: list[str] = []
        for node in self.config["nodes"]:
            if node[1].get("type", "") == "start":
                start_message: str = node[1]["attribute"]["value"]
                break
        history.append({"role": self.worker_prefix, "content": start_message})

        for user_text in self.TEST_CASES[1]["user_utterance"]:
            print(f"User: {user_text}")
            output, params, hitl = self._get_api_bot_response(
                user_text, history, params
            )
            print(f"Bot: {output}")
            nodes.append(params["taskgraph"]["curr_node"])
            final_state.append(hitl if hitl else "")
            history.append({"role": self.user_prefix, "content": user_text})
            history.append({"role": self.worker_prefix, "content": output})
        print(f"{self.TEST_CASES[1]['criteria']}")
        # print(json.dumps(history, indent=4))
        self.assertEqual(nodes, self.TEST_CASES[1]["trajectory"])
        self.assertEqual(final_state, self.TEST_CASES[1]["final_state"])


if __name__ == "__main__":
    unittest.main()
