# Go to the parent folder of this file (calendar), then Run python -m unittest test.py to test the code in this file.
import json
import os
import sys
import unittest
from typing import Any, Dict, List, Tuple

from arklex.orchestrator.orchestrator import AgentOrg
from arklex.env.env import Env
from arklex.env.tools.tools import logger

# May not need after pip install agentorg
sys.path.insert(0, os.path.abspath("../../"))
print(sys.path)


class Logic_Test(unittest.TestCase):
    file_path: str = "test_cases.json"
    with open(file_path, "r", encoding="UTF-8") as f:
        TEST_CASES: List[Dict[str, Any]] = json.load(f)

    @classmethod
    def setUpClass(cls) -> None:
        """Method to prepare the test fixture. Run BEFORE the test methods."""
        cls.user_prefix: str = "user"
        cls.worker_prefix: str = "assistant"
        file_path: str = "taskgraph.json"
        with open(file_path, "r", encoding="UTF-8") as f:
            cls.config: Dict[str, Any] = json.load(f)
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
        pass

    def _get_api_bot_response(
        self, user_text: str, history: List[Dict[str, str]], params: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        data: Dict[str, Any] = {
            "text": user_text,
            "chat_history": history,
            "parameters": params,
        }
        orchestrator = AgentOrg(config=self.config, env=self.env)
        result: Dict[str, Any] = orchestrator.get_response(data)

        return result["answer"], result["parameters"]

    def test_Unittest0(self) -> None:
        logger.info("\n=============Unit Test 0=============")
        logger.info(f"{self.TEST_CASES[0]['description']}")
        history: List[Dict[str, str]] = []
        params: Dict[str, Any] = {}
        nodes: List[str] = []
        for node in self.config["nodes"]:
            if node[1].get("type", "") == "start":
                start_message: str = node[1]["attribute"]["value"]
                break
        history.append({"role": self.worker_prefix, "content": start_message})

        for user_text in self.TEST_CASES[0]["user_utterance"]:
            print(f"User: {user_text}")
            output, params = self._get_api_bot_response(user_text, history, params)
            print(f"Bot: {output}")
            nodes.append(params.get("curr_node"))
            history.append({"role": self.user_prefix, "content": user_text})
            history.append({"role": self.worker_prefix, "content": output})
        print(f"{self.TEST_CASES[0]['criteria']}")
        # print(json.dumps(history, indent=4))
        self.assertEqual(nodes, self.TEST_CASES[0]["trajectory"])

    def test_Unittest1(self) -> None:
        print("\n=============Unit Test 1=============")
        print(f"{self.TEST_CASES[1]['description']}")
        history: List[Dict[str, str]] = []
        params: Dict[str, Any] = {}
        nodes: List[str] = []
        for node in self.config["nodes"]:
            if node[1].get("type", "") == "start":
                start_message: str = node[1]["attribute"]["value"]
                break
        history.append({"role": self.worker_prefix, "content": start_message})

        for user_text in self.TEST_CASES[1]["user_utterance"]:
            print(f"User: {user_text}")
            output, params = self._get_api_bot_response(user_text, history, params)
            print(f"Bot: {output}")
            nodes.append(params.get("curr_node"))
            history.append({"role": self.user_prefix, "content": user_text})
            history.append({"role": self.worker_prefix, "content": output})
        print(f"{self.TEST_CASES[1]['criteria']}")
        # print(json.dumps(history, indent=4))
        self.assertEqual(nodes, self.TEST_CASES[1]["trajectory"])

    def test_Unittest2(self) -> None:
        print("\n=============Unit Test 2=============")
        print(f"{self.TEST_CASES[2]['description']}")
        history: List[Dict[str, str]] = []
        params: Dict[str, Any] = {}
        nodes: List[str] = []
        for node in self.config["nodes"]:
            if node[1].get("type", "") == "start":
                start_message: str = node[1]["attribute"]["value"]
                break
        history.append({"role": self.worker_prefix, "content": start_message})

        for user_text in self.TEST_CASES[2]["user_utterance"]:
            print(f"User: {user_text}")
            output, params = self._get_api_bot_response(user_text, history, params)
            print(f"Bot: {output}")
            nodes.append(params.get("curr_node"))
            history.append({"role": self.user_prefix, "content": user_text})
            history.append({"role": self.worker_prefix, "content": output})
        print(f"{self.TEST_CASES[2]['criteria']}")
        # print(json.dumps(history, indent=4))
        self.assertEqual(nodes, self.TEST_CASES[2]["trajectory"])


if __name__ == "__main__":
    unittest.main()
