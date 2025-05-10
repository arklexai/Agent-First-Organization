# Install required packages in the root directory: pip install -e .
# Go to the parent folder of this file (shopify), then Run python -m unittest test.py to test the code in this file.

import unittest
import json
from typing import Dict
import os
import time
import warnings

from arklex.orchestrator.orchestrator import AgentOrg
from arklex.env.env import Env
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import LLM_PROVIDERS

# Wait this many seconds between tests to avoid token rate-limiting
WAIT_TIME_BETWEEN_TESTS_SEC = 5 # Set to None or 0 for no wait time

class Logic_Test(unittest.TestCase):
    file_path = "test_cases_shopify_advanced.json"
    with open(file_path, "r", encoding="UTF-8") as f:
        TEST_CASES = json.load(f)

    @classmethod
    def setUpClass(cls):
        """Method to prepare the test fixture. Run BEFORE the test methods."""
        cls.user_prefix = "user"
        cls.worker_prefix = "assistant"
        cls.config = None
        cls.env = None
        cls.total_tests_run = 0
        
    @classmethod
    def tearDownClass(cls):
        """Method to tear down the test fixture. Run AFTER the test methods."""
        pass
    
    def _get_api_bot_response(self, user_text, history, params):

        data = {"text": user_text, 'chat_history': history, 'parameters': params}
        orchestrator = AgentOrg(config=self.config, env=self.env)
        result = orchestrator.get_response(data)

        return result['answer'], result['parameters']

    def _check_task_completion(self, output: str, params: Dict, test_case: Dict):
        expected_output = test_case.get("expected_output", {})
        contains = expected_output.get("contains", {})
        contains_all = contains.get("all", [])
        contains_any = contains.get("any", [])

        if len(contains_all) > 0:
            for text in contains_all:
                failure_message = f"FAILED: Expected text '{text}' not found in final output ('{output}'). params['memory']['function_calling_trajectory'] = {params['memory']['function_calling_trajectory']}"
                self.assertTrue(text.lower() in output.lower(), failure_message)

        if len(contains_any) > 0:
            contains_flags = [text.lower() in output.lower() for text in contains_any]
            failure_message = f"FAILED: None of {contains_any} were found in final output ('{output}'). params['memory']['function_calling_trajectory'] = {params['memory']['function_calling_trajectory']}"
            self.assertTrue(True in contains_flags, failure_message)

    def _check_tool_calls(self, params: Dict, env: Env, test_case: Dict):
        _expected_tool_calls = test_case.get("expected_tool_calls", {})
        correct_tool_calls = False

        # Check if multiple possible tool call sequences are allowed to pass this test
        # (E.g., get_products or get_web_product)
        _allowed_tool_calls = _expected_tool_calls.get("options", None)
        if _allowed_tool_calls is None:
            _allowed_tool_calls = [_expected_tool_calls]

        # Reformat tool/worker names to match those found in conversation history (which
        # are not necessarily the same names as those found in taskgraph.json)
        expected_tool_calls = []
        for tool_set in _allowed_tool_calls:
            expected_tool_set = {}
            for tool_name in tool_set:
                for resource_name in env.planner.all_resources_info.keys():
                    if tool_name in resource_name:
                        expected_tool_set[resource_name] = tool_set[tool_name]
                        break
            expected_tool_calls.append(expected_tool_set)

        # Get actual tool calls from conversation history (ignore DefaultWorker because planner can't
        # call it)
        actual_tool_calls = {}
        for msg in params["memory"]["function_calling_trajectory"]:
            if msg["role"] == "tool":
                tool_name = msg["name"]

                if tool_name != "DefaultWorker" and tool_name in actual_tool_calls:
                    actual_tool_calls[tool_name] += 1
                elif tool_name != "DefaultWorker":
                    actual_tool_calls[tool_name] = 1

        # If only one set of tool calls is allowed to pass this test, check that actual tool
        # calls match these exactly
        if len(expected_tool_calls) == 1:
            expected_tool_calls = expected_tool_calls[0]
            failure_message = (
                "FAILED: Planner expected tool calls != actual tool calls." +
                f"\nexpected_tool_calls = {json.dumps(expected_tool_calls, indent=2)}" +
                f"\nactual_tool_calls = {json.dumps(actual_tool_calls, indent=2)}" +
                f"\nparams['memory']['function_calling_trajectory'] = {params['memory']['function_calling_trajectory']}"
            )
            self.assertEqual(expected_tool_calls, actual_tool_calls, failure_message)

        # If multiple possible tool call sequences are allowed, check that actual tool calls
        # matches at least one of these
        else:
            failure_message = (
                "FAILED: Planner allowed tool calls != actual tool calls." +
                f"\nActual tool calls was expected to be one of the following:"
            )
            for tool_set in expected_tool_calls:
                failure_message += f"\n{json.dumps(tool_set, indent=2)}"
            failure_message += (
                f"\nInstead, actual_tool_calls were: {json.dumps(actual_tool_calls, indent=2)}" +
                f"\nparams['memory']['function_calling_trajectory'] = {params['memory']['function_calling_trajectory']}"
            )

            tool_call_matches = [actual_tool_calls == tool_set for tool_set in expected_tool_calls]
            self.assertTrue(True in tool_call_matches, failure_message)

    def _check_success_criteria(self, output: str, params: Dict, test_case: Dict):
        self._check_tool_calls(params, self.env, test_case)
        self._check_task_completion(output, params, test_case)

    def _run_test_case(self, idx: int):

        # Wait to avoid token rate-limiting
        if WAIT_TIME_BETWEEN_TESTS_SEC is not None and self.total_tests_run > 0:
            print(f"\nWaiting {WAIT_TIME_BETWEEN_TESTS_SEC} sec between tests to avoid token rate-limiting...")
            time.sleep(WAIT_TIME_BETWEEN_TESTS_SEC)

        print(f"\n=============Unit Test {idx}=============")

        test_case = self.TEST_CASES[idx]
        print(f"Task description: {test_case['description']}")

        # Initialize config and env
        file_path = test_case['taskgraph']
        input_dir, _ = os.path.split(file_path) 
        with open (file_path, "r", encoding="UTF_8") as f:
            self.config = json.load(f)
        self.env = Env(
            tools = self.config.get("tools", []),
            workers = self.config.get("workers", []),
            slotsfillapi = self.config["slotfillapi"],
            planner_enabled=True
        )

        history = []
        params = {}
        for node in self.config['nodes']:
            if node[1].get("type", "") == 'start':
                start_message = node[1]['attribute']["value"]
                break
        history.append({"role": self.worker_prefix, "content": start_message})
        
        for user_text in test_case["user_utterance"]:
            print(f"User: {user_text}")
            output, params = self._get_api_bot_response(user_text, history, params)
            print(f"Bot: {output}")
            history.append({"role": self.user_prefix, "content": user_text})
            history.append({"role": self.worker_prefix, "content": output})

        print(f"Success criteria: {test_case['criteria']}")
        final_output = history[-1]["content"]
        self._check_success_criteria(final_output, params, test_case)

        self.total_tests_run += 1

    def test_Unittest00(self):
        self._run_test_case(0)

    def test_Unittest01(self):
        self._run_test_case(1)

    def test_Unittest02(self):
        self._run_test_case(2)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        unittest.main()