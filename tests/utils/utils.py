import json
from abc import ABC, abstractmethod

from arklex.env.env import Env
from arklex.orchestrator.orchestrator import AgentOrg


class MockOrchestrator(ABC):
    def __init__(self, config_file_path: str, fixed_args: dict = {}):
        self.user_prefix = "user"
        self.assistant_prefix = "assistant"
        config = json.load(open(config_file_path))
        if fixed_args:
            for tool in config["tools"]:
                tool["fixed_args"].update(fixed_args)
        self.config = config

    def _get_test_response(self, user_text: str, history: list, params: dict) -> dict:
        data = {"text": user_text, 'chat_history': history, 'parameters': params}
        orchestrator = AgentOrg(
            config=self.config,
            env=Env(
                tools=self.config["tools"],
                workers=self.config["workers"],
                slotsfillapi=self.config["slotfillapi"]
            )
        )
        return orchestrator.get_response(data)
    
    def _initialize_test(self) -> tuple[list, dict]:
        history = []
        params = {}
        start_message = None
        for node in self.config["nodes"]:
            if node[1].get("type", "") == 'start':
                start_message = node[1]['attribute']["value"]
                break
        history.append({"role": self.assistant_prefix, "content": start_message})
        return history, params

    def _execute_conversation(self, test_case: dict, history: list, params: dict) -> tuple[list, dict]:
        for user_text in test_case["user_utterance"]:
            result = self._get_test_response(user_text, history, params)
            answer, params = result["answer"], result["parameters"]
            history.append({"role": self.user_prefix, "content": user_text})
            history.append({"role": self.assistant_prefix, "content": answer})
        return history, params

    @abstractmethod
    def _validate_result(self, test_case: dict, history: list, params: dict):
        # NOTE: change the assert to check the result
        pass

    def run_single_test(self, test_case: dict):
        history, params = self._initialize_test()
        history, params = self._execute_conversation(test_case, history, params)
        self._validate_result(test_case, history, params)
