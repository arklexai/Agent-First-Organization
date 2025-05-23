import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from arklex.env.env import Env
from arklex.orchestrator.orchestrator import AgentOrg


class MockOrchestrator(ABC):
    def __init__(self, config_file_path: str, fixed_args: Dict[str, Any] = {}) -> None:
        self.user_prefix: str = "user"
        self.assistant_prefix: str = "assistant"
        config: Dict[str, Any] = json.load(open(config_file_path))
        if fixed_args:
            for tool in config["tools"]:
                tool["fixed_args"].update(fixed_args)
        self.config: Dict[str, Any] = config

    def _get_test_response(
        self, user_text: str, history: List[Dict[str, str]], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "text": user_text,
            "chat_history": history,
            "parameters": params,
        }
        orchestrator = AgentOrg(
            config=self.config,
            env=Env(
                tools=self.config["tools"],
                workers=self.config["workers"],
                slotsfillapi=self.config["slotfillapi"],
            ),
        )
        return orchestrator.get_response(data)

    def _initialize_test(self) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        history: List[Dict[str, str]] = []
        params: Dict[str, Any] = {}
        start_message: Optional[str] = None
        for node in self.config["nodes"]:
            if node[1].get("type", "") == "start":
                start_message = node[1]["attribute"]["value"]
                break
        if start_message:
            history.append({"role": self.assistant_prefix, "content": start_message})
        return history, params

    def _execute_conversation(
        self,
        test_case: Dict[str, Any],
        history: List[Dict[str, str]],
        params: Dict[str, Any],
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        for user_text in test_case["user_utterance"]:
            result: Dict[str, Any] = self._get_test_response(user_text, history, params)
            answer: str = result["answer"]
            params = result["parameters"]
            history.append({"role": self.user_prefix, "content": user_text})
            history.append({"role": self.assistant_prefix, "content": answer})
        return history, params

    @abstractmethod
    def _validate_result(
        self,
        test_case: Dict[str, Any],
        history: List[Dict[str, str]],
        params: Dict[str, Any],
    ) -> None:
        # NOTE: change the assert to check the result
        pass

    def run_single_test(self, test_case: Dict[str, Any]) -> None:
        history, params = self._initialize_test()
        history, params = self._execute_conversation(test_case, history, params)
        self._validate_result(test_case, history, params)
