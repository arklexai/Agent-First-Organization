"""Core test utilities for the Arklex framework.

This module provides core test utilities for the Arklex framework, including
the MockOrchestrator base class for testing orchestrator behavior. It includes
methods for initializing tests, executing conversations, and validating results,
with support for custom validation through abstract methods.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from arklex.env.env import Env
from arklex.orchestrator.orchestrator import AgentOrg


class MockOrchestrator(ABC):
    def __init__(self, config_file_path: str, fixed_args: Dict[str, Any] = {}) -> None:
        """Initialize the mock orchestrator.

        Args:
            config_file_path (str): Path to the configuration file.
            fixed_args (Dict[str, Any], optional): Fixed arguments to update
                tool configurations. Defaults to empty dict.
        """
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
        """Get a test response from the orchestrator.

        This function simulates a conversation by sending user text to the
        orchestrator and getting a response.

        Args:
            user_text (str): The user's input text.
            history (List[Dict[str, str]]): Conversation history.
            params (Dict[str, Any]): Parameters for the conversation.

        Returns:
            Dict[str, Any]: The orchestrator's response containing the answer
                and updated parameters.
        """
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
        """Initialize a test conversation.

        This function sets up the initial conversation state by creating an
        empty history and parameters, and adding the start message if one
        exists in the configuration.

        Returns:
            Tuple[List[Dict[str, str]], Dict[str, Any]]: A tuple containing
                the initial conversation history and parameters.
        """
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
        """Execute a test conversation.

        This function simulates a conversation by processing each user utterance
        in the test case, getting responses from the orchestrator, and updating
        the conversation history and parameters.

        Args:
            test_case (Dict[str, Any]): Test case containing user utterances.
            history (List[Dict[str, str]]): Initial conversation history.
            params (Dict[str, Any]): Initial parameters.

        Returns:
            Tuple[List[Dict[str, str]], Dict[str, Any]]: A tuple containing
                the updated conversation history and parameters.
        """
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
        """Validate the test results.

        This abstract method should be implemented by subclasses to validate
        the results of a test case. The implementation should check that the
        conversation history and parameters match the expected values.

        Args:
            test_case (Dict[str, Any]): Test case containing expected values.
            history (List[Dict[str, str]]): Conversation history to validate.
            params (Dict[str, Any]): Parameters to validate.
        """
        # NOTE: change the assert to check the result
        pass

    def run_single_test(self, test_case: Dict[str, Any]) -> None:
        """Run a single test case.

        This function executes a test case by initializing the conversation,
        executing the conversation, and validating the results.

        Args:
            test_case (Dict[str, Any]): Test case to run.

        Raises:
            AssertionError: If the test validation fails.
        """
        history, params = self._initialize_test()
        history, params = self._execute_conversation(test_case, history, params)
        self._validate_result(test_case, history, params)
