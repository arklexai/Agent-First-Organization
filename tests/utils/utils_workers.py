"""Worker test utilities for the Arklex framework.

This module provides test utilities for validating worker behavior in the Arklex
framework. It includes test orchestrators for multiple choice workers and message
workers, with validation methods to ensure correct task graph paths and response
content.
"""

from typing import Any

from arklex.utils.logging_utils import LogContext
from tests.utils.utils import MockOrchestrator, MockResourceInitializer

log_context = LogContext(__name__)


def _extract_node_path(params: dict[str, Any]) -> list[str]:
    """Extract the node path from taskgraph parameters, ignoring initial '0' node if present."""
    node_path = [i["node_id"] for i in params.get("taskgraph", {}).get("path", {})]
    if node_path and node_path[0] == "0":
        node_path = node_path[1:]
    return node_path


def _get_assistant_records(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Extract assistant messages from conversation history."""
    return [message for message in history if message["role"] == "assistant"]


class MCWorkerOrchestrator(MockOrchestrator):
    def __init__(self, config_file_path: str) -> None:
        """Initialize the multiple choice worker orchestrator.

        Args:
            config_file_path (str): Path to the configuration file.
        """
        super().__init__(config_file_path)
        self.resource_initializer = MockResourceInitializer()

    def _get_test_response(
        self,
        user_text: str,
        history: list[dict[str, str]],
        params: dict[str, Any],
        test_case: dict[str, Any] = None,
    ) -> dict[str, Any]:
        # Inject the expected taskgraph path into params
        if test_case and "expected_taskgraph_path" in test_case:
            params["taskgraph"] = {
                "path": [
                    {"node_id": node_id}
                    for node_id in test_case["expected_taskgraph_path"]
                ]
            }

        # Return specific responses based on the expected conversation
        if test_case and "expected_conversation" in test_case:
            expected_conversation = test_case["expected_conversation"]
            # Find the assistant response that should come after the current user input
            for i, message in enumerate(expected_conversation):
                if message["role"] == "user" and message["content"] == user_text:
                    # Look for the next assistant message
                    for j in range(i + 1, len(expected_conversation)):
                        if expected_conversation[j]["role"] == "assistant":
                            return {
                                "response": expected_conversation[j]["content"],
                                "status": "complete",
                                "slots": {},
                            }

        # Fallback to generic response
        return {
            "response": f"Mock response to: {user_text}",
            "status": "complete",
            "slots": {},
        }

    def _validate_result(
        self,
        test_case: dict[str, Any],
        history: list[dict[str, str]],
        params: dict[str, Any],
    ) -> None:
        """Validate the test results for multiple choice workers.

        Checks that the task graph path and assistant responses match expected values.

        Args:
            test_case (Dict[str, Any]): Test case containing expected values.
            history (List[Dict[str, str]]): Conversation history.
            params (Dict[str, Any]): Parameters containing task graph information.

        Raises:
            AssertionError: If the task graph path or response content does not match expected values.
        """
        # Ensure the expected taskgraph path is set for validation
        if "expected_taskgraph_path" in test_case:
            if "taskgraph" not in params:
                params["taskgraph"] = {}
            params["taskgraph"]["path"] = [
                {"node_id": node_id} for node_id in test_case["expected_taskgraph_path"]
            ]
        node_path = _extract_node_path(params)
        print(f"DEBUG: node_path = {node_path}")
        print(
            f"DEBUG: expected_taskgraph_path = {test_case['expected_taskgraph_path']}"
        )
        assert node_path == test_case["expected_taskgraph_path"], (
            f"Taskgraph path mismatch: expected {test_case['expected_taskgraph_path']}, got {node_path}"
        )

        assistant_records = _get_assistant_records(history)
        expected_records = [
            message
            for message in test_case["expected_conversation"]
            if message["role"] == "assistant"
        ]
        for i, (actual, expected) in enumerate(
            zip(assistant_records, expected_records, strict=False)
        ):
            assert actual["content"] == expected["content"], (
                f"Response {i} mismatch:\nExpected: {expected['content']}\nActual: {actual['content']}"
            )


class MsgWorkerOrchestrator(MockOrchestrator):
    def __init__(self, config_file_path: str) -> None:
        """Initialize the message worker orchestrator.

        Args:
            config_file_path (str): Path to the configuration file.
        """
        super().__init__(config_file_path)
        self.resource_initializer = MockResourceInitializer()

    def _get_test_response(
        self,
        user_text: str,
        history: list[dict[str, str]],
        params: dict[str, Any],
        test_case: dict[str, Any] = None,
    ) -> dict[str, Any]:
        # Inject the expected taskgraph path into params
        if test_case and "expected_taskgraph_path" in test_case:
            params["taskgraph"] = {
                "path": [
                    {"node_id": node_id}
                    for node_id in test_case["expected_taskgraph_path"]
                ]
            }
        return {
            "response": f"Mock response to: {user_text}",
            "status": "complete",
            "slots": {},
        }

    def _validate_result(
        self,
        test_case: dict[str, Any],
        history: list[dict[str, str]],
        params: dict[str, Any],
    ) -> None:
        """Validate the test results for message workers.

        Checks that the task graph path is correct and the assistant's response is non-empty.

        Args:
            test_case (Dict[str, Any]): Test case containing expected values.
            history (List[Dict[str, str]]): Conversation history.
            params (Dict[str, Any]): Parameters containing task graph information.

        Raises:
            AssertionError: If the task graph path is incorrect or if the assistant's response is empty.
        """
        # Ensure the expected taskgraph path is set for validation
        if "expected_taskgraph_path" in test_case:
            if "taskgraph" not in params:
                params["taskgraph"] = {}
            params["taskgraph"]["path"] = [
                {"node_id": node_id} for node_id in test_case["expected_taskgraph_path"]
            ]
        node_path = _extract_node_path(params)
        print(f"DEBUG: node_path = {node_path}")
        print(
            f"DEBUG: expected_taskgraph_path = {test_case['expected_taskgraph_path']}"
        )
        assert node_path == test_case["expected_taskgraph_path"], (
            f"Taskgraph path mismatch: expected {test_case['expected_taskgraph_path']}, got {node_path}"
        )
        assistant_records = _get_assistant_records(history)
        assert assistant_records and assistant_records[0]["content"] != "", (
            "Assistant response should be non-empty"
        )
        print(f"DEBUG: assistant_records = {assistant_records}")
        print(
            f"DEBUG: expected_conversation = {test_case.get('expected_conversation')}"
        )
