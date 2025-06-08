"""Worker test utilities for the Arklex framework.

This module provides test utilities for validating worker behavior in the Arklex
framework. It includes test orchestrators for multiple choice workers and message
workers, with validation methods to ensure correct task graph paths and response
content.
"""

import logging
from typing import Any, Dict, List

from tests.utils.utils import MockOrchestrator, MockResourceInitializer

logger = logging.getLogger(__name__)


def _extract_node_path(params: Dict[str, Any]) -> List[str]:
    """Extract the node path from taskgraph parameters, ignoring initial '0' node if present."""
    node_path = [i["node_id"] for i in params.get("taskgraph", {}).get("path", {})]
    if node_path and node_path[0] == "0":
        node_path = node_path[1:]
    return node_path


def _get_assistant_records(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
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

    def _validate_result(
        self,
        test_case: Dict[str, Any],
        history: List[Dict[str, str]],
        params: Dict[str, Any],
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
            zip(assistant_records, expected_records)
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

    def _validate_result(
        self,
        test_case: Dict[str, Any],
        history: List[Dict[str, str]],
        params: Dict[str, Any],
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
