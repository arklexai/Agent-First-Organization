"""Tool test utilities for the Arklex framework.

This module provides test utilities for validating tool behavior in the Arklex
framework. It includes test orchestrators for Shopify tools, with validation
methods to ensure correct task graph paths and node status.
"""

import os
import json
from typing import Any, Dict, List

from tests.utils.utils import MockOrchestrator


class ShopifyToolOrchestrator(MockOrchestrator):
    def __init__(self, config_file_path: str) -> None:
        """Initialize the Shopify tool orchestrator.

        Args:
            config_file_path (str): Path to the configuration file.
        """
        fixed_args: str = os.environ["SHOPIFY_FIXED_ARGS"]
        self.fixed_args: Dict[str, Any] = json.loads(fixed_args)
        super().__init__(config_file_path, self.fixed_args)

    def _validate_result(
        self,
        test_case: Dict[str, Any],
        history: List[Dict[str, str]],
        params: Dict[str, Any],
    ) -> None:
        """Validate the test results for Shopify tools.

        This function validates that the task graph path and node status match
        the expected values from the test case.

        Args:
            test_case (Dict[str, Any]): Test case containing expected values.
            history (List[Dict[str, str]]): Conversation history.
            params (Dict[str, Any]): Parameters containing task graph information.

        Raises:
            AssertionError: If the task graph path or node status does not match
                the expected values.
        """
        # Check taskgraph path
        node_path: List[str] = [
            i["node_id"] for i in params.get("taskgraph", {}).get("path", {})
        ]
        assert node_path == test_case["expected_taskgraph_path"]
        # Check node status
        node_status: Dict[str, Any] = params.get("taskgraph", {}).get("node_status")
        assert node_status == test_case["expected_node_status"]
