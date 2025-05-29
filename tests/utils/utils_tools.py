import os
import json
from typing import Any, Dict, List

from tests.utils.utils import MockOrchestrator


class ShopifyToolOrchestrator(MockOrchestrator):
    def __init__(self, config_file_path: str) -> None:
        fixed_args: str = os.environ["SHOPIFY_FIXED_ARGS"]
        self.fixed_args: Dict[str, Any] = json.loads(fixed_args)
        super().__init__(config_file_path, self.fixed_args)

    def _validate_result(
        self,
        test_case: Dict[str, Any],
        history: List[Dict[str, str]],
        params: Dict[str, Any],
    ) -> None:
        # Check taskgraph path
        node_path: List[str] = [
            i["node_id"] for i in params.get("taskgraph", {}).get("path", {})
        ]
        assert node_path == test_case["expected_taskgraph_path"]
        # Check node status
        node_status: Dict[str, Any] = params.get("taskgraph", {}).get("node_status")
        assert node_status == test_case["expected_node_status"]
