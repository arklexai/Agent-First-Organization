import os
import json

from tests.utils.utils import MockOrchestrator


class ShopifyToolOrchestrator(MockOrchestrator):
    def __init__(self, config_file_path: str):
        fixed_args = os.environ["SHOPIFY_FIXED_ARGS"]
        self.fixed_args = json.loads(fixed_args)
        super().__init__(config_file_path, self.fixed_args)


    def _validate_result(self, test_case: dict, history: list, params: dict):
        # Check taskgraph path
        node_path = [i["node_id"] for i in params.get("taskgraph", {}).get("path", {})]
        assert node_path == test_case["expected_taskgraph_path"]
        # Check node status
        node_status = params.get("taskgraph", {}).get("node_status")
        assert node_status == test_case["expected_node_status"]
