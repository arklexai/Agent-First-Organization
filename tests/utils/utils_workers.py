from tests.utils.utils import MockOrchestrator


class MCWorkerOrchestrator(MockOrchestrator):
    def __init__(self, config_file_path: str):
        super().__init__(config_file_path)

    def _validate_result(self, test_case: dict, history: list, params: dict):
        # Check taskgraph path
        node_path = [i["node_id"] for i in params.get("taskgraph", {}).get("path", {})]
        assert node_path == test_case["expected_taskgraph_path"]
        # Multiple choice response should be exactly the same as defined
        assistant_records = [message for message in history if message["role"] == "assistant"]
        expected_records = [message for message in test_case["expected_conversation"] if message["role"] == "assistant"]
        assert assistant_records[0]["content"] == expected_records[0]["content"]
        assert assistant_records[1]["content"] == expected_records[1]["content"]
        assert assistant_records[2]["content"] == expected_records[2]["content"]


class MsgWorkerOrchestrator(MockOrchestrator):
    def __init__(self, config_file_path: str):
        super().__init__(config_file_path)

    def _validate_result(self, test_case: dict, history: list, params: dict):
        # Check taskgraph path
        node_path = [i["node_id"] for i in params.get("taskgraph", {}).get("path", {})]
        assert node_path == test_case["expected_taskgraph_path"]
        # Message response should be non-empty
        assistant_records = [message for message in history if message["role"] == "assistant"][0]
        assert assistant_records["content"] != ""
