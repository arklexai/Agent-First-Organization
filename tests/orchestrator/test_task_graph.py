import pytest
from arklex.orchestrator.task_graph import TaskGraph
from arklex.utils.graph_state import LLMConfig


@pytest.fixture
def dummy_config():
    return {
        "nodes": [],
        "edges": [],
        "intent_api": {
            "model_name": "dummy",
            "api_key": "dummy",
            "endpoint": "http://dummy",
            "model_type_or_path": "dummy-path",
            "llm_provider": "dummy",
        },
        "slotfillapi": "",
    }


@pytest.fixture
def dummy_llm_config():
    return LLMConfig(
        model_name="dummy",
        api_key="dummy",
        endpoint="http://dummy",
        model_type_or_path="dummy-path",
        llm_provider="dummy",
    )


def test_task_graph_requires_model_service(dummy_config, dummy_llm_config) -> None:
    with pytest.raises(
        ValueError, match="model_service is required for TaskGraph and cannot be None"
    ):
        TaskGraph(
            name="test",
            product_kwargs=dummy_config,
            llm_config=dummy_llm_config,
            model_service=None,
        )
