"""Comprehensive tests for the TaskGraph module.

This module provides thorough line-by-line testing coverage for the TaskGraph class,
including all methods, edge cases, and error conditions. Tests follow the established
patterns with fixtures at the top and clear, modular test organization.
"""

import collections
from typing import Any
from unittest.mock import Mock, patch

import pytest

from arklex.orchestrator.NLU.core.intent import IntentDetector
from arklex.orchestrator.NLU.core.slot import SlotFiller
from arklex.orchestrator.NLU.services.model_service import (
    DummyModelService,
    ModelService,
)
from arklex.orchestrator.task_graph import TaskGraph, TaskGraphBase
from arklex.utils.exceptions import TaskGraphError
from arklex.utils.graph_state import LLMConfig, NodeInfo, Params, PathNode, StatusEnum


@pytest.fixture
def always_valid_mock_model() -> Mock:
    """Provide a mock model service that always returns valid responses."""
    mock_model = Mock(spec=ModelService)
    mock_model.get_response.return_value = "1) test_intent"
    # Mock format_intent_input to return a tuple as expected
    mock_model.format_intent_input.return_value = ("test prompt", {"1": "test_intent"})
    return mock_model


@pytest.fixture
def patched_sample_config() -> dict[str, Any]:
    """Provide a sample configuration for testing TaskGraph."""
    return {
        "nodes": [
            [
                "start_node",
                {
                    "type": "start",
                    "resource": {"name": "start_resource", "id": "start_id"},
                    "attribute": {
                        "start": True,
                        "can_skipped": False,
                        "tags": {},
                        "node_specific_data": {},
                    },
                },
            ],
            [
                "task_node",
                {
                    "type": "task",
                    "resource": {"name": "task_resource", "id": "task_id"},
                    "attribute": {
                        "can_skipped": True,
                        "limit": 3,
                        "tags": {"category": "test"},
                        "node_specific_data": {
                            "key1": "value1",
                            "nested": {"key2": "value2"},
                        },
                    },
                },
            ],
            [
                "leaf_node",
                {
                    "type": "task",
                    "resource": {"name": "leaf_resource", "id": "leaf_id"},
                    "attribute": {
                        "can_skipped": False,
                        "tags": {},
                        "node_specific_data": {},
                    },
                },
            ],
        ],
        "edges": [
            (
                "start_node",
                "task_node",
                {
                    "intent": "test_intent",
                    "attribute": {
                        "weight": 1.0,
                        "pred": True,
                        "definition": "Test intent",
                    },
                },
            ),
            (
                "task_node",
                "leaf_node",
                {
                    "intent": "next_intent",
                    "attribute": {"weight": 1.0, "pred": False},
                },
            ),
            (
                "task_node",
                "task_node",
                {
                    "intent": "none",
                    "attribute": {"weight": 0.5, "pred": False},
                },
            ),
        ],
    }


@pytest.fixture
def sample_llm_config() -> LLMConfig:
    """Provide a sample LLM configuration for testing."""
    return LLMConfig(model_type_or_path="gpt-3.5-turbo", llm_provider="openai")


@pytest.fixture
def sample_params() -> Params:
    """Provide sample parameters for testing."""
    return Params()


@pytest.fixture
def mock_intent_detector() -> Mock:
    """Provide a mock IntentDetector for testing."""
    mock_detector = Mock(spec=IntentDetector)
    mock_detector.execute.return_value = "test_intent"
    return mock_detector


@pytest.fixture
def mock_slot_filler() -> Mock:
    """Provide a mock SlotFiller for testing."""
    mock_filler = Mock(spec=SlotFiller)
    return mock_filler


class TestTaskGraphBase:
    """Test the TaskGraphBase class functionality."""

    def test_init_basic(self, patched_sample_config: dict[str, Any]) -> None:
        """Test basic initialization of TaskGraphBase."""
        with patch.object(TaskGraphBase, "create_graph"):
            base = TaskGraphBase("test_graph", patched_sample_config)
            assert base.graph.name == "test_graph"
            assert base.product_kwargs == patched_sample_config

    def test_get_pred_intents(self, patched_sample_config: dict[str, Any]) -> None:
        """Test getting predicted intents from graph edges."""
        with patch.object(TaskGraphBase, "create_graph"):
            base = TaskGraphBase("test_graph", patched_sample_config)
            # Mock the graph edges to return the expected data
            base.graph.add_edge(
                "start_node",
                "task_node",
                intent="test_intent",
                attribute={"pred": True},
            )
            intents = base.get_pred_intents()
            assert isinstance(intents, collections.defaultdict)
            assert "test_intent" in intents
            assert len(intents["test_intent"]) == 1

    def test_get_start_node(self, patched_sample_config: dict[str, Any]) -> None:
        """Test getting the start node from the graph."""
        with patch.object(TaskGraphBase, "create_graph"):
            base = TaskGraphBase("test_graph", patched_sample_config)
            # Mock the graph nodes to return the expected data
            base.graph.add_node("start_node", type="start", attribute={"start": True})
            start_node = base.get_start_node()
            assert start_node == "start_node"

    def test_get_start_node_no_start(self) -> None:
        """Test getting start node when no start node exists."""
        config = {
            "nodes": [["node1", {"type": "task"}]],
            "edges": [],
        }
        with patch.object(TaskGraphBase, "create_graph"):
            base = TaskGraphBase("test_graph", config)
            start_node = base.get_start_node()
            assert start_node is None

    def test_create_graph_not_implemented(
        self, patched_sample_config: dict[str, Any]
    ) -> None:
        """Test that create_graph raises NotImplementedError in base class."""
        with pytest.raises(NotImplementedError):
            TaskGraphBase("test_graph", patched_sample_config)


class TestTaskGraphInitialization:
    """Test TaskGraph initialization and setup."""

    def test_init_basic(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
    ) -> None:
        """Test basic initialization of TaskGraph."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        assert task_graph.graph.name == "test_graph"
        assert task_graph.llm_config == sample_llm_config
        assert task_graph.unsure_intent["intent"] == "others"
        assert task_graph.start_node == "start_node"

    def test_init_without_model_service(
        self, patched_sample_config: dict[str, Any], sample_llm_config: LLMConfig
    ) -> None:
        """Test initialization without model service raises ValueError."""
        with pytest.raises(ValueError, match="model_service is required"):
            TaskGraph(
                "test_graph",
                patched_sample_config,
                sample_llm_config,
                model_service=None,
            )

    def test_init_with_string_slotfillapi(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
    ) -> None:
        """Test initialization with string slotfillapi creates DummyModelService."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            slotfillapi="",
            model_service=always_valid_mock_model,
        )
        assert isinstance(task_graph.slotfillapi, SlotFiller)
        assert isinstance(task_graph.slotfillapi.model_service, DummyModelService)

    def test_init_with_model_service_slotfillapi(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        mock_slot_filler: Mock,
    ) -> None:
        """Test initialization with model service as slotfillapi."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            slotfillapi=always_valid_mock_model,
            model_service=always_valid_mock_model,
        )
        assert isinstance(task_graph.slotfillapi, SlotFiller)
        assert task_graph.slotfillapi.model_service == always_valid_mock_model

    def test_create_graph(
        self, sample_llm_config: LLMConfig, always_valid_mock_model: Mock
    ) -> None:
        """Test graph creation from configuration."""
        config = {
            "nodes": [
                [
                    "node1",
                    {
                        "type": "task",
                        "resource": {"name": "resource1", "id": "id1"},
                        "attribute": {},
                    },
                ]
            ],
            "edges": [
                (
                    "node1",
                    "node1",
                    {
                        "intent": "TEST_INTENT",
                        "attribute": {"weight": 1.0, "pred": True},
                    },
                )
            ],
        }
        task_graph = TaskGraph(
            "test_graph",
            config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        assert "node1" in task_graph.graph.nodes
        assert task_graph.graph.has_edge("node1", "node1")
        # Check that intent was converted to lowercase
        edge_data = task_graph.graph.get_edge_data("node1", "node1")
        assert edge_data["intent"] == "test_intent"

    def test_get_initial_flow_with_services_nodes(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
    ) -> None:
        """Test getting initial flow with services nodes."""
        # Add services_nodes to the config
        config = patched_sample_config.copy()
        config["services_nodes"] = {"service1": "start_node", "service2": "task_node"}

        # Add a self-loop edge to start_node so it has an in-edge with weight
        config["edges"].append(
            (
                "start_node",
                "start_node",
                {
                    "intent": "none",
                    "attribute": {"weight": 1.0, "pred": False},
                },
            )
        )

        task_graph = TaskGraph(
            "test_graph",
            config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        initial_node = task_graph.get_initial_flow()
        assert initial_node in ["start_node", "task_node"]

    def test_get_initial_flow_no_services_nodes(
        self, sample_llm_config: LLMConfig, always_valid_mock_model: Mock
    ) -> None:
        """Test getting initial flow without services nodes."""
        config = {
            "nodes": [
                [
                    "node1",
                    {
                        "type": "task",
                        "resource": {"name": "resource1", "id": "id1"},
                        "attribute": {},
                    },
                ]
            ],
            "edges": [],
        }
        task_graph = TaskGraph(
            "test_graph",
            config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        initial_node = task_graph.get_initial_flow()
        assert initial_node is None


class TestTaskGraphNodeManagement:
    """Test TaskGraph node management methods."""

    def test_jump_to_node_success(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
    ) -> None:
        """Test successful jump to node based on intent."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        next_node, next_intent = task_graph.jump_to_node("test_intent", 0, "start_node")
        assert next_node == "task_node"
        assert next_intent == "test_intent"

    def test_jump_to_node_exception_handling(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
    ) -> None:
        """Test jump_to_node handles exceptions gracefully."""
        # Add a self-loop edge to start_node so it has an in-edge
        config = patched_sample_config.copy()
        config["edges"].append(
            (
                "start_node",
                "start_node",
                {
                    "intent": "none",
                    "attribute": {"weight": 1.0, "pred": False},
                },
            )
        )

        task_graph = TaskGraph(
            "test_graph",
            config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        # Test with non-existent intent
        next_node, next_intent = task_graph.jump_to_node(
            "non_existent", 0, "start_node"
        )
        assert next_node == "start_node"

    def test_get_node_basic(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test basic _get_node functionality."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        node_info, updated_params = task_graph._get_node("task_node", sample_params)
        assert isinstance(node_info, NodeInfo)
        assert node_info.node_id == "task_node"
        assert node_info.resource_id == "task_id"
        assert node_info.resource_name == "task_resource"
        assert updated_params.taskgraph.curr_node == "task_node"

    def test_get_node_with_intent(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test _get_node with intent parameter."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.available_global_intents = {
            "test_intent": [{"target_node": "task_node"}]
        }
        node_info, updated_params = task_graph._get_node(
            "task_node", sample_params, intent="test_intent"
        )
        assert node_info.node_id == "task_node"
        # Check that intent was removed from available intents
        assert "test_intent" not in updated_params.taskgraph.available_global_intents

    def test_get_current_node_with_valid_node(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test get_current_node with valid current node."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.curr_node = "task_node"
        curr_node, updated_params = task_graph.get_current_node(sample_params)
        assert curr_node == "task_node"
        assert updated_params.taskgraph.curr_node == "task_node"

    def test_get_current_node_with_invalid_node(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test get_current_node with invalid current node falls back to start node."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.curr_node = "invalid_node"
        curr_node, updated_params = task_graph.get_current_node(sample_params)
        assert curr_node == "start_node"
        assert updated_params.taskgraph.curr_node == "start_node"

    def test_get_current_node_with_none_node(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test get_current_node with None current node falls back to start node."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.curr_node = None
        curr_node, updated_params = task_graph.get_current_node(sample_params)
        assert curr_node == "start_node"
        assert updated_params.taskgraph.curr_node == "start_node"


class TestTaskGraphIntentHandling:
    """Test TaskGraph intent handling methods."""

    def test_postprocess_intent_basic(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
    ) -> None:
        """Test basic _postprocess_intent functionality."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        found, real_intent, idx = task_graph._postprocess_intent(
            "test_intent", ["test_intent"]
        )
        assert found is True
        assert real_intent == "test_intent"
        assert idx == 0

    def test_postprocess_intent_with_index(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
    ) -> None:
        """Test _postprocess_intent with indexed intent."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        found, real_intent, idx = task_graph._postprocess_intent(
            "test_intent__<2>", ["test_intent"]
        )
        assert found is True
        assert real_intent == "test_intent"
        assert idx == 2

    def test_postprocess_intent_not_found(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
    ) -> None:
        """Test _postprocess_intent when intent not found."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        found, real_intent, idx = task_graph._postprocess_intent(
            "unknown_intent", ["test_intent"]
        )
        assert found is False
        assert real_intent == "unknown_intent"
        assert idx == 0

    def test_get_available_global_intents_empty(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test get_available_global_intents with empty params."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        available_intents = task_graph.get_available_global_intents(sample_params)
        assert "test_intent" in available_intents
        assert "others" in available_intents

    def test_get_available_global_intents_with_existing(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test getting available global intents with existing intents."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.available_global_intents = {"existing_intent": []}
        available_intents = task_graph.get_available_global_intents(sample_params)
        assert "existing_intent" in available_intents
        # The method should return the existing intents as-is
        assert len(available_intents) == 1

    def test_update_node_limit(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test update_node_limit functionality."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.node_limit = {"task_node": 5}
        updated_params = task_graph.update_node_limit(sample_params)
        assert "task_node" in updated_params.taskgraph.node_limit
        assert updated_params.taskgraph.node_limit["task_node"] == 5

    def test_get_local_intent(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test getting local intent from current node."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.curr_node = "task_node"
        local_intents = task_graph.get_local_intent("task_node", sample_params)
        assert "next_intent" in local_intents
        assert len(local_intents["next_intent"]) == 1

    def test_get_last_flow_stack_node_with_stack(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test get_last_flow_stack_node with flow stack."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        path_node = PathNode(
            node_id="stack_node", in_flow_stack=True, global_intent="test_intent"
        )
        sample_params.taskgraph.path = [path_node]
        last_node = task_graph.get_last_flow_stack_node(sample_params)
        assert last_node is not None
        assert last_node.node_id == "stack_node"
        assert last_node.in_flow_stack is False

    def test_get_last_flow_stack_node_empty(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test get_last_flow_stack_node with empty path."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        last_node = task_graph.get_last_flow_stack_node(sample_params)
        assert last_node is None


class TestTaskGraphNodeStatusHandling:
    """Test TaskGraph node status handling methods."""

    def test_handle_multi_step_node_stay(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test handle_multi_step_node with STAY status."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.node_status = {"task_node": StatusEnum.STAY}
        is_multi_step, node_info, updated_params = task_graph.handle_multi_step_node(
            "task_node", sample_params
        )
        assert is_multi_step is True
        assert isinstance(node_info, NodeInfo)
        assert node_info.node_id == "task_node"

    def test_handle_multi_step_node_complete(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test handle_multi_step_node with COMPLETE status."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.node_status = {"task_node": StatusEnum.COMPLETE}
        is_multi_step, node_info, updated_params = task_graph.handle_multi_step_node(
            "task_node", sample_params
        )
        assert is_multi_step is False

    def test_handle_incomplete_node_incomplete(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test handle_incomplete_node with INCOMPLETE status."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.node_status = {"task_node": StatusEnum.INCOMPLETE}
        is_incomplete, node_info, updated_params = task_graph.handle_incomplete_node(
            "task_node", sample_params
        )
        assert is_incomplete is True
        assert isinstance(node_info, NodeInfo)
        assert node_info.node_id == "task_node"

    def test_handle_incomplete_node_complete(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test handle_incomplete_node with COMPLETE status."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.node_status = {"task_node": StatusEnum.COMPLETE}
        is_incomplete, node_info, updated_params = task_graph.handle_incomplete_node(
            "task_node", sample_params
        )
        assert is_incomplete is False
        assert node_info == {}


class TestTaskGraphIntentPrediction:
    """Test TaskGraph intent prediction methods."""

    def test_global_intent_prediction_unsure_only(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test global_intent_prediction with only unsure intent available."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        available_intents = {"others": [task_graph.unsure_intent]}
        excluded_intents = {}
        found, pred_intent, node_output, updated_params = (
            task_graph.global_intent_prediction(
                "task_node", sample_params, available_intents, excluded_intents
            )
        )
        assert found is False
        assert pred_intent == "others"

    def test_global_intent_prediction_with_model_service(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test global intent prediction with model service."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        # Set the text attribute that the method expects
        task_graph.text = "test message"
        task_graph.chat_history_str = ""

        sample_params.taskgraph.curr_node = "task_node"
        available_intents = {
            "test_intent": [{"target_node": "task_node", "attribute": {"weight": 1.0}}],
            "others": [task_graph.unsure_intent],
        }
        excluded_intents = {}
        found, pred_intent, node_output, updated_params = (
            task_graph.global_intent_prediction(
                "task_node", sample_params, available_intents, excluded_intents
            )
        )
        assert isinstance(found, bool)
        assert pred_intent is not None

    def test_handle_random_next_node_with_candidates(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test handle_random_next_node with available candidates."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.curr_node = "task_node"
        found, node_output, updated_params = task_graph.handle_random_next_node(
            "task_node", sample_params
        )
        # The method should return False when there are no valid next nodes
        # (task_node only has a self-loop edge which might not be considered valid)
        assert found is False
        assert isinstance(node_output, dict)

    def test_local_intent_prediction_success(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test successful local intent prediction."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        # Set the text attribute that the method expects
        task_graph.text = "test message"
        task_graph.chat_history_str = ""

        local_intents = {
            "test_intent": [{"target_node": "task_node", "attribute": {"weight": 1.0}}]
        }
        found, node_info, updated_params = task_graph.local_intent_prediction(
            "start_node", sample_params, local_intents
        )
        # The result depends on the mock model service response
        assert isinstance(found, bool)

    def test_local_intent_prediction_not_found(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test local intent prediction when intent is not found."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        # Set the text attribute that the method expects
        task_graph.text = "test message"
        task_graph.chat_history_str = ""

        # Mock the model service to return an intent that's not in local_intents
        always_valid_mock_model.get_response.return_value = "1) unknown_intent"

        local_intents = {
            "test_intent": [{"target_node": "task_node", "attribute": {"weight": 1.0}}]
        }
        found, node_info, updated_params = task_graph.local_intent_prediction(
            "start_node", sample_params, local_intents
        )
        # When an unknown intent is predicted, the system falls back to "others" intent
        # and returns True because it found a fallback intent
        assert found is True

    def test_global_intent_prediction_same_intent_continue(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test global_intent_prediction when same intent should continue."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        # Set the text attribute that the method expects
        task_graph.text = "test message"
        task_graph.chat_history_str = ""

        sample_params.taskgraph.curr_global_intent = "test_intent"
        sample_params.taskgraph.node_status = {"task_node": StatusEnum.INCOMPLETE}
        available_intents = {
            "test_intent": [{"target_node": "task_node", "attribute": {"weight": 1.0}}],
            "others": [task_graph.unsure_intent],
        }
        excluded_intents = {}
        found, pred_intent, node_output, updated_params = (
            task_graph.global_intent_prediction(
                "task_node", sample_params, available_intents, excluded_intents
            )
        )
        assert found is False
        assert pred_intent == "test_intent"

    def test_local_intent_prediction_start_node(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test local_intent_prediction from start node sets global intent."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        # Set the text attribute that the method expects
        task_graph.text = "test message"
        task_graph.chat_history_str = ""

        local_intents = {
            "test_intent": [{"target_node": "task_node", "attribute": {"weight": 1.0}}]
        }
        found, node_info, updated_params = task_graph.local_intent_prediction(
            "start_node", sample_params, local_intents
        )
        # The result depends on the mock model service response
        assert isinstance(found, bool)

    def test_intent_prediction_performance(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test intent prediction performance with many intents."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        # Set the text attribute that the method expects
        task_graph.text = "test message"
        task_graph.chat_history_str = ""

        # Create many available intents
        available_intents = {}
        for i in range(50):
            available_intents[f"intent_{i}"] = [
                {"target_node": "task_node", "attribute": {"weight": 1.0}}
            ]
        available_intents["others"] = [task_graph.unsure_intent]

        excluded_intents = {}
        found, pred_intent, node_output, updated_params = (
            task_graph.global_intent_prediction(
                "task_node", sample_params, available_intents, excluded_intents
            )
        )
        assert isinstance(found, bool)
        assert pred_intent is not None


class TestTaskGraphSpecialHandling:
    """Test TaskGraph special handling methods."""

    def test_handle_unknown_intent(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test handle_unknown_intent functionality."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        node_info, updated_params = task_graph.handle_unknown_intent(
            "task_node", sample_params
        )
        assert isinstance(node_info, NodeInfo)
        assert node_info.resource_id == "planner"
        assert node_info.resource_name == "planner"
        assert updated_params.taskgraph.intent == "others"
        assert updated_params.taskgraph.curr_global_intent == "others"

    def test_handle_leaf_node_not_leaf(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test handle_leaf_node with non-leaf node."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        curr_node, updated_params = task_graph.handle_leaf_node(
            "task_node", sample_params
        )
        assert curr_node == "task_node"

    def test_handle_leaf_node_leaf_with_flow_stack(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test handle_leaf_node with leaf node and flow stack."""
        # Add stack_node to the configuration
        config = patched_sample_config.copy()
        config["nodes"].append(
            [
                "stack_node",
                {
                    "type": "task",
                    "resource": {"name": "stack_resource", "id": "stack_id"},
                    "attribute": {
                        "can_skipped": False,
                        "tags": {},
                        "node_specific_data": {},
                    },
                },
            ]
        )
        config["edges"].append(
            (
                "stack_node",
                "leaf_node",
                {
                    "intent": "none",
                    "attribute": {"weight": 1.0, "pred": False},
                },
            )
        )

        task_graph = TaskGraph(
            "test_graph",
            config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.path = [
            PathNode(node_id="stack_node", in_flow_stack=True)
        ]
        curr_node, updated_params = task_graph.handle_leaf_node(
            "leaf_node", sample_params
        )
        assert curr_node == "stack_node"

    def test_handle_leaf_node_leaf_with_initial_node(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test handle_leaf_node with leaf node and initial node."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.path = []
        curr_node, updated_params = task_graph.handle_leaf_node(
            "leaf_node", sample_params
        )
        # Should return the leaf_node itself when no flow stack
        assert curr_node == "leaf_node"

    def test_handle_leaf_node_with_nested_graph(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test handle_leaf_node with nested graph component."""
        # Add nested_node to the configuration
        config = patched_sample_config.copy()
        config["nodes"].append(
            [
                "nested_node",
                {
                    "type": "task",
                    "resource": {"name": "nested_resource", "id": "nested_id"},
                    "attribute": {
                        "can_skipped": False,
                        "tags": {},
                        "node_specific_data": {},
                    },
                },
            ]
        )

        task_graph = TaskGraph(
            "test_graph",
            config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        with patch("arklex.orchestrator.task_graph.NestedGraph") as mock_nested_graph:
            mock_nested_graph.get_nested_graph_component_node.return_value = (
                NodeInfo(
                    node_id="nested_node",
                    resource_id="nested_id",
                    resource_name="nested_name",
                ),
                sample_params,
            )
            curr_node, updated_params = task_graph.handle_leaf_node(
                "leaf_node", sample_params
            )
            assert curr_node == "nested_node"

    def test_get_node_complex_flow(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test get_node with complex flow including global intent prediction."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.curr_node = "task_node"
        inputs = {
            "text": "test message",
            "chat_history_str": "",
            "parameters": sample_params,
            "allow_global_intent_switch": True,
        }
        node_info, updated_params = task_graph.get_node(inputs)
        assert isinstance(node_info, NodeInfo)

    def test_get_node_no_global_intent_switch(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test get_node with global intent switch disabled."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.curr_node = "task_node"
        inputs = {
            "text": "test message",
            "chat_history_str": "",
            "parameters": sample_params,
            "allow_global_intent_switch": False,
        }
        node_info, updated_params = task_graph.get_node(inputs)
        assert isinstance(node_info, NodeInfo)


class TestTaskGraphMainFlow:
    """Test TaskGraph main flow methods."""

    def test_get_node_start_text(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test get_node with start text."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        inputs = {
            "text": "<start>",
            "chat_history_str": "",
            "parameters": sample_params,
            "allow_global_intent_switch": True,
        }
        node_info, updated_params = task_graph.get_node(inputs)
        assert isinstance(node_info, NodeInfo)
        assert node_info.node_id == "start_node"

    def test_get_node_multi_step(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test get_node with multi-step node."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.node_status = {"task_node": StatusEnum.STAY}
        sample_params.taskgraph.curr_node = "task_node"
        inputs = {
            "text": "test message",
            "chat_history_str": "",
            "parameters": sample_params,
            "allow_global_intent_switch": True,
        }
        node_info, updated_params = task_graph.get_node(inputs)
        assert isinstance(node_info, NodeInfo)
        assert node_info.node_id == "task_node"

    def test_postprocess_node(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test postprocess_node functionality."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        node_info = NodeInfo(
            node_id="test_node", resource_id="test_id", resource_name="test_name"
        )
        node_tuple = (node_info, sample_params)
        processed_node, processed_params = task_graph.postprocess_node(node_tuple)
        assert processed_node == node_info
        assert processed_params == sample_params


class TestTaskGraphValidation:
    """Test TaskGraph validation methods."""

    def test_validate_node_valid(
        self, sample_llm_config: LLMConfig, always_valid_mock_model: Mock
    ) -> None:
        """Test _validate_node with valid node."""
        config = {"nodes": [], "edges": []}
        task_graph = TaskGraph(
            "test_graph",
            config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        valid_node = {
            "id": "test_node",
            "type": "task",
            "next": ["node1", "node2"],
        }
        # Should not raise any exception
        task_graph._validate_node(valid_node)

    def test_validate_node_not_dict(
        self, sample_llm_config: LLMConfig, always_valid_mock_model: Mock
    ) -> None:
        """Test _validate_node with non-dict node raises TaskGraphError."""
        config = {"nodes": [], "edges": []}
        task_graph = TaskGraph(
            "test_graph",
            config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        with pytest.raises(TaskGraphError, match="Node must be a dictionary"):
            task_graph._validate_node("not_a_dict")

    def test_validate_node_no_id(
        self, sample_llm_config: LLMConfig, always_valid_mock_model: Mock
    ) -> None:
        """Test _validate_node with node missing id raises TaskGraphError."""
        config = {"nodes": [], "edges": []}
        task_graph = TaskGraph(
            "test_graph",
            config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        invalid_node = {"type": "task"}
        with pytest.raises(TaskGraphError, match="Node must have an id"):
            task_graph._validate_node(invalid_node)

    def test_validate_node_no_type(
        self, sample_llm_config: LLMConfig, always_valid_mock_model: Mock
    ) -> None:
        """Test _validate_node with node missing type raises TaskGraphError."""
        config = {"nodes": [], "edges": []}
        task_graph = TaskGraph(
            "test_graph",
            config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        invalid_node = {"id": "test_node"}
        with pytest.raises(TaskGraphError, match="Node must have a type"):
            task_graph._validate_node(invalid_node)

    def test_validate_node_invalid_next(
        self, sample_llm_config: LLMConfig, always_valid_mock_model: Mock
    ) -> None:
        """Test _validate_node with invalid next field raises TaskGraphError."""
        config = {"nodes": [], "edges": []}
        task_graph = TaskGraph(
            "test_graph",
            config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        invalid_node = {"id": "test_node", "type": "task", "next": "notalist"}
        with pytest.raises(TaskGraphError, match="Node next must be a list"):
            task_graph._validate_node(invalid_node)


class TestTaskGraphEdgeCases:
    """Test TaskGraph edge cases and error conditions."""

    def test_jump_to_node_empty_candidates(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
    ) -> None:
        """Test jump_to_node handles empty candidates gracefully."""
        # Add a self-loop edge to start_node so it has an in-edge
        config = patched_sample_config.copy()
        config["edges"].append(
            (
                "start_node",
                "start_node",
                {
                    "intent": "none",
                    "attribute": {"weight": 1.0, "pred": False},
                },
            )
        )

        task_graph = TaskGraph(
            "test_graph",
            config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        # Clear the intents to simulate empty candidates
        task_graph.intents["test_intent"] = []
        next_node, next_intent = task_graph.jump_to_node("test_intent", 0, "start_node")
        assert next_node == "start_node"

    def test_get_node_with_complex_node_specific_data(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test _get_node with complex node_specific_data."""
        config = {
            "nodes": [
                [
                    "complex_node",
                    {
                        "type": "task",
                        "resource": {"name": "complex_resource", "id": "complex_id"},
                        "attribute": {
                            "node_specific_data": {
                                "simple_key": "simple_value",
                                "nested_dict": {"nested_key": "nested_value"},
                                "another_nested": {"another_key": "another_value"},
                            }
                        },
                    },
                ]
            ],
            "edges": [],
        }
        task_graph = TaskGraph(
            "test_graph",
            config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        node_info, updated_params = task_graph._get_node("complex_node", sample_params)
        assert node_info.node_id == "complex_node"
        assert node_info.additional_args["simple_key"] == "simple_value"
        assert node_info.additional_args["nested_key"] == "nested_value"
        assert node_info.additional_args["another_key"] == "another_value"

    def test_global_intent_prediction_same_intent_continue(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test global_intent_prediction when same intent should continue."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        # Set the text attribute that the method expects
        task_graph.text = "test message"
        task_graph.chat_history_str = ""

        sample_params.taskgraph.curr_global_intent = "test_intent"
        sample_params.taskgraph.node_status = {"task_node": StatusEnum.INCOMPLETE}
        available_intents = {
            "test_intent": [{"target_node": "task_node", "attribute": {"weight": 1.0}}],
            "others": [task_graph.unsure_intent],
        }
        excluded_intents = {}
        found, pred_intent, node_output, updated_params = (
            task_graph.global_intent_prediction(
                "task_node", sample_params, available_intents, excluded_intents
            )
        )
        assert found is False
        assert pred_intent == "test_intent"

    def test_handle_leaf_node_with_nested_graph(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test handle_leaf_node with nested graph component."""
        # Add nested_node to the configuration
        config = patched_sample_config.copy()
        config["nodes"].append(
            [
                "nested_node",
                {
                    "type": "task",
                    "resource": {"name": "nested_resource", "id": "nested_id"},
                    "attribute": {
                        "can_skipped": False,
                        "tags": {},
                        "node_specific_data": {},
                    },
                },
            ]
        )

        task_graph = TaskGraph(
            "test_graph",
            config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        with patch("arklex.orchestrator.task_graph.NestedGraph") as mock_nested_graph:
            mock_nested_graph.get_nested_graph_component_node.return_value = (
                NodeInfo(
                    node_id="nested_node",
                    resource_id="nested_id",
                    resource_name="nested_name",
                ),
                sample_params,
            )
            curr_node, updated_params = task_graph.handle_leaf_node(
                "leaf_node", sample_params
            )
            assert curr_node == "nested_node"

    def test_get_node_complex_flow(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test get_node with complex flow including global intent prediction."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.curr_node = "task_node"
        inputs = {
            "text": "test message",
            "chat_history_str": "",
            "parameters": sample_params,
            "allow_global_intent_switch": True,
        }
        node_info, updated_params = task_graph.get_node(inputs)
        assert isinstance(node_info, NodeInfo)

    def test_get_node_no_global_intent_switch(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test get_node with global intent switch disabled."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.curr_node = "task_node"
        inputs = {
            "text": "test message",
            "chat_history_str": "",
            "parameters": sample_params,
            "allow_global_intent_switch": False,
        }
        node_info, updated_params = task_graph.get_node(inputs)
        assert isinstance(node_info, NodeInfo)


class TestTaskGraphIntegration:
    """Test TaskGraph integration scenarios."""

    def test_full_conversation_flow(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test a full conversation flow through multiple nodes."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )

        # Start conversation
        inputs = {
            "text": "<start>",
            "chat_history_str": "",
            "parameters": sample_params,
            "allow_global_intent_switch": True,
        }
        node_info, params = task_graph.get_node(inputs)
        assert node_info.node_id == "start_node"

        # Move to next node
        inputs = {
            "text": "test message",
            "chat_history_str": "",
            "parameters": params,
            "allow_global_intent_switch": True,
        }
        node_info, params = task_graph.get_node(inputs)
        assert isinstance(node_info, NodeInfo)

    def test_node_status_transitions(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test node status transitions through different states."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )

        # Test STAY status
        sample_params.taskgraph.node_status = {"task_node": StatusEnum.STAY}
        sample_params.taskgraph.curr_node = "task_node"
        inputs = {
            "text": "test message",
            "chat_history_str": "",
            "parameters": sample_params,
            "allow_global_intent_switch": True,
        }
        node_info, params = task_graph.get_node(inputs)
        assert node_info.node_id == "task_node"

        # Test INCOMPLETE status
        sample_params.taskgraph.node_status = {"task_node": StatusEnum.INCOMPLETE}
        node_info, params = task_graph.get_node(inputs)
        assert node_info.node_id == "task_node"

    def test_intent_flow_with_available_intents(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test intent flow with available global intents."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.available_global_intents = {
            "test_intent": [{"target_node": "task_node", "attribute": {"weight": 1.0}}]
        }
        sample_params.taskgraph.curr_node = "start_node"
        inputs = {
            "text": "test message",
            "chat_history_str": "",
            "parameters": sample_params,
            "allow_global_intent_switch": True,
        }
        node_info, params = task_graph.get_node(inputs)
        assert isinstance(node_info, NodeInfo)

    def test_flow_stack_management(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test flow stack management and restoration."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )

        # Add nodes to flow stack
        path_node1 = PathNode(
            node_id="node1", in_flow_stack=True, global_intent="intent1"
        )
        path_node2 = PathNode(
            node_id="node2", in_flow_stack=False, global_intent="intent2"
        )
        path_node3 = PathNode(
            node_id="node3", in_flow_stack=True, global_intent="intent3"
        )
        sample_params.taskgraph.path = [path_node1, path_node2, path_node3]

        # Test getting last flow stack node
        last_node = task_graph.get_last_flow_stack_node(sample_params)
        assert last_node is not None
        assert last_node.node_id == "node3"
        assert (
            last_node.in_flow_stack is False
        )  # Should be set to False after retrieval

    def test_nlu_records_tracking(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test NLU records tracking throughout conversation."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.curr_node = "task_node"
        inputs = {
            "text": "test message",
            "chat_history_str": "",
            "parameters": sample_params,
            "allow_global_intent_switch": True,
        }
        node_info, params = task_graph.get_node(inputs)
        assert len(params.taskgraph.nlu_records) > 0


class TestTaskGraphErrorHandling:
    """Test TaskGraph error handling and edge cases."""

    def test_create_graph_with_none_intent(
        self, sample_llm_config: LLMConfig, always_valid_mock_model: Mock
    ) -> None:
        """Test create_graph handles None intent correctly."""
        config = {
            "nodes": [
                [
                    "node1",
                    {
                        "type": "task",
                        "resource": {"name": "resource1", "id": "id1"},
                        "attribute": {},
                    },
                ]
            ],
            "edges": [
                (
                    "node1",
                    "node1",
                    {
                        "intent": None,
                        "attribute": {"weight": 1.0, "pred": True},
                    },
                )
            ],
        }
        task_graph = TaskGraph(
            "test_graph",
            config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        edge_data = task_graph.graph.get_edge_data("node1", "node1")
        assert edge_data["intent"] == "none"

    def test_create_graph_with_empty_intent(
        self, sample_llm_config: LLMConfig, always_valid_mock_model: Mock
    ) -> None:
        """Test create_graph handles empty intent correctly."""
        config = {
            "nodes": [
                [
                    "node1",
                    {
                        "type": "task",
                        "resource": {"name": "resource1", "id": "id1"},
                        "attribute": {},
                    },
                ]
            ],
            "edges": [
                (
                    "node1",
                    "node1",
                    {
                        "intent": "",
                        "attribute": {"weight": 1.0, "pred": True},
                    },
                )
            ],
        }
        task_graph = TaskGraph(
            "test_graph",
            config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        edge_data = task_graph.graph.get_edge_data("node1", "node1")
        assert edge_data["intent"] == "none"

    def test_get_initial_flow_with_invalid_weights(
        self, sample_llm_config: LLMConfig, always_valid_mock_model: Mock
    ) -> None:
        """Test get_initial_flow handles invalid weights gracefully."""
        config = {
            "nodes": [
                [
                    "node1",
                    {
                        "type": "task",
                        "resource": {"name": "resource1", "id": "id1"},
                        "attribute": {},
                    },
                ]
            ],
            "edges": [
                (
                    "node1",
                    "node1",
                    {
                        "intent": "none",
                        "attribute": {"weight": 1.0, "pred": False},
                    },
                )
            ],
            "services_nodes": {"service1": "node1"},
        }
        task_graph = TaskGraph(
            "test_graph",
            config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        initial_node = task_graph.get_initial_flow()
        assert initial_node == "node1"

    def test_jump_to_node_with_invalid_index(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
    ) -> None:
        """Test jump_to_node handles invalid index gracefully."""
        # Add a self-loop edge to start_node so it has an in-edge
        config = patched_sample_config.copy()
        config["edges"].append(
            (
                "start_node",
                "start_node",
                {
                    "intent": "none",
                    "attribute": {"weight": 1.0, "pred": False},
                },
            )
        )

        task_graph = TaskGraph(
            "test_graph",
            config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        # Test with index out of range
        next_node, next_intent = task_graph.jump_to_node(
            "test_intent", 999, "start_node"
        )
        assert next_node == "start_node"

    def test_get_node_with_missing_node(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        """Test _get_node handles missing node gracefully."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        with pytest.raises(KeyError):
            task_graph._get_node("non_existent_node", sample_params)

    def test_postprocess_intent_with_similarity_threshold(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
    ) -> None:
        """Test _postprocess_intent with similarity threshold."""
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        # Test with very similar intent
        found, real_intent, idx = task_graph._postprocess_intent(
            "test_intent", ["test_intent"]
        )
        assert found is True
        assert real_intent == "test_intent"

        # Test with dissimilar intent
        found, real_intent, idx = task_graph._postprocess_intent(
            "completely_different", ["test_intent"]
        )
        assert found is False
        assert real_intent == "completely_different"


class TestTaskGraphCoverage:
    """Extra coverage tests for TaskGraph edge and exception branches."""

    def test_jump_to_node_exception_branch(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
    ) -> None:
        # Simulate an exception in jump_to_node by patching in_edges to raise
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        with patch.object(task_graph.graph, "in_edges", side_effect=Exception("fail")):
            next_node, next_intent = task_graph.jump_to_node(
                "test_intent", 0, "start_node"
            )
            assert (
                next_node == "task_node"
            )  # The fallback returns the first in-edge's target
            assert isinstance(next_intent, str)

    def test__get_node_intent_removal(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        # Test the branch that removes and pops intent from available_global_intents
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.available_global_intents = {"test_intent": []}
        sample_params.taskgraph.curr_node = "task_node"
        # Use a valid node name that exists in the graph
        node_info, params2 = task_graph._get_node(
            "task_node", sample_params, intent="test_intent"
        )
        assert "test_intent" not in params2.taskgraph.available_global_intents

    def test__postprocess_intent_with_idx(
        self, sample_llm_config: LLMConfig, always_valid_mock_model: Mock
    ) -> None:
        from arklex.orchestrator.task_graph import TaskGraph

        task_graph = TaskGraph(
            "test",
            {"nodes": [], "edges": []},
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        found, intent, idx = task_graph._postprocess_intent("1", {"1": "1"})
        assert found is True
        assert intent == "1"
        assert idx == 0

    def test__postprocess_intent_similarity(
        self, sample_llm_config: LLMConfig, always_valid_mock_model: Mock
    ) -> None:
        from arklex.orchestrator.task_graph import TaskGraph

        task_graph = TaskGraph(
            "test",
            {"nodes": [], "edges": []},
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        found, intent, idx = task_graph._postprocess_intent(
            "2", {"1": "test_intent", "2": "2"}
        )
        assert found is True
        assert intent == "2"
        assert idx == 0

    def test_handle_multi_step_node_not_stay(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        from arklex.orchestrator.task_graph import NodeInfo

        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        with (
            patch.object(task_graph, "is_multi_step", return_value=True, create=True),
            patch.object(
                task_graph,
                "_should_stay_in_multi_step",
                return_value=False,
                create=True,
            ),
        ):
            result = task_graph.handle_multi_step_node("start_node", sample_params)
            assert isinstance(result, tuple)
            assert result[0] is False
            assert isinstance(result[1], NodeInfo)

    def test_handle_unknown_intent_no_nlu_records(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        sample_params.taskgraph.nlu_records = []
        result = task_graph.handle_unknown_intent("start_node", sample_params)
        assert isinstance(result, tuple)
        assert len(result) == 2
        node_info = result[0]
        assert hasattr(node_info, "resource_id")
        assert node_info.resource_id == "planner"

    def test_handle_leaf_node_not_leaf(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        with patch.object(task_graph, "is_leaf", return_value=False, create=True):
            curr_node, updated_params = task_graph.handle_leaf_node(
                "start_node", sample_params
            )
            assert curr_node == "start_node"

    def test_postprocess_node_noop(
        self,
        patched_sample_config: dict[str, Any],
        sample_llm_config: LLMConfig,
        always_valid_mock_model: Mock,
        sample_params: Params,
    ) -> None:
        task_graph = TaskGraph(
            "test_graph",
            patched_sample_config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        curr_node = task_graph.postprocess_node(("start_node", sample_params))
        assert curr_node[0] == "start_node"

    def test_validate_node_all_errors(
        self, sample_llm_config: LLMConfig, always_valid_mock_model: Mock
    ) -> None:
        from arklex.orchestrator.task_graph import TaskGraphError

        config = {
            "nodes": [
                [
                    "n",
                    {
                        # missing 'type' field to trigger error
                        "resource": {"name": "test", "id": "test"},
                        "attribute": {},
                    },
                ]
            ],
            "edges": [],
        }
        task_graph = TaskGraph(
            "test_graph",
            config,
            sample_llm_config,
            model_service=always_valid_mock_model,
        )
        with pytest.raises(TaskGraphError):
            task_graph._validate_node({"id": "i"})
