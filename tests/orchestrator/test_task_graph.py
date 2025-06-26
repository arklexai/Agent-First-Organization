import pytest
from unittest.mock import Mock, patch, MagicMock
from arklex.orchestrator.task_graph import TaskGraph, TaskGraphError
from arklex.orchestrator.orchestrator import NodeInfo, PathNode
from arklex.utils.graph_state import (
    LLMConfig,
    Params,
    NodeInfo,
    StatusEnum,
    PathNode,
    Taskgraph,
    Memory,
    Metadata,
)
from arklex.orchestrator.NLU.services.model_service import DummyModelService
from arklex.orchestrator.NLU.core.slot import SlotFiller
from contextlib import ExitStack


@pytest.fixture
def dummy_config():
    return {
        "nodes": [
            (
                "start",
                {
                    "type": "start",
                    "resource": {"name": "start_resource", "id": "start_id"},
                    "attribute": {"start": True, "weight": 1.0},
                },
            ),
            (
                "node1",
                {
                    "type": "task",
                    "resource": {"name": "task1", "id": "task1_id"},
                    "attribute": {"weight": 1.0, "can_skipped": False},
                },
            ),
            (
                "node2",
                {
                    "type": "task",
                    "resource": {"name": "task2", "id": "task2_id"},
                    "attribute": {"weight": 1.0, "can_skipped": True},
                },
            ),
            (
                "end",
                {
                    "type": "end",
                    "resource": {"name": "end_resource", "id": "end_id"},
                    "attribute": {"weight": 1.0},
                },
            ),
        ],
        "edges": [
            (
                "start",
                "node1",
                {"intent": "start", "attribute": {"weight": 1.0}, "pred": True},
            ),
            (
                "node1",
                "node2",
                {"intent": "continue", "attribute": {"weight": 1.0}, "pred": True},
            ),
            (
                "node2",
                "end",
                {"intent": "finish", "attribute": {"weight": 1.0}, "pred": True},
            ),
            (
                "node1",
                "end",
                {"intent": "skip", "attribute": {"weight": 0.5}, "pred": True},
            ),
            # Add a dummy incoming edge for start node to prevent IndexError
            (
                "dummy",
                "start",
                {"intent": "dummy", "attribute": {"weight": 1.0}, "pred": True},
            ),
        ],
        "intent_api": {
            "model_name": "dummy",
            "api_key": "dummy",
            "endpoint": "http://dummy",
            "model_type_or_path": "dummy-path",
            "llm_provider": "dummy",
        },
        "slotfillapi": "",
        "services_nodes": {"service1": "node1", "service2": "node2"},
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


@pytest.fixture
def mock_model_service():
    return DummyModelService(
        {
            "model_name": "dummy",
            "api_key": "dummy",
            "endpoint": "http://dummy",
            "model_type_or_path": "dummy-path",
            "llm_provider": "dummy",
        }
    )


@pytest.fixture
def task_graph(dummy_config, dummy_llm_config, mock_model_service):
    task_graph = TaskGraph(
        name="test_graph",
        product_kwargs=dummy_config,
        llm_config=dummy_llm_config,
        model_service=mock_model_service,
    )
    # Set required attributes that are normally set in get_node method
    task_graph.text = "test text"
    task_graph.chat_history_str = "test history"
    return task_graph


@pytest.fixture
def params():
    return Params(taskgraph=Taskgraph(), memory=Memory(), metadata=Metadata())


class TestTaskGraph:
    """Test the TaskGraph class functionality."""

    def test_task_graph_requires_model_service(
        self, dummy_config, dummy_llm_config
    ) -> None:
        """Test that TaskGraph requires a model service."""
        with pytest.raises(
            ValueError,
            match="model_service is required for TaskGraph and cannot be None",
        ):
            TaskGraph(
                name="test",
                product_kwargs=dummy_config,
                llm_config=dummy_llm_config,
                model_service=None,
            )

    def test_task_graph_initialization(self, task_graph) -> None:
        """Test TaskGraph initialization."""
        assert task_graph.graph.name == "test_graph"
        assert task_graph.unsure_intent["intent"] == "others"
        assert task_graph.initial_node is not None
        # The intent_detector and slotfillapi are real objects, just check they're not None
        assert task_graph.intent_detector is not None
        assert task_graph.slotfillapi is not None

    def test_create_graph(self, task_graph) -> None:
        """Test graph creation from nodes and edges."""
        # Check that nodes were added
        assert "start" in task_graph.graph.nodes
        assert "node1" in task_graph.graph.nodes
        assert "node2" in task_graph.graph.nodes
        assert "end" in task_graph.graph.nodes

        # Check that edges were added with lowercase intents
        edges = list(task_graph.graph.edges(data=True))
        intents = [edge[2]["intent"] for edge in edges]
        assert all(intent == intent.lower() for intent in intents)

    def test_get_initial_flow_with_services_nodes(self, task_graph) -> None:
        """Test getting initial flow with services_nodes configuration."""
        # Mock numpy.random.choice to return predictable result
        with patch("numpy.random.choice") as mock_choice:
            mock_choice.return_value = "node1"
            initial_flow = task_graph.get_initial_flow()
            assert initial_flow == "node1"

    def test_get_initial_flow_without_services_nodes(
        self, dummy_config, dummy_llm_config, mock_model_service
    ) -> None:
        """Test getting initial flow without services_nodes configuration."""
        config_without_services = dummy_config.copy()
        config_without_services.pop("services_nodes", None)

        task_graph = TaskGraph(
            name="test",
            product_kwargs=config_without_services,
            llm_config=dummy_llm_config,
            model_service=mock_model_service,
        )

        initial_flow = task_graph.get_initial_flow()
        assert initial_flow is None

    def test_jump_to_node_success(self, task_graph) -> None:
        """Test successful jump to node based on intent."""
        # Set up the intents structure properly
        task_graph.intents = {
            "start": [{"target_node": "node1", "attribute": {"weight": 1.0}}]
        }

        with patch("numpy.random.choice") as mock_choice:
            mock_choice.return_value = "node1"
            next_node, next_intent = task_graph.jump_to_node("start", 0, "start")
            assert next_node == "node1"
            assert next_intent == "start"

    def test_jump_to_node_exception_handling(self, task_graph) -> None:
        """Test jump_to_node handles exceptions gracefully."""
        # Mock intents to cause an exception
        task_graph.intents = {}

        next_node, next_intent = task_graph.jump_to_node("nonexistent", 0, "start")
        assert next_node == "start"  # Should return current node on error
        # The next_intent should be the intent from the incoming edge
        assert next_intent == "dummy"  # From the dummy edge we added

    def test_get_node(self, task_graph, params) -> None:
        """Test getting node information."""
        # Setup params with available global intents
        params.taskgraph.available_global_intents = {
            "start": [{"target_node": "node1"}]
        }

        node_info, updated_params = task_graph._get_node("node1", params, "start")

        assert isinstance(node_info, NodeInfo)
        assert node_info.node_id == "node1"
        assert node_info.resource_name == "task1"
        assert node_info.resource_id == "task1_id"
        assert updated_params.taskgraph.curr_node == "node1"

    def test_get_node_without_intent(self, task_graph, params) -> None:
        """Test getting node information without intent."""
        node_info, updated_params = task_graph._get_node("node1", params)

        assert isinstance(node_info, NodeInfo)
        assert node_info.node_id == "node1"
        assert updated_params.taskgraph.curr_node == "node1"

    def test_postprocess_intent_basic(self, task_graph) -> None:
        """Test postprocessing intent without special formatting."""
        found, real_intent, idx = task_graph._postprocess_intent(
            "start", ["start", "continue"]
        )
        assert found is True
        assert real_intent == "start"
        assert idx == 0

    def test_postprocess_intent_with_index(self, task_graph) -> None:
        """Test postprocessing intent with __<idx> format."""
        found, real_intent, idx = task_graph._postprocess_intent(
            "start__<2>", ["start", "continue"]
        )
        assert found is True
        assert real_intent == "start"
        assert idx == 2

    def test_postprocess_intent_not_found(self, task_graph) -> None:
        """Test postprocessing intent that's not in available intents."""
        found, real_intent, idx = task_graph._postprocess_intent(
            "unknown", ["start", "continue"]
        )
        assert found is False
        assert real_intent == "unknown"
        assert idx == 0

    def test_get_current_node_with_valid_node(self, task_graph, params) -> None:
        """Test getting current node when it's valid."""
        params.taskgraph.curr_node = "node1"
        curr_node, updated_params = task_graph.get_current_node(params)
        assert curr_node == "node1"
        assert updated_params.taskgraph.curr_node == "node1"

    def test_get_current_node_with_invalid_node(self, task_graph, params) -> None:
        """Test getting current node when it's invalid."""
        params.taskgraph.curr_node = "nonexistent"
        curr_node, updated_params = task_graph.get_current_node(params)
        assert curr_node == "start"  # Should fall back to start node
        assert updated_params.taskgraph.curr_node == "start"

    def test_get_current_node_with_none(self, task_graph, params) -> None:
        """Test getting current node when it's None."""
        params.taskgraph.curr_node = None
        curr_node, updated_params = task_graph.get_current_node(params)
        assert curr_node == "start"
        assert updated_params.taskgraph.curr_node == "start"

    def test_get_available_global_intents_empty(self, task_graph, params) -> None:
        """Test getting available global intents when empty."""
        params.taskgraph.available_global_intents = {}
        intents = task_graph.get_available_global_intents(params)
        assert "others" in intents

    def test_get_available_global_intents_existing(self, task_graph, params) -> None:
        """Test getting available global intents when they already exist."""
        existing_intents = {"custom": [{"intent": "custom"}]}
        params.taskgraph.available_global_intents = existing_intents
        intents = task_graph.get_available_global_intents(params)

        assert intents == existing_intents  # Should return existing intents

    def test_update_node_limit(self, task_graph, params) -> None:
        """Test updating node limits."""
        params.taskgraph.node_limit = {"node1": 5}
        updated_params = task_graph.update_node_limit(params)

        assert "node1" in updated_params.taskgraph.node_limit
        assert updated_params.taskgraph.node_limit["node1"] == 5

    def test_get_local_intent(self, task_graph, params) -> None:
        """Test getting local intents for a node."""
        local_intents = task_graph.get_local_intent("node1", params)

        assert "continue" in local_intents
        assert "skip" in local_intents
        assert len(local_intents["continue"]) == 1
        assert len(local_intents["skip"]) == 1

    def test_get_local_intent_no_intents(self, task_graph, params) -> None:
        """Test getting local intents for a node with no outgoing intents."""
        # Create a node with no outgoing edges with intents
        task_graph.graph.add_node(
            "isolated",
            type="task",
            resource={"name": "isolated", "id": "isolated_id"},
            attribute={},
        )

        local_intents = task_graph.get_local_intent("isolated", params)
        assert local_intents == {}

    def test_get_last_flow_stack_node(self, task_graph, params) -> None:
        """Test getting last flow stack node from path."""
        # Create path with flow stack nodes
        path_node1 = PathNode(node_id="node1", in_flow_stack=True)
        path_node2 = PathNode(node_id="node2", in_flow_stack=False)
        path_node3 = PathNode(node_id="node3", in_flow_stack=True)

        params.taskgraph.path = [path_node1, path_node2, path_node3]

        last_flow_node = task_graph.get_last_flow_stack_node(params)
        assert last_flow_node.node_id == "node3"
        assert last_flow_node.in_flow_stack is False  # Should be set to False

    def test_get_last_flow_stack_node_not_found(self, task_graph, params) -> None:
        """Test getting last flow stack node when none exists."""
        path_node1 = PathNode(node_id="node1", in_flow_stack=False)
        path_node2 = PathNode(node_id="node2", in_flow_stack=False)

        params.taskgraph.path = [path_node1, path_node2]

        last_flow_node = task_graph.get_last_flow_stack_node(params)
        assert last_flow_node is None

    def test_handle_multi_step_node_stay_status(self, task_graph, params) -> None:
        """Test handling multi-step node with STAY status."""
        params.taskgraph.node_status = {"node1": StatusEnum.STAY}

        should_stay, node_info, updated_params = task_graph.handle_multi_step_node(
            "node1", params
        )

        assert should_stay is True
        assert isinstance(node_info, NodeInfo)
        assert node_info.node_id == "node1"

    def test_handle_multi_step_node_complete_status(self, task_graph, params) -> None:
        """Test handling multi-step node with COMPLETE status."""
        params.taskgraph.node_status = {"node1": StatusEnum.COMPLETE}

        should_stay, node_info, updated_params = task_graph.handle_multi_step_node(
            "node1", params
        )

        assert should_stay is False
        # The implementation returns a default NodeInfo, not None
        assert isinstance(node_info, NodeInfo)
        assert node_info.node_id is None

    def test_handle_incomplete_node(self, task_graph, params) -> None:
        """Test handling incomplete node."""
        params.taskgraph.node_status = {"node1": StatusEnum.INCOMPLETE}

        is_incomplete, node_data, updated_params = task_graph.handle_incomplete_node(
            "node1", params
        )

        assert is_incomplete is True
        # The implementation returns a NodeInfo, not a dict
        assert isinstance(node_data, NodeInfo)

    def test_handle_incomplete_node_complete_status(self, task_graph, params) -> None:
        """Test handling incomplete node with COMPLETE status."""
        params.taskgraph.node_status = {"node1": StatusEnum.COMPLETE}

        is_incomplete, node_data, updated_params = task_graph.handle_incomplete_node(
            "node1", params
        )

        assert is_incomplete is False
        # The implementation returns an empty dict, not None
        assert node_data == {}

    def test_handle_random_next_node(self, task_graph, params) -> None:
        """Test handling random next node selection."""
        with patch("numpy.random.choice") as mock_choice:
            mock_choice.return_value = "node2"

            has_random, node_data, updated_params = task_graph.handle_random_next_node(
                "node1", params
            )

            # The dummy graph does not support random next node, so expect False
            assert has_random is False

    def test_handle_unknown_intent(self, task_graph, params) -> None:
        """Test handling unknown intent."""
        node_info, updated_params = task_graph.handle_unknown_intent("node1", params)

        assert isinstance(node_info, NodeInfo)
        # The fallback returns a default NodeInfo with node_id None
        assert node_info.node_id is None

    def test_handle_leaf_node(self, task_graph, params) -> None:
        """Test handling leaf node."""
        # Mock NestedGraph.get_nested_graph_component_node
        with patch(
            "arklex.orchestrator.task_graph.NestedGraph.get_nested_graph_component_node"
        ) as mock_nested:
            mock_nested.return_value = (None, params)

            # Mock get_last_flow_stack_node to return a node
            with patch.object(task_graph, "get_last_flow_stack_node") as mock_flow:
                mock_flow_node = Mock(spec=PathNode)
                mock_flow_node.node_id = "node1"  # Match the actual node id
                mock_flow_node.global_intent = "test_intent"
                mock_flow.return_value = mock_flow_node

                # Mock initial_node to be None so it doesn't interfere
                with patch.object(task_graph, "initial_node", None):
                    # Patch graph.successors to return empty list (leaf node)
                    with patch.object(task_graph.graph, "successors", return_value=[]):
                        result_node, result_params = task_graph.handle_leaf_node(
                            "curr_node", params
                        )
                        assert result_node == "node1"

    def test_validate_node_valid(self, task_graph) -> None:
        """Test validating a valid node."""
        valid_node = {
            "id": "test",
            "type": "task",
            "resource": {"name": "test", "id": "test"},
            "attribute": {},
        }

        # Should not raise any exception
        task_graph._validate_node(valid_node)

    def test_validate_node_missing_id(self, task_graph) -> None:
        """Test validating a node missing required fields."""
        invalid_node = {
            "type": "task",
            "resource": {"name": "test", "id": "test"},
            "attribute": {},
        }

        from arklex.utils.exceptions import TaskGraphError

        with pytest.raises(TaskGraphError):
            task_graph._validate_node(invalid_node)

    def test_validate_node_missing_attribute(self, task_graph) -> None:
        """Test _validate_node with missing attribute."""
        node = {"id": "test_id"}
        with pytest.raises(Exception):  # TaskGraphError
            task_graph._validate_node(node)

    def test_global_intent_prediction_with_excluded_intents(
        self, task_graph, params
    ) -> None:
        """Test global_intent_prediction with excluded intents."""
        available_global_intents = {"intent1": [{"name": "intent1"}]}
        excluded_intents = {"intent1": True}

        # Mock the _get_node method
        with patch.object(task_graph, "_get_node") as mock_get_node:
            mock_node_info = NodeInfo(
                node_id="next_node",
                type="test",
                resource_id="test",
                resource_name="test",
            )
            mock_get_node.return_value = (mock_node_info, params)

            # Mock graph.successors to return a non-empty list
            with patch.object(
                task_graph.graph, "successors", return_value=["next_node"]
            ):
                result, pred_intent, node_info, params = (
                    task_graph.global_intent_prediction(
                        "curr_node", params, available_global_intents, excluded_intents
                    )
                )
                assert result is False

    def test_global_intent_prediction_with_no_candidates(
        self, task_graph, params
    ) -> None:
        """Test global_intent_prediction with no candidate intents."""
        available_global_intents = {}
        excluded_intents = {}

        result, pred_intent, node_info, params = task_graph.global_intent_prediction(
            "curr_node", params, available_global_intents, excluded_intents
        )
        assert result is False
        assert pred_intent == "others"

    def test_global_intent_prediction_with_unsure_intent_not_available(
        self, task_graph, params
    ) -> None:
        """Test global_intent_prediction when unsure_intent is not in available intents (lines 475-480)."""
        available_global_intents = {"intent1": [{"intent": "intent1"}]}
        excluded_intents = {}

        with (
            patch.object(task_graph.intent_detector, "execute", return_value="intent1"),
            patch.object(
                task_graph, "_postprocess_intent", return_value=(True, "intent1", 0)
            ),
            patch.object(
                task_graph, "jump_to_node", return_value=("next_node", "intent1")
            ),
            patch.object(task_graph.graph, "successors", return_value=["next_node"]),
            patch.object(task_graph, "_get_node") as mock_get_node,
        ):
            mock_node_info = NodeInfo(
                node_id="next_node",
                type="task",
                resource_name="test",
            )
            mock_get_node.return_value = (mock_node_info, params)

            result, pred_intent, node_info, params = (
                task_graph.global_intent_prediction(
                    "curr_node", params, available_global_intents, excluded_intents
                )
            )
            assert result is True
            assert pred_intent == "intent1"

    def test_global_intent_prediction_same_intent_complete_status(
        self, task_graph, params
    ) -> None:
        """Test global_intent_prediction with same intent but COMPLETE status (lines 541-542)."""
        available_global_intents = {"intent1": [{"intent": "intent1"}]}
        excluded_intents = {}
        params.taskgraph.node_status = {"curr_node": StatusEnum.COMPLETE}

        with (
            patch.object(task_graph.intent_detector, "execute", return_value="intent1"),
            patch.object(
                task_graph, "_postprocess_intent", return_value=(True, "intent1", 0)
            ),
            patch.object(
                task_graph, "jump_to_node", return_value=("next_node", "intent1")
            ),
            patch.object(task_graph.graph, "successors", return_value=["next_node"]),
            patch.object(task_graph, "_get_node") as mock_get_node,
        ):
            mock_node_info = NodeInfo(
                node_id="next_node",
                type="task",
                resource_name="test",
            )
            mock_get_node.return_value = (mock_node_info, params)

            result, pred_intent, node_info, params = (
                task_graph.global_intent_prediction(
                    "curr_node", params, available_global_intents, excluded_intents
                )
            )
            assert result is True
            assert pred_intent == "intent1"

    def test_global_intent_prediction_same_intent_leaf_node(
        self, task_graph, params
    ) -> None:
        """Test global_intent_prediction with same intent but leaf node (lines 541-542)."""
        available_global_intents = {"intent1": [{"intent": "intent1"}]}
        excluded_intents = {}
        params.taskgraph.node_status = {"curr_node": StatusEnum.INCOMPLETE}

        with (
            patch.object(task_graph.intent_detector, "execute", return_value="intent1"),
            patch.object(
                task_graph, "_postprocess_intent", return_value=(True, "intent1", 0)
            ),
            patch.object(
                task_graph, "jump_to_node", return_value=("next_node", "intent1")
            ),
            patch.object(task_graph.graph, "successors", return_value=[]),
            patch.object(task_graph, "_get_node") as mock_get_node,
        ):
            mock_node_info = NodeInfo(
                node_id="next_node",
                type="task",
                resource_name="test",
            )
            mock_get_node.return_value = (mock_node_info, params)

            result, pred_intent, node_info, params = (
                task_graph.global_intent_prediction(
                    "curr_node", params, available_global_intents, excluded_intents
                )
            )
            assert result is True
            assert pred_intent == "intent1"

    def test_global_intent_prediction_same_intent_incomplete_status(
        self, task_graph, params
    ) -> None:
        """Test global_intent_prediction with same intent but INCOMPLETE status (lines 541-542)."""
        available_global_intents = {"intent1": [{"intent": "intent1"}]}
        excluded_intents = {}
        params.taskgraph.node_status = {"curr_node": StatusEnum.INCOMPLETE}

        with (
            patch.object(task_graph.intent_detector, "execute", return_value="intent1"),
            patch.object(
                task_graph, "_postprocess_intent", return_value=(True, "intent1", 0)
            ),
            patch.object(
                task_graph, "jump_to_node", return_value=("next_node", "intent1")
            ),
            patch.object(task_graph.graph, "successors", return_value=["next_node"]),
            patch.object(task_graph, "_get_node") as mock_get_node,
        ):
            mock_node_info = NodeInfo(
                node_id="next_node",
                type="task",
                resource_name="test",
            )
            mock_get_node.return_value = (mock_node_info, params)

            result, pred_intent, node_info, params = (
                task_graph.global_intent_prediction(
                    "curr_node", params, available_global_intents, excluded_intents
                )
            )
            assert result is True
            assert pred_intent == "intent1"

    def test_global_intent_prediction_add_flow_stack(self, task_graph, params) -> None:
        """Test global_intent_prediction with add_flow_stack logic (lines 554-570)."""
        available_global_intents = {"intent1": [{"intent": "intent1"}]}
        excluded_intents = {}

        with patch.object(
            task_graph.intent_detector, "execute", return_value="intent1"
        ):
            with patch.object(
                task_graph, "_postprocess_intent", return_value=(True, "intent1", 0)
            ):
                with patch.object(
                    task_graph, "jump_to_node", return_value=("next_node", "intent1")
                ):
                    with patch.object(task_graph, "_get_node") as mock_get_node:
                        mock_node_info = NodeInfo(
                            node_id="next_node",
                            type="test",
                            resource_id="test",
                            resource_name="test",
                        )
                        mock_get_node.return_value = (mock_node_info, params)

                        with patch.object(
                            task_graph.graph, "successors", return_value=["next_node"]
                        ):
                            result, pred_intent, node_info, params = (
                                task_graph.global_intent_prediction(
                                    "curr_node",
                                    params,
                                    available_global_intents,
                                    excluded_intents,
                                )
                            )
                            assert result is True
                            assert pred_intent == "intent1"
                            assert node_info.add_flow_stack is True

    def test_handle_random_next_node_with_candidates(self, task_graph, params) -> None:
        """Test handle_random_next_node with candidate samples (lines 619-621)."""
        # Add nodes with proper structure
        task_graph.graph.add_node(
            "next_node1",
            type="task",
            resource={"name": "next_node1", "id": "next_node1"},
            attribute={},
        )
        task_graph.graph.add_node(
            "next_node2",
            type="task",
            resource={"name": "next_node2", "id": "next_node2"},
            attribute={},
        )

        # Add edges with "none" intent
        task_graph.graph.add_edge(
            "curr_node", "next_node1", intent="none", attribute={"weight": 1.0}
        )
        task_graph.graph.add_edge(
            "curr_node", "next_node2", intent="none", attribute={"weight": 2.0}
        )

        with patch("numpy.random.choice") as mock_choice:
            mock_choice.return_value = "next_node1"
            has_random, node_info, params = task_graph.handle_random_next_node(
                "curr_node", params
            )
            assert has_random is True
            assert node_info.node_id == "next_node1"

    def test_handle_random_next_node_with_nlu_records(self, task_graph, params) -> None:
        """Test handle_random_next_node with existing nlu_records (lines 640)."""
        # Add node with proper structure
        task_graph.graph.add_node(
            "next_node1",
            type="task",
            resource={"name": "next_node1", "id": "next_node1"},
            attribute={},
        )

        # Add edges with "none" intent
        task_graph.graph.add_edge(
            "curr_node", "next_node1", intent="none", attribute={"weight": 1.0}
        )
        params.taskgraph.nlu_records = [{"test": "record"}]

        with patch("numpy.random.choice") as mock_choice:
            mock_choice.return_value = "next_node1"
            has_random, node_info, params = task_graph.handle_random_next_node(
                "curr_node", params
            )
            assert has_random is True
            assert params.taskgraph.nlu_records[-1]["no_intent"] is True

    def test_handle_random_next_node_without_nlu_records(
        self, task_graph, params
    ) -> None:
        """Test handle_random_next_node without existing nlu_records (lines 640)."""
        # Add node with proper structure
        task_graph.graph.add_node(
            "next_node1",
            type="task",
            resource={"name": "next_node1", "id": "next_node1"},
            attribute={},
        )

        # Add edges with "none" intent
        task_graph.graph.add_edge(
            "curr_node", "next_node1", intent="none", attribute={"weight": 1.0}
        )
        params.taskgraph.nlu_records = []

        with patch("numpy.random.choice") as mock_choice:
            mock_choice.return_value = "next_node1"
            has_random, node_info, params = task_graph.handle_random_next_node(
                "curr_node", params
            )
            assert has_random is True
            assert len(params.taskgraph.nlu_records) == 1
            assert params.taskgraph.nlu_records[0]["no_intent"] is True

    def test_handle_leaf_node_with_flow_stack_node(self, task_graph, params) -> None:
        """Test handle_leaf_node with flow stack node (lines 748-758)."""
        # Create a flow stack node
        flow_stack_node = PathNode(
            node_id="flow_node", global_intent="flow_intent", in_flow_stack=True
        )
        params.taskgraph.path = [flow_stack_node]

        with (
            patch.object(task_graph.graph, "successors", return_value=[]),
            patch.object(
                task_graph, "get_last_flow_stack_node", return_value=flow_stack_node
            ),
        ):
            result_node, result_params = task_graph.handle_leaf_node(
                "curr_node", params
            )
            # The actual implementation returns the initial_node because it overrides the flow stack node
            # Check that it returns one of the possible initial nodes from the fixture
            assert result_node in ["node1", "node2"]  # From services_nodes in fixture

    def test_handle_leaf_node_without_flow_stack_node(self, task_graph, params) -> None:
        """Test handle_leaf_node without flow stack node (lines 748-758)."""
        # Set the initial node to match what the implementation expects
        task_graph.initial_node = "initial_node"

        with (
            patch.object(task_graph.graph, "successors", return_value=[]),
            patch.object(task_graph, "get_last_flow_stack_node", return_value=None),
        ):
            result_node, result_params = task_graph.handle_leaf_node(
                "curr_node", params
            )
            assert result_node == "initial_node"

    def test_get_start_node_with_start_type(self, task_graph) -> None:
        """Test get_start_node with type='start' (lines 96-99)."""
        # Add a node with type='start'
        task_graph.graph.add_node(
            "start_node",
            type="start",
            resource={"name": "start", "id": "start"},
            attribute={},
        )

        start_node = task_graph.get_start_node()
        # The implementation returns the resource name, not the node id
        assert start_node == "start"

    def test_get_start_node_with_start_attribute(self, task_graph) -> None:
        """Test get_start_node with start attribute (lines 96-99)."""
        # Add a node with start attribute
        task_graph.graph.add_node(
            "start_node",
            type="task",
            resource={"name": "start", "id": "start"},
            attribute={"start": True},
        )

        start_node = task_graph.get_start_node()
        # The implementation returns the resource name, not the node id
        assert start_node == "start"

    def test_get_start_node_no_start_node(self, task_graph) -> None:
        """Test get_start_node when no start node exists (line 106)."""
        # Remove all nodes and add only non-start nodes
        task_graph.graph.clear()
        task_graph.graph.add_node(
            "regular_node",
            type="task",
            resource={"name": "regular", "id": "regular"},
            attribute={},
        )

        start_node = task_graph.get_start_node()
        assert start_node is None

    def test_get_initial_flow_with_empty_services_nodes(self, task_graph) -> None:
        """Test get_initial_flow with empty services_nodes (line 187)."""
        # Set empty services_nodes
        task_graph.product_kwargs["services_nodes"] = {}

        initial_flow = task_graph.get_initial_flow()
        assert initial_flow is None

    def test_get_pred_intents_with_pred_edges(self, task_graph) -> None:
        """Test get_pred_intents method (lines 92-100)."""
        # Add edges with pred=True attribute
        task_graph.graph.add_edge(
            "start", "node1", intent="test", attribute={"pred": True, "weight": 1.0}
        )
        task_graph.graph.add_edge(
            "node1",
            "node2",
            intent="continue",
            attribute={"pred": False, "weight": 1.0},
        )

        intents = task_graph.get_pred_intents()
        assert "test" in intents
        assert len(intents["test"]) == 1
        assert intents["test"][0]["source_node"] == "start"
        assert intents["test"][0]["target_node"] == "node1"

    def test_get_start_node_with_start_type(self, task_graph) -> None:
        """Test get_start_node with type='start' (lines 102-106)."""
        # Clear existing nodes and add a start type node
        task_graph.graph.clear()
        task_graph.graph.add_node(
            "start_node",
            type="start",
            resource={"name": "start", "id": "start"},
            attribute={},
        )

        start_node = task_graph.get_start_node()
        assert start_node == "start_node"

    def test_get_start_node_with_start_attribute(self, task_graph) -> None:
        """Test get_start_node with start attribute (lines 102-106)."""
        # Clear existing nodes and add a node with start attribute
        task_graph.graph.clear()
        task_graph.graph.add_node(
            "start_node",
            type="task",
            resource={"name": "start", "id": "start"},
            attribute={"start": True},
        )

        start_node = task_graph.get_start_node()
        assert start_node == "start_node"

    def test_jump_to_node_with_empty_candidates(self, task_graph) -> None:
        """Test jump_to_node when candidates_nodes is empty (lines 235-236)."""
        # Set up intents to have empty list for the intent
        task_graph.intents = {"test_intent": []}

        with patch("numpy.random.choice") as mock_choice:
            mock_choice.return_value = "start"
            next_node, next_intent = task_graph.jump_to_node("test_intent", 0, "start")
            assert next_node == "start"
            assert next_intent == "dummy"  # From the dummy edge

    def test_jump_to_node_exception_with_no_in_edges(self, task_graph) -> None:
        """Test jump_to_node exception handling when node has no incoming edges (lines 235-236)."""
        # Create a node with no incoming edges
        task_graph.graph.add_node("isolated_node")

        with pytest.raises(IndexError):
            task_graph.jump_to_node("nonexistent", 0, "isolated_node")

    def test_get_node_with_node_specific_data_dict(self, task_graph, params) -> None:
        """Test _get_node with node_specific_data containing dict values (lines 275-284)."""
        # Add a node with node_specific_data containing dict values
        task_graph.graph.add_node(
            "test_node",
            type="test",
            resource={"name": "test", "id": "test"},
            attribute={
                "node_specific_data": {
                    "dict_value": {"nested_key": "nested_value"},
                    "simple_value": "simple",
                }
            },
        )

        node_info, updated_params = task_graph._get_node("test_node", params)
        assert node_info.node_id == "test_node"
        assert node_info.additional_args["nested_key"] == "nested_value"
        assert node_info.additional_args["simple_value"] == "simple"

    def test_get_node_with_node_specific_data_mixed(self, task_graph, params) -> None:
        """Test _get_node with mixed node_specific_data (lines 275-284)."""
        # Add a node with mixed node_specific_data
        task_graph.graph.add_node(
            "test_node",
            type="test",
            resource={"name": "test", "id": "test"},
            attribute={
                "node_specific_data": {
                    "dict_value": {"key1": "value1", "key2": "value2"},
                    "simple_value": "simple",
                    "another_dict": {"key3": "value3"},
                }
            },
        )

        node_info, updated_params = task_graph._get_node("test_node", params)
        assert node_info.node_id == "test_node"
        assert node_info.additional_args["key1"] == "value1"
        assert node_info.additional_args["key2"] == "value2"
        assert node_info.additional_args["key3"] == "value3"
        assert node_info.additional_args["simple_value"] == "simple"

    def test_global_intent_prediction_only_unsure_intent(
        self, task_graph, params
    ) -> None:
        """Test global_intent_prediction when only unsure_intent is available (lines 470-475)."""
        available_global_intents = {"others": [{"intent": "others"}]}
        excluded_intents = {}

        result, pred_intent, node_info, params = task_graph.global_intent_prediction(
            "curr_node", params, available_global_intents, excluded_intents
        )
        assert result is False
        assert pred_intent == "others"

    def test_global_intent_prediction_same_intent_complete_status(
        self, task_graph, params
    ) -> None:
        """Test global_intent_prediction with same intent but COMPLETE status (lines 515)."""
        available_global_intents = {"intent1": [{"intent": "intent1"}]}
        excluded_intents = {}
        params.taskgraph.node_status = {"curr_node": StatusEnum.COMPLETE}

        with (
            patch.object(task_graph.intent_detector, "execute", return_value="intent1"),
            patch.object(
                task_graph, "_postprocess_intent", return_value=(True, "intent1", 0)
            ),
            patch.object(
                task_graph, "jump_to_node", return_value=("next_node", "intent1")
            ),
            patch.object(task_graph.graph, "successors", return_value=["next_node"]),
            patch.object(task_graph, "_get_node") as mock_get_node,
        ):
            mock_node_info = NodeInfo(
                node_id="next_node",
                type="task",
                resource_name="test",
            )
            mock_get_node.return_value = (mock_node_info, params)

            result, pred_intent, node_info, params = (
                task_graph.global_intent_prediction(
                    "curr_node", params, available_global_intents, excluded_intents
                )
            )
            assert result is True
            assert pred_intent == "intent1"

    def test_global_intent_prediction_same_intent_leaf_node(
        self, task_graph, params
    ) -> None:
        """Test global_intent_prediction with same intent but leaf node (lines 515)."""
        available_global_intents = {"intent1": [{"intent": "intent1"}]}
        excluded_intents = {}
        params.taskgraph.node_status = {"curr_node": StatusEnum.INCOMPLETE}

        with (
            patch.object(task_graph.intent_detector, "execute", return_value="intent1"),
            patch.object(
                task_graph, "_postprocess_intent", return_value=(True, "intent1", 0)
            ),
            patch.object(
                task_graph, "jump_to_node", return_value=("next_node", "intent1")
            ),
            patch.object(task_graph.graph, "successors", return_value=[]),  # Leaf node
            patch.object(task_graph, "_get_node") as mock_get_node,
        ):
            mock_node_info = NodeInfo(
                node_id="next_node",
                type="task",
                resource_name="test",
            )
            mock_get_node.return_value = (mock_node_info, params)

            result, pred_intent, node_info, params = (
                task_graph.global_intent_prediction(
                    "curr_node", params, available_global_intents, excluded_intents
                )
            )
            assert result is True
            assert pred_intent == "intent1"

    def test_global_intent_prediction_same_intent_incomplete_status(
        self, task_graph, params
    ) -> None:
        """Test global_intent_prediction with same intent but INCOMPLETE status (lines 515)."""
        available_global_intents = {"intent1": [{"intent": "intent1"}]}
        excluded_intents = {}
        params.taskgraph.node_status = {"curr_node": StatusEnum.INCOMPLETE}
        params.taskgraph.curr_global_intent = "intent1"

        with (
            patch.object(task_graph.intent_detector, "execute", return_value="intent1"),
            patch.object(
                task_graph, "_postprocess_intent", return_value=(True, "intent1", 0)
            ),
            patch.object(task_graph.graph, "successors", return_value=["next_node"]),
        ):
            result, pred_intent, node_info, params = (
                task_graph.global_intent_prediction(
                    "curr_node", params, available_global_intents, excluded_intents
                )
            )
            assert result is False
            assert pred_intent == "intent1"

    def test_handle_random_next_node_no_candidates(self, task_graph, params) -> None:
        """Test handle_random_next_node with no candidates (lines 582-628)."""
        # Add a node with no outgoing edges with "none" intent
        task_graph.graph.add_node(
            "leaf_node",
            type="task",
            resource={"name": "leaf", "id": "leaf"},
            attribute={},
        )

        has_random, node_info, params = task_graph.handle_random_next_node(
            "leaf_node", params
        )
        assert has_random is False
        assert node_info == {}

    def test_handle_random_next_node_same_node(self, task_graph, params) -> None:
        """Test handle_random_next_node when next_node equals curr_node (lines 640)."""
        # Add a node with outgoing edge to itself
        task_graph.graph.add_node(
            "self_loop_node",
            type="task",
            resource={"name": "self_loop", "id": "self_loop"},
            attribute={},
        )
        task_graph.graph.add_edge(
            "self_loop_node", "self_loop_node", intent="none", attribute={"weight": 1.0}
        )

        with patch("numpy.random.choice") as mock_choice:
            mock_choice.return_value = "self_loop_node"
            has_random, node_info, params = task_graph.handle_random_next_node(
                "self_loop_node", params
            )
            assert has_random is False
            assert node_info == {}

    def test_get_node_with_multi_step_node(self, task_graph, params) -> None:
        """Test get_node with multi-step node (lines 724)."""
        inputs = {
            "text": "test text",
            "chat_history_str": "",
            "parameters": params,
            "allow_global_intent_switch": False,
        }

        with (
            patch.object(
                task_graph, "get_current_node", return_value=("curr_node", params)
            ),
            patch.object(task_graph, "handle_multi_step_node") as mock_multi_step,
        ):
            mock_node_info = NodeInfo(
                node_id="multi_step_node",
                type="task",
                resource_name="test",
            )
            mock_multi_step.return_value = (True, mock_node_info, params)

            node_info, params = task_graph.get_node(inputs)
            assert node_info.node_id == "multi_step_node"

    def test_get_initial_flow_with_no_in_edges(self, task_graph) -> None:
        """Test get_initial_flow when nodes have no incoming edges (line 187)."""
        # Create a config with services_nodes but nodes that have no incoming edges
        task_graph.product_kwargs["services_nodes"] = {"service1": "isolated_node"}
        task_graph.graph.add_node(
            "isolated_node",
            type="task",
            resource={"name": "isolated", "id": "isolated"},
            attribute={},
        )

        with pytest.raises(IndexError):
            task_graph.get_initial_flow()

    def test_jump_to_node_with_empty_intents(self, task_graph) -> None:
        """Test jump_to_node with empty intents (lines 235-236)."""
        # Set up intents to have empty list for the intent
        task_graph.intents = {"test_intent": []}

        with patch("numpy.random.choice") as mock_choice:
            mock_choice.return_value = "start"
            next_node, next_intent = task_graph.jump_to_node("test_intent", 0, "start")
            assert next_node == "start"
            assert next_intent == "dummy"  # From the dummy edge

    def test_jump_to_node_with_index_error(self, task_graph) -> None:
        """Test jump_to_node with IndexError when getting in_edges (lines 235-236)."""
        # Create a node with no incoming edges
        task_graph.graph.add_node("isolated_node")

        with pytest.raises(IndexError):
            task_graph.jump_to_node("nonexistent", 0, "isolated_node")

    def test_get_node_with_global_intent_switch_disabled(
        self, task_graph, params
    ) -> None:
        """Test get_node with global intent switch disabled (lines 795-804)."""
        inputs = {
            "text": "test text",
            "chat_history_str": "",
            "parameters": params,
            "allow_global_intent_switch": False,
        }

        with (
            patch.object(
                task_graph, "get_current_node", return_value=("curr_node", params)
            ),
            patch.object(
                task_graph, "handle_multi_step_node", return_value=(False, None, params)
            ),
            patch.object(
                task_graph, "handle_leaf_node", return_value=("curr_node", params)
            ),
            patch.object(task_graph, "get_available_global_intents", return_value={}),
            patch.object(
                task_graph,
                "get_local_intent",
                return_value={"intent1": [{"intent": "intent1"}]},
            ),
            patch.object(
                task_graph, "handle_incomplete_node", return_value=(False, {}, params)
            ),
            patch.object(
                task_graph, "local_intent_prediction", return_value=(False, {}, params)
            ),
            patch.object(task_graph, "handle_unknown_intent") as mock_unknown,
            patch.object(task_graph.graph, "successors", return_value=[]),
        ):
            mock_node_info = NodeInfo(
                node_id="planner",
                type="task",
                resource_name="test",
            )
            mock_unknown.return_value = (mock_node_info, params)

            node_info, params = task_graph.get_node(inputs)
            assert node_info.node_id == "planner"

    def test_get_node_with_global_intent_switch_and_no_random_fallback(
        self, task_graph, params
    ) -> None:
        """Test get_node with global intent switch and no random fallback (lines 795-804)."""
        inputs = {
            "text": "test text",
            "chat_history_str": "",
            "parameters": params,
            "allow_global_intent_switch": True,
        }

        with (
            patch.object(
                task_graph, "get_current_node", return_value=("curr_node", params)
            ),
            patch.object(
                task_graph, "handle_multi_step_node", return_value=(False, None, params)
            ),
            patch.object(
                task_graph, "handle_leaf_node", return_value=("curr_node", params)
            ),
            patch.object(
                task_graph,
                "get_available_global_intents",
                return_value={"intent1": [{"intent": "intent1"}]},
            ),
            patch.object(
                task_graph,
                "get_local_intent",
                return_value={"intent1": [{"intent": "intent1"}]},
            ),
            patch.object(
                task_graph, "handle_incomplete_node", return_value=(False, {}, params)
            ),
            patch.object(
                task_graph, "local_intent_prediction", return_value=(False, {}, params)
            ),
            patch.object(task_graph, "global_intent_prediction") as mock_global,
            patch.object(
                task_graph, "handle_random_next_node", return_value=(False, {}, params)
            ),
            patch.object(task_graph, "handle_unknown_intent") as mock_unknown,
            patch.object(task_graph.graph, "successors", return_value=[]),
        ):
            # Mock global intent prediction to return a non-unsure intent
            mock_global.return_value = (False, "intent1", {}, params)

            mock_node_info = NodeInfo(
                node_id="planner",
                type="task",
                resource_name="test",
            )
            mock_unknown.return_value = (mock_node_info, params)

            node_info, params = task_graph.get_node(inputs)
            assert node_info.node_id == "planner"

    def test_get_node_with_global_intent_switch_and_unsure_intent(
        self, task_graph, params
    ) -> None:
        """Test get_node with global intent switch and unsure intent (lines 795-804)."""
        inputs = {
            "text": "test text",
            "chat_history_str": "",
            "parameters": params,
            "allow_global_intent_switch": True,
        }

        with (
            patch.object(
                task_graph, "get_current_node", return_value=("curr_node", params)
            ),
            patch.object(
                task_graph, "handle_multi_step_node", return_value=(False, None, params)
            ),
            patch.object(
                task_graph, "handle_leaf_node", return_value=("curr_node", params)
            ),
            patch.object(
                task_graph,
                "get_available_global_intents",
                return_value={"intent1": [{"intent": "intent1"}]},
            ),
            patch.object(
                task_graph,
                "get_local_intent",
                return_value={"intent1": [{"intent": "intent1"}]},
            ),
            patch.object(
                task_graph, "handle_incomplete_node", return_value=(False, {}, params)
            ),
            patch.object(
                task_graph, "local_intent_prediction", return_value=(False, {}, params)
            ),
            patch.object(task_graph, "global_intent_prediction") as mock_global,
            patch.object(
                task_graph, "handle_random_next_node", return_value=(False, {}, params)
            ),
            patch.object(task_graph, "handle_unknown_intent") as mock_unknown,
            patch.object(task_graph.graph, "successors", return_value=[]),
        ):
            # Mock global intent prediction to return unsure intent
            mock_global.return_value = (False, "others", {}, params)

            mock_node_info = NodeInfo(
                node_id="planner",
                type="task",
                resource_name="test",
            )
            mock_unknown.return_value = (mock_node_info, params)

            node_info, params = task_graph.get_node(inputs)
            assert node_info.node_id == "planner"

    def test_get_node_with_global_intent_switch_and_no_pred_intent(
        self, task_graph, params
    ) -> None:
        """Test get_node with global intent switch and no pred_intent (lines 795-804)."""
        inputs = {
            "text": "test text",
            "chat_history_str": "",
            "parameters": params,
            "allow_global_intent_switch": True,
        }

        with (
            patch.object(
                task_graph, "get_current_node", return_value=("curr_node", params)
            ),
            patch.object(
                task_graph, "handle_multi_step_node", return_value=(False, None, params)
            ),
            patch.object(
                task_graph, "handle_leaf_node", return_value=("curr_node", params)
            ),
            patch.object(
                task_graph,
                "get_available_global_intents",
                return_value={"intent1": [{"intent": "intent1"}]},
            ),
            patch.object(
                task_graph,
                "get_local_intent",
                return_value={"intent1": [{"intent": "intent1"}]},
            ),
            patch.object(
                task_graph, "handle_incomplete_node", return_value=(False, {}, params)
            ),
            patch.object(
                task_graph, "local_intent_prediction", return_value=(False, {}, params)
            ),
            patch.object(task_graph, "global_intent_prediction") as mock_global,
            patch.object(
                task_graph, "handle_random_next_node", return_value=(False, {}, params)
            ),
            patch.object(task_graph, "handle_unknown_intent") as mock_unknown,
            patch.object(task_graph.graph, "successors", return_value=[]),
        ):
            # Mock global intent prediction to return None pred_intent
            mock_global.return_value = (False, None, {}, params)

            mock_node_info = NodeInfo(
                node_id="planner",
                type="task",
                resource_name="test",
            )
            mock_unknown.return_value = (mock_node_info, params)

            node_info, params = task_graph.get_node(inputs)
            assert node_info.node_id == "planner"

    def test_get_pred_intents_with_edge_without_intent(self, task_graph) -> None:
        """Test get_pred_intents with edge that has no intent (line 90)."""
        # Add an edge with pred=True but no intent
        task_graph.graph.add_edge(
            "start", "node1", intent=None, attribute={"pred": True, "weight": 1.0}
        )

        intents = task_graph.get_pred_intents()
        # Should handle None intent gracefully
        assert None in intents
        assert len(intents[None]) == 1

    def test_get_initial_flow_with_node_without_incoming_edges(
        self, task_graph
    ) -> None:
        """Test get_initial_flow with node that has no incoming edges (line 187)."""
        # Create a config with services_nodes but nodes that have no incoming edges
        task_graph.product_kwargs["services_nodes"] = {"service1": "isolated_node"}
        task_graph.graph.add_node(
            "isolated_node",
            type="task",
            resource={"name": "isolated", "id": "isolated"},
            attribute={},
        )

        with pytest.raises(IndexError):
            task_graph.get_initial_flow()

    def test_jump_to_node_with_empty_candidates_and_exception(self, task_graph) -> None:
        """Test jump_to_node with empty candidates and exception handling (lines 235-236)."""
        # Set up intents to have empty list for the intent
        task_graph.intents = {"test_intent": []}

        # Create a node with no incoming edges to trigger IndexError
        task_graph.graph.add_node("isolated_node")

        with pytest.raises(IndexError):
            task_graph.jump_to_node("test_intent", 0, "isolated_node")

    def test_handle_random_next_node_with_candidates_and_nlu_records(
        self, task_graph, params
    ) -> None:
        """Test handle_random_next_node with candidates and NLU records (lines 582-628)."""
        # Add nodes with proper structure
        task_graph.graph.add_node(
            "next_node1",
            type="task",
            resource={"name": "next_node1", "id": "next_node1"},
            attribute={},
        )
        task_graph.graph.add_node(
            "next_node2",
            type="task",
            resource={"name": "next_node2", "id": "next_node2"},
            attribute={},
        )

        # Add edges with "none" intent
        task_graph.graph.add_edge(
            "curr_node", "next_node1", intent="none", attribute={"weight": 1.0}
        )
        task_graph.graph.add_edge(
            "curr_node", "next_node2", intent="none", attribute={"weight": 1.0}
        )

        # Set up NLU records
        params.taskgraph.nlu_records = [{"test": "record"}]

        with patch("numpy.random.choice") as mock_choice:
            mock_choice.return_value = "next_node1"
            has_random, node_info, params = task_graph.handle_random_next_node(
                "curr_node", params
            )
            assert has_random is True
            assert node_info.node_id == "next_node1"
            assert params.taskgraph.nlu_records[-1]["no_intent"] is True

    def test_handle_random_next_node_with_candidates_and_no_nlu_records(
        self, task_graph, params
    ) -> None:
        """Test handle_random_next_node with candidates but no NLU records (lines 582-628)."""
        # Add nodes with proper structure
        task_graph.graph.add_node(
            "next_node1",
            type="task",
            resource={"name": "next_node1", "id": "next_node1"},
            attribute={},
        )

        # Add edge with "none" intent
        task_graph.graph.add_edge(
            "curr_node", "next_node1", intent="none", attribute={"weight": 1.0}
        )

        # No NLU records
        params.taskgraph.nlu_records = []

        with patch("numpy.random.choice") as mock_choice:
            mock_choice.return_value = "next_node1"
            has_random, node_info, params = task_graph.handle_random_next_node(
                "curr_node", params
            )
            assert has_random is True
            assert node_info.node_id == "next_node1"
            assert len(params.taskgraph.nlu_records) == 1
            assert params.taskgraph.nlu_records[0]["no_intent"] is True

    def test_handle_random_next_node_with_same_node_and_nlu_records(
        self, task_graph, params
    ) -> None:
        """Test handle_random_next_node when next_node equals curr_node with NLU records (line 640)."""
        # Add a node with outgoing edge to itself
        task_graph.graph.add_node(
            "self_loop_node",
            type="task",
            resource={"name": "self_loop", "id": "self_loop"},
            attribute={},
        )
        task_graph.graph.add_edge(
            "self_loop_node", "self_loop_node", intent="none", attribute={"weight": 1.0}
        )

        # Set up NLU records
        params.taskgraph.nlu_records = [{"test": "record"}]

        with patch("numpy.random.choice") as mock_choice:
            mock_choice.return_value = "self_loop_node"
            has_random, node_info, params = task_graph.handle_random_next_node(
                "self_loop_node", params
            )
            assert has_random is False
            assert node_info == {}

    def test_get_node_with_local_intent_prediction_success(
        self, task_graph, params
    ) -> None:
        """Test get_node with successful local intent prediction (line 804)."""
        inputs = {
            "text": "test text",
            "chat_history_str": "",
            "parameters": params,
            "allow_global_intent_switch": False,
        }

        with (
            patch.object(
                task_graph, "get_current_node", return_value=("curr_node", params)
            ),
            patch.object(
                task_graph, "handle_multi_step_node", return_value=(False, None, params)
            ),
            patch.object(
                task_graph, "handle_leaf_node", return_value=("curr_node", params)
            ),
            patch.object(task_graph, "get_available_global_intents", return_value={}),
            patch.object(
                task_graph,
                "get_local_intent",
                return_value={"intent1": [{"intent": "intent1"}]},
            ),
            patch.object(
                task_graph, "handle_incomplete_node", return_value=(False, {}, params)
            ),
            patch.object(task_graph, "local_intent_prediction") as mock_local,
            patch.object(task_graph, "handle_unknown_intent") as mock_unknown,
            patch.object(task_graph.graph, "successors", return_value=[]),
        ):
            mock_node_info = NodeInfo(
                node_id="local_node",
                type="task",
                resource_name="test",
            )
            mock_local.return_value = (True, mock_node_info, params)

            node_info, params = task_graph.get_node(inputs)
            assert node_info.node_id == "local_node"

    def test_get_node_with_global_intent_switch_and_local_intents(
        self, task_graph, params
    ) -> None:
        """Test get_node with global intent switch enabled and local intents available (line 804)."""
        inputs = {
            "text": "test text",
            "chat_history_str": "",
            "parameters": params,
            "allow_global_intent_switch": True,
        }

        with (
            patch.object(
                task_graph, "get_current_node", return_value=("curr_node", params)
            ),
            patch.object(
                task_graph, "handle_multi_step_node", return_value=(False, None, params)
            ),
            patch.object(
                task_graph, "handle_leaf_node", return_value=("curr_node", params)
            ),
            patch.object(
                task_graph,
                "get_available_global_intents",
                return_value={"intent1": [{"intent": "intent1"}]},
            ),
            patch.object(
                task_graph,
                "get_local_intent",
                return_value={"intent1": [{"intent": "intent1"}]},
            ),
            patch.object(
                task_graph, "handle_incomplete_node", return_value=(False, {}, params)
            ),
            patch.object(
                task_graph, "local_intent_prediction", return_value=(False, {}, params)
            ),
            patch.object(task_graph, "global_intent_prediction") as mock_global,
            patch.object(
                task_graph, "handle_random_next_node", return_value=(False, {}, params)
            ),
            patch.object(task_graph, "handle_unknown_intent") as mock_unknown,
            patch.object(task_graph.graph, "successors", return_value=[]),
        ):
            # Mock global intent prediction to return a non-unsure intent
            mock_global.return_value = (False, "intent1", {}, params)

            mock_node_info = NodeInfo(
                node_id="planner",
                type="task",
                resource_name="test",
            )
            mock_unknown.return_value = (mock_node_info, params)

            node_info, params = task_graph.get_node(inputs)
            assert node_info.node_id == "planner"

    def test_get_node_with_global_intent_switch_and_unsure_intent_fallback(
        self, task_graph, params
    ) -> None:
        """Test get_node with global intent switch and unsure intent fallback (line 804)."""
        inputs = {
            "text": "test text",
            "chat_history_str": "",
            "parameters": params,
            "allow_global_intent_switch": True,
        }

        with (
            patch.object(
                task_graph, "get_current_node", return_value=("curr_node", params)
            ),
            patch.object(
                task_graph, "handle_multi_step_node", return_value=(False, None, params)
            ),
            patch.object(
                task_graph, "handle_leaf_node", return_value=("curr_node", params)
            ),
            patch.object(
                task_graph,
                "get_available_global_intents",
                return_value={"intent1": [{"intent": "intent1"}]},
            ),
            patch.object(
                task_graph,
                "get_local_intent",
                return_value={"intent1": [{"intent": "intent1"}]},
            ),
            patch.object(
                task_graph, "handle_incomplete_node", return_value=(False, {}, params)
            ),
            patch.object(
                task_graph, "local_intent_prediction", return_value=(False, {}, params)
            ),
            patch.object(task_graph, "global_intent_prediction") as mock_global,
            patch.object(
                task_graph, "handle_random_next_node", return_value=(False, {}, params)
            ),
            patch.object(task_graph, "handle_unknown_intent") as mock_unknown,
            patch.object(task_graph.graph, "successors", return_value=[]),
        ):
            # Mock global intent prediction to return unsure intent
            mock_global.return_value = (False, "others", {}, params)

            mock_node_info = NodeInfo(
                node_id="planner",
                type="task",
                resource_name="test",
            )
            mock_unknown.return_value = (mock_node_info, params)

            node_info, params = task_graph.get_node(inputs)
            assert node_info.node_id == "planner"

    def test_get_node_with_global_intent_switch_and_no_pred_intent_fallback(
        self, task_graph, params
    ) -> None:
        """Test get_node with global intent switch and no pred_intent fallback (line 804)."""
        inputs = {
            "text": "test text",
            "chat_history_str": "",
            "parameters": params,
            "allow_global_intent_switch": True,
        }

        with (
            patch.object(
                task_graph, "get_current_node", return_value=("curr_node", params)
            ),
            patch.object(
                task_graph, "handle_multi_step_node", return_value=(False, None, params)
            ),
            patch.object(
                task_graph, "handle_leaf_node", return_value=("curr_node", params)
            ),
            patch.object(
                task_graph,
                "get_available_global_intents",
                return_value={"intent1": [{"intent": "intent1"}]},
            ),
            patch.object(
                task_graph,
                "get_local_intent",
                return_value={"intent1": [{"intent": "intent1"}]},
            ),
            patch.object(
                task_graph, "handle_incomplete_node", return_value=(False, {}, params)
            ),
            patch.object(
                task_graph, "local_intent_prediction", return_value=(False, {}, params)
            ),
            patch.object(task_graph, "global_intent_prediction") as mock_global,
            patch.object(
                task_graph, "handle_random_next_node", return_value=(False, {}, params)
            ),
            patch.object(task_graph, "handle_unknown_intent") as mock_unknown,
            patch.object(task_graph.graph, "successors", return_value=[]),
        ):
            # Mock global intent prediction to return None pred_intent
            mock_global.return_value = (False, None, {}, params)

            mock_node_info = NodeInfo(
                node_id="planner",
                type="task",
                resource_name="test",
            )
            mock_unknown.return_value = (mock_node_info, params)

            node_info, params = task_graph.get_node(inputs)
            assert node_info.node_id == "planner"

    def test_get_node_with_global_intent_switch_and_random_fallback_after_pred_intent(
        self, task_graph, params
    ) -> None:
        """Test get_node with global intent switch and random fallback after pred_intent (line 804)."""
        inputs = {
            "text": "test text",
            "chat_history_str": "",
            "parameters": params,
            "allow_global_intent_switch": True,
        }

        with (
            patch.object(
                task_graph, "get_current_node", return_value=("curr_node", params)
            ),
            patch.object(
                task_graph, "handle_multi_step_node", return_value=(False, None, params)
            ),
            patch.object(
                task_graph, "handle_leaf_node", return_value=("curr_node", params)
            ),
            patch.object(
                task_graph,
                "get_available_global_intents",
                return_value={"intent1": [{"intent": "intent1"}]},
            ),
            patch.object(
                task_graph,
                "get_local_intent",
                return_value={"intent1": [{"intent": "intent1"}]},
            ),
            patch.object(
                task_graph, "handle_incomplete_node", return_value=(False, {}, params)
            ),
            patch.object(
                task_graph, "local_intent_prediction", return_value=(False, {}, params)
            ),
            patch.object(task_graph, "global_intent_prediction") as mock_global,
            patch.object(task_graph, "handle_random_next_node") as mock_random,
            patch.object(task_graph, "handle_unknown_intent") as mock_unknown,
            patch.object(task_graph.graph, "successors", return_value=[]),
        ):
            # Mock global intent prediction to return a non-unsure intent
            mock_global.return_value = (False, "intent1", {}, params)

            mock_node_info = NodeInfo(
                node_id="random_node",
                type="task",
                resource_name="test",
            )
            mock_random.return_value = (True, mock_node_info, params)

            node_info, params = task_graph.get_node(inputs)
            assert node_info.node_id == "random_node"

    def test_get_node_with_global_intent_switch_and_no_random_fallback_after_pred_intent(
        self, task_graph, params
    ) -> None:
        """Test get_node with global intent switch and no random fallback after pred intent."""
        # Set up params with no local intents and allow global intent switch
        params.taskgraph.curr_node = "node1"
        params.taskgraph.available_global_intents = {
            "continue": [
                {
                    "target_node": "node2",
                    "attribute": {"weight": 1.0},
                    "source_node": "node1",
                }
            ]
        }

        # Mock get_local_intent to return empty dict to force global intent prediction
        with patch.object(task_graph, "get_local_intent") as mock_local_intent:
            mock_local_intent.return_value = {}

            # Mock handle_incomplete_node to return False
            with patch.object(task_graph, "handle_incomplete_node") as mock_incomplete:
                mock_incomplete.return_value = (False, {}, params)

                # Mock handle_random_next_node to return False
                with patch.object(task_graph, "handle_random_next_node") as mock_random:
                    mock_random.return_value = (False, {}, params)

                    # Mock global_intent_prediction to return a pred_intent that's not unsure
                    with patch.object(
                        task_graph, "global_intent_prediction"
                    ) as mock_global:
                        mock_global.return_value = (False, "continue", {}, params)

                        inputs = {
                            "text": "test message",
                            "chat_history_str": "test history",
                            "parameters": params,
                            "allow_global_intent_switch": True,
                        }

                        result = task_graph.get_node(inputs)

                        # Verify that handle_random_next_node was called
                        assert mock_random.call_count == 2

    def test_task_graph_base_create_graph_not_implemented(self) -> None:
        """Test that TaskGraphBase.create_graph raises NotImplementedError."""
        from arklex.orchestrator.task_graph import TaskGraphBase

        # Test that create_graph raises NotImplementedError
        with pytest.raises(NotImplementedError):
            # Create a minimal config for TaskGraphBase
            config = {"nodes": [], "edges": []}
            base_graph = TaskGraphBase("test", config)

    def test_task_graph_init_with_non_string_slotfillapi(
        self, dummy_config, dummy_llm_config, mock_model_service
    ) -> None:
        """Test TaskGraph initialization with non-string slotfillapi."""
        # Create a mock model service for slotfillapi
        mock_slotfill_service = Mock()

        task_graph = TaskGraph(
            name="test",
            product_kwargs=dummy_config,
            llm_config=dummy_llm_config,
            slotfillapi=mock_slotfill_service,
            model_service=mock_model_service,
        )

        # Verify that the slotfillapi was set correctly
        assert task_graph.slotfillapi is not None
        assert isinstance(task_graph.slotfillapi, SlotFiller)

    def test_jump_to_node_protection_branch(self, task_graph) -> None:
        """Test jump_to_node protection branch when candidates_nodes is empty."""
        # Mock self.intents to return empty list for the pred_intent
        task_graph.intents = {"test_intent": []}

        # Mock the graph to have in_edges that return a valid intent
        with patch.object(task_graph.graph, "in_edges") as mock_in_edges:
            mock_in_edges.return_value = [("source", "curr_node", "test_intent")]

            result = task_graph.jump_to_node("test_intent", 0, "curr_node")

            # Should return current node and the intent from in_edges
            assert result == ("curr_node", "test_intent")

    def test_local_intent_prediction_with_unsure_intent_already_present(
        self, task_graph, params
    ) -> None:
        """Test local_intent_prediction when unsure intent is already present in curr_local_intents."""
        # Set up params
        params.taskgraph.curr_node = "node1"
        params.taskgraph.nlu_records = []

        # Mock the intent detector
        with patch.object(task_graph.intent_detector, "execute") as mock_execute:
            mock_execute.return_value = "others"

            # Set up curr_local_intents with unsure intent already present
            curr_local_intents = {
                "others": [{"target_node": "node2", "attribute": {"weight": 1.0}}]
            }

            # Mock _postprocess_intent to return found_pred_in_avil=True
            with patch.object(task_graph, "_postprocess_intent") as mock_postprocess:
                mock_postprocess.return_value = (True, "others", 0)

                # Mock _get_node
                with patch.object(task_graph, "_get_node") as mock_get_node:
                    mock_node_info = NodeInfo(
                        node_id="node2",
                        type="task",
                        resource_id="test_id",
                        resource_name="test_name",
                        can_skipped=False,
                        is_leaf=False,
                        attributes={},
                        additional_args={},
                    )
                    mock_get_node.return_value = (mock_node_info, params)

                    result = task_graph.local_intent_prediction(
                        "node1", params, curr_local_intents
                    )

                    # Should return True, node_info, params
                    assert result[0] is True
                    assert isinstance(result[1], NodeInfo)
                    assert result[2] == params

    def test_handle_unknown_intent_with_existing_nlu_records(
        self, task_graph, params
    ) -> None:
        """Test handle_unknown_intent when nlu_records already exist."""
        # Set up params with existing nlu_records
        params.taskgraph.nlu_records = [
            {
                "candidate_intents": [],
                "pred_intent": "test",
                "no_intent": False,
                "global_intent": False,
            }
        ]

        result = task_graph.handle_unknown_intent("node1", params)

        # Verify that the last nlu_record was updated
        assert params.taskgraph.nlu_records[-1]["no_intent"] is True
        assert result[0].resource_name == "planner"
        assert result[1] == params

    def test_get_node_with_start_text(self, task_graph, params) -> None:
        """Test get_node with start text."""
        # Mock _get_node
        with patch.object(task_graph, "_get_node") as mock_get_node:
            mock_node_info = NodeInfo(
                node_id="start",
                type="start",
                resource_id="start_id",
                resource_name="start_name",
                can_skipped=False,
                is_leaf=False,
                attributes={},
                additional_args={},
            )
            mock_get_node.return_value = (mock_node_info, params)

            inputs = {
                "text": "<start>",
                "chat_history_str": "test history",
                "parameters": params,
                "allow_global_intent_switch": True,
            }

            result = task_graph.get_node(inputs)

            # Should return the start node
            assert result[0].node_id == "start"
            assert result[1] == params

    def test_get_node_with_no_local_intents_and_global_intent_switch_disabled(
        self, task_graph, params
    ) -> None:
        """Test get_node when no local intents and global intent switch is disabled."""
        # Set up params
        params.taskgraph.curr_node = "node1"

        # Mock get_local_intent to return empty dict
        with patch.object(task_graph, "get_local_intent") as mock_local_intent:
            mock_local_intent.return_value = {}

            # Mock handle_incomplete_node to return False
            with patch.object(task_graph, "handle_incomplete_node") as mock_incomplete:
                mock_incomplete.return_value = (False, {}, params)

                # Mock handle_random_next_node to return False
                with patch.object(task_graph, "handle_random_next_node") as mock_random:
                    mock_random.return_value = (False, {}, params)

                    # Mock local_intent_prediction to return False
                    with patch.object(
                        task_graph, "local_intent_prediction"
                    ) as mock_local_pred:
                        mock_local_pred.return_value = (False, {}, params)

                        # Mock global_intent_prediction to return False
                        with patch.object(
                            task_graph, "global_intent_prediction"
                        ) as mock_global_pred:
                            mock_global_pred.return_value = (False, None, {}, params)

                            # Mock handle_unknown_intent
                            with patch.object(
                                task_graph, "handle_unknown_intent"
                            ) as mock_unknown:
                                mock_node_info = NodeInfo(
                                    node_id=None,
                                    type="",
                                    resource_id="planner",
                                    resource_name="planner",
                                    can_skipped=False,
                                    is_leaf=False,
                                    attributes={"value": "", "direct": False},
                                    additional_args={"tags": {}},
                                )
                                mock_unknown.return_value = (mock_node_info, params)

                                inputs = {
                                    "text": "test message",
                                    "chat_history_str": "test history",
                                    "parameters": params,
                                    "allow_global_intent_switch": False,
                                }

                                result = task_graph.get_node(inputs)

                                # Should return the planner node
                                assert result[0].resource_name == "planner"
                                assert result[1] == params

    def test_get_node_with_pred_intent_not_unsure_and_random_fallback(
        self, task_graph, params
    ) -> None:
        """Test get_node when pred_intent is not unsure and random fallback is available."""
        # Set up params
        params.taskgraph.curr_node = "node1"

        # Mock get_local_intent to return empty dict
        with patch.object(task_graph, "get_local_intent") as mock_local_intent:
            mock_local_intent.return_value = {}

            # Mock handle_incomplete_node to return False
            with patch.object(task_graph, "handle_incomplete_node") as mock_incomplete:
                mock_incomplete.return_value = (False, {}, params)

                # Mock handle_random_next_node to return False initially
                with patch.object(task_graph, "handle_random_next_node") as mock_random:
                    mock_random.return_value = (False, {}, params)

                    # Mock local_intent_prediction to return False
                    with patch.object(
                        task_graph, "local_intent_prediction"
                    ) as mock_local_pred:
                        mock_local_pred.return_value = (False, {}, params)

                        # Mock global_intent_prediction to return a pred_intent that's not unsure
                        with patch.object(
                            task_graph, "global_intent_prediction"
                        ) as mock_global_pred:
                            mock_global_pred.return_value = (
                                False,
                                "continue",
                                {},
                                params,
                            )

                            # Mock handle_random_next_node to return True for the second call
                            mock_random.return_value = (
                                True,
                                {"target_node": "node2"},
                                params,
                            )

                            inputs = {
                                "text": "test message",
                                "chat_history_str": "test history",
                                "parameters": params,
                                "allow_global_intent_switch": True,
                            }

                            result = task_graph.get_node(inputs)

                            # Should return the random next node
                            assert result[0]["target_node"] == "node2"
                            assert result[1] == params

    def test_get_node_with_pred_intent_unsure(self, task_graph, params) -> None:
        """Test get_node when pred_intent is unsure."""
        # Set up params
        params.taskgraph.curr_node = "node1"

        # Mock get_local_intent to return empty dict
        with patch.object(task_graph, "get_local_intent") as mock_local_intent:
            mock_local_intent.return_value = {}

            # Mock handle_incomplete_node to return False
            with patch.object(task_graph, "handle_incomplete_node") as mock_incomplete:
                mock_incomplete.return_value = (False, {}, params)

                # Mock handle_random_next_node to return False
                with patch.object(task_graph, "handle_random_next_node") as mock_random:
                    mock_random.return_value = (False, {}, params)

                    # Mock local_intent_prediction to return False
                    with patch.object(
                        task_graph, "local_intent_prediction"
                    ) as mock_local_pred:
                        mock_local_pred.return_value = (False, {}, params)

                        # Mock global_intent_prediction to return unsure intent
                        with patch.object(
                            task_graph, "global_intent_prediction"
                        ) as mock_global_pred:
                            mock_global_pred.return_value = (
                                False,
                                "others",
                                {},
                                params,
                            )

                            # Mock handle_unknown_intent
                            with patch.object(
                                task_graph, "handle_unknown_intent"
                            ) as mock_unknown:
                                mock_node_info = NodeInfo(
                                    node_id=None,
                                    type="",
                                    resource_id="planner",
                                    resource_name="planner",
                                    can_skipped=False,
                                    is_leaf=False,
                                    attributes={"value": "", "direct": False},
                                    additional_args={"tags": {}},
                                )
                                mock_unknown.return_value = (mock_node_info, params)

                                inputs = {
                                    "text": "test message",
                                    "chat_history_str": "test history",
                                    "parameters": params,
                                    "allow_global_intent_switch": True,
                                }

                                result = task_graph.get_node(inputs)

                                # Should return the planner node
                                assert result[0].resource_name == "planner"
                                assert result[1] == params

    def test_validate_node_not_dict(self, task_graph) -> None:
        """Test _validate_node with non-dict input."""
        with pytest.raises(TaskGraphError, match="Node must be a dictionary"):
            task_graph._validate_node("not a dict")

    def test_validate_node_missing_id(self, task_graph) -> None:
        """Test _validate_node with missing id."""
        with pytest.raises(TaskGraphError, match="Node must have an id"):
            task_graph._validate_node({"type": "test"})

    def test_validate_node_missing_type(self, task_graph) -> None:
        """Test _validate_node with missing type."""
        with pytest.raises(TaskGraphError, match="Node must have a type"):
            task_graph._validate_node({"id": "test"})

    def test_validate_node_invalid_next(self, task_graph) -> None:
        """Test _validate_node with invalid next field."""
        with pytest.raises(TaskGraphError, match="Node next must be a list"):
            task_graph._validate_node(
                {"id": "test", "type": "test", "next": "not a list"}
            )

    def test_jump_to_node_with_exception_and_no_in_edges(self, task_graph) -> None:
        """Test jump_to_node when exception occurs and no in_edges exist."""
        # Mock self.intents to return empty list for the pred_intent
        task_graph.intents = {"test_intent": []}

        # Mock the graph to have no in_edges
        with patch.object(task_graph.graph, "in_edges") as mock_in_edges:
            mock_in_edges.return_value = []

            # This should raise an IndexError when trying to access [0][2]
            with pytest.raises(IndexError):
                task_graph.jump_to_node("test_intent", 0, "curr_node")

    def test_local_intent_prediction_with_unsure_intent_not_present(
        self, task_graph, params
    ) -> None:
        """Test local_intent_prediction when unsure intent is not present in curr_local_intents."""
        # Set up params
        params.taskgraph.curr_node = "node1"
        params.taskgraph.nlu_records = []

        # Mock the intent detector
        with patch.object(task_graph.intent_detector, "execute") as mock_execute:
            mock_execute.return_value = "others"

            # Set up curr_local_intents without unsure intent
            curr_local_intents = {
                "continue": [{"target_node": "node2", "attribute": {"weight": 1.0}}]
            }

            # Mock _postprocess_intent to return found_pred_in_avil=False
            with patch.object(task_graph, "_postprocess_intent") as mock_postprocess:
                mock_postprocess.return_value = (False, "others", 0)

                result = task_graph.local_intent_prediction(
                    "node1", params, curr_local_intents
                )

                # Should return False, {}, params
                assert result[0] is False
                assert result[1] == {}
                assert result[2] == params

    def test_handle_leaf_node_with_nested_graph_leaf(self, task_graph, params) -> None:
        """Test handle_leaf_node with nested graph that returns a leaf node."""
        # Set initial_node to None to prevent it from overriding the nested graph result
        task_graph.initial_node = None

        # Mock NestedGraph.get_nested_graph_component_node to return a leaf node
        with patch(
            "arklex.orchestrator.task_graph.NestedGraph.get_nested_graph_component_node"
        ) as mock_nested:
            mock_node_info = NodeInfo(
                node_id="nested_leaf",
                type="task",
                resource_id="test_id",
                resource_name="test_name",
                can_skipped=False,
                is_leaf=True,
                attributes={},
                additional_args={},
            )
            mock_nested.return_value = (mock_node_info, params)

            # Mock the graph to make curr_node a leaf
            with patch.object(task_graph.graph, "successors") as mock_successors:
                mock_successors.return_value = []

                result_node, result_params = task_graph.handle_leaf_node(
                    "curr_node", params
                )

                # Should return the nested leaf node
                assert result_node == "nested_leaf"
                assert result_params == params

    def test_jump_to_node_else_branch(self, task_graph) -> None:
        """Covers the else branch in jump_to_node (lines 239-240)."""
        # Setup: no candidates for the intent
        task_graph.intents = {"no_such_intent": []}
        # Patch in_edges to return a valid intent
        with patch.object(task_graph.graph, "in_edges") as mock_in_edges:
            mock_in_edges.return_value = [("src", "curr", "intent_from_edge")]
            next_node, next_intent = task_graph.jump_to_node(
                "no_such_intent", 0, "curr"
            )
            assert next_node == "curr"
            assert next_intent == "intent_from_edge"

    def test_local_intent_prediction_found_pred_in_avil_false(
        self, task_graph, params
    ) -> None:
        """Covers found_pred_in_avil False in local_intent_prediction (lines 624-625, 630)."""
        params.taskgraph.curr_node = "node1"
        params.taskgraph.nlu_records = []
        curr_local_intents = {
            "intentA": [{"target_node": "node2", "attribute": {"weight": 1.0}}]
        }
        with patch.object(task_graph.intent_detector, "execute", return_value="others"):
            with patch.object(
                task_graph, "_postprocess_intent", return_value=(False, "others", 0)
            ):
                result = task_graph.local_intent_prediction(
                    "node1", params, curr_local_intents
                )
                assert result == (False, {}, params)

    def test_handle_leaf_node_not_leaf_and_nested_none(
        self, task_graph, params
    ) -> None:
        """Covers the not-leaf and nested_graph returns None branch in handle_leaf_node (line 688)."""
        # Add a non-leaf node
        task_graph.graph.add_node(
            "not_leaf",
            type="task",
            resource={"name": "test", "id": "test"},
            attribute={},
        )
        task_graph.graph.add_edge(
            "not_leaf", "next", intent="test", attribute={"weight": 1.0}
        )
        # Patch NestedGraph.get_nested_graph_component_node to return (None, params)
        with patch(
            "arklex.orchestrator.task_graph.NestedGraph.get_nested_graph_component_node",
            return_value=(None, params),
        ):
            node, _ = task_graph.handle_leaf_node("not_leaf", params)
            assert node == "not_leaf"

    def test_get_node_final_fallback(self, task_graph, params) -> None:
        """Covers the final fallback in get_node (lines 808, 831-834, 846-850, 867-871)."""
        # Patch all intent and node handlers to return fall-throughs
        with (
            patch.object(task_graph, "get_current_node", return_value=("curr", params)),
            patch.object(
                task_graph, "handle_multi_step_node", return_value=(False, None, params)
            ),
            patch.object(task_graph, "handle_leaf_node", return_value=("curr", params)),
            patch.object(task_graph, "get_available_global_intents", return_value={}),
            patch.object(task_graph, "get_local_intent", return_value={}),
            patch.object(
                task_graph, "handle_incomplete_node", return_value=(False, {}, params)
            ),
            patch.object(
                task_graph, "handle_random_next_node", return_value=(False, {}, params)
            ),
            patch.object(
                task_graph, "local_intent_prediction", return_value=(False, {}, params)
            ),
            patch.object(
                task_graph,
                "global_intent_prediction",
                return_value=(False, None, {}, params),
            ),
            patch.object(
                task_graph,
                "handle_unknown_intent",
                return_value=(
                    NodeInfo(
                        node_id=None,
                        type="",
                        resource_id="planner",
                        resource_name="planner",
                        can_skipped=False,
                        is_leaf=False,
                        attributes={},
                        additional_args={},
                    ),
                    params,
                ),
            ),
        ):
            inputs = {
                "text": "irrelevant",
                "chat_history_str": "",
                "parameters": params,
                "allow_global_intent_switch": True,
            }
            node_info, _ = task_graph.get_node(inputs)
            assert node_info.resource_name == "planner"

    def test_validate_node_all_error_branches(self, task_graph) -> None:
        """Covers all error branches in _validate_node (lines 831-834, 846-850, 867-871)."""
        # Not a dict
        with pytest.raises(TaskGraphError):
            task_graph._validate_node("notadict")
        # Missing id
        with pytest.raises(TaskGraphError):
            task_graph._validate_node({"type": "t"})
        # Missing type
        with pytest.raises(TaskGraphError):
            task_graph._validate_node({"id": "i"})
        # next is not a list
        with pytest.raises(TaskGraphError):
            task_graph._validate_node({"id": "i", "type": "t", "next": "notalist"})

    def test_jump_to_node_exception_branch(self, task_graph) -> None:
        # Simulate an exception in jump_to_node by patching in_edges to raise
        with patch.object(task_graph.graph, "in_edges", side_effect=Exception("fail")):
            # This should handle the exception gracefully
            with pytest.raises(Exception, match="fail"):
                next_node, next_intent = task_graph.jump_to_node("intent", 0, "node1")

    def test_handle_leaf_node_with_no_successors(self, task_graph, params) -> None:
        # Should handle node with no successors
        node = "end"
        result, updated_params = task_graph.handle_leaf_node(node, params)
        assert isinstance(result, str)

    def test_handle_leaf_node_with_successors(self, task_graph, params) -> None:
        # Should handle node with successors
        node = "node1"
        result, updated_params = task_graph.handle_leaf_node(node, params)
        assert isinstance(result, str)

    def test_handle_leaf_node_with_nested_graph(self, task_graph, params) -> None:
        # Should handle node with nested_graph attribute
        node = "node1"
        task_graph.graph.nodes[node]["attribute"]["nested_graph"] = True
        result, updated_params = task_graph.handle_leaf_node(node, params)
        assert isinstance(result, str)
        del task_graph.graph.nodes[node]["attribute"]["nested_graph"]

    def test_validate_node_handles_missing_fields(self, task_graph) -> None:
        # Should raise for missing id/type/attribute/next
        node = {"resource": {}}
        with pytest.raises(Exception):
            task_graph._validate_node(node)

    def test_validate_node_handles_all_error_branches(self, task_graph) -> None:
        # Should raise for all error branches
        node = {
            "id": None,
            "type": None,
            "attribute": None,
            "next": None,
            "resource": {},
        }
        with pytest.raises(Exception):
            task_graph._validate_node(node)
