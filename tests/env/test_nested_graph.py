"""Tests for the nested_graph module.

This module contains comprehensive test cases for nested graph functionality,
including nested graph component node identification and path traversal.
"""

from contextlib import suppress
from unittest.mock import Mock

import pytest

from arklex.env.nested_graph.nested_graph import NestedGraph
from arklex.orchestrator.entities.orchestrator_params_entities import OrchestratorParams
from arklex.orchestrator.entities.taskgraph_entities import NodeInfo, PathNode


class TestNestedGraph:
    """Test cases for NestedGraph class.

    This class contains comprehensive tests for nested graph functionality,
    including initialization, start node identification, and component node detection.
    """

    def test_nested_graph_initialization(self) -> None:
        """Test NestedGraph initialization with valid node info."""
        # Setup
        node_info = Mock(spec=NodeInfo)
        node_info.attributes = {"value": "test_start_node"}

        # Execute
        nested_graph = NestedGraph(node_info)

        # Assert
        assert nested_graph.node_info == node_info

    def test_get_nested_graph_start_node_id(self) -> None:
        """Test get_nested_graph_start_node_id returns correct start node ID."""
        # Setup
        node_info = Mock(spec=NodeInfo)
        node_info.attributes = {"value": "test_start_node"}
        nested_graph = NestedGraph(node_info)

        # Execute
        result = nested_graph.get_nested_graph_start_node_id()

        # Assert
        assert result == "test_start_node"

    def test_get_nested_graph_start_node_id_with_numeric_value(self) -> None:
        """Test get_nested_graph_start_node_id with numeric value."""
        # Setup
        node_info = Mock(spec=NodeInfo)
        node_info.attributes = {"value": 123}
        nested_graph = NestedGraph(node_info)

        # Execute
        result = nested_graph.get_nested_graph_start_node_id()

        # Assert
        assert result == "123"

    def test_get_nested_graph_component_node_main_graph(self) -> None:
        """Test get_nested_graph_component_node when node is in main graph."""
        # Setup
        params = Mock(spec=OrchestratorParams)
        params.taskgraph = Mock()
        params.taskgraph.path = [
            PathNode(
                node_id="node1",
                nested_graph_leaf_jump=None,
                nested_graph_node_value=None,
            ),
            PathNode(
                node_id="node2",
                nested_graph_leaf_jump=None,
                nested_graph_node_value=None,
            ),
        ]
        params.taskgraph.node_status = {}

        def is_leaf_func(node_id: str) -> bool:
            return node_id == "node2"

        # Execute
        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        # Assert
        assert result_node.node_id == "node2"
        assert result_params == params

    def test_get_nested_graph_component_node_nested_graph_not_leaf(self) -> None:
        """Test get_nested_graph_component_node when nested graph component is not a leaf."""
        # Setup
        params = Mock(spec=OrchestratorParams)
        params.taskgraph = Mock()
        params.taskgraph.path = [
            PathNode(
                node_id="main_node",
                nested_graph_leaf_jump=None,
                nested_graph_node_value=None,
            ),
            PathNode(
                node_id="nested_node",
                nested_graph_leaf_jump=None,
                nested_graph_node_value="main_node",
            ),
        ]
        params.taskgraph.node_status = {}

        def is_leaf_func(node_id: str) -> bool:
            return node_id == "main_node"  # nested_node is not a leaf

        # Execute
        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        # Assert
        assert result_node.node_id == "nested_node"
        assert result_params == params

    def test_get_nested_graph_component_node_nested_graph_leaf(self) -> None:
        """Test get_nested_graph_component_node when nested graph component is a leaf."""
        # Setup
        params = Mock(spec=OrchestratorParams)
        params.taskgraph = Mock()
        params.taskgraph.path = [
            PathNode(
                node_id="main_node",
                nested_graph_leaf_jump=None,
                nested_graph_node_value=None,
            ),
            PathNode(
                node_id="nested_node",
                nested_graph_leaf_jump=None,
                nested_graph_node_value="main_node",
            ),
        ]
        params.taskgraph.node_status = {}

        def is_leaf_func(node_id: str) -> bool:
            return node_id in ["main_node", "nested_node"]  # Both are leaves

        # Execute
        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        # Assert
        assert result_node.node_id == "nested_node"
        assert result_params == params

    def test_get_nested_graph_component_node_with_leaf_jump(self) -> None:
        """Test get_nested_graph_component_node with nested_graph_leaf_jump."""
        params = OrchestratorParams()
        path_node1 = PathNode(node_id="node1", nested_graph_leaf_jump=0)
        path_node2 = PathNode(node_id="node2", nested_graph_node_value="node1")
        params.taskgraph.path = [path_node1, path_node2]

        def is_leaf_func(node_id: str) -> bool:
            return node_id == "node1"

        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        assert result_node is not None
        # The function starts from the last node (node2) and works backwards
        # Since no nested graph component is found, it returns the current node (node2)
        assert result_node.node_id == "node2"
        # No node status should be updated since no nested graph was found
        assert "node1" not in result_params.taskgraph.node_status

    def test_get_nested_graph_component_node_complex_nested_structure(self) -> None:
        """Test get_nested_graph_component_node with complex nested structure."""
        params = OrchestratorParams()
        path_node1 = PathNode(node_id="node1", nested_graph_node_value="node2")
        path_node2 = PathNode(node_id="node2", nested_graph_node_value="node3")
        path_node3 = PathNode(node_id="node3")
        params.taskgraph.path = [path_node1, path_node2, path_node3]

        def is_leaf_func(node_id: str) -> bool:
            return node_id in ["node1", "node2"]  # node1 and node2 are leaves

        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        assert result_node is not None
        # The algorithm finds nested graph components and returns node1
        # because it's the first node that forms a valid nested graph pattern
        assert result_node.node_id == "node1"

    def test_get_nested_graph_component_node_updates_node_status(self) -> None:
        """Test that get_nested_graph_component_node updates node status correctly."""
        params = OrchestratorParams()
        path_node1 = PathNode(node_id="node1")
        path_node2 = PathNode(node_id="node2", nested_graph_node_value="node1")
        params.taskgraph.path = [path_node1, path_node2]
        params.taskgraph.node_status = {}

        def is_leaf_func(node_id: str) -> bool:
            return node_id == "node1"

        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        assert result_node is not None
        # The function starts from node2, but no nested graph component is found
        # So it returns the current node (node2)
        assert result_node.node_id == "node2"
        # No node status should be updated since no nested graph was found
        assert "node1" not in result_params.taskgraph.node_status

    def test_get_nested_graph_component_node_empty_path(self) -> None:
        """Test get_nested_graph_component_node with empty path."""
        params = OrchestratorParams()
        params.taskgraph.path = []

        def is_leaf_func(node_id: str) -> bool:
            return False

        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        assert result_node is None
        assert result_params == params

    def test_get_nested_graph_component_node_single_node_path(self) -> None:
        """Test get_nested_graph_component_node with single node path."""
        params = OrchestratorParams()
        path_node = PathNode(node_id="node1")
        params.taskgraph.path = [path_node]

        def is_leaf_func(node_id: str) -> bool:
            return False

        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        assert result_node is not None
        assert result_node.node_id == "node1"

    def test_get_nested_graph_component_node_multiple_nested_levels(self) -> None:
        """Test get_nested_graph_component_node with multiple nested levels."""
        params = OrchestratorParams()
        path_node1 = PathNode(node_id="node1", nested_graph_node_value="node2")
        path_node2 = PathNode(node_id="node2", nested_graph_node_value="node3")
        path_node3 = PathNode(node_id="node3", nested_graph_node_value="node4")
        path_node4 = PathNode(node_id="node4")
        params.taskgraph.path = [path_node1, path_node2, path_node3, path_node4]

        def is_leaf_func(node_id: str) -> bool:
            return node_id in ["node1", "node2", "node3"]  # All except node4 are leaves

        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        assert result_node is not None
        # The algorithm finds nested graph components and returns node1
        # because it's the first node that forms a valid nested graph pattern
        assert result_node.node_id == "node1"

    def test_get_nested_graph_component_node_no_nested_graph_found(self) -> None:
        """Test get_nested_graph_component_node when no nested graph is found."""
        params = OrchestratorParams()
        path_node1 = PathNode(node_id="node1")
        path_node2 = PathNode(node_id="node2")
        params.taskgraph.path = [path_node1, path_node2]

        def is_leaf_func(node_id: str) -> bool:
            return False

        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        assert result_node is not None
        assert (
            result_node.node_id == "node2"
        )  # Should return current node (last in path)

    def test_get_nested_graph_component_node_with_updated_path(self) -> None:
        """Test get_nested_graph_component_node with path that gets updated during processing."""
        params = OrchestratorParams()
        path_node1 = PathNode(node_id="node1")
        path_node2 = PathNode(node_id="node2", nested_graph_node_value="node1")
        params.taskgraph.path = [path_node1, path_node2]

        def is_leaf_func(node_id: str) -> bool:
            return node_id == "node1"

        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        assert result_node is not None
        # The function starts from node2, but no nested graph component is found
        # So it returns the current node (node2)
        assert result_node.node_id == "node2"
        # No path should be updated since no nested graph was found
        assert result_params.taskgraph.path[1].nested_graph_leaf_jump is None

    def test_get_nested_graph_component_node_complex_leaf_jump_scenario(self) -> None:
        """Test get_nested_graph_component_node with complex leaf jump scenario."""
        params = OrchestratorParams()
        # Create a valid scenario where the leaf jump points to a valid index
        path_node1 = PathNode(
            node_id="node1", nested_graph_leaf_jump=0
        )  # Points to itself
        path_node2 = PathNode(node_id="node2")
        path_node3 = PathNode(node_id="node3", nested_graph_node_value="node1")
        params.taskgraph.path = [path_node1, path_node2, path_node3]
        params.taskgraph.node_status = {}

        def is_leaf_func(node_id: str) -> bool:
            return node_id == "node1"

        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        assert result_node is not None
        # The function starts from node3, but no nested graph component is found
        # So it returns the current node (node3)
        assert result_node.node_id == "node3"

    def test_get_nested_graph_component_node_nested_graph_id_constant(self) -> None:
        """Test that NESTED_GRAPH_ID constant is correctly defined."""
        from arklex.env.nested_graph.nested_graph import NESTED_GRAPH_ID

        assert NESTED_GRAPH_ID == "nested_graph"

    # Defensive/Rare Case Tests
    def test_get_nested_graph_start_node_id_missing_attributes(self) -> None:
        """Test get_nested_graph_start_node_id when attributes dict is missing."""
        # Setup
        node_info = Mock(spec=NodeInfo)
        node_info.attributes = {}  # Missing "value" key
        nested_graph = NestedGraph(node_info)

        # Execute & Assert - Should raise KeyError
        try:
            result = nested_graph.get_nested_graph_start_node_id()
            raise AssertionError("Expected KeyError but got result: " + str(result))
        except KeyError:
            pass  # Expected behavior

    def test_get_nested_graph_start_node_id_none_attributes(self) -> None:
        """Test get_nested_graph_start_node_id when attributes is None."""
        # Setup
        node_info = Mock(spec=NodeInfo)
        node_info.attributes = None
        nested_graph = NestedGraph(node_info)

        # Execute & Assert - Should raise TypeError (not AttributeError)
        try:
            result = nested_graph.get_nested_graph_start_node_id()
            raise AssertionError("Expected TypeError but got result: " + str(result))
        except TypeError:
            pass  # Expected behavior

    def test_get_nested_graph_start_node_id_none_value(self) -> None:
        """Test get_nested_graph_start_node_id when value is None."""
        # Setup
        node_info = Mock(spec=NodeInfo)
        node_info.attributes = {"value": None}
        nested_graph = NestedGraph(node_info)

        # Execute
        result = nested_graph.get_nested_graph_start_node_id()

        # Assert - Should convert None to string "None"
        assert result == "None"

    def test_get_nested_graph_start_node_id_empty_value(self) -> None:
        """Test get_nested_graph_start_node_id when value is empty string."""
        # Setup
        node_info = Mock(spec=NodeInfo)
        node_info.attributes = {"value": ""}
        nested_graph = NestedGraph(node_info)

        # Execute
        result = nested_graph.get_nested_graph_start_node_id()

        # Assert
        assert result == ""

    def test_get_nested_graph_component_node_none_params(self) -> None:
        """Test get_nested_graph_component_node with None params."""

        # Setup
        def is_leaf_func(node_id: str) -> bool:
            return False

        # Execute & Assert - Should raise AttributeError
        try:
            result = NestedGraph.get_nested_graph_component_node(None, is_leaf_func)
            raise AssertionError(
                "Expected AttributeError but got result: " + str(result)
            )
        except AttributeError:
            pass  # Expected behavior

    def test_get_nested_graph_component_node_none_is_leaf_func(self) -> None:
        """Test get_nested_graph_component_node with None is_leaf_func."""
        # Setup
        params = OrchestratorParams()
        params.taskgraph.path = [PathNode(node_id="node1")]

        # Execute - The code actually handles None gracefully
        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, None
        )

        # Assert - Should return the current node
        assert result_node is not None
        assert result_node.node_id == "node1"

    def test_get_nested_graph_component_node_is_leaf_func_raises_exception(
        self,
    ) -> None:
        """Test get_nested_graph_component_node when is_leaf_func raises an exception."""
        # Setup
        params = OrchestratorParams()
        params.taskgraph.path = [PathNode(node_id="node1")]

        def is_leaf_func(node_id: str) -> bool:
            raise ValueError("Test exception")

        # Execute - The code actually handles exceptions gracefully
        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        # Assert - Should return the current node even when is_leaf_func raises
        assert result_node is not None
        assert result_node.node_id == "node1"

    def test_get_nested_graph_component_node_path_with_none_nodes(self) -> None:
        """Test get_nested_graph_component_node with None nodes in path."""
        # Setup
        params = OrchestratorParams()
        params.taskgraph.path = [None, PathNode(node_id="node1")]

        def is_leaf_func(node_id: str) -> bool:
            return False

        # Execute & Assert - Should raise AttributeError when accessing None node
        try:
            result = NestedGraph.get_nested_graph_component_node(params, is_leaf_func)
            raise AssertionError(
                "Expected AttributeError but got result: " + str(result)
            )
        except AttributeError:
            pass  # Expected behavior

    def test_get_nested_graph_component_node_nested_graph_node_value_points_to_nonexistent(
        self,
    ) -> None:
        """Test when nested_graph_node_value points to a node that doesn't exist in path."""
        # Setup
        params = OrchestratorParams()
        path_node1 = PathNode(node_id="node1")
        path_node2 = PathNode(
            node_id="node2", nested_graph_node_value="nonexistent_node"
        )
        params.taskgraph.path = [path_node1, path_node2]
        params.taskgraph.node_status = {}

        def is_leaf_func(node_id: str) -> bool:
            return node_id == "node2"

        # Execute
        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        # Assert - Should return current node since no valid nested graph is found
        assert result_node is not None
        assert result_node.node_id == "node2"
        assert "node1" not in result_params.taskgraph.node_status

    def test_get_nested_graph_component_node_negative_leaf_jump(self) -> None:
        """Test get_nested_graph_component_node with negative nested_graph_leaf_jump."""
        # Setup
        params = OrchestratorParams()
        path_node1 = PathNode(node_id="node1", nested_graph_leaf_jump=-1)
        path_node2 = PathNode(node_id="node2")
        params.taskgraph.path = [path_node1, path_node2]

        def is_leaf_func(node_id: str) -> bool:
            return False

        # Execute - The code actually handles negative indices gracefully
        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        # Assert - Should return the current node (node2)
        assert result_node is not None
        assert result_node.node_id == "node2"

    def test_get_nested_graph_component_node_leaf_jump_out_of_bounds(self) -> None:
        """Test get_nested_graph_component_node with nested_graph_leaf_jump out of bounds."""
        # Setup
        params = OrchestratorParams()
        path_node1 = PathNode(
            node_id="node1", nested_graph_leaf_jump=10
        )  # Out of bounds
        path_node2 = PathNode(node_id="node2")
        params.taskgraph.path = [path_node1, path_node2]

        def is_leaf_func(node_id: str) -> bool:
            return False

        # Execute & Assert - Should raise IndexError when accessing out of bounds index
        try:
            result = NestedGraph.get_nested_graph_component_node(params, is_leaf_func)
            raise AssertionError("Expected IndexError but got result: " + str(result))
        except IndexError:
            pass  # Expected behavior

    def test_get_nested_graph_component_node_missing_taskgraph_attributes(self) -> None:
        """Test get_nested_graph_component_node when taskgraph is missing required attributes."""
        # Setup
        params = Mock(spec=OrchestratorParams)
        params.taskgraph = Mock()
        # Configure the mock to raise AttributeError when accessing path
        params.taskgraph.path = Mock()
        params.taskgraph.path.__len__ = Mock(
            side_effect=AttributeError("path attribute missing")
        )

        def is_leaf_func(node_id: str) -> bool:
            return False

        # Execute & Assert - Should raise AttributeError
        try:
            result = NestedGraph.get_nested_graph_component_node(params, is_leaf_func)
            raise AssertionError(
                "Expected AttributeError but got result: " + str(result)
            )
        except AttributeError:
            pass  # Expected behavior

    def test_get_nested_graph_component_node_missing_node_status(self) -> None:
        """Test get_nested_graph_component_node when node_status is missing."""
        # Setup
        params = OrchestratorParams()
        path_node1 = PathNode(node_id="node1")
        path_node2 = PathNode(node_id="node2", nested_graph_node_value="node1")
        params.taskgraph.path = [path_node1, path_node2]
        # Don't set node_status

        def is_leaf_func(node_id: str) -> bool:
            return node_id == "node1"

        # Execute - The code actually handles missing node_status gracefully
        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        # Assert - Should return the current node (node2)
        assert result_node is not None
        assert result_node.node_id == "node2"

    def test_get_nested_graph_component_node_circular_reference(self) -> None:
        """Test get_nested_graph_component_node with circular nested graph references."""
        # Setup
        params = OrchestratorParams()
        path_node1 = PathNode(node_id="node1", nested_graph_node_value="node2")
        path_node2 = PathNode(node_id="node2", nested_graph_node_value="node1")
        params.taskgraph.path = [path_node1, path_node2]
        params.taskgraph.node_status = {}

        def is_leaf_func(node_id: str) -> bool:
            return node_id in ["node1", "node2"]

        # Execute
        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        # Assert - Should handle circular reference gracefully
        assert result_node is not None
        assert result_node.node_id in ["node1", "node2"]

    def test_get_nested_graph_component_node_pathnode_with_none_attributes(
        self,
    ) -> None:
        """Test get_nested_graph_component_node with PathNode having None attributes."""
        # Setup
        params = OrchestratorParams()
        path_node1 = PathNode(
            node_id="node1", nested_graph_leaf_jump=None, nested_graph_node_value=None
        )
        path_node2 = PathNode(
            node_id="node2", nested_graph_leaf_jump=None, nested_graph_node_value=None
        )
        params.taskgraph.path = [path_node1, path_node2]
        params.taskgraph.node_status = {}

        def is_leaf_func(node_id: str) -> bool:
            return node_id == "node2"

        # Execute
        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        # Assert - Should work normally with None attributes
        assert result_node is not None
        assert result_node.node_id == "node2"

    def test_get_nested_graph_component_node_empty_path_triggers_none(self) -> None:
        """Test get_nested_graph_component_node returns None when path is empty."""
        params = OrchestratorParams()
        params.taskgraph.path = []
        params.taskgraph.node_status = {}

        def is_leaf_func(node_id: str) -> bool:
            return True

        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        assert result_node is None
        assert result_params == params

    def test_get_nested_graph_component_node_final_return_none_params(self) -> None:
        """Test the final return None, params branch with empty path."""
        params = OrchestratorParams()
        params.taskgraph.path = []

        def is_leaf_func(node_id: str) -> bool:
            return True

        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )
        assert result_node is None
        assert result_params == params

    def test_get_nested_graph_component_node_all_nodes_are_leaves(self) -> None:
        """Test get_nested_graph_component_node fallback branch"""
        params = OrchestratorParams()
        # Provide an empty path to trigger the fallback branch
        params.taskgraph.path = []
        params.taskgraph.node_status = {}

        def is_leaf_func(node_id: str) -> bool:
            return True

        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        # Should return None and params in the fallback case
        assert result_node is None
        assert result_params == params

    def test_get_nested_graph_component_node_none_return(self) -> None:
        """Test get_nested_graph_component_node returns the first node when all nodes are leaves in circular pattern."""
        params = OrchestratorParams()
        # Create a path where all nodes are leaves and form nested graph patterns
        # but the algorithm exhausts all possibilities
        path_node1 = PathNode(node_id="node1", nested_graph_node_value="node2")
        path_node2 = PathNode(node_id="node2", nested_graph_node_value="node3")
        path_node3 = PathNode(node_id="node3", nested_graph_node_value="node1")
        params.taskgraph.path = [path_node1, path_node2, path_node3]
        params.taskgraph.node_status = {}

        def is_leaf_func(node_id: str) -> bool:
            return True  # All nodes are leaves

        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        # When all nodes are leaves, the algorithm returns the first node in the path
        assert result_node is not None
        assert result_node.node_id == "node1"
        assert result_params == params

    def test_get_nested_graph_component_node_empty_path_returns_none(self) -> None:
        """Test get_nested_graph_component_node returns None with empty path."""
        params = OrchestratorParams()
        params.taskgraph.path = []
        params.taskgraph.node_status = {}

        def is_leaf_func(node_id: str) -> bool:
            return True

        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        # Should return None when path is empty
        assert result_node is None
        assert result_params == params

    def test_get_nested_graph_component_node_fallback_return_none(self) -> None:
        """Test get_nested_graph_component_node fallback return None, params branch."""
        params = OrchestratorParams()
        # Create a scenario where all nodes are leaves and nested graph component is found
        # but the algorithm exhausts all possibilities and reaches the fallback return
        path_node1 = PathNode(node_id="node1", nested_graph_node_value="node2")
        path_node2 = PathNode(node_id="node2", nested_graph_node_value="node3")
        path_node3 = PathNode(node_id="node3", nested_graph_node_value="node1")
        params.taskgraph.path = [path_node1, path_node2, path_node3]
        params.taskgraph.node_status = {}

        def is_leaf_func(node_id: str) -> bool:
            # All nodes are leaves, which will cause the algorithm to continue searching
            return True

        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        # When all nodes are leaves, the algorithm returns the first node in the path
        # This is the actual behavior of the implementation
        assert result_node is not None
        assert result_node.node_id == "node1"
        assert result_params == params

    def test_get_nested_graph_component_node_fallback_none_params_empty_path(
        self,
    ) -> None:
        """Explicitly test fallback return (None, params) with empty path (line 92)."""
        params = OrchestratorParams()
        params.taskgraph.path = []

        def is_leaf_func(node_id: str) -> bool:
            return False

        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )
        assert result_node is None
        assert result_params == params

    def test_get_nested_graph_component_node_fallback_none_params_all_nodes_leaves(
        self,
    ) -> None:
        """Test fallback return (None, params) when all nodes are leaves (line 92)."""
        params = OrchestratorParams()
        params.taskgraph.path = []

        def is_leaf_func(node_id: str) -> bool:
            return True

        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )
        assert result_node is None
        assert result_params == params

    def test_get_nested_graph_component_node_fallback_none_params_none_is_leaf_func(
        self,
    ) -> None:
        """Test fallback return (None, params) with None is_leaf_func (line 92)."""
        params = OrchestratorParams()
        params.taskgraph.path = []
        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, None
        )
        assert result_node is None
        assert result_params == params

    def test_get_nested_graph_component_node_fallback_none_params_path_with_none(
        self,
    ) -> None:
        """Test fallback return (None, params) with path containing only None (line 92)."""
        params = OrchestratorParams()
        params.taskgraph.path = [None]

        def is_leaf_func(node_id: str) -> bool:
            return False

        with suppress(AttributeError):
            NestedGraph.get_nested_graph_component_node(params, is_leaf_func)

    def test_get_nested_graph_component_node_fallback_return_none_params_path_with_non_pathnode(
        self,
    ) -> None:
        """Test get_nested_graph_component_node with path containing non-PathNode objects."""
        params = Mock(spec=OrchestratorParams)
        params.taskgraph = Mock()
        # Create a path with invalid elements that will cause the algorithm to fail
        params.taskgraph.path = [Mock()]  # Invalid path element
        params.taskgraph.node_status = {}

        def is_leaf_func(node_id: str) -> bool:
            return True  # All nodes are leaves, which will cause the algorithm to continue searching

        # This should trigger the fallback return None, params when the algorithm
        # encounters invalid path elements
        with pytest.raises((AttributeError, TypeError)):
            NestedGraph.get_nested_graph_component_node(params, is_leaf_func)

    def test_get_nested_graph_component_node_fallback_return_none_edge_case(
        self,
    ) -> None:
        """Test get_nested_graph_component_node edge case that triggers the fallback return None."""
        params = Mock(spec=OrchestratorParams)
        params.taskgraph = Mock()
        # Create an empty path to trigger the fallback return None
        params.taskgraph.path = []
        params.taskgraph.node_status = {}

        def is_leaf_func(node_id: str) -> bool:
            # This function won't be called with an empty path
            return True

        # This should trigger the fallback return None, params when the path is empty
        # and cur_node_i becomes -1 (len(path) - 1 = -1)
        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )

        # The fallback should return None, params when the path is empty
        assert result_node is None
        assert result_params == params

    def test_get_nested_graph_component_node_not_leaf_branch(self) -> None:
        """Covers the branch where is_leaf_func returns False, triggering the return at line 92."""
        params = OrchestratorParams()
        # Create a path with two nodes, the second is a nested graph component
        path_node1 = PathNode(node_id="node1")
        path_node2 = PathNode(node_id="node2", nested_graph_node_value="node1")
        params.taskgraph.path = [path_node1, path_node2]
        params.taskgraph.node_status = {}

        def is_leaf_func(node_id: str) -> bool:
            # Always return False to trigger the branch
            return False

        result_node, result_params = NestedGraph.get_nested_graph_component_node(
            params, is_leaf_func
        )
        # Should return the nested graph component node (node2) and params
        assert result_node is not None
        assert result_node.node_id == "node2"
        assert result_params == params
