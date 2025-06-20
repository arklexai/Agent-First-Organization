"""Test suite for nested graph linking functionality.

This module tests the logic for properly linking main graphs to nested graphs,
ensuring that the nested graph is not orphaned and has proper connectivity.
"""

import pytest
from arklex.orchestrator.generator.formatting.task_graph_formatter import (
    TaskGraphFormatter,
)


class TestNestedGraphLinking:
    """Test suite for nested graph linking functionality."""

    def test_link_main_graph_to_nested_graph(self) -> None:
        """Test linking main graph to nested graph."""
        formatter = TaskGraphFormatter()

        # Create a main graph with a nested graph node
        main_graph = {
            "nodes": [
                [
                    "0",
                    {
                        "resource": {"id": "MessageWorker", "name": "MessageWorker"},
                        "attribute": {
                            "value": "Hello",
                            "task": "start",
                            "directed": False,
                        },
                    },
                ],
                [
                    "1",
                    {
                        "resource": {"id": "nested_graph", "name": "NestedGraph"},
                        "attribute": {
                            "value": "0",
                            "task": "Nested Graph",
                            "directed": False,
                        },
                        "limit": 1,
                    },
                ],
                [
                    "2",
                    {
                        "resource": {"id": "MessageWorker", "name": "MessageWorker"},
                        "attribute": {
                            "value": "Task 1",
                            "task": "Task 1",
                            "directed": False,
                        },
                    },
                ],
            ],
            "edges": [
                [
                    "0",
                    "1",
                    {
                        "intent": "start",
                        "attribute": {
                            "weight": 1,
                            "pred": True,
                            "definition": "",
                            "sample_utterances": [],
                        },
                    },
                ],
                [
                    "1",
                    "0",
                    {
                        "intent": None,
                        "attribute": {
                            "weight": 1,
                            "pred": False,
                            "definition": "",
                            "sample_utterances": [],
                        },
                    },
                ],
            ],
        }

        # Create a nested graph
        nested_graph = {
            "nodes": [
                [
                    "0",
                    {
                        "resource": {"id": "MessageWorker", "name": "MessageWorker"},
                        "attribute": {
                            "value": "Nested Start",
                            "task": "Nested Start",
                            "directed": False,
                        },
                    },
                ],
                [
                    "1",
                    {
                        "resource": {"id": "MessageWorker", "name": "MessageWorker"},
                        "attribute": {
                            "value": "Nested Task 1",
                            "task": "Nested Task 1",
                            "directed": False,
                        },
                    },
                ],
                [
                    "2",
                    {
                        "resource": {"id": "MessageWorker", "name": "MessageWorker"},
                        "attribute": {
                            "value": "Nested Task 2",
                            "task": "Nested Task 2",
                            "directed": False,
                        },
                    },
                ],
            ],
            "edges": [
                [
                    "0",
                    "1",
                    {
                        "intent": "depends_on",
                        "attribute": {
                            "weight": 1,
                            "pred": True,
                            "definition": "",
                            "sample_utterances": [],
                        },
                    },
                ],
                [
                    "1",
                    "2",
                    {
                        "intent": "depends_on",
                        "attribute": {
                            "weight": 1,
                            "pred": True,
                            "definition": "",
                            "sample_utterances": [],
                        },
                    },
                ],
            ],
        }

        # Link the graphs
        combined_graph = formatter.link_main_graph_to_nested_graph(
            main_graph, nested_graph
        )

        # Verify the edge to node 0 was removed
        edge_to_node_0 = [
            edge
            for edge in combined_graph["edges"]
            if edge[0] == "1" and edge[1] == "0"
        ]
        assert len(edge_to_node_0) == 0, (
            "Edge from nested graph to node 0 should be removed"
        )

        # Verify edges were added to subgraph start nodes
        nested_graph_edges = [
            edge for edge in combined_graph["edges"] if edge[0] == "1"
        ]
        assert len(nested_graph_edges) > 0, "Nested graph should have outgoing edges"

        # Verify the nested graph node value was updated
        nested_graph_node = None
        for node_id, node_data in combined_graph["nodes"]:
            if node_id == "1":
                nested_graph_node = node_data
                break

        assert nested_graph_node is not None, "Nested graph node should exist"
        assert nested_graph_node["attribute"]["value"] == "2", (
            "Nested graph should point to first task node"
        )
        assert nested_graph_node["attribute"]["task"] == "TBD", (
            "Nested graph task should be TBD"
        )
        assert nested_graph_node["attribute"]["directed"] is True, (
            "Nested graph directed should be True"
        )

    def test_ensure_nested_graph_connectivity(self) -> None:
        """Test ensuring nested graph connectivity."""
        formatter = TaskGraphFormatter()

        # Create a graph with orphaned nested graph
        graph = {
            "nodes": [
                [
                    "0",
                    {
                        "resource": {"id": "MessageWorker", "name": "MessageWorker"},
                        "attribute": {
                            "value": "Hello",
                            "task": "start",
                            "directed": False,
                        },
                    },
                ],
                [
                    "1",
                    {
                        "resource": {"id": "nested_graph", "name": "NestedGraph"},
                        "attribute": {
                            "value": "0",
                            "task": "Nested Graph",
                            "directed": False,
                        },
                        "limit": 1,
                    },
                ],
                [
                    "2",
                    {
                        "resource": {"id": "MessageWorker", "name": "MessageWorker"},
                        "attribute": {
                            "value": "Task 1",
                            "task": "Task 1",
                            "directed": False,
                        },
                    },
                ],
                [
                    "3",
                    {
                        "resource": {"id": "MessageWorker", "name": "MessageWorker"},
                        "attribute": {
                            "value": "Task 2",
                            "task": "Task 2",
                            "directed": False,
                        },
                    },
                ],
            ],
            "edges": [
                [
                    "0",
                    "2",
                    {
                        "intent": "start",
                        "attribute": {
                            "weight": 1,
                            "pred": True,
                            "definition": "",
                            "sample_utterances": [],
                        },
                    },
                ],
                [
                    "2",
                    "3",
                    {
                        "intent": "depends_on",
                        "attribute": {
                            "weight": 1,
                            "pred": True,
                            "definition": "",
                            "sample_utterances": [],
                        },
                    },
                ],
            ],
        }

        # Fix the connectivity
        fixed_graph = formatter.ensure_nested_graph_connectivity(graph)

        # Verify nested graph has incoming edges
        incoming_to_nested = [edge for edge in fixed_graph["edges"] if edge[1] == "1"]
        assert len(incoming_to_nested) > 0, "Nested graph should have incoming edges"

        # Verify nested graph has outgoing edges to leaf nodes
        outgoing_from_nested = [edge for edge in fixed_graph["edges"] if edge[0] == "1"]
        assert len(outgoing_from_nested) > 0, "Nested graph should have outgoing edges"

        # Verify no edges to node 0
        edges_to_node_0 = [edge for edge in fixed_graph["edges"] if edge[1] == "0"]
        assert len(edges_to_node_0) == 0, "No edges should point to node 0"

    def test_nested_graph_with_no_leaf_nodes(self) -> None:
        """Test nested graph connectivity when there are no clear leaf nodes."""
        formatter = TaskGraphFormatter()

        # Create a graph with a cycle (no leaf nodes)
        graph = {
            "nodes": [
                [
                    "0",
                    {
                        "resource": {"id": "MessageWorker", "name": "MessageWorker"},
                        "attribute": {
                            "value": "Hello",
                            "task": "start",
                            "directed": False,
                        },
                    },
                ],
                [
                    "1",
                    {
                        "resource": {"id": "nested_graph", "name": "NestedGraph"},
                        "attribute": {
                            "value": "0",
                            "task": "Nested Graph",
                            "directed": False,
                        },
                        "limit": 1,
                    },
                ],
                [
                    "2",
                    {
                        "resource": {"id": "MessageWorker", "name": "MessageWorker"},
                        "attribute": {
                            "value": "Task 1",
                            "task": "Task 1",
                            "directed": False,
                        },
                    },
                ],
                [
                    "3",
                    {
                        "resource": {"id": "MessageWorker", "name": "MessageWorker"},
                        "attribute": {
                            "value": "Task 2",
                            "task": "Task 2",
                            "directed": False,
                        },
                    },
                ],
            ],
            "edges": [
                [
                    "2",
                    "3",
                    {
                        "intent": "depends_on",
                        "attribute": {
                            "weight": 1,
                            "pred": True,
                            "definition": "",
                            "sample_utterances": [],
                        },
                    },
                ],
                [
                    "3",
                    "2",
                    {
                        "intent": "depends_on",
                        "attribute": {
                            "weight": 1,
                            "pred": True,
                            "definition": "",
                            "sample_utterances": [],
                        },
                    },
                ],
            ],
        }

        # Fix the connectivity
        fixed_graph = formatter.ensure_nested_graph_connectivity(graph)

        # Verify nested graph still has outgoing edges (to task nodes as fallback)
        outgoing_from_nested = [edge for edge in fixed_graph["edges"] if edge[0] == "1"]
        assert len(outgoing_from_nested) > 0, (
            "Nested graph should have outgoing edges even with cycles"
        )

    def test_nested_graph_without_nested_graph_node(self) -> None:
        """Test behavior when no nested graph node exists."""
        formatter = TaskGraphFormatter()

        # Create a graph without a nested graph node
        graph = {
            "nodes": [
                [
                    "0",
                    {
                        "resource": {"id": "MessageWorker", "name": "MessageWorker"},
                        "attribute": {
                            "value": "Hello",
                            "task": "start",
                            "directed": False,
                        },
                    },
                ],
                [
                    "1",
                    {
                        "resource": {"id": "MessageWorker", "name": "MessageWorker"},
                        "attribute": {
                            "value": "Task 1",
                            "task": "Task 1",
                            "directed": False,
                        },
                    },
                ],
            ],
            "edges": [
                [
                    "0",
                    "1",
                    {
                        "intent": "start",
                        "attribute": {
                            "weight": 1,
                            "pred": True,
                            "definition": "",
                            "sample_utterances": [],
                        },
                    },
                ]
            ],
        }

        # Fix the connectivity (should return unchanged)
        fixed_graph = formatter.ensure_nested_graph_connectivity(graph)

        # Verify the graph is unchanged
        assert len(fixed_graph["nodes"]) == len(graph["nodes"])
        assert len(fixed_graph["edges"]) == len(graph["edges"])

    def test_nested_graph_edge_structure(self) -> None:
        """Test that nested graph edges have the correct structure."""
        formatter = TaskGraphFormatter()

        # Create a simple graph
        graph = {
            "nodes": [
                [
                    "0",
                    {
                        "resource": {"id": "MessageWorker", "name": "MessageWorker"},
                        "attribute": {
                            "value": "Hello",
                            "task": "start",
                            "directed": False,
                        },
                    },
                ],
                [
                    "1",
                    {
                        "resource": {"id": "nested_graph", "name": "NestedGraph"},
                        "attribute": {
                            "value": "0",
                            "task": "Nested Graph",
                            "directed": False,
                        },
                        "limit": 1,
                    },
                ],
                [
                    "2",
                    {
                        "resource": {"id": "MessageWorker", "name": "MessageWorker"},
                        "attribute": {
                            "value": "Task 1",
                            "task": "Task 1",
                            "directed": False,
                        },
                    },
                ],
            ],
            "edges": [],
        }

        # Fix the connectivity
        fixed_graph = formatter.ensure_nested_graph_connectivity(graph)

        # Verify edge structure
        nested_graph_edges = [edge for edge in fixed_graph["edges"] if edge[0] == "1"]
        assert len(nested_graph_edges) > 0, "Should have nested graph edges"

        for edge in nested_graph_edges:
            assert len(edge) == 3, "Edge should have 3 elements: [from, to, attributes]"
            assert edge[0] == "1", "Edge should start from nested graph node"
            assert edge[2]["intent"] is None, "Nested graph edge intent should be None"
            assert edge[2]["attribute"]["weight"] == 1
            assert edge[2]["attribute"]["pred"] is False
            assert edge[2]["attribute"]["definition"] != ""
            assert edge[2]["attribute"]["sample_utterances"] == []
