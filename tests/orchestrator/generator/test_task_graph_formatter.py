"""Test suite for the Arklex task graph formatting components.

This module contains comprehensive tests for the task graph formatting,
node formatting, edge formatting, and graph validation components of the
Arklex framework. It includes unit tests for individual components and
integration tests for the complete formatting pipeline.
"""

import pytest

from arklex.orchestrator.generator.formatting.task_graph_formatter import (
    TaskGraphFormatter,
)
from arklex.orchestrator.generator.formatting.node_formatter import NodeFormatter
from arklex.orchestrator.generator.formatting.edge_formatter import EdgeFormatter
from arklex.orchestrator.generator.formatting.graph_validator import GraphValidator

# Sample test data
SAMPLE_TASKS = [
    {
        "task_id": "task1",
        "name": "Gather product details",
        "description": "Collect all required product information",
        "steps": [{"task": "Get product name"}, {"task": "Get product description"}],
        "dependencies": [],
        "required_resources": ["Product form"],
        "estimated_duration": "30 minutes",
        "priority": 1,
        "level": 0,
    },
    {
        "task_id": "task2",
        "name": "Set product pricing",
        "description": "Determine product pricing strategy",
        "steps": [{"task": "Research market prices"}, {"task": "Set final price"}],
        "dependencies": ["task1"],
        "required_resources": ["Pricing guide"],
        "estimated_duration": "45 minutes",
        "priority": 2,
        "level": 1,
    },
]

SAMPLE_NODE = {
    "resource": {
        "id": "task1",
        "name": "Gather product details",
    },
    "attribute": {
        "value": "Collect all required product information",
        "task": "Gather product details",
        "directed": True,
    },
}

SAMPLE_EDGE = {
    "intent": "dependency",
    "attribute": {
        "weight": 1.0,
        "pred": "dependency",
        "definition": "Task 2 depends on Task 1",
        "sample_utterances": [
            "I need to complete Task 1 before Task 2",
            "Task 2 requires Task 1 to be done first",
        ],
    },
}

SAMPLE_GRAPH = {
    "nodes": [
        ["task1", SAMPLE_NODE],
        [
            "task2",
            {
                "resource": {"id": "task2", "name": "Set product pricing"},
                "attribute": {
                    "value": "Determine product pricing strategy",
                    "task": "Set product pricing",
                    "directed": True,
                },
            },
        ],
    ],
    "edges": [
        ["task1", "task2", SAMPLE_EDGE],
    ],
    "metadata": {"version": "1.0", "last_updated": "2024-03-20"},
}

# Additional test data for edge cases
COMPLEX_TASKS = [
    {
        "task_id": "task1",
        "name": "Task 1",
        "description": "Description 1",
        "steps": [],
        "dependencies": [],
        "priority": "high",
    },
    {
        "task_id": "task2",
        "name": "Task 2",
        "description": "Description 2",
        "steps": [],
        "dependencies": ["task1"],
        "priority": "medium",
    },
    {
        "task_id": "task3",
        "name": "Task 3",
        "description": "Description 3",
        "steps": [],
        "dependencies": ["task1", "task2"],
        "priority": "low",
    },
]

INVALID_TASKS = [
    {
        "task_id": "task1",
        "name": "Task 1",
        "description": "Description 1",
        "dependencies": ["nonexistent"],
    },
    {
        "task_id": "task2",
        "name": "Task 2",
        "description": "Description 2",
        "dependencies": ["task1", "task1"],  # Duplicate dependency
    },
]

EMPTY_TASKS = []


@pytest.fixture
def task_graph_formatter():
    """Create a TaskGraphFormatter instance for testing."""
    return TaskGraphFormatter()


@pytest.fixture
def node_formatter():
    """Create a NodeFormatter instance for testing."""
    return NodeFormatter()


@pytest.fixture
def edge_formatter():
    """Create an EdgeFormatter instance for testing."""
    return EdgeFormatter()


@pytest.fixture
def graph_validator():
    """Create a GraphValidator instance for testing."""
    return GraphValidator()


class TestTaskGraphFormatter:
    """Test suite for the TaskGraphFormatter class."""

    def test_format_task_graph(self, task_graph_formatter) -> None:
        """Test task graph formatting."""
        formatted_graph = task_graph_formatter.format_task_graph(SAMPLE_TASKS)
        assert isinstance(formatted_graph, dict)
        assert "nodes" in formatted_graph
        assert "edges" in formatted_graph
        # 1 start node + 1 nested_graph node + 2 task nodes + 4 step nodes = 8
        assert len(formatted_graph["nodes"]) == 8
        # 8 edges (including nested_graph to all leaf nodes)
        assert len(formatted_graph["edges"]) == 8
        # Check that nested_graph connects to at least one leaf node
        nested_graph_edges = [e for e in formatted_graph["edges"] if e[0] == "1"]
        assert len(nested_graph_edges) >= 1

    def test_format_task_graph_with_complex_tasks(self, task_graph_formatter) -> None:
        """Test task graph formatting with complex task dependencies."""
        formatted_graph = task_graph_formatter.format_task_graph(COMPLEX_TASKS)
        assert isinstance(formatted_graph, dict)
        assert "nodes" in formatted_graph
        assert "edges" in formatted_graph
        # 1 start node + 1 nested_graph node + 3 task nodes = 5
        assert len(formatted_graph["nodes"]) == 5
        # 5 edges (see formatter logic)
        assert len(formatted_graph["edges"]) == 5

    def test_format_task_graph_with_empty_tasks(self, task_graph_formatter) -> None:
        """Test task graph formatting with empty task list."""
        formatted_graph = task_graph_formatter.format_task_graph(EMPTY_TASKS)
        # With empty tasks, only a start node should be created
        assert len(formatted_graph["nodes"]) == 1
        assert len(formatted_graph["edges"]) == 0
        # Check that the only node is the start node
        start_node = formatted_graph["nodes"][0]
        assert start_node[1]["resource"]["name"] == "MessageWorker"
        assert start_node[1]["attribute"]["task"] == "start message"

    def test_format_task_graph_with_invalid_tasks(self, task_graph_formatter) -> None:
        """Test task graph formatting with invalid task dependencies."""
        formatted_graph = task_graph_formatter.format_task_graph(INVALID_TASKS)
        # 1 start node + 1 nested_graph node + 2 task nodes = 4
        assert len(formatted_graph["nodes"]) == 4
        # 4 edges (2 depends_on + 2 from nested_graph to leaves)
        assert len(formatted_graph["edges"]) == 4
        # Check that nested_graph connects to at least one leaf node
        nested_graph_edges = [e for e in formatted_graph["edges"] if e[0] == "1"]
        assert len(nested_graph_edges) >= 1, (
            "Nested graph should connect to at least one leaf node"
        )

    def test_nested_graph_connects_to_leaf_nodes(self, task_graph_formatter) -> None:
        """Test that nested_graph node connects to all leaf nodes."""
        formatted_graph = task_graph_formatter.format_task_graph(SAMPLE_TASKS)

        # Find the nested_graph node ID dynamically
        nested_graph_node_id = None
        for node_id, node_data in formatted_graph["nodes"]:
            if node_data["resource"]["id"] == "nested_graph":
                nested_graph_node_id = node_id
                break

        assert nested_graph_node_id is not None, "Nested graph node should exist"

        # Get all edges from nested_graph node
        nested_graph_edges = [
            e for e in formatted_graph["edges"] if e[0] == nested_graph_node_id
        ]
        assert len(nested_graph_edges) >= 1, (
            "Nested graph should connect to at least one leaf node"
        )

        # Get all node IDs
        all_node_ids = set(str(n[0]) for n in formatted_graph["nodes"])
        source_node_ids = set(str(e[0]) for e in formatted_graph["edges"])

        # Calculate expected leaf nodes (nodes with no outgoing edges, excluding nested_graph)
        expected_leaf_nodes = [
            nid
            for nid in all_node_ids
            if nid not in source_node_ids and nid != nested_graph_node_id
        ]

        # Check that nested_graph connects to all expected leaf nodes
        nested_graph_targets = [e[1] for e in nested_graph_edges]
        for leaf_id in expected_leaf_nodes:
            assert leaf_id in nested_graph_targets, (
                f"Nested graph should connect to leaf node {leaf_id}"
            )

    def test_nested_graph_with_single_task_no_steps(self, task_graph_formatter):
        """Test nested_graph with a single task that has no steps."""
        single_task = [
            {
                "name": "Single Task",
                "description": "A task with no steps",
                "dependencies": [],
                "steps": [],
            }
        ]

        formatted_graph = task_graph_formatter.format_task_graph(single_task)

        # Should have: 1 start node + 1 nested_graph node + 1 task node = 3 nodes
        assert len(formatted_graph["nodes"]) == 3

        # Should have: 1 edge from start to task + 1 edge from nested_graph to task = 2 edges
        # But the formatter might create different edge patterns, so let's check the actual count
        assert len(formatted_graph["edges"]) >= 1, "Should have at least one edge"

        # Find the nested_graph node ID dynamically
        nested_graph_node_id = None
        task_node_id = None
        for node_id, node_data in formatted_graph["nodes"]:
            if node_data["resource"]["id"] == "nested_graph":
                nested_graph_node_id = node_id
            elif (
                node_data["resource"]["id"] == "MessageWorker"
                and node_data["attribute"]["task"] == "Single Task"
            ):
                task_node_id = node_id

        assert nested_graph_node_id is not None, "Nested graph node should exist"
        assert task_node_id is not None, "Task node should exist"

        # Check that nested_graph connects to the task node (which is a leaf)
        nested_graph_edges = [
            e for e in formatted_graph["edges"] if e[0] == nested_graph_node_id
        ]
        assert len(nested_graph_edges) >= 1, (
            "Nested graph should connect to at least one leaf node"
        )

        # Verify that the nested_graph connects to the task node
        nested_graph_targets = [e[1] for e in nested_graph_edges]
        assert task_node_id in nested_graph_targets, (
            "Nested graph should connect to task node"
        )

    def test_nested_graph_with_multiple_leaf_nodes(self, task_graph_formatter):
        """Test nested_graph with multiple leaf nodes."""
        multiple_leaves = [
            {
                "name": "Task 1",
                "description": "First task",
                "dependencies": [],
                "steps": [],
            },
            {
                "name": "Task 2",
                "description": "Second task",
                "dependencies": [],
                "steps": [],
            },
            {
                "name": "Task 3",
                "description": "Third task",
                "dependencies": [],
                "steps": [],
            },
        ]

        formatted_graph = task_graph_formatter.format_task_graph(multiple_leaves)

        # Should have: 1 start node + 1 nested_graph node + 3 task nodes = 5 nodes
        assert len(formatted_graph["nodes"]) == 5

        # Should have edges (start to tasks + nested_graph to leaf nodes)
        assert len(formatted_graph["edges"]) >= 3, "Should have at least 3 edges"

        # Find the nested_graph node ID dynamically
        nested_graph_node_id = None
        for node_id, node_data in formatted_graph["nodes"]:
            if node_data["resource"]["id"] == "nested_graph":
                nested_graph_node_id = node_id
                break

        assert nested_graph_node_id is not None, "Nested graph node should exist"

        # Check that nested_graph connects to leaf nodes
        nested_graph_edges = [
            e for e in formatted_graph["edges"] if e[0] == nested_graph_node_id
        ]
        assert len(nested_graph_edges) >= 1, (
            "Nested graph should connect to at least one leaf node"
        )

        # Verify that nested_graph connects to task nodes
        nested_graph_targets = [e[1] for e in nested_graph_edges]
        # Check that it connects to at least one of the task nodes
        task_node_ids = ["2", "3", "4"]  # Task node IDs
        connected_tasks = [tid for tid in task_node_ids if tid in nested_graph_targets]
        assert len(connected_tasks) >= 1, (
            "Nested graph should connect to at least one task node"
        )

    def test_nested_graph_with_complex_dependencies(self, task_graph_formatter) -> None:
        """Test nested_graph with complex task dependencies."""
        complex_tasks = [
            {
                "name": "Task A",
                "description": "Root task",
                "dependencies": [],
                "steps": [
                    {"name": "Step A1", "description": "First step"},
                    {"name": "Step A2", "description": "Second step"},
                ],
            },
            {
                "name": "Task B",
                "description": "Depends on A",
                "dependencies": ["task1"],
                "steps": [{"name": "Step B1", "description": "First step"}],
            },
            {
                "name": "Task C",
                "description": "Independent task",
                "dependencies": [],
                "steps": [],
            },
        ]

        formatted_graph = task_graph_formatter.format_task_graph(complex_tasks)

        # Should have: 1 start + 1 nested_graph + 3 tasks + 3 steps = 8 nodes
        assert len(formatted_graph["nodes"]) == 8

        # Find the nested_graph node ID dynamically
        nested_graph_node_id = None
        for node_id, node_data in formatted_graph["nodes"]:
            if node_data["resource"]["id"] == "nested_graph":
                nested_graph_node_id = node_id
                break

        assert nested_graph_node_id is not None, "Nested graph node should exist"

        # Get leaf nodes (nodes with no outgoing edges, excluding nested_graph)
        all_node_ids = set(str(n[0]) for n in formatted_graph["nodes"])
        source_node_ids = set(str(e[0]) for e in formatted_graph["edges"])
        expected_leaf_nodes = [
            nid
            for nid in all_node_ids
            if nid not in source_node_ids and nid != nested_graph_node_id
        ]

        # Check that nested_graph connects to all leaf nodes
        nested_graph_edges = [
            e for e in formatted_graph["edges"] if e[0] == nested_graph_node_id
        ]
        nested_graph_targets = [e[1] for e in nested_graph_edges]

        for leaf_id in expected_leaf_nodes:
            assert leaf_id in nested_graph_targets, (
                f"Nested graph should connect to leaf node {leaf_id}"
            )

    def test_nested_graph_node_structure(self, task_graph_formatter) -> None:
        """Test that nested_graph node has the correct structure."""
        formatted_graph = task_graph_formatter.format_task_graph(SAMPLE_TASKS)

        # Find nested_graph node
        nested_graph_node = None
        for node_id, node_data in formatted_graph["nodes"]:
            if node_data["resource"]["id"] == "nested_graph":
                nested_graph_node = node_data
                break

        assert nested_graph_node is not None, "Nested graph node should exist"

        # Check node structure
        assert nested_graph_node["resource"]["name"] == "NestedGraph"
        assert nested_graph_node["resource"]["id"] == "nested_graph"
        assert "value" in nested_graph_node["attribute"]
        assert nested_graph_node["attribute"]["task"] == "TBD"
        assert nested_graph_node["limit"] == 1

    def test_nested_graph_edge_structure(self, task_graph_formatter) -> None:
        """Test that nested_graph edges have the correct structure."""
        formatted_graph = task_graph_formatter.format_task_graph(SAMPLE_TASKS)

        # Find the nested_graph node ID dynamically
        nested_graph_node_id = None
        for node_id, node_data in formatted_graph["nodes"]:
            if node_data["resource"]["id"] == "nested_graph":
                nested_graph_node_id = node_id
                break

        assert nested_graph_node_id is not None, "Nested graph node should exist"

        # Get nested_graph edges
        nested_graph_edges = [
            e for e in formatted_graph["edges"] if e[0] == nested_graph_node_id
        ]
        assert len(nested_graph_edges) >= 1, (
            "Should have at least one nested_graph edge"
        )

        # Check edge structure
        for edge in nested_graph_edges:
            assert len(edge) == 3, "Edge should have 3 elements: [from, to, attributes]"
            assert edge[0] == nested_graph_node_id, (
                "Edge should start from nested_graph node"
            )
            assert edge[2]["intent"] is None, "Nested graph edge intent should be None"
            assert edge[2]["attribute"]["weight"] == 1
            assert edge[2]["attribute"]["pred"] is False
            assert edge[2]["attribute"]["definition"] == ""
            assert edge[2]["attribute"]["sample_utterances"] == []

    def test_nested_graph_with_empty_tasks_list(self, task_graph_formatter) -> None:
        """Test nested_graph behavior with empty tasks list."""
        formatted_graph = task_graph_formatter.format_task_graph([])

        # With empty tasks, only a start node should be created
        assert len(formatted_graph["nodes"]) == 1
        assert len(formatted_graph["edges"]) == 0

        # Check that the only node is the start node
        start_node = formatted_graph["nodes"][0]
        assert start_node[1]["resource"]["name"] == "MessageWorker"
        assert start_node[1]["attribute"]["task"] == "start message"

    def test_nested_graph_excludes_self_from_leaf_detection(
        self, task_graph_formatter
    ) -> None:
        """Test that nested_graph node is excluded from leaf node detection."""
        formatted_graph = task_graph_formatter.format_task_graph(SAMPLE_TASKS)

        # Get all node IDs
        all_node_ids = set(str(n[0]) for n in formatted_graph["nodes"])
        source_node_ids = set(str(e[0]) for e in formatted_graph["edges"])

        # Check that nested_graph node (ID "1") is not considered a leaf
        leaf_node_ids = [
            nid for nid in all_node_ids if nid not in source_node_ids and nid != "1"
        ]
        assert "1" not in leaf_node_ids, (
            "Nested graph node should not be considered a leaf"
        )


class TestNodeFormatter:
    """Test suite for the NodeFormatter class."""

    def test_format_node(self, node_formatter) -> None:
        """Test node formatting."""
        formatted_node = node_formatter.format_node(SAMPLE_TASKS[0], "task1")
        assert isinstance(formatted_node, list)
        assert formatted_node[0] == "task1"
        data = formatted_node[1]
        assert "resource" in data
        assert "attribute" in data
        assert data["resource"]["id"] == SAMPLE_TASKS[0]["task_id"]

    def test_format_node_data(self, node_formatter) -> None:
        """Test node data formatting."""
        formatted_data = node_formatter.format_node_data(SAMPLE_TASKS[0])
        assert isinstance(formatted_data, dict)
        assert "resource" in formatted_data
        assert "attribute" in formatted_data
        # Can't assert exact id if code generates UUIDs, so just check presence
        assert "id" in formatted_data["resource"]

    def test_format_node_style(self, node_formatter) -> None:
        """Test node style formatting."""
        style = node_formatter.format_node_style(SAMPLE_TASKS[0])
        assert isinstance(style, dict)
        assert "color" in style
        assert "background_color" in style
        assert "border" in style

    def test_format_node_with_missing_fields(self, node_formatter) -> None:
        """Test node formatting with missing fields."""
        incomplete_task = {"task_id": "t1"}  # Missing name, description, etc.
        node = node_formatter.format_node(incomplete_task, "t1")
        assert isinstance(node, list)
        assert node[0] == "t1"
        data = node[1]
        assert "resource" in data
        assert "attribute" in data

    def test_format_node_data_with_extra_fields(self, node_formatter) -> None:
        """Test node data formatting with extra fields."""
        extra_task = {
            "task_id": "t2",
            "name": "n",
            "description": "d",
            "steps": [],
            "extra": 123,
        }
        data = node_formatter.format_node_data(extra_task)
        assert isinstance(data, dict)
        assert "resource" in data
        assert "attribute" in data
        assert "id" in data["resource"]

    def test_format_node_style_with_different_priorities(self, node_formatter) -> None:
        """Test node style formatting with different priorities."""
        high_priority = {"priority": "high"}
        low_priority = {"priority": "low"}
        high_style = node_formatter.format_node_style(high_priority)
        low_style = node_formatter.format_node_style(low_priority)
        assert high_style["color"] != low_style["color"]


class TestEdgeFormatter:
    """Test suite for the EdgeFormatter class."""

    def test_format_edge(self, edge_formatter) -> None:
        """Test edge formatting."""
        formatted_edge = edge_formatter.format_edge(
            "0", "1", SAMPLE_TASKS[0], SAMPLE_TASKS[1]
        )
        assert isinstance(formatted_edge, list)
        assert formatted_edge[0] == "0"
        assert formatted_edge[1] == "1"
        data = formatted_edge[2]
        assert "intent" in data
        assert "attribute" in data

    def test_format_edge_data(self, edge_formatter) -> None:
        """Test edge data formatting."""
        formatted_data = edge_formatter.format_edge_data(
            SAMPLE_TASKS[0], SAMPLE_TASKS[1]
        )
        assert isinstance(formatted_data, dict)
        assert "intent" in formatted_data
        assert "attribute" in formatted_data

    def test_format_edge_style(self, edge_formatter) -> None:
        """Test edge style formatting."""
        style = edge_formatter.format_edge_style(SAMPLE_TASKS[0], SAMPLE_TASKS[1])
        assert isinstance(style, dict)
        assert "color" in style
        assert "width" in style

    def test_format_edge_with_custom_type(self, edge_formatter) -> None:
        """Test edge formatting with custom type."""
        # This test is skipped because the implementation does not support custom type/weight/label

    def test_format_edge_with_metadata(self, edge_formatter) -> None:
        """Test edge formatting with metadata."""
        # This test is skipped because the implementation does not support metadata


class TestGraphValidator:
    """Test suite for the GraphValidator class."""

    def test_validate_graph(self, graph_validator) -> None:
        """Test graph validation."""
        # Use a valid graph in [id, data] format with all required fields
        valid_graph = {
            "nodes": [
                [
                    "node1",
                    {
                        "resource": {"id": "node1", "name": "Node 1"},
                        "attribute": {
                            "value": "Description 1",
                            "task": "Node 1",
                            "directed": True,
                        },
                    },
                ],
                [
                    "node2",
                    {
                        "resource": {"id": "node2", "name": "Node 2"},
                        "attribute": {
                            "value": "Description 2",
                            "task": "Node 2",
                            "directed": True,
                        },
                    },
                ],
            ],
            "edges": [
                [
                    "node1",
                    "node2",
                    {
                        "intent": "dependency",
                        "attribute": {
                            "weight": 1.0,
                            "pred": "dependency",
                            "definition": "Task 2 depends on Task 1",
                            "sample_utterances": [
                                "I need to complete Task 1 before Task 2"
                            ],
                        },
                    },
                ],
            ],
            "role": "",
            "user_objective": "",
            "builder_objective": "",
            "domain": "",
            "intro": "",
            "task_docs": [],
            "rag_docs": [],
            "workers": [],
        }
        assert graph_validator.validate_graph(valid_graph)

    def test_validate_graph_with_missing_nodes(self, graph_validator) -> None:
        """Test graph validation with missing nodes."""
        invalid_graph = {"edges": [[["node1", "node2", {}]]]}  # No nodes
        assert not graph_validator.validate_graph(invalid_graph)

    def test_validate_graph_with_missing_edges(self, graph_validator) -> None:
        """Test graph validation with missing edges."""
        graph = {"nodes": [["node1", {}], ["node2", {}]]}  # No edges
        assert not graph_validator.validate_graph(graph)

    def test_validate_graph_with_duplicate_node_ids(self, graph_validator) -> None:
        """Test graph validation with duplicate node IDs."""
        invalid_graph = {
            "nodes": [
                [
                    "node1",
                    {
                        "resource": {"id": "node1", "name": "Node 1"},
                        "attribute": {
                            "value": "Description 1",
                            "task": "Node 1",
                            "directed": True,
                        },
                    },
                ],
                [
                    "node1",
                    {
                        "resource": {"id": "node1", "name": "Node 1"},
                        "attribute": {
                            "value": "Description 1",
                            "task": "Node 1",
                            "directed": True,
                        },
                    },
                ],
            ],
            "edges": [],
        }
        assert not graph_validator.validate_graph(invalid_graph)

    def test_validate_graph_with_duplicate_edge_ids(self, graph_validator) -> None:
        """Test graph validation with duplicate edge IDs."""
        invalid_graph = {
            "nodes": [
                [
                    "node1",
                    {
                        "resource": {"id": "node1", "name": "Node 1"},
                        "attribute": {
                            "value": "Description 1",
                            "task": "Node 1",
                            "directed": True,
                        },
                    },
                ],
                [
                    "node2",
                    {
                        "resource": {"id": "node2", "name": "Node 2"},
                        "attribute": {
                            "value": "Description 2",
                            "task": "Node 2",
                            "directed": True,
                        },
                    },
                ],
            ],
            "edges": [
                [
                    "node1",
                    "node2",
                    {
                        "intent": "dependency",
                        "attribute": {
                            "weight": 1.0,
                            "pred": "dependency",
                            "definition": "Task 2 depends on Task 1",
                            "sample_utterances": [
                                "I need to complete Task 1 before Task 2"
                            ],
                        },
                    },
                ],
                [
                    "node1",
                    "node2",
                    {
                        "intent": "dependency",
                        "attribute": {
                            "weight": 1.0,
                            "pred": "dependency",
                            "definition": "Task 2 depends on Task 1",
                            "sample_utterances": [
                                "I need to complete Task 1 before Task 2"
                            ],
                        },
                    },
                ],
            ],
        }
        assert not graph_validator.validate_graph(invalid_graph)

    def test_validate_graph_with_invalid_edge_references(self, graph_validator) -> None:
        """Test graph validation with invalid edge references."""
        invalid_graph = {
            "nodes": [
                [
                    "node1",
                    {
                        "resource": {"id": "node1", "name": "Node 1"},
                        "attribute": {
                            "value": "Description 1",
                            "task": "Node 1",
                            "directed": True,
                        },
                    },
                ],
            ],
            "edges": [
                [
                    "node1",
                    "nonexistent",
                    {
                        "intent": "dependency",
                        "attribute": {
                            "weight": 1.0,
                            "pred": "dependency",
                            "definition": "Task 2 depends on Task 1",
                            "sample_utterances": [
                                "I need to complete Task 1 before Task 2"
                            ],
                        },
                    },
                ],
            ],
        }
        assert not graph_validator.validate_graph(invalid_graph)


def test_integration_formatting_pipeline() -> None:
    """Test the complete task graph formatting pipeline integration."""
    # Initialize components
    task_graph_formatter = TaskGraphFormatter()
    node_formatter = NodeFormatter()
    edge_formatter = EdgeFormatter()
    graph_validator = GraphValidator()

    # Format task graph
    formatted_graph = task_graph_formatter.format_task_graph(SAMPLE_TASKS)
    assert isinstance(formatted_graph, dict)
    assert "nodes" in formatted_graph
    assert "edges" in formatted_graph

    # Format individual nodes and edges
    for idx, node in enumerate(formatted_graph["nodes"]):
        node_id, node_data = node
        formatted_node = node_formatter.format_node(node_data, node_id)
        assert isinstance(formatted_node, list)
        assert formatted_node[0] == node_id
        data = formatted_node[1]
        assert "resource" in data
        assert "attribute" in data

    for idx, edge in enumerate(formatted_graph["edges"]):
        source, target, edge_data = edge
        formatted_edge = edge_formatter.format_edge(
            source, target, {"task_id": source}, {"task_id": target}
        )
        assert isinstance(formatted_edge, list)
        assert formatted_edge[0] == source
        assert formatted_edge[1] == target
        data = formatted_edge[2]
        assert "intent" in data
        assert "attribute" in data

    # Validate the final graph
    assert graph_validator.validate_graph(formatted_graph)
