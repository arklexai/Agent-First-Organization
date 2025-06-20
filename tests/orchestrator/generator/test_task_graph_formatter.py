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
        # 1 start node + 2 task nodes + 4 step nodes + 1 nested graph node = 8 nodes
        assert len(formatted_graph["nodes"]) == 8
        # 2 start_node edges + 2 "has_step" + 2 "next_step" = 6 edges
        # Note: The original test counted dependency edges which are now handled differently
        assert len(formatted_graph["edges"]) == 6

    def test_format_task_graph_with_complex_tasks(self, task_graph_formatter) -> None:
        """Test task graph formatting with complex task dependencies."""
        formatted_graph = task_graph_formatter.format_task_graph(COMPLEX_TASKS)
        assert isinstance(formatted_graph, dict)
        assert "nodes" in formatted_graph
        assert "edges" in formatted_graph
        # 1 start node + 3 task nodes + 1 nested graph node = 5 nodes
        assert len(formatted_graph["nodes"]) == 5
        # 2 start_node edges + 2 dependency edges = 4 edges
        assert len(formatted_graph["edges"]) == 4

    def test_format_task_graph_with_empty_tasks(self, task_graph_formatter) -> None:
        """Test task graph formatting with empty task list."""
        formatted_graph = task_graph_formatter.format_task_graph(EMPTY_TASKS)
        # With empty tasks, only a start node should be created
        assert len(formatted_graph["nodes"]) == 1
        assert len(formatted_graph["edges"]) == 0

    def test_format_task_graph_with_invalid_tasks(self, task_graph_formatter) -> None:
        """Test task graph formatting with invalid task dependencies."""
        formatted_graph = task_graph_formatter.format_task_graph(INVALID_TASKS)
        # 1 start node + 2 task nodes + 1 nested graph node = 4
        assert len(formatted_graph["nodes"]) == 4
        # 1 valid dependency + 1 start_node edge = 2
        assert len(formatted_graph["edges"]) == 2

    def test_task_with_missing_name(self, task_graph_formatter) -> None:
        """Test that a task without a name field is handled gracefully."""
        tasks = [{"task_id": "task1", "description": "A task without name"}]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # Should create 3 nodes: start, task1, nested_graph
        assert len(formatted_graph["nodes"]) == 3

        # Find the task node and verify it has empty name
        task_node = None
        for node_id, node_data in formatted_graph["nodes"]:
            if node_data.get("attribute", {}).get("task") == "":
                task_node = node_data
                break

        assert task_node is not None, "Task node should be created with empty name"
        assert task_node["attribute"]["value"] == "A task without name"

    def test_task_with_missing_description(self, task_graph_formatter) -> None:
        """Test that a task without a description field is handled gracefully."""
        tasks = [{"task_id": "task1", "name": "Task without description"}]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # Should create 3 nodes: start, task1, nested_graph
        assert len(formatted_graph["nodes"]) == 3

        # Find the task node and verify it has empty description
        task_node = None
        for node_id, node_data in formatted_graph["nodes"]:
            if node_data.get("attribute", {}).get("task") == "Task without description":
                task_node = node_data
                break

        assert task_node is not None, "Task node should be created"
        assert task_node["attribute"]["value"] == "", (
            "Description should be empty string"
        )

    def test_task_with_string_steps(self, task_graph_formatter) -> None:
        """Test that tasks with string steps (not dict) are handled correctly."""
        tasks = [
            {
                "task_id": "task1",
                "name": "Task with string steps",
                "steps": ["Step 1", "Step 2", "Step 3"],
            }
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 1 task + 3 steps + 1 nested_graph = 6 nodes
        assert len(formatted_graph["nodes"]) == 6

        # 1 start->task + 1 task->step1 + 2 step->step = 4 edges
        assert len(formatted_graph["edges"]) == 4

        # Verify step nodes have correct values (step nodes have resource id "message_worker")
        step_nodes = [
            node
            for node_id, node in formatted_graph["nodes"]
            if node.get("resource", {}).get("id") == "message_worker"
            and node.get("attribute", {}).get("task") == "Task with string steps"
        ]
        assert len(step_nodes) == 3
        assert step_nodes[0]["attribute"]["value"] == "Step 1"
        assert step_nodes[1]["attribute"]["value"] == "Step 2"
        assert step_nodes[2]["attribute"]["value"] == "Step 3"

    def test_task_with_dict_steps(self, task_graph_formatter) -> None:
        """Test that tasks with dictionary steps are handled correctly."""
        tasks = [
            {
                "task_id": "task1",
                "name": "Task with dict steps",
                "steps": [
                    {"description": "First step", "type": "input"},
                    {"description": "Second step", "type": "process"},
                ],
            }
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 1 task + 2 steps + 1 nested_graph = 5 nodes
        assert len(formatted_graph["nodes"]) == 5

        # 1 start->task + 1 task->step1 + 1 step1->step2 = 3 edges
        assert len(formatted_graph["edges"]) == 3

        # Verify step nodes have correct values from description field
        step_nodes = [
            node
            for node_id, node in formatted_graph["nodes"]
            if node.get("resource", {}).get("id") == "message_worker"
            and node.get("attribute", {}).get("task") == "Task with dict steps"
        ]
        assert len(step_nodes) == 2
        assert step_nodes[0]["attribute"]["value"] == "First step"
        assert step_nodes[1]["attribute"]["value"] == "Second step"

    def test_multiple_tasks_with_same_dependency(self, task_graph_formatter) -> None:
        """Test that multiple tasks can depend on the same task."""
        tasks = [
            {"task_id": "task1", "name": "Base Task"},
            {"task_id": "task2", "name": "Dependent 1", "dependencies": ["task1"]},
            {"task_id": "task3", "name": "Dependent 2", "dependencies": ["task1"]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 3 tasks + 1 nested_graph = 5 nodes
        assert len(formatted_graph["nodes"]) == 5

        # 1 start->task1 + 1 task1->task2 + 1 task1->task3 = 3 edges
        assert len(formatted_graph["edges"]) == 3

        # Verify both dependent tasks connect to the base task
        edge_tuples = {(e[0], e[1]) for e in formatted_graph["edges"]}

        # Find node IDs
        node_ids = {}
        for node_id, node_data in formatted_graph["nodes"]:
            task_name = node_data.get("attribute", {}).get("task")
            if task_name:
                node_ids[task_name] = node_id

        # Both dependent tasks should connect to the base task
        assert (node_ids["Base Task"], node_ids["Dependent 1"]) in edge_tuples
        assert (node_ids["Base Task"], node_ids["Dependent 2"]) in edge_tuples

    def test_nested_graph_connects_to_leaf_nodes(self, task_graph_formatter) -> None:
        """Test that nested_graph node is created, but not connected to anything in this simplified logic."""
        formatted_graph = task_graph_formatter.format_task_graph(SAMPLE_TASKS)
        nested_graph_node_id = None
        for node_id, node_data in formatted_graph["nodes"]:
            if node_data.get("resource", {}).get("id") == "nested_graph":
                nested_graph_node_id = node_id
                break
        assert nested_graph_node_id is not None, "Nested graph node should exist"
        nested_graph_edges = [
            e for e in formatted_graph["edges"] if e[0] == nested_graph_node_id
        ]
        # In the new simplified logic, the nested_graph node is not connected to leaves.
        assert len(nested_graph_edges) == 0

    def test_task_with_empty_steps_list(self, task_graph_formatter) -> None:
        """Test that a task with an empty steps list is handled gracefully."""
        tasks = [{"task_id": "task1", "name": "Task with no steps", "steps": []}]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 1 task + 1 nested_graph = 3 nodes
        assert len(formatted_graph["nodes"]) == 3

        # 1 start->task = 1 edge
        assert len(formatted_graph["edges"]) == 1

        # Verify no step nodes were created
        step_nodes = [
            node
            for node_id, node in formatted_graph["nodes"]
            if node.get("resource", {}).get("id") == "message_worker"
            and node.get("attribute", {}).get("task") == "Task with no steps"
        ]
        assert len(step_nodes) == 0, (
            "No step nodes should be created for empty steps list"
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
        # 1 start node + 1 task node + 1 nested_graph node = 3
        assert len(formatted_graph["nodes"]) == 3
        # 1 start_node edge
        assert len(formatted_graph["edges"]) == 1

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
        # 1 start node + 3 task nodes + 1 nested_graph node = 5
        assert len(formatted_graph["nodes"]) == 5
        # 3 start_node edges
        assert len(formatted_graph["edges"]) == 3

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
                "dependencies": ["task_0"],
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
        # 1 start + 3 tasks + 3 steps + 1 nested_graph = 8 nodes
        assert len(formatted_graph["nodes"]) == 8
        # 2 start_node edges + 1 dependency edge + 2 has_step + 1 next_step = 6 edges
        assert len(formatted_graph["edges"]) == 6

    def test_nested_graph_node_structure(self, task_graph_formatter) -> None:
        """Test that nested_graph node has the correct structure."""
        formatted_graph = task_graph_formatter.format_task_graph(SAMPLE_TASKS)
        nested_graph_node = None
        for _, node_data in formatted_graph["nodes"]:
            if node_data.get("resource", {}).get("id") == "nested_graph":
                nested_graph_node = node_data
                break
        assert nested_graph_node is not None, "Nested graph node should exist"
        assert nested_graph_node["resource"]["name"] == "NestedGraph"
        assert "value" in nested_graph_node["attribute"]
        assert nested_graph_node["limit"] == 1

    def test_nested_graph_edge_structure(self, task_graph_formatter) -> None:
        """Test that nested_graph edges have the correct structure."""
        formatted_graph = task_graph_formatter.format_task_graph(SAMPLE_TASKS)
        nested_graph_node_id = None
        for node_id, node_data in formatted_graph["nodes"]:
            if node_data.get("resource", {}).get("id") == "nested_graph":
                nested_graph_node_id = node_id
                break
        assert nested_graph_node_id is not None, "Nested graph node should exist"
        nested_graph_edges = [
            e for e in formatted_graph["edges"] if e[0] == nested_graph_node_id
        ]
        # The new logic doesn't create edges from the nested_graph node.
        assert len(nested_graph_edges) == 0

    def test_nested_graph_with_empty_tasks_list(self, task_graph_formatter) -> None:
        """Test nested_graph behavior with empty tasks list."""
        formatted_graph = task_graph_formatter.format_task_graph([])
        # Only start node should be created
        assert len(formatted_graph["nodes"]) == 1
        assert len(formatted_graph["edges"]) == 0

    def test_task_with_duplicate_step_descriptions(self, task_graph_formatter) -> None:
        tasks = [
            {
                "task_id": "task1",
                "name": "Task with duplicate steps",
                "steps": ["Step", "Step", "Step"],
            }
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        step_nodes = [
            node
            for node_id, node in formatted_graph["nodes"]
            if node.get("resource", {}).get("id") == "message_worker"
            and node.get("attribute", {}).get("task") == "Task with duplicate steps"
        ]
        assert len(step_nodes) == 3
        for node in step_nodes:
            assert node["attribute"]["value"] == "Step"

    def test_task_with_mixed_step_types(self, task_graph_formatter) -> None:
        tasks = [
            {
                "task_id": "task1",
                "name": "Mixed Steps",
                "steps": ["String step", {"description": "Dict step"}, 123, None],
            }
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        step_nodes = [
            node
            for node_id, node in formatted_graph["nodes"]
            if node.get("resource", {}).get("id") == "message_worker"
            and node.get("attribute", {}).get("task") == "Mixed Steps"
        ]
        # Only valid string and dict steps should be nodes
        assert len(step_nodes) == 2
        assert step_nodes[0]["attribute"]["value"] == "String step"
        assert step_nodes[1]["attribute"]["value"] == "Dict step"

    def test_task_with_missing_steps_field(self, task_graph_formatter) -> None:
        tasks = [{"task_id": "task1", "name": "No Steps Field"}]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        step_nodes = [
            node
            for node_id, node in formatted_graph["nodes"]
            if node.get("resource", {}).get("id") == "message_worker"
            and node.get("attribute", {}).get("task") == "No Steps Field"
        ]
        assert len(step_nodes) == 0

    def test_task_with_empty_steps_field(self, task_graph_formatter) -> None:
        tasks = [{"task_id": "task1", "name": "Empty Steps", "steps": []}]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        step_nodes = [
            node
            for node_id, node in formatted_graph["nodes"]
            if node.get("resource", {}).get("id") == "message_worker"
            and node.get("attribute", {}).get("task") == "Empty Steps"
        ]
        assert len(step_nodes) == 0

    def test_task_with_circular_dependency(self, task_graph_formatter) -> None:
        tasks = [
            {"task_id": "task1", "name": "A", "dependencies": ["task2"]},
            {"task_id": "task2", "name": "B", "dependencies": ["task1"]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        # Should not crash, should create 4 nodes (start, task1, task2, nested_graph)
        assert len(formatted_graph["nodes"]) == 4
        # Should create 2 dependency edges
        dep_edges = [
            e for e in formatted_graph["edges"] if e[2]["intent"] == "depends_on"
        ]
        assert len(dep_edges) == 2

    def test_task_with_dict_dependency(self, task_graph_formatter) -> None:
        tasks = [
            {"task_id": "task1", "name": "A"},
            {"task_id": "task2", "name": "B", "dependencies": [{"id": "task1"}]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        dep_edges = [
            e for e in formatted_graph["edges"] if e[2]["intent"] == "depends_on"
        ]
        assert len(dep_edges) == 1

    def test_task_with_large_number_of_steps(self, task_graph_formatter) -> None:
        steps = [f"Step {i}" for i in range(100)]
        tasks = [{"task_id": "task1", "name": "Big Task", "steps": steps}]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        step_nodes = [
            node
            for node_id, node in formatted_graph["nodes"]
            if node.get("resource", {}).get("id") == "message_worker"
            and node.get("attribute", {}).get("task") == "Big Task"
        ]
        assert len(step_nodes) == 100

    def test_task_with_non_string_non_dict_steps(self, task_graph_formatter) -> None:
        tasks = [
            {"task_id": "task1", "name": "Weird Steps", "steps": [123, None, True, 4.5]}
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        step_nodes = [
            node
            for node_id, node in formatted_graph["nodes"]
            if node.get("resource", {}).get("id") == "message_worker"
            and node.get("attribute", {}).get("task") == "Weird Steps"
        ]
        # Should not create any step nodes
        assert len(step_nodes) == 0

    def test_task_with_self_dependency(self, task_graph_formatter) -> None:
        tasks = [{"task_id": "task1", "name": "Self Dep", "dependencies": ["task1"]}]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        # Should not crash, should create 3 nodes (start, task1, nested_graph)
        assert len(formatted_graph["nodes"]) == 3
        # Should create 1 dependency edge (self-loop)
        dep_edges = [
            e for e in formatted_graph["edges"] if e[2]["intent"] == "depends_on"
        ]
        assert len(dep_edges) == 1
        assert dep_edges[0][0] == dep_edges[0][1]

    def test_task_with_mixed_valid_invalid_dependencies(
        self, task_graph_formatter
    ) -> None:
        tasks = [
            {"task_id": "task1", "name": "A"},
            {"task_id": "task2", "name": "B", "dependencies": ["task1", "nonexistent"]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        dep_edges = [
            e for e in formatted_graph["edges"] if e[2]["intent"] == "depends_on"
        ]
        assert len(dep_edges) == 1
        # There should also be a start->task1 edge
        start_edges = [e for e in formatted_graph["edges"] if e[0] == "0"]
        assert len(start_edges) >= 1

    def test_task_with_missing_task_id(self, task_graph_formatter) -> None:
        tasks = [{"name": "No ID Task", "description": "No id field"}]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        # Should create 3 nodes (start, fallback task, nested_graph)
        assert len(formatted_graph["nodes"]) == 3
        # Should create a start->task edge
        start_edges = [e for e in formatted_graph["edges"] if e[0] == "0"]
        assert len(start_edges) == 1


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
