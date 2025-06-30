"""Comprehensive tests for GraphValidator.

This module provides extensive test coverage for the GraphValidator class,
including all methods, edge cases, error conditions, and validation scenarios.
"""

import pytest

from arklex.orchestrator.generator.formatting.graph_validator import GraphValidator

# =============================================================================
# FIXTURES - Core Test Data
# =============================================================================


@pytest.fixture
def graph_validator() -> GraphValidator:
    """Create a GraphValidator instance for testing."""
    return GraphValidator()


@pytest.fixture
def valid_graph() -> dict:
    """Sample valid graph for testing."""
    return {
        "role": "test_role",
        "user_objective": "test_objective",
        "builder_objective": "test_builder_objective",
        "domain": "test_domain",
        "intro": "test_intro",
        "task_docs": [],
        "rag_docs": [],
        "workers": [],
        "nodes": [
            ["node1", {"id": "node1", "type": "task"}],
            ["node2", {"id": "node2", "type": "task"}],
        ],
        "edges": [
            ["node1", "node2", {"intent": "continue"}],
        ],
    }


@pytest.fixture
def invalid_graph_not_dict() -> str:
    """Sample invalid graph that is not a dictionary."""
    return "not a dictionary"


@pytest.fixture
def invalid_graph_no_nodes() -> dict:
    """Sample invalid graph with nodes field missing."""
    return {
        "role": "test_role",
        "user_objective": "test_objective",
        "builder_objective": "test_builder_objective",
        "domain": "test_domain",
        "intro": "test_intro",
        "task_docs": [],
        "rag_docs": [],
        "workers": [],
        # Missing "nodes" field entirely
        "edges": [],
    }


@pytest.fixture
def invalid_graph_nodes_not_list() -> dict:
    """Sample invalid graph with nodes not being a list."""
    return {
        "role": "test_role",
        "user_objective": "test_objective",
        "builder_objective": "test_builder_objective",
        "domain": "test_domain",
        "intro": "test_intro",
        "task_docs": [],
        "rag_docs": [],
        "workers": [],
        "nodes": "not a list",
        "edges": [],
    }


@pytest.fixture
def invalid_graph_no_edges() -> dict:
    """Sample invalid graph with edges field missing."""
    return {
        "role": "test_role",
        "user_objective": "test_objective",
        "builder_objective": "test_builder_objective",
        "domain": "test_domain",
        "intro": "test_intro",
        "task_docs": [],
        "rag_docs": [],
        "workers": [],
        "nodes": [
            ["node1", {"id": "node1", "type": "task"}],
        ],
        # Missing "edges" field entirely
    }


@pytest.fixture
def invalid_graph_edges_not_list() -> dict:
    """Sample invalid graph with edges not being a list."""
    return {
        "role": "test_role",
        "user_objective": "test_objective",
        "builder_objective": "test_builder_objective",
        "domain": "test_domain",
        "intro": "test_intro",
        "task_docs": [],
        "rag_docs": [],
        "workers": [],
        "nodes": [
            ["node1", {"id": "node1", "type": "task"}],
        ],
        "edges": "not a list",
    }


@pytest.fixture
def invalid_graph_edge_wrong_format() -> dict:
    """Sample invalid graph with edge in wrong format."""
    return {
        "role": "test_role",
        "user_objective": "test_objective",
        "builder_objective": "test_builder_objective",
        "domain": "test_domain",
        "intro": "test_intro",
        "task_docs": [],
        "rag_docs": [],
        "workers": [],
        "nodes": [
            ["node1", {"id": "node1", "type": "task"}],
            ["node2", {"id": "node2", "type": "task"}],
        ],
        "edges": [
            ["node1", "node2"],  # Missing data
        ],
    }


@pytest.fixture
def invalid_graph_edge_source_not_found() -> dict:
    """Sample invalid graph with edge source not found in nodes."""
    return {
        "role": "test_role",
        "user_objective": "test_objective",
        "builder_objective": "test_builder_objective",
        "domain": "test_domain",
        "intro": "test_intro",
        "task_docs": [],
        "rag_docs": [],
        "workers": [],
        "nodes": [
            ["node1", {"id": "node1", "type": "task"}],
        ],
        "edges": [
            ["nonexistent_node", "node1", {"intent": "continue"}],
        ],
    }


@pytest.fixture
def invalid_graph_edge_target_not_found() -> dict:
    """Sample invalid graph with edge target not found in nodes."""
    return {
        "role": "test_role",
        "user_objective": "test_objective",
        "builder_objective": "test_builder_objective",
        "domain": "test_domain",
        "intro": "test_intro",
        "task_docs": [],
        "rag_docs": [],
        "workers": [],
        "nodes": [
            ["node1", {"id": "node1", "type": "task"}],
        ],
        "edges": [
            ["node1", "nonexistent_node", {"intent": "continue"}],
        ],
    }


@pytest.fixture
def invalid_graph_edge_data_not_dict() -> dict:
    """Sample invalid graph with edge data not being a dictionary."""
    return {
        "role": "test_role",
        "user_objective": "test_objective",
        "builder_objective": "test_builder_objective",
        "domain": "test_domain",
        "intro": "test_intro",
        "task_docs": [],
        "rag_docs": [],
        "workers": [],
        "nodes": [
            ["node1", {"id": "node1", "type": "task"}],
            ["node2", {"id": "node2", "type": "task"}],
        ],
        "edges": [
            ["node1", "node2", "not a dict"],
        ],
    }


@pytest.fixture
def invalid_graph_missing_required_field() -> dict:
    """Sample invalid graph with missing required field."""
    return {
        "role": "test_role",
        "user_objective": "test_objective",
        "builder_objective": "test_builder_objective",
        "domain": "test_domain",
        "intro": "test_intro",
        "task_docs": [],
        "rag_docs": [],
        # Missing "workers" field
        "nodes": [
            ["node1", {"id": "node1", "type": "task"}],
        ],
        "edges": [],
    }


# =============================================================================
# TEST CLASSES
# =============================================================================


class TestGraphValidator:
    """Test the GraphValidator class."""

    def test_graph_validator_initialization(
        self, graph_validator: GraphValidator
    ) -> None:
        """Test GraphValidator initialization."""
        assert isinstance(graph_validator._errors, list)
        assert len(graph_validator._errors) == 0

    def test_validate_graph_valid(
        self, graph_validator: GraphValidator, valid_graph: dict
    ) -> None:
        """Test validation of a valid graph."""
        result = graph_validator.validate_graph(valid_graph)
        assert result is True
        assert len(graph_validator._errors) == 0

    def test_validate_graph_not_dict(
        self, graph_validator: GraphValidator, invalid_graph_not_dict: dict
    ) -> None:
        """Test validation of a graph that is not a dictionary."""
        result = graph_validator.validate_graph(invalid_graph_not_dict)
        assert result is False
        assert "Graph must be a dictionary" in graph_validator._errors

    def test_validate_graph_missing_nodes_field(
        self, graph_validator: GraphValidator, invalid_graph_no_nodes: dict
    ) -> None:
        """Test validation of a graph with missing nodes field (should be valid as it defaults to empty list)."""
        result = graph_validator.validate_graph(invalid_graph_no_nodes)
        assert (
            result is True
        )  # Missing nodes field defaults to empty list, which is valid

    def test_validate_graph_nodes_not_list(
        self, graph_validator: GraphValidator, invalid_graph_nodes_not_list: dict
    ) -> None:
        """Test validation of a graph with nodes not being a list."""
        result = graph_validator.validate_graph(invalid_graph_nodes_not_list)
        assert result is False
        assert "Nodes must be a list" in graph_validator._errors

    def test_validate_graph_missing_edges_field(
        self, graph_validator: GraphValidator, invalid_graph_no_edges: dict
    ) -> None:
        """Test validation of a graph with missing edges field (should be valid as it defaults to empty list)."""
        result = graph_validator.validate_graph(invalid_graph_no_edges)
        assert (
            result is True
        )  # Missing edges field defaults to empty list, which is valid

    def test_validate_graph_edges_not_list(
        self, graph_validator: GraphValidator, invalid_graph_edges_not_list: dict
    ) -> None:
        """Test validation of a graph with edges not being a list."""
        result = graph_validator.validate_graph(invalid_graph_edges_not_list)
        assert result is False
        assert "Edges must be a list" in graph_validator._errors

    def test_validate_graph_edge_wrong_format(
        self, graph_validator: GraphValidator, invalid_graph_edge_wrong_format: dict
    ) -> None:
        """Test validation of a graph with edge in wrong format."""
        result = graph_validator.validate_graph(invalid_graph_edge_wrong_format)
        assert result is False
        assert (
            "Edge must be a list of [source, target, data]" in graph_validator._errors
        )

    def test_validate_graph_edge_source_not_found(
        self, graph_validator: GraphValidator, invalid_graph_edge_source_not_found: dict
    ) -> None:
        """Test validation of a graph with edge source not found in nodes."""
        result = graph_validator.validate_graph(invalid_graph_edge_source_not_found)
        assert result is False
        assert (
            "Edge source nonexistent_node not found in nodes" in graph_validator._errors
        )

    def test_validate_graph_edge_target_not_found(
        self, graph_validator: GraphValidator, invalid_graph_edge_target_not_found: dict
    ) -> None:
        """Test validation of a graph with edge target not found in nodes."""
        result = graph_validator.validate_graph(invalid_graph_edge_target_not_found)
        assert result is False
        assert (
            "Edge target nonexistent_node not found in nodes" in graph_validator._errors
        )

    def test_validate_graph_edge_data_not_dict(
        self, graph_validator: GraphValidator, invalid_graph_edge_data_not_dict: dict
    ) -> None:
        """Test validation of a graph with edge data not being a dictionary."""
        result = graph_validator.validate_graph(invalid_graph_edge_data_not_dict)
        assert result is False
        assert "Edge data must be a dictionary" in graph_validator._errors

    def test_validate_graph_missing_required_field(
        self,
        graph_validator: GraphValidator,
        invalid_graph_missing_required_field: dict,
    ) -> None:
        """Test validation of a graph with missing required field."""
        result = graph_validator.validate_graph(invalid_graph_missing_required_field)
        assert result is False
        assert "Missing required field: workers" in graph_validator._errors

    def test_validate_graph_multiple_edge_errors(
        self, graph_validator: GraphValidator
    ) -> None:
        """Test validation of a graph with multiple edge errors."""
        graph = {
            "role": "test_role",
            "user_objective": "test_objective",
            "builder_objective": "test_builder_objective",
            "domain": "test_domain",
            "intro": "test_intro",
            "task_docs": [],
            "rag_docs": [],
            "workers": [],
            "nodes": [
                ["node1", {"id": "node1", "type": "task"}],
            ],
            "edges": [
                ["node1", "node2", {"intent": "continue"}],  # Target not found
                ["node2", "node1", "not a dict"],  # Data not dict
                ["node1", "node1"],  # Wrong format
            ],
        }
        result = graph_validator.validate_graph(graph)
        assert result is False
        assert "Edge target node2 not found in nodes" in graph_validator._errors
        assert "Edge data must be a dictionary" in graph_validator._errors
        assert (
            "Edge must be a list of [source, target, data]" in graph_validator._errors
        )

    def test_validate_graph_multiple_missing_fields(
        self, graph_validator: GraphValidator
    ) -> None:
        """Test validation of a graph with multiple missing required fields."""
        graph = {
            "role": "test_role",
            "user_objective": "test_objective",
            # Missing builder_objective, domain, intro, task_docs, rag_docs, workers
            "nodes": [],
            "edges": [],
        }
        result = graph_validator.validate_graph(graph)
        assert result is False
        assert "Missing required field: builder_objective" in graph_validator._errors
        assert "Missing required field: domain" in graph_validator._errors
        assert "Missing required field: intro" in graph_validator._errors
        assert "Missing required field: task_docs" in graph_validator._errors
        assert "Missing required field: rag_docs" in graph_validator._errors
        assert "Missing required field: workers" in graph_validator._errors

    def test_validate_graph_edge_with_none_intent(
        self, graph_validator: GraphValidator
    ) -> None:
        """Test validation of a graph with edge having None intent."""
        graph = {
            "role": "test_role",
            "user_objective": "test_objective",
            "builder_objective": "test_builder_objective",
            "domain": "test_domain",
            "intro": "test_intro",
            "task_docs": [],
            "rag_docs": [],
            "workers": [],
            "nodes": [
                ["node1", {"id": "node1", "type": "task"}],
                ["node2", {"id": "node2", "type": "task"}],
            ],
            "edges": [
                ["node1", "node2", {"intent": None}],
            ],
        }
        result = graph_validator.validate_graph(graph)
        assert result is True  # None intent is valid

    def test_validate_graph_empty_nodes_and_edges(
        self, graph_validator: GraphValidator
    ) -> None:
        """Test validation of a graph with empty nodes and edges."""
        graph = {
            "role": "test_role",
            "user_objective": "test_objective",
            "builder_objective": "test_builder_objective",
            "domain": "test_domain",
            "intro": "test_intro",
            "task_docs": [],
            "rag_docs": [],
            "workers": [],
            "nodes": [],
            "edges": [],
        }
        result = graph_validator.validate_graph(graph)
        assert result is True

    def test_validate_graph_complex_valid_graph(
        self, graph_validator: GraphValidator
    ) -> None:
        """Test validation of a complex valid graph."""
        graph = {
            "role": "test_role",
            "user_objective": "test_objective",
            "builder_objective": "test_builder_objective",
            "domain": "test_domain",
            "intro": "test_intro",
            "task_docs": [{"doc": "test"}],
            "rag_docs": [{"rag": "test"}],
            "workers": [{"worker": "test"}],
            "nodes": [
                ["start", {"id": "start", "type": "start"}],
                ["task1", {"id": "task1", "type": "task"}],
                ["task2", {"id": "task2", "type": "task"}],
                ["end", {"id": "end", "type": "end"}],
            ],
            "edges": [
                ["start", "task1", {"intent": "start", "weight": 1}],
                ["task1", "task2", {"intent": "continue", "weight": 1}],
                ["task2", "end", {"intent": "finish", "weight": 1}],
                ["task1", "end", {"intent": "skip", "weight": 0.5}],
            ],
        }
        result = graph_validator.validate_graph(graph)
        assert result is True
        assert len(graph_validator._errors) == 0

    def test_get_error_messages_empty(self, graph_validator: GraphValidator) -> None:
        """Test getting error messages when no errors exist."""
        errors = graph_validator.get_error_messages()
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_get_error_messages_with_errors(
        self, graph_validator: GraphValidator, invalid_graph_not_dict: dict
    ) -> None:
        """Test getting error messages when errors exist."""
        graph_validator.validate_graph(invalid_graph_not_dict)
        errors = graph_validator.get_error_messages()
        assert isinstance(errors, list)
        assert len(errors) == 1
        assert "Graph must be a dictionary" in errors

    def test_validate_graph_resets_errors(
        self,
        graph_validator: GraphValidator,
        invalid_graph_not_dict: dict,
        valid_graph: dict,
    ) -> None:
        """Test that validate_graph resets errors between calls."""
        # First validation with invalid graph
        result1 = graph_validator.validate_graph(invalid_graph_not_dict)
        assert result1 is False
        assert len(graph_validator._errors) == 1

        # Second validation with valid graph
        result2 = graph_validator.validate_graph(valid_graph)
        assert result2 is True
        assert len(graph_validator._errors) == 0

    def test_validate_graph_edge_with_empty_data_dict(
        self, graph_validator: GraphValidator
    ) -> None:
        """Test validation of a graph with edge having empty data dictionary."""
        graph = {
            "role": "test_role",
            "user_objective": "test_objective",
            "builder_objective": "test_builder_objective",
            "domain": "test_domain",
            "intro": "test_intro",
            "task_docs": [],
            "rag_docs": [],
            "workers": [],
            "nodes": [
                ["node1", {"id": "node1", "type": "task"}],
                ["node2", {"id": "node2", "type": "task"}],
            ],
            "edges": [
                ["node1", "node2", {}],  # Empty dict is valid
            ],
        }
        result = graph_validator.validate_graph(graph)
        assert result is True

    def test_validate_graph_node_with_complex_data(
        self, graph_validator: GraphValidator
    ) -> None:
        """Test validation of a graph with nodes containing complex data."""
        graph = {
            "role": "test_role",
            "user_objective": "test_objective",
            "builder_objective": "test_builder_objective",
            "domain": "test_domain",
            "intro": "test_intro",
            "task_docs": [],
            "rag_docs": [],
            "workers": [],
            "nodes": [
                [
                    "node1",
                    {"id": "node1", "type": "task", "complex": {"nested": "data"}},
                ],
                ["node2", {"id": "node2", "type": "task", "list": [1, 2, 3]}],
            ],
            "edges": [
                [
                    "node1",
                    "node2",
                    {"intent": "continue", "metadata": {"key": "value"}},
                ],
            ],
        }
        result = graph_validator.validate_graph(graph)
        assert result is True
