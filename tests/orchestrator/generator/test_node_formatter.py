"""Tests for the NodeFormatter class.

This module provides comprehensive tests for the NodeFormatter class,
covering formatting logic, validation, and edge cases.
"""

from typing import Dict, Any

import pytest

from arklex.orchestrator.generator.formatting.node_formatter import NodeFormatter


# --- Fixtures ---


@pytest.fixture
def node_formatter() -> NodeFormatter:
    """Create a NodeFormatter instance for testing."""
    return NodeFormatter(default_directed=True)


@pytest.fixture
def sample_task() -> Dict[str, Any]:
    """Create a sample task for testing."""
    return {
        "task_id": "task_001",
        "name": "Test Task",
        "description": "A test task description",
        "priority": "high",
        "steps": [
            {"step_id": "step_1", "description": "Step 1", "order": 1},
            {"step_id": "step_2", "description": "Step 2", "order": 2},
        ],
        "dependencies": ["task_002", "task_003"],
        "required_resources": ["resource_1", "resource_2"],
        "estimated_duration": "2 hours",
        "type": "analysis",
        "limit": {"max_retries": 3},
    }


@pytest.fixture
def complex_task() -> Dict[str, Any]:
    """Create a complex task with nested structures for testing."""
    return {
        "task_id": "complex_task_001",
        "name": "Complex Analysis Task",
        "description": "A complex task with multiple components",
        "priority": "medium",
        "steps": [
            {
                "step_id": "complex_step_1",
                "description": "Initialize analysis",
                "order": 1,
                "metadata": {"subtasks": ["subtask_1", "subtask_2"]},
            },
            {
                "step_id": "complex_step_2",
                "description": "Process data",
                "order": 2,
                "metadata": {"algorithm": "ml_model_v1"},
            },
        ],
        "dependencies": ["prep_task", "data_task"],
        "required_resources": ["gpu", "database", "api_key"],
        "estimated_duration": "4 hours",
        "type": "machine_learning",
        "limit": {"timeout": 3600, "memory": "8GB"},
        "metadata": {
            "complexity": "high",
            "team": ["analyst", "engineer"],
            "tags": ["ml", "analysis", "critical"],
        },
    }


@pytest.fixture
def minimal_task() -> Dict[str, Any]:
    """Create a minimal task with only required fields."""
    return {
        "task_id": "minimal_task",
        "name": "Minimal Task",
    }


@pytest.fixture
def extended_task() -> Dict[str, Any]:
    """Create a task with extra fields beyond standard ones."""
    return {
        "task_id": "extended_task",
        "name": "Extended Task",
        "description": "Task with extra fields",
        "limit": {"max_retries": 5},
        "type": "custom_type",
        "extra_field_1": "extra_value_1",
        "extra_field_2": {"nested": "value"},
        "extra_field_3": [1, 2, 3],
    }


@pytest.fixture
def valid_node() -> Dict[str, Any]:
    """Create a valid node for testing."""
    return {
        "id": "node_001",
        "type": "task",
        "data": {
            "name": "Valid Task",
            "description": "A valid task description",
            "priority": "high",
        },
    }


@pytest.fixture
def invalid_source() -> Dict[str, Any]:
    """Create an invalid source task (missing ID) for testing."""
    return {
        "name": "Invalid Source",
        "description": "Source task without ID",
    }


@pytest.fixture
def invalid_target() -> Dict[str, Any]:
    """Create an invalid target task (missing ID) for testing."""
    return {
        "name": "Invalid Target",
        "description": "Target task without ID",
    }


# --- Test Classes ---


class TestNodeFormatterInitialization:
    """Test NodeFormatter initialization and basic functionality."""

    def test_import_node_formatter_module(self) -> None:
        """Should successfully import NodeFormatter module."""
        assert NodeFormatter is not None

    def test_node_formatter_initialization(self, node_formatter: NodeFormatter) -> None:
        """Should initialize NodeFormatter with default settings."""
        assert node_formatter is not None
        assert hasattr(node_formatter, "format_node")
        assert hasattr(node_formatter, "format_node_data")
        assert hasattr(node_formatter, "format_node_style")
        assert hasattr(node_formatter, "validate_node")


class TestNodeFormatterFormatting:
    """Test node formatting functionality."""

    def test_format_node_basic(
        self, node_formatter: NodeFormatter, sample_task: Dict[str, Any]
    ) -> None:
        """Should format a basic task node correctly."""
        result = node_formatter.format_node(sample_task, "node_id")
        assert len(result) == 2
        assert result[0] == "node_id"
        assert "resource" in result[1]
        assert "attribute" in result[1]

    def test_format_node_with_complex_steps(
        self, node_formatter: NodeFormatter, complex_task: Dict[str, Any]
    ) -> None:
        """Should format node with complex step structures."""
        node_id = "node_001"
        result = node_formatter.format_node(complex_task, node_id)

        assert len(result) == 2
        assert result[0] == node_id
        assert "resource" in result[1]
        assert "attribute" in result[1]
        assert result[1]["resource"]["id"] == complex_task["task_id"]
        assert result[1]["resource"]["name"] == complex_task["name"]

    def test_format_node_with_missing_fields(
        self, node_formatter: NodeFormatter, minimal_task: Dict[str, Any]
    ) -> None:
        """Should format node with missing optional fields."""
        node_id = "minimal_node"
        result = node_formatter.format_node(minimal_task, node_id)

        assert len(result) == 2
        assert result[0] == node_id
        assert "resource" in result[1]
        assert "attribute" in result[1]
        assert "limit" not in result[1]
        assert "type" not in result[1]

    def test_format_node_with_extra_fields(
        self, node_formatter: NodeFormatter, extended_task: Dict[str, Any]
    ) -> None:
        """Should format node with extra fields beyond standard ones."""
        node_id = "extended_node"
        result = node_formatter.format_node(extended_task, node_id)

        assert len(result) == 2
        assert result[0] == node_id
        assert "resource" in result[1]
        assert "attribute" in result[1]
        assert "limit" in result[1]
        assert "type" in result[1]
        # Extra fields should not be included in the formatted result
        assert "extra_field_1" not in result[1]

    def test_format_node_error_case(self, node_formatter: NodeFormatter) -> None:
        """Should handle empty task gracefully."""
        empty_task = {}
        result = node_formatter.format_node(empty_task, "node_id")
        assert len(result) == 2
        assert result[0] == "node_id"
        assert "resource" in result[1]
        assert "attribute" in result[1]

    def test_format_node_data_with_complex_task(
        self, node_formatter: NodeFormatter, complex_task: Dict[str, Any]
    ) -> None:
        """Should format node data with complex task structure."""
        result = node_formatter.format_node_data(complex_task)

        assert "resource" in result
        assert "attribute" in result
        assert result["resource"]["id"] == "26bb6634-3bee-417d-ad75-23269ac17bc3"
        assert result["resource"]["name"] == "MessageWorker"
        assert result["attribute"]["value"] == complex_task["description"]
        assert result["attribute"]["task"] == complex_task["name"]
        assert result["attribute"]["directed"] is True

    def test_format_node_data_with_minimal_task(
        self, node_formatter: NodeFormatter, minimal_task: Dict[str, Any]
    ) -> None:
        """Should format node data with minimal task data."""
        result = node_formatter.format_node_data(minimal_task)

        assert "resource" in result
        assert "attribute" in result
        assert result["attribute"]["value"] == ""
        assert result["attribute"]["task"] == "Minimal Task"

    def test_format_node_style_with_different_priorities(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should format node style with different priority levels."""
        # Test high priority
        high_priority_task = {"priority": "high"}
        high_style = node_formatter.format_node_style(high_priority_task)
        assert high_style["color"] == "#ff0000"

        # Test medium priority
        medium_priority_task = {"priority": "medium"}
        medium_style = node_formatter.format_node_style(medium_priority_task)
        assert medium_style["color"] == "#ffa500"

        # Test low priority
        low_priority_task = {"priority": "low"}
        low_style = node_formatter.format_node_style(low_priority_task)
        assert low_style["color"] == "#00ff00"

        # Test unknown priority
        unknown_priority_task = {"priority": "unknown"}
        unknown_style = node_formatter.format_node_style(unknown_priority_task)
        assert unknown_style["color"] == "#808080"

        # Test no priority specified
        no_priority_task = {}
        default_style = node_formatter.format_node_style(no_priority_task)
        assert default_style["color"] == "#ffa500"  # Defaults to medium priority

    def test_format_node_style_structure(
        self, node_formatter: NodeFormatter, sample_task: Dict[str, Any]
    ) -> None:
        """Should have complete structure in format_node_style output."""
        style = node_formatter.format_node_style(sample_task)

        # Check all expected style properties
        assert "color" in style
        assert "border" in style
        assert "padding" in style
        assert "text_color" in style
        assert "background_color" in style
        assert "opacity" in style

        # Check border structure
        assert "color" in style["border"]
        assert "width" in style["border"]
        assert "style" in style["border"]
        assert "radius" in style["border"]

        # Check padding structure
        assert "top" in style["padding"]
        assert "right" in style["padding"]
        assert "bottom" in style["padding"]
        assert "left" in style["padding"]

        # Check data types
        assert isinstance(style["color"], str)
        assert isinstance(style["border"]["width"], int)
        assert isinstance(style["opacity"], float)


class TestNodeFormatterValidation:
    """Test node validation functionality."""

    def test_validate_node_with_complete_data(
        self, node_formatter: NodeFormatter, valid_node: Dict[str, Any]
    ) -> None:
        """Should validate node with complete and valid node data."""
        assert node_formatter.validate_node(valid_node) is True

    def test_validate_node_with_missing_id(self, node_formatter: NodeFormatter) -> None:
        """Should reject node with missing ID."""
        invalid_node = {
            "type": "task",
            "data": {"name": "Task", "description": "Description"},
        }
        assert node_formatter.validate_node(invalid_node) is False

    def test_validate_node_with_invalid_id_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should reject node with invalid ID type."""
        invalid_node = {
            "id": 123,  # Should be string
            "type": "task",
            "data": {"name": "Task", "description": "Description"},
        }
        assert node_formatter.validate_node(invalid_node) is False

    def test_validate_node_with_missing_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should reject node with missing type."""
        invalid_node = {
            "id": "node_001",
            "data": {"name": "Task", "description": "Description"},
        }
        assert node_formatter.validate_node(invalid_node) is False

    def test_validate_node_with_invalid_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should reject node with invalid type."""
        invalid_node = {
            "id": "node_001",
            "type": 123,  # Should be string
            "data": {"name": "Task", "description": "Description"},
        }
        assert node_formatter.validate_node(invalid_node) is False

    def test_validate_node_with_invalid_data_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should reject node with invalid data type."""
        invalid_node = {
            "id": "node_001",
            "type": "task",
            "data": "not_a_dict",  # Should be dict
        }
        assert node_formatter.validate_node(invalid_node) is False

    def test_validate_node_with_invalid_name_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should reject node with invalid name type in data."""
        invalid_node = {
            "id": "node_001",
            "type": "task",
            "data": {"name": 123, "description": "Test desc"},  # name should be string
        }
        assert node_formatter.validate_node(invalid_node) is False

    def test_validate_node_with_invalid_description_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should reject node with invalid description type in data."""
        invalid_node = {
            "id": "node_001",
            "type": "task",
            "data": {
                "name": "Test",
                "description": 123,
            },  # description should be string
        }
        assert node_formatter.validate_node(invalid_node) is False

    def test_validate_node_with_invalid_priority_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should reject node with invalid priority type in data."""
        invalid_node = {
            "id": "node_001",
            "type": "task",
            "data": {"name": "Test", "priority": []},  # priority should be string
        }
        assert node_formatter.validate_node(invalid_node) is False

    def test_validate_node_without_data(self, node_formatter: NodeFormatter) -> None:
        """Should validate node without data field."""
        node_without_data = {"id": "node_001", "type": "task"}
        assert node_formatter.validate_node(node_without_data) is True

    def test_validate_node_with_empty_data(self, node_formatter: NodeFormatter) -> None:
        """Should validate node with empty data dict."""
        node_with_empty_data = {"id": "node_001", "type": "task", "data": {}}
        assert node_formatter.validate_node(node_with_empty_data) is True


class TestNodeFormatterAttributeValidation:
    """Test attribute validation functionality."""

    def test_validate_attribute_steps_valid(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should validate valid steps attribute."""
        valid_steps = [
            {"step_id": "step1", "description": "Step 1", "order": 1},
            {"step_id": "step2", "description": "Step 2", "order": 2},
        ]
        assert node_formatter._validate_attribute(valid_steps, "steps") is True

    def test_validate_attribute_steps_invalid_list(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should reject steps that are not a list."""
        invalid_steps = "not_a_list"
        assert node_formatter._validate_attribute(invalid_steps, "steps") is False

    def test_validate_attribute_steps_invalid_step_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should reject steps with invalid step type."""
        invalid_steps = [{"step_id": "step1"}, "not_a_dict"]
        assert node_formatter._validate_attribute(invalid_steps, "steps") is False

    def test_validate_attribute_steps_missing_step_id(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should reject steps missing step_id."""
        invalid_steps = [{"description": "Step 1", "order": 1}]
        assert node_formatter._validate_attribute(invalid_steps, "steps") is False

    def test_validate_attribute_steps_missing_description(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should reject steps missing description."""
        invalid_steps = [{"step_id": "step1", "order": 1}]
        assert node_formatter._validate_attribute(invalid_steps, "steps") is False

    def test_validate_attribute_steps_missing_order(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should reject steps missing order."""
        invalid_steps = [{"step_id": "step1", "description": "Step 1"}]
        assert node_formatter._validate_attribute(invalid_steps, "steps") is False

    def test_validate_attribute_dependencies_valid(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should validate valid dependencies attribute."""
        valid_dependencies = ["dep1", "dep2", "dep3"]
        assert (
            node_formatter._validate_attribute(valid_dependencies, "dependencies")
            is True
        )

    def test_validate_attribute_dependencies_invalid_list(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should reject dependencies that are not a list."""
        invalid_dependencies = "not_a_list"
        assert (
            node_formatter._validate_attribute(invalid_dependencies, "dependencies")
            is False
        )

    def test_validate_attribute_dependencies_invalid_dependency_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should reject dependencies with invalid dependency type."""
        invalid_dependencies = ["dep1", 123, "dep3"]
        assert (
            node_formatter._validate_attribute(invalid_dependencies, "dependencies")
            is False
        )

    def test_validate_attribute_required_resources_valid(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should validate valid required_resources attribute."""
        valid_resources = ["resource1", "resource2"]
        assert (
            node_formatter._validate_attribute(valid_resources, "required_resources")
            is True
        )

    def test_validate_attribute_required_resources_invalid_list(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should reject required_resources that are not a list."""
        invalid_resources = "not_a_list"
        assert (
            node_formatter._validate_attribute(invalid_resources, "required_resources")
            is False
        )

    def test_validate_attribute_required_resources_invalid_resource_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should reject required_resources with invalid resource type."""
        invalid_resources = ["resource1", 123]
        assert (
            node_formatter._validate_attribute(invalid_resources, "required_resources")
            is False
        )

    def test_validate_attribute_estimated_duration_valid(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should validate valid estimated_duration attribute."""
        valid_duration = "2 hours"
        assert (
            node_formatter._validate_attribute(valid_duration, "estimated_duration")
            is True
        )

    def test_validate_attribute_estimated_duration_invalid_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should reject estimated_duration that is not a string."""
        invalid_duration = 123
        assert (
            node_formatter._validate_attribute(invalid_duration, "estimated_duration")
            is False
        )

    def test_validate_attribute_priority_valid(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should validate valid priority values."""
        assert node_formatter._validate_attribute("high", "priority") is True
        assert node_formatter._validate_attribute("medium", "priority") is True
        assert node_formatter._validate_attribute("low", "priority") is True

    def test_validate_attribute_priority_invalid_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should reject priority that is not a string."""
        assert node_formatter._validate_attribute(123, "priority") is False

    def test_validate_attribute_priority_invalid_value(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should reject invalid priority values."""
        assert node_formatter._validate_attribute("invalid", "priority") is False

    def test_validate_attribute_unknown_attribute(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should return True for unknown attributes."""
        assert node_formatter._validate_attribute("any_value", "unknown_attr") is True


class TestNodeFormatterEdgeCases:
    """Test edge cases and error handling."""

    def test_format_node_with_empty_strings(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should handle empty strings gracefully."""
        task_with_empty_strings = {
            "task_id": "empty_task",
            "name": "",
            "description": "",
            "priority": "",
        }
        result = node_formatter.format_node(task_with_empty_strings, "node_id")
        assert len(result) == 2
        assert result[0] == "node_id"
        assert "resource" in result[1]
        assert "attribute" in result[1]

    def test_format_node_with_none_values(self, node_formatter: NodeFormatter) -> None:
        """Should handle None values gracefully."""
        task_with_none_values = {
            "task_id": "none_task",
            "name": None,
            "description": None,
            "priority": None,
        }
        result = node_formatter.format_node(task_with_none_values, "node_id")
        assert len(result) == 2
        assert result[0] == "node_id"
        assert "resource" in result[1]
        assert "attribute" in result[1]

    def test_format_node_style_with_none_priority(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should handle None priority in style formatting."""
        task_with_none_priority = {"priority": None}
        style = node_formatter.format_node_style(task_with_none_priority)
        assert "color" in style
        assert style["color"] == "#808080"  # Should default to gray for None priority

    def test_validate_node_with_none_values(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should handle None values in validation."""
        node_with_none_values = {
            "id": None,
            "type": None,
            "data": {"name": None, "description": None},
        }
        assert node_formatter.validate_node(node_with_none_values) is False

    def test_validate_node_with_empty_strings(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should handle empty strings in validation."""
        node_with_empty_strings = {
            "id": "",
            "type": "",
            "data": {"name": "", "description": ""},
        }
        assert (
            node_formatter.validate_node(node_with_empty_strings) is True
        )  # Empty strings are valid

    def test_format_node_data_with_none_values(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Should handle None values in node data formatting."""
        task_with_none_values = {
            "name": None,
            "description": None,
        }
        result = node_formatter.format_node_data(task_with_none_values)
        assert "resource" in result
        assert "attribute" in result
        assert (
            result["attribute"]["value"] is None
        )  # Should return None for None values
        assert result["attribute"]["task"] is None  # Should return None for None values
