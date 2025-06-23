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


class TestNodeFormatterFormatting:
    """Test node formatting functionality."""

    def test_format_node_with_complex_steps(
        self, node_formatter: NodeFormatter, complex_task: Dict[str, Any]
    ) -> None:
        """Test formatting node with complex step structures."""
        node_id = "node_001"
        result = node_formatter.format_node(complex_task, node_id)

        assert len(result) == 2
        assert result[0] == node_id
        assert "resource" in result[1]
        assert "attribute" in result[1]
        assert result[1]["resource"]["id"] == complex_task["task_id"]
        assert result[1]["resource"]["name"] == complex_task["name"]

    def test_format_node_with_missing_fields(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test formatting node with missing optional fields."""
        minimal_task = {
            "task_id": "minimal_task",
            "name": "Minimal Task",
        }
        node_id = "minimal_node"
        result = node_formatter.format_node(minimal_task, node_id)

        assert len(result) == 2
        assert result[0] == node_id
        assert "resource" in result[1]
        assert "attribute" in result[1]
        assert "limit" not in result[1]
        assert "type" not in result[1]

    def test_format_node_with_extra_fields(self, node_formatter: NodeFormatter) -> None:
        """Test formatting node with extra fields beyond standard ones."""
        extended_task = {
            "task_id": "extended_task",
            "name": "Extended Task",
            "description": "Task with extra fields",
            "limit": {"max_retries": 5},
            "type": "custom_type",
            "extra_field_1": "extra_value_1",
            "extra_field_2": {"nested": "value"},
            "extra_field_3": [1, 2, 3],
        }
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

    def test_format_node_data_with_complex_task(
        self, node_formatter: NodeFormatter, complex_task: Dict[str, Any]
    ) -> None:
        """Test format_node_data with complex task structure."""
        result = node_formatter.format_node_data(complex_task)

        assert "resource" in result
        assert "attribute" in result
        assert result["resource"]["id"] == "26bb6634-3bee-417d-ad75-23269ac17bc3"
        assert result["resource"]["name"] == "MessageWorker"
        assert result["attribute"]["value"] == complex_task["description"]
        assert result["attribute"]["task"] == complex_task["name"]
        assert result["attribute"]["directed"] is True

    def test_format_node_data_with_minimal_task(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test format_node_data with minimal task data."""
        minimal_task = {"name": "Minimal Task"}
        result = node_formatter.format_node_data(minimal_task)

        assert "resource" in result
        assert "attribute" in result
        assert result["attribute"]["value"] == ""
        assert result["attribute"]["task"] == "Minimal Task"

    def test_format_node_style_with_different_priorities(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test format_node_style with different priority levels."""
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
        """Test the complete structure of format_node_style output."""
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
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test validate_node with complete and valid node data."""
        valid_node = {
            "id": "node_001",
            "type": "task",
            "data": {
                "name": "Valid Task",
                "description": "A valid task description",
                "priority": "high",
            },
        }
        assert node_formatter.validate_node(valid_node) is True

    def test_validate_node_with_missing_id(self, node_formatter: NodeFormatter) -> None:
        """Test validate_node with missing node ID."""
        invalid_node = {
            "type": "task",
            "data": {"name": "Task", "description": "Description"},
        }
        assert node_formatter.validate_node(invalid_node) is False

    def test_validate_node_with_invalid_id_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test validate_node with invalid ID type."""
        invalid_node = {
            "id": 123,  # Should be string
            "type": "task",
            "data": {"name": "Task", "description": "Description"},
        }
        assert node_formatter.validate_node(invalid_node) is False

    def test_validate_node_with_missing_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test validate_node with missing node type."""
        invalid_node = {
            "id": "node_001",
            "data": {"name": "Task", "description": "Description"},
        }
        assert node_formatter.validate_node(invalid_node) is False

    def test_validate_node_with_invalid_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test validate_node with invalid type."""
        invalid_node = {
            "id": "node_001",
            "type": 123,  # Should be string
            "data": {"name": "Task", "description": "Description"},
        }
        assert node_formatter.validate_node(invalid_node) is False

    def test_validate_node_with_invalid_data_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test validate_node with invalid data type."""
        invalid_node = {
            "id": "node_001",
            "type": "task",
            "data": "not_a_dict",  # Should be dict
        }
        assert node_formatter.validate_node(invalid_node) is False

    def test_validate_node_with_invalid_name_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test validate_node with invalid name type in data."""
        invalid_node = {
            "id": "node_001",
            "type": "task",
            "data": {
                "name": 123,
                "description": "Description",
            },  # name should be string
        }
        assert node_formatter.validate_node(invalid_node) is False

    def test_validate_node_with_invalid_description_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test validate_node with invalid description type in data."""
        invalid_node = {
            "id": "node_001",
            "type": "task",
            "data": {
                "name": "Task",
                "description": 123,
            },  # description should be string
        }
        assert node_formatter.validate_node(invalid_node) is False

    def test_validate_node_with_invalid_priority_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test validate_node with invalid priority type in data."""
        invalid_node = {
            "id": "node_001",
            "type": "task",
            "data": {
                "name": "Task",
                "description": "Description",
                "priority": [],
            },  # priority should be string/int
        }
        assert node_formatter.validate_node(invalid_node) is False

    def test_validate_node_without_data(self, node_formatter: NodeFormatter) -> None:
        """Test validate_node without data field."""
        valid_node = {
            "id": "node_001",
            "type": "task",
        }
        assert node_formatter.validate_node(valid_node) is True

    def test_validate_node_with_empty_data(self, node_formatter: NodeFormatter) -> None:
        """Test validate_node with empty data dict."""
        valid_node = {
            "id": "node_001",
            "type": "task",
            "data": {},
        }
        assert node_formatter.validate_node(valid_node) is True


class TestNodeFormatterAttributeValidation:
    """Test attribute validation functionality."""

    def test_validate_attribute_steps_valid(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test _validate_attribute with valid steps."""
        valid_steps = [
            {"step_id": "step_1", "description": "Step 1", "order": 1},
            {"step_id": "step_2", "description": "Step 2", "order": 2},
        ]
        assert node_formatter._validate_attribute(valid_steps, "steps") is True

    def test_validate_attribute_steps_invalid_list(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test _validate_attribute with invalid steps (not a list)."""
        invalid_steps = "not_a_list"
        assert node_formatter._validate_attribute(invalid_steps, "steps") is False

    def test_validate_attribute_steps_invalid_step_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test _validate_attribute with invalid step type."""
        invalid_steps = [
            {"step_id": "step_1", "description": "Step 1", "order": 1},
            "not_a_dict",  # Invalid step
        ]
        assert node_formatter._validate_attribute(invalid_steps, "steps") is False

    def test_validate_attribute_steps_missing_step_id(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test _validate_attribute with steps missing step_id."""
        invalid_steps = [
            {"description": "Step 1", "order": 1},  # Missing step_id
        ]
        assert node_formatter._validate_attribute(invalid_steps, "steps") is False

    def test_validate_attribute_steps_missing_description(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test _validate_attribute with steps missing description."""
        invalid_steps = [
            {"step_id": "step_1", "order": 1},  # Missing description
        ]
        assert node_formatter._validate_attribute(invalid_steps, "steps") is False

    def test_validate_attribute_steps_missing_order(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test _validate_attribute with steps missing order."""
        invalid_steps = [
            {"step_id": "step_1", "description": "Step 1"},  # Missing order
        ]
        assert node_formatter._validate_attribute(invalid_steps, "steps") is False

    def test_validate_attribute_dependencies_valid(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test _validate_attribute with valid dependencies."""
        valid_dependencies = ["task_1", "task_2", "task_3"]
        assert (
            node_formatter._validate_attribute(valid_dependencies, "dependencies")
            is True
        )

    def test_validate_attribute_dependencies_invalid_list(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test _validate_attribute with invalid dependencies (not a list)."""
        invalid_dependencies = "not_a_list"
        assert (
            node_formatter._validate_attribute(invalid_dependencies, "dependencies")
            is False
        )

    def test_validate_attribute_dependencies_invalid_dependency_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test _validate_attribute with invalid dependency type."""
        invalid_dependencies = ["task_1", 123, "task_3"]  # 123 is not a string
        assert (
            node_formatter._validate_attribute(invalid_dependencies, "dependencies")
            is False
        )

    def test_validate_attribute_required_resources_valid(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test _validate_attribute with valid required_resources."""
        valid_resources = ["resource_1", "resource_2", "resource_3"]
        assert (
            node_formatter._validate_attribute(valid_resources, "required_resources")
            is True
        )

    def test_validate_attribute_required_resources_invalid_list(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test _validate_attribute with invalid required_resources (not a list)."""
        invalid_resources = "not_a_list"
        assert (
            node_formatter._validate_attribute(invalid_resources, "required_resources")
            is False
        )

    def test_validate_attribute_required_resources_invalid_resource_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test _validate_attribute with invalid resource type."""
        invalid_resources = ["resource_1", 123, "resource_3"]  # 123 is not a string
        assert (
            node_formatter._validate_attribute(invalid_resources, "required_resources")
            is False
        )

    def test_validate_attribute_estimated_duration_valid(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test _validate_attribute with valid estimated_duration."""
        valid_duration = "2 hours"
        assert (
            node_formatter._validate_attribute(valid_duration, "estimated_duration")
            is True
        )

    def test_validate_attribute_estimated_duration_invalid_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test _validate_attribute with invalid estimated_duration type."""
        invalid_duration = 120  # Should be string
        assert (
            node_formatter._validate_attribute(invalid_duration, "estimated_duration")
            is False
        )

    def test_validate_attribute_priority_valid(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test _validate_attribute with valid priority values."""
        valid_priorities = ["high", "medium", "low"]
        for priority in valid_priorities:
            assert node_formatter._validate_attribute(priority, "priority") is True

    def test_validate_attribute_priority_invalid_type(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test _validate_attribute with invalid priority type."""
        invalid_priority = 123  # Should be string
        assert node_formatter._validate_attribute(invalid_priority, "priority") is False

    def test_validate_attribute_priority_invalid_value(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test _validate_attribute with invalid priority value."""
        invalid_priority = "invalid_priority"
        assert node_formatter._validate_attribute(invalid_priority, "priority") is False

    def test_validate_attribute_unknown_attribute(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test _validate_attribute with unknown attribute type."""
        unknown_value = "some_value"
        assert (
            node_formatter._validate_attribute(unknown_value, "unknown_attribute")
            is True
        )


class TestNodeFormatterEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_format_node_with_empty_strings(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test format_node with empty string values."""
        empty_task = {
            "task_id": "",
            "name": "",
            "description": "",
        }
        node_id = "empty_node"
        result = node_formatter.format_node(empty_task, node_id)

        assert len(result) == 2
        assert result[0] == node_id
        assert result[1]["resource"]["id"] == ""
        assert result[1]["resource"]["name"] == ""
        assert result[1]["attribute"]["value"] == ""

    def test_format_node_with_none_values(self, node_formatter: NodeFormatter) -> None:
        """Test format_node with None values."""
        none_task = {
            "task_id": None,
            "name": None,
            "description": None,
        }
        node_id = "none_node"
        result = node_formatter.format_node(none_task, node_id)

        assert len(result) == 2
        assert result[0] == node_id
        assert result[1]["resource"]["id"] is None
        assert result[1]["resource"]["name"] is None
        assert result[1]["attribute"]["value"] is None

    def test_format_node_style_with_none_priority(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test format_node_style with None priority."""
        none_priority_task = {"priority": None}
        style = node_formatter.format_node_style(none_priority_task)
        assert style["color"] == "#808080"  # Default color for unknown priority

    def test_validate_node_with_none_values(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test validate_node with None values."""
        none_node = {
            "id": None,
            "type": None,
            "data": None,
        }
        assert node_formatter.validate_node(none_node) is False

    def test_validate_node_with_empty_strings(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test validate_node with empty string values."""
        empty_node = {
            "id": "",
            "type": "",
            "data": {},
        }
        assert (
            node_formatter.validate_node(empty_node) is True
        )  # Empty strings are valid

    def test_format_node_data_with_none_values(
        self, node_formatter: NodeFormatter
    ) -> None:
        """Test format_node_data with None values."""
        none_task = {
            "name": None,
            "description": None,
        }
        result = node_formatter.format_node_data(none_task)

        assert result["attribute"]["value"] is None
        assert result["attribute"]["task"] is None
        assert result["attribute"]["directed"] is True
