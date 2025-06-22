"""Tests for the EdgeFormatter class.

This module provides comprehensive tests for the EdgeFormatter class,
including edge formatting, validation, and style management functionality.
"""

from typing import Dict, Any
import pytest

from arklex.orchestrator.generator.formatting.edge_formatter import EdgeFormatter


# --- Fixtures ---


@pytest.fixture
def edge_formatter() -> EdgeFormatter:
    """Create an EdgeFormatter instance for testing."""
    return EdgeFormatter(default_intent="depends_on", default_weight=2)


@pytest.fixture
def sample_source() -> Dict[str, Any]:
    """Create a sample source task for testing."""
    return {"task_id": "source_1", "id": "source_1", "name": "Source Task"}


@pytest.fixture
def sample_target() -> Dict[str, Any]:
    """Create a sample target task for testing."""
    return {"task_id": "target_1", "id": "target_1", "name": "Target Task"}


@pytest.fixture
def invalid_source() -> Dict[str, Any]:
    """Create an invalid source task (missing ID) for testing."""
    return {"name": "No ID"}


@pytest.fixture
def invalid_target() -> Dict[str, Any]:
    """Create an invalid target task (missing ID) for testing."""
    return {"name": "No ID"}


@pytest.fixture
def edge_types() -> Dict[str, str]:
    """Define edge types and their expected colors for testing."""
    return {
        "depends_on": "#ff0000",
        "blocks": "#ffa500",
        "related_to": "#00ff00",
        "part_of": "#0000ff",
        "custom_type": "#808080",  # Unknown type defaults to gray
    }


# --- Test Classes ---


class TestEdgeFormatting:
    """Test suite for edge formatting functionality."""

    def test_format_edge_style_types(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: Dict[str, Any],
        sample_target: Dict[str, Any],
        edge_types: Dict[str, str],
    ) -> None:
        """Test format_edge_style with different edge types."""
        for edge_type, expected_color in edge_types.items():
            style = edge_formatter.format_edge_style(
                sample_source, sample_target, type=edge_type
            )
            assert style["color"] == expected_color, (
                f"Failed for edge type: {edge_type}"
            )

    def test_format_edge_style_structure(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: Dict[str, Any],
        sample_target: Dict[str, Any],
    ) -> None:
        """Test the structure of format_edge_style output."""
        style = edge_formatter.format_edge_style(sample_source, sample_target)

        # Check all required style properties exist
        required_properties = [
            "color",
            "width",
            "style",
            "arrow_size",
            "arrow_style",
            "label_color",
            "label_font_size",
            "label_font_family",
            "label_font_weight",
            "opacity",
        ]

        for prop in required_properties:
            assert prop in style, f"Missing property: {prop}"

        # Check data types
        assert isinstance(style["width"], int)
        assert isinstance(style["opacity"], float)


class TestEdgeValidation:
    """Test suite for edge validation functionality."""

    def test_validate_edge_valid_inputs(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: Dict[str, Any],
        sample_target: Dict[str, Any],
    ) -> None:
        """Test validate_edge with various valid inputs."""
        # Test with dict inputs
        assert (
            edge_formatter.validate_edge(
                sample_source,
                sample_target,
                type="depends_on",
                weight=1.0,
                label="",
                metadata=None,
            )
            is True
        )

        # Test with string inputs
        assert (
            edge_formatter.validate_edge(
                "source_1",
                "target_1",
                type="blocks",
                weight=2,
                label="label",
                metadata={},
            )
            is True
        )

    def test_validate_edge_invalid_source(
        self,
        edge_formatter: EdgeFormatter,
        invalid_source: Dict[str, Any],
        sample_target: Dict[str, Any],
    ) -> None:
        """Test validate_edge with invalid source input."""
        assert edge_formatter.validate_edge(invalid_source, sample_target) is False

    def test_validate_edge_invalid_target(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: Dict[str, Any],
        invalid_target: Dict[str, Any],
    ) -> None:
        """Test validate_edge with invalid target input."""
        assert edge_formatter.validate_edge(sample_source, invalid_target) is False

    def test_validate_edge_invalid_type(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: Dict[str, Any],
        sample_target: Dict[str, Any],
    ) -> None:
        """Test validate_edge with invalid type input."""
        assert (
            edge_formatter.validate_edge(sample_source, sample_target, type=123)
            is False
        )

    def test_validate_edge_invalid_weight(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: Dict[str, Any],
        sample_target: Dict[str, Any],
    ) -> None:
        """Test validate_edge with invalid weight input."""
        assert (
            edge_formatter.validate_edge(
                sample_source, sample_target, weight="not_a_number"
            )
            is False
        )

    def test_validate_edge_invalid_label(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: Dict[str, Any],
        sample_target: Dict[str, Any],
    ) -> None:
        """Test validate_edge with invalid label input."""
        assert (
            edge_formatter.validate_edge(sample_source, sample_target, label=123)
            is False
        )

    def test_validate_edge_invalid_metadata(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: Dict[str, Any],
        sample_target: Dict[str, Any],
    ) -> None:
        """Test validate_edge with invalid metadata input."""
        assert (
            edge_formatter.validate_edge(
                sample_source, sample_target, metadata="not_a_dict"
            )
            is False
        )

    def test_validate_edge_with_string_ids(self, edge_formatter: EdgeFormatter) -> None:
        """Test validate_edge with string source and target IDs."""
        assert edge_formatter.validate_edge("source_id", "target_id") is True

    def test_validate_edge_with_missing_ids(
        self, edge_formatter: EdgeFormatter
    ) -> None:
        """Test validate_edge with dicts missing 'id' and 'task_id'."""
        source = {"foo": "bar"}
        target = {"baz": "qux"}
        assert edge_formatter.validate_edge(source, target) is False
