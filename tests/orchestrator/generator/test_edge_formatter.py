"""Tests for the EdgeFormatter class.

This module provides comprehensive tests for the EdgeFormatter class,
including edge formatting, validation, and style management functionality.
"""

from typing import Any

import pytest

from arklex.orchestrator.generator.formatting.edge_formatter import EdgeFormatter

# --- Fixtures ---


@pytest.fixture
def edge_formatter() -> EdgeFormatter:
    """Create an EdgeFormatter instance for testing."""
    return EdgeFormatter(default_intent="depends_on", default_weight=2)


@pytest.fixture
def edge_formatter_custom() -> EdgeFormatter:
    """Create an EdgeFormatter instance with custom default intent."""
    return EdgeFormatter(default_intent="custom_type")


@pytest.fixture
def sample_source() -> dict[str, Any]:
    """Create a sample source task for testing."""
    return {"task_id": "source_1", "id": "source_1", "name": "Source Task"}


@pytest.fixture
def sample_target() -> dict[str, Any]:
    """Create a sample target task for testing."""
    return {"task_id": "target_1", "id": "target_1", "name": "Target Task"}


@pytest.fixture
def invalid_source() -> dict[str, Any]:
    """Create an invalid source task (missing ID) for testing."""
    return {"name": "No ID"}


@pytest.fixture
def invalid_target() -> dict[str, Any]:
    """Create an invalid target task (missing ID) for testing."""
    return {"name": "No ID"}


@pytest.fixture
def edge_types() -> dict[str, str]:
    """Define edge types and their expected colors for testing."""
    return {
        "depends_on": "#ff0000",
        "blocks": "#ffa500",
        "related_to": "#00ff00",
        "part_of": "#0000ff",
        "custom_type": "#808080",  # Unknown type defaults to gray
    }


@pytest.fixture
def edge_style_properties() -> list:
    """Define required edge style properties for testing."""
    return [
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


# --- Test Classes ---


class TestEdgeFormatterInitialization:
    """Test suite for EdgeFormatter initialization and basic functionality."""

    def test_import_edge_formatter_module(self) -> None:
        """Test that the EdgeFormatter module can be imported."""
        assert EdgeFormatter is not None

    def test_edge_formatter_initialization(self, edge_formatter: EdgeFormatter) -> None:
        """Test EdgeFormatter initialization with default parameters."""
        # Test that the formatter can format edges with default behavior
        source = {"name": "Source Task"}
        target = {"name": "Target Task"}
        result = edge_formatter.format_edge("source", "target", source, target)
        assert len(result) == 3
        assert result[2]["intent"] == "depends_on"

    def test_edge_formatter_custom_initialization(
        self, edge_formatter_custom: EdgeFormatter
    ) -> None:
        """Test EdgeFormatter initialization with custom parameters."""
        # Test that the formatter uses custom default intent
        source = {"name": "Source Task"}
        target = {"name": "Target Task"}
        result = edge_formatter_custom.format_edge("source", "target", source, target)
        assert len(result) == 3
        assert result[2]["intent"] == "custom_type"


class TestEdgeFormatting:
    """Test suite for edge formatting functionality."""

    def test_format_edge_basic(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: dict[str, Any],
        sample_target: dict[str, Any],
    ) -> None:
        """Test basic edge formatting with valid inputs."""
        result = edge_formatter.format_edge(
            "source", "target", sample_source, sample_target
        )
        assert len(result) == 3
        assert result[0] == "source"
        assert result[1] == "target"
        assert "intent" in result[2]

    def test_format_edge_missing_fields(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: dict[str, Any],
        sample_target: dict[str, Any],
    ) -> None:
        """Test edge formatting with missing optional fields."""
        result = edge_formatter.format_edge(
            "source", "target", sample_source, sample_target
        )
        assert len(result) == 3
        assert result[0] == "source"
        assert result[1] == "target"
        assert "intent" in result[2]

    def test_format_edge_custom_type(
        self,
        edge_formatter_custom: EdgeFormatter,
        sample_source: dict[str, Any],
        sample_target: dict[str, Any],
    ) -> None:
        """Test edge formatting with custom edge type."""
        result = edge_formatter_custom.format_edge(
            "source", "target", sample_source, sample_target
        )
        assert len(result) == 3
        assert result[0] == "source"
        assert result[1] == "target"
        assert result[2]["intent"] == "custom_type"

    def test_format_edge_none_source(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: dict[str, Any],
        sample_target: dict[str, Any],
    ) -> None:
        """Test edge formatting with None as source_idx."""
        result = edge_formatter.format_edge(
            None, "target", sample_source, sample_target
        )
        assert len(result) == 3
        assert result[0] is None
        assert result[1] == "target"
        assert "intent" in result[2]

    def test_format_edge_data(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: dict[str, Any],
        sample_target: dict[str, Any],
    ) -> None:
        """Test edge data formatting."""
        result = edge_formatter.format_edge_data(sample_source, sample_target)
        assert "intent" in result
        assert "attribute" in result
        assert result["intent"] == "depends_on"
        assert "weight" in result["attribute"]

    def test_format_edge_data_label_fallback(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: dict[str, Any],
        sample_target: dict[str, Any],
    ) -> None:
        """Test edge data formatting with empty label fallback."""
        result = edge_formatter.format_edge_data(sample_source, sample_target, label="")
        assert result["intent"] == "depends_on"  # Should use default

    def test_format_edge_style(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: dict[str, Any],
        sample_target: dict[str, Any],
    ) -> None:
        """Test basic edge style formatting."""
        result = edge_formatter.format_edge_style(sample_source, sample_target)
        assert "color" in result
        assert "width" in result
        assert "style" in result

    def test_format_edge_style_types(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: dict[str, Any],
        sample_target: dict[str, Any],
        edge_types: dict[str, str],
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
        sample_source: dict[str, Any],
        sample_target: dict[str, Any],
        edge_style_properties: list,
    ) -> None:
        """Test the structure of format_edge_style output."""
        style = edge_formatter.format_edge_style(sample_source, sample_target)

        # Check all required style properties exist
        for prop in edge_style_properties:
            assert prop in style, f"Missing property: {prop}"

        # Check data types
        assert isinstance(style["width"], int)
        assert isinstance(style["opacity"], float)


class TestEdgeValidation:
    """Test suite for edge validation functionality."""

    def test_validate_edge_valid_inputs(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: dict[str, Any],
        sample_target: dict[str, Any],
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
        invalid_source: dict[str, Any],
        sample_target: dict[str, Any],
    ) -> None:
        """Test validate_edge with invalid source input."""
        assert edge_formatter.validate_edge(invalid_source, sample_target) is False

    def test_validate_edge_invalid_target(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: dict[str, Any],
        invalid_target: dict[str, Any],
    ) -> None:
        """Test validate_edge with invalid target input."""
        assert edge_formatter.validate_edge(sample_source, invalid_target) is False

    def test_validate_edge_invalid_type(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: dict[str, Any],
        sample_target: dict[str, Any],
    ) -> None:
        """Test validate_edge with invalid type input."""
        assert (
            edge_formatter.validate_edge(sample_source, sample_target, type=123)
            is False
        )

    def test_validate_edge_invalid_weight(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: dict[str, Any],
        sample_target: dict[str, Any],
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
        sample_source: dict[str, Any],
        sample_target: dict[str, Any],
    ) -> None:
        """Test validate_edge with invalid label input."""
        assert (
            edge_formatter.validate_edge(sample_source, sample_target, label=123)
            is False
        )

    def test_validate_edge_invalid_metadata(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: dict[str, Any],
        sample_target: dict[str, Any],
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

    def test_validate_edge_metadata_and_id_variants(
        self, edge_formatter: EdgeFormatter
    ) -> None:
        """Test validate_edge with various metadata and ID combinations."""
        # Test with string IDs
        assert edge_formatter.validate_edge("n1", "n2", metadata={"foo": "bar"})

        # Test with dict IDs
        assert edge_formatter.validate_edge(
            {"id": "n1"}, {"id": "n2"}, metadata={"foo": "bar"}
        )

        # Test with missing IDs
        assert not edge_formatter.validate_edge(
            {"name": "n1"}, {"id": "n2"}, metadata={"foo": "bar"}
        )
        assert not edge_formatter.validate_edge(
            {"id": "n1"}, {"name": "n2"}, metadata={"foo": "bar"}
        )

    def test_validate_edge_comprehensive(self, edge_formatter: EdgeFormatter) -> None:
        """Test validate_edge with comprehensive validation scenarios."""
        # Valid edge - both are strings
        assert edge_formatter.validate_edge("n1", "n2")

        # Valid edge - both are dicts with id
        assert edge_formatter.validate_edge({"id": "n1"}, {"id": "n2"})

        # Invalid edge - source is None
        assert not edge_formatter.validate_edge(None, "n2")

        # Invalid edge - target is None
        assert not edge_formatter.validate_edge("n1", None)

        # Invalid edge - source is dict without id
        assert not edge_formatter.validate_edge({"name": "n1"}, {"id": "n2"})

        # Invalid edge - target is dict without id
        assert not edge_formatter.validate_edge({"id": "n1"}, {"name": "n2"})

        # Invalid edge - wrong type
        assert not edge_formatter.validate_edge("n1", "n2", type=123)

        # Invalid edge - wrong weight
        assert not edge_formatter.validate_edge("n1", "n2", weight="not_a_number")

        # Invalid edge - wrong label
        assert not edge_formatter.validate_edge("n1", "n2", label=123)

        # Invalid edge - wrong metadata
        assert not edge_formatter.validate_edge("n1", "n2", metadata=[1, 2, 3])
        # Valid: metadata is a dict
        assert edge_formatter.validate_edge("n1", "n2", metadata={"foo": "bar"})


# --- Parametrized Tests ---


@pytest.mark.parametrize(
    "etype,expected_color",
    [
        ("depends_on", "#ff0000"),
        ("blocks", "#ffa500"),
        ("related_to", "#00ff00"),
        ("part_of", "#0000ff"),
        ("unknown_type", "#808080"),
    ],
)
def test_format_edge_style_types_parametrized(
    edge_formatter: EdgeFormatter,
    sample_source: dict[str, Any],
    sample_target: dict[str, Any],
    etype: str,
    expected_color: str,
) -> None:
    """Test format_edge_style with different edge types using parametrization."""
    result = edge_formatter.format_edge_style(sample_source, sample_target, type=etype)
    assert result["color"] == expected_color
