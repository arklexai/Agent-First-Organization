"""Core generator tests for the main Generator class.

These tests verify the core functionality of the Generator class,
including initialization, component integration, and error handling.
"""

import pytest
from unittest.mock import MagicMock, patch
from arklex.orchestrator.generator.core.generator import Generator
from arklex.env.env import BaseResourceInitializer, DefaultResourceInitializer

# --- Fixtures for common mocks and configs ---


@pytest.fixture
def mock_model():
    """Create a mock language model for testing."""
    return MagicMock()


@pytest.fixture
def minimal_config():
    """Create a minimal valid configuration for testing."""
    return {
        "role": "test_role",
        "user_objective": "test_objective",
        "builder_objective": "test_builder_objective",
        "intro": "test_intro",
        "instruction_docs": [],
        "task_docs": [],
        "rag_docs": [],
        "user_tasks": [],
        "example_conversations": [],
        "product_kwargs": {
            "tools": [],
            "workers": [],
        },
    }


@pytest.fixture
def mock_document_loader():
    """Create a mock document loader."""
    loader = MagicMock()
    loader.load_task_document.return_value = "docs"
    loader.load_instructions.return_value = "instructions"
    return loader


@pytest.fixture
def mock_task_generator():
    """Create a mock task generator."""
    generator = MagicMock()
    generator.add_provided_tasks.return_value = []
    generator.generate_tasks.return_value = []
    return generator


@pytest.fixture
def mock_best_practice_manager():
    """Create a mock best practice manager."""
    manager = MagicMock()
    manager.generate_best_practices.return_value = []
    return manager


@pytest.fixture
def mock_reusable_task_manager():
    """Create a mock reusable task manager."""
    manager = MagicMock()
    manager.generate_reusable_tasks.return_value = {}
    return manager


@pytest.fixture
def mock_task_graph_formatter():
    """Create a mock task graph formatter."""
    formatter = MagicMock()
    formatter.format_task_graph.return_value = {"nodes": [], "edges": []}
    formatter.ensure_nested_graph_connectivity.return_value = {"nodes": [], "edges": []}
    return formatter


# --- Test Functions ---


def test_generator_initialization(minimal_config, mock_model) -> None:
    """Test generator initialization with basic configuration."""
    gen = Generator(config=minimal_config, model=mock_model)
    assert gen.role == "test_role"
    assert gen.user_objective == "test_objective"
    assert gen.builder_objective == "test_builder_objective"
    assert gen.intro == "test_intro"
    assert isinstance(gen.resource_initializer, DefaultResourceInitializer)


def test_generator_generate_calls_components(
    minimal_config,
    mock_model,
    mock_document_loader,
    mock_task_generator,
    mock_best_practice_manager,
    mock_reusable_task_manager,
    mock_task_graph_formatter,
) -> None:
    """Test that generate method calls all required components."""
    gen = Generator(config=minimal_config, model=mock_model)
    with (
        patch.object(
            gen, "_initialize_document_loader", return_value=mock_document_loader
        ),
        patch.object(
            gen, "_initialize_task_generator", return_value=mock_task_generator
        ),
        patch.object(
            gen,
            "_initialize_best_practice_manager",
            return_value=mock_best_practice_manager,
        ),
        patch.object(
            gen,
            "_initialize_reusable_task_manager",
            return_value=mock_reusable_task_manager,
        ),
        patch.object(
            gen,
            "_initialize_task_graph_formatter",
            return_value=mock_task_graph_formatter,
        ),
        patch("arklex.orchestrator.generator.core.generator.UI_AVAILABLE", False),
    ):
        result = gen.generate()
        assert isinstance(result, dict)
        assert "nodes" in result and "edges" in result


def test_generator_save_task_graph(tmp_path, minimal_config, mock_model) -> None:
    """Test saving task graph to file."""
    gen = Generator(config=minimal_config, model=mock_model, output_dir=str(tmp_path))
    task_graph = {"nodes": [], "edges": []}
    output_path = gen.save_task_graph(task_graph)
    assert output_path.endswith(".json")
    import os

    assert os.path.exists(output_path)


def test_generator_with_invalid_resource_initializer(
    minimal_config, mock_model
) -> None:
    """Test generator fallback to DefaultResourceInitializer when None is provided."""
    gen = Generator(config=minimal_config, model=mock_model, resource_initializer=None)
    assert isinstance(gen.resource_initializer, DefaultResourceInitializer)


def test_generator_with_custom_resource_initializer(minimal_config, mock_model) -> None:
    """Test generator with custom resource initializer."""

    class CustomInitializer(BaseResourceInitializer):
        def init_workers(self, workers):
            return {"custom_worker": {}}

        def init_tools(self, tools):
            return {"custom_tool": {}}

    gen = Generator(
        config=minimal_config,
        model=mock_model,
        resource_initializer=CustomInitializer(),
    )
    assert isinstance(gen.resource_initializer, CustomInitializer)


def test_generator_document_instruction_type_conversion(
    minimal_config, mock_model
) -> None:
    """Test that documents and instructions are converted from lists to strings."""
    config_with_lists = minimal_config.copy()
    config_with_lists["task_docs"] = ["doc1.txt", "doc2.txt"]
    config_with_lists["instruction_docs"] = ["instruction1.txt", "instruction2.txt"]

    gen = Generator(config=config_with_lists, model=mock_model)
    mock_document_loader = MagicMock()
    mock_document_loader.load_task_document.side_effect = [
        "Document 1 content",
        "Document 2 content",
    ]
    mock_document_loader.load_instruction_document.side_effect = [
        "Instruction 1 content",
        "Instruction 2 content",
    ]

    with patch.object(
        gen, "_initialize_document_loader", return_value=mock_document_loader
    ):
        with (
            patch.object(gen, "_initialize_task_generator", return_value=MagicMock()),
            patch.object(
                gen, "_initialize_best_practice_manager", return_value=MagicMock()
            ),
            patch.object(
                gen, "_initialize_reusable_task_manager", return_value=MagicMock()
            ),
            patch.object(
                gen, "_initialize_task_graph_formatter", return_value=MagicMock()
            ),
            patch("arklex.orchestrator.generator.core.generator.UI_AVAILABLE", False),
        ):
            gen.generate()
            assert isinstance(gen.documents, str)
            assert isinstance(gen.instructions, str)
            assert "Document 1 content" in gen.documents
            assert "Document 2 content" in gen.documents
            assert "Instruction 1 content" in gen.instructions
            assert "Instruction 2 content" in gen.instructions
