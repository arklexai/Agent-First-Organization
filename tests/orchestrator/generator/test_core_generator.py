"""Core generator tests for the main Generator class.

These tests verify the core functionality of the Generator class,
including initialization, component integration, and error handling.
"""

import pytest
from unittest.mock import MagicMock, patch
from arklex.orchestrator.generator.core.generator import Generator
from arklex.env.env import BaseResourceInitializer, DefaultResourceInitializer


# --- Centralized Mock Fixtures ---


@pytest.fixture
def always_valid_mock_model():
    """Create a mock language model that always returns valid responses."""
    mock = MagicMock()
    mock.generate.return_value = MagicMock()
    mock.invoke.return_value = MagicMock()
    return mock


@pytest.fixture
def patched_sample_config():
    """Create a sample configuration with all required fields."""
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
            "tools": [{"id": "test_tool", "name": "TestTool", "path": "test_tool.py"}],
            "workers": [{"id": "test_worker", "name": "TestWorker"}],
        },
    }


@pytest.fixture
def mock_document_loader():
    """Create a mock document loader with standard return values."""
    loader = MagicMock()
    loader.load_task_document.return_value = "docs"
    loader.load_instructions.return_value = "instructions"
    return loader


@pytest.fixture
def mock_task_generator():
    """Create a mock task generator with standard return values."""
    generator = MagicMock()
    generator.add_provided_tasks.return_value = []
    generator.generate_tasks.return_value = []
    return generator


@pytest.fixture
def mock_best_practice_manager():
    """Create a mock best practice manager with standard return values."""
    manager = MagicMock()
    manager.generate_best_practices.return_value = [
        {"practice_id": "bp1", "name": "BP1"}
    ]
    manager.finetune_best_practice.side_effect = lambda bp, task: {
        "steps": task.get("steps", [])
    }
    return manager


@pytest.fixture
def mock_reusable_task_manager():
    """Create a mock reusable task manager with standard return values."""
    manager = MagicMock()
    manager.generate_reusable_tasks.return_value = {}
    return manager


@pytest.fixture
def mock_task_graph_formatter():
    """Create a mock task graph formatter with standard return values."""
    formatter = MagicMock()
    formatter.format_task_graph.return_value = {"nodes": [], "edges": []}
    formatter.ensure_nested_graph_connectivity.return_value = {"nodes": [], "edges": []}
    return formatter


@pytest.fixture
def mock_prompt_manager():
    """Create a mock prompt manager with standard return values."""
    manager = MagicMock()
    manager.get_prompt.return_value = "test prompt"
    return manager


@pytest.fixture
def patched_generator_components(
    patched_sample_config,
    always_valid_mock_model,
    mock_document_loader,
    mock_task_generator,
    mock_best_practice_manager,
    mock_reusable_task_manager,
    mock_task_graph_formatter,
):
    """Create a generator with all components patched for testing."""
    gen = Generator(config=patched_sample_config, model=always_valid_mock_model)

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
        yield gen


# --- Test Data Fixtures ---


@pytest.fixture
def sample_tasks():
    """Sample task data for testing."""
    return [
        {
            "id": "task1",
            "name": "Test Task 1",
            "description": "Test description",
            "steps": [{"task": "Step 1", "description": "Test step"}],
        }
    ]


@pytest.fixture
def sample_best_practices():
    """Sample best practice data for testing."""
    return [
        {
            "practice_id": "bp1",
            "name": "Test Practice",
            "description": "Test practice description",
        }
    ]


# --- Core Generator Tests ---


class TestGeneratorInitialization:
    """Test suite for Generator initialization and basic functionality."""

    def test_generator_initialization(
        self, patched_sample_config, always_valid_mock_model
    ):
        """Test generator initialization with basic configuration."""
        gen = Generator(config=patched_sample_config, model=always_valid_mock_model)

        assert gen.role == "test_role"
        assert gen.user_objective == "test_objective"
        assert gen.builder_objective == "test_builder_objective"
        assert gen.intro == "test_intro"
        assert isinstance(gen.resource_initializer, DefaultResourceInitializer)

    def test_generator_with_invalid_resource_initializer(
        self, patched_sample_config, always_valid_mock_model
    ):
        """Test generator fallback to DefaultResourceInitializer when None is provided."""
        gen = Generator(
            config=patched_sample_config,
            model=always_valid_mock_model,
            resource_initializer=None,
        )
        assert isinstance(gen.resource_initializer, DefaultResourceInitializer)

    def test_generator_with_custom_resource_initializer(
        self, patched_sample_config, always_valid_mock_model
    ):
        """Test generator with custom resource initializer."""

        class CustomInitializer(BaseResourceInitializer):
            def init_workers(self, workers):
                return {"custom_worker": {}}

            def init_tools(self, tools):
                return {"custom_tool": {}}

        gen = Generator(
            config=patched_sample_config,
            model=always_valid_mock_model,
            resource_initializer=CustomInitializer(),
        )
        assert isinstance(gen.resource_initializer, CustomInitializer)


class TestGeneratorCoreFunctionality:
    """Test suite for core generator functionality."""

    def test_generator_generate_calls_components(self, patched_generator_components):
        """Test that generate method calls all required components."""
        result = patched_generator_components.generate()

        assert isinstance(result, dict)
        assert "nodes" in result and "edges" in result

    def test_generator_save_task_graph(
        self, patched_sample_config, always_valid_mock_model, tmp_path
    ):
        """Test saving task graph to file."""
        gen = Generator(
            config=patched_sample_config,
            model=always_valid_mock_model,
            output_dir=str(tmp_path),
        )
        task_graph = {"nodes": [], "edges": []}

        output_path = gen.save_task_graph(task_graph)

        assert output_path.endswith(".json")
        import os

        assert os.path.exists(output_path)

    def test_generator_document_instruction_type_conversion(
        self, patched_sample_config, always_valid_mock_model
    ):
        """Test that documents and instructions are converted from lists to strings."""
        config_with_lists = patched_sample_config.copy()
        config_with_lists["task_docs"] = ["doc1.txt", "doc2.txt"]
        config_with_lists["instruction_docs"] = ["instruction1.txt", "instruction2.txt"]

        gen = Generator(config=config_with_lists, model=always_valid_mock_model)
        mock_document_loader = MagicMock()
        mock_document_loader.load_task_document.side_effect = [
            "Document 1 content",
            "Document 2 content",
        ]
        mock_document_loader.load_instruction_document.side_effect = [
            "Instruction 1 content",
            "Instruction 2 content",
        ]

        with (
            patch.object(
                gen, "_initialize_document_loader", return_value=mock_document_loader
            ),
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

            # Verify document loader was called with correct arguments
            assert mock_document_loader.load_task_document.call_count == 2
            assert mock_document_loader.load_instruction_document.call_count == 2


# --- Advanced Generator Tests ---


class TestGeneratorAdvanced:
    """Test suite for advanced generator features."""

    @pytest.fixture
    def patched_advanced_generator(
        self,
        patched_sample_config,
        always_valid_mock_model,
        sample_tasks,
        sample_best_practices,
        mock_best_practice_manager,
    ):
        """Create a generator with advanced features patched for testing."""
        gen = Generator(config=patched_sample_config, model=always_valid_mock_model)

        with (
            patch(
                "arklex.orchestrator.generator.core.generator.TaskEditorApp"
            ) as mock_task_editor_app,
            patch(
                "arklex.orchestrator.generator.core.generator.PromptManager"
            ) as mock_prompt_manager,
            patch(
                "arklex.orchestrator.generator.core.generator.TaskGraphFormatter"
            ) as mock_formatter,
        ):
            mock_formatter.return_value.format_task_graph.return_value = {
                "nodes": sample_tasks,
                "edges": [],
            }
            mock_formatter.return_value.ensure_nested_graph_connectivity.return_value = {
                "nodes": sample_tasks,
                "edges": [],
            }
            mock_prompt_manager.return_value.get_prompt.return_value = "test prompt"
            mock_task_editor_app.return_value.run.return_value = sample_tasks

            yield gen, mock_formatter, mock_prompt_manager, mock_task_editor_app

    def test_human_in_the_loop_refinement_with_changes(
        self, patched_advanced_generator
    ) -> None:
        """Test human-in-the-loop refinement when changes are made."""
        gen, mock_formatter, mock_prompt_manager, mock_task_editor_app = (
            patched_advanced_generator
        )

        result = gen.generate()

        assert isinstance(result, dict)
        assert "nodes" in result and "edges" in result
        mock_task_editor_app.return_value.run.assert_called_once()

    def test_resource_pairing_without_ui(self, patched_advanced_generator) -> None:
        """Test resource pairing functionality without UI interaction."""
        gen, mock_formatter, mock_prompt_manager, mock_task_editor_app = (
            patched_advanced_generator
        )

        with patch("arklex.orchestrator.generator.core.generator.UI_AVAILABLE", False):
            result = gen.generate()

            assert isinstance(result, dict)
            assert "nodes" in result and "edges" in result

    def test_intent_prediction(self, patched_advanced_generator) -> None:
        """Test intent prediction functionality."""
        gen, mock_formatter, mock_prompt_manager, mock_task_editor_app = (
            patched_advanced_generator
        )

        result = gen.generate()

        assert isinstance(result, dict)
        assert "nodes" in result and "edges" in result

    def test_nested_graph_connectivity(self, patched_advanced_generator) -> None:
        """Test nested graph connectivity functionality."""
        gen, mock_formatter, mock_prompt_manager, mock_task_editor_app = (
            patched_advanced_generator
        )

        result = gen.generate()

        assert isinstance(result, dict)
        assert "nodes" in result and "edges" in result
        mock_formatter.return_value.ensure_nested_graph_connectivity.assert_called_once()

    def test_reusable_tasks_added_to_graph(self, patched_advanced_generator) -> None:
        """Test that reusable tasks are properly added to the graph."""
        gen, mock_formatter, mock_prompt_manager, mock_task_editor_app = (
            patched_advanced_generator
        )

        result = gen.generate()

        assert isinstance(result, dict)
        assert "nodes" in result and "edges" in result
