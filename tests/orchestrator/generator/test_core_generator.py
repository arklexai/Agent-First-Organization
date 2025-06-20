import pytest
from unittest.mock import Mock, patch
from arklex.orchestrator.generator.core.generator import Generator
from arklex.env.env import BaseResourceInitializer, DefaultResourceInitializer


@pytest.fixture
def mock_model():
    model = Mock()
    return model


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


def test_generator_initialization(minimal_config, mock_model) -> None:
    gen = Generator(config=minimal_config, model=mock_model)
    assert gen.role == "test_role"
    assert gen.user_objective == "test_objective"
    assert gen.builder_objective == "test_builder_objective"
    assert gen.intro == "test_intro"
    assert isinstance(gen.resource_initializer, DefaultResourceInitializer)


def test_generator_generate_calls_components(minimal_config, mock_model) -> None:
    gen = Generator(config=minimal_config, model=mock_model)
    # Patch component initializers to return mocks and UI_AVAILABLE to False
    with (
        patch.object(gen, "_initialize_document_loader") as doc_loader_patch,
        patch.object(gen, "_initialize_task_generator") as task_gen_patch,
        patch.object(gen, "_initialize_best_practice_manager") as bpm_patch,
        patch.object(gen, "_initialize_reusable_task_manager") as rtm_patch,
        patch.object(gen, "_initialize_task_graph_formatter") as tgf_patch,
        patch("arklex.orchestrator.generator.core.generator.UI_AVAILABLE", False),
    ):
        doc_loader = Mock()
        doc_loader.load_task_document.return_value = "docs"
        doc_loader.load_instructions.return_value = "instructions"
        doc_loader_patch.return_value = doc_loader
        task_gen = Mock()
        task_gen.add_provided_tasks.return_value = []
        task_gen.generate_tasks.return_value = []
        task_gen_patch.return_value = task_gen
        bpm = Mock()
        bpm.generate_best_practices.return_value = []
        bpm_patch.return_value = bpm
        rtm = Mock()
        rtm.generate_reusable_tasks.return_value = {}
        rtm_patch.return_value = rtm
        tgf = Mock()
        tgf.format_task_graph.return_value = {"nodes": [], "edges": []}
        tgf_patch.return_value = tgf
        result = gen.generate()
        assert isinstance(result, dict)
        assert "nodes" in result and "edges" in result


def test_generator_save_task_graph(tmp_path, minimal_config, mock_model) -> None:
    gen = Generator(config=minimal_config, model=mock_model, output_dir=str(tmp_path))
    task_graph = {"nodes": [], "edges": []}
    output_path = gen.save_task_graph(task_graph)
    assert output_path.endswith(".json")
    import os

    assert os.path.exists(output_path)


def test_generator_with_invalid_resource_initializer(
    minimal_config, mock_model
) -> None:
    # Should fallback to DefaultResourceInitializer if None is provided
    gen = Generator(config=minimal_config, model=mock_model, resource_initializer=None)
    assert isinstance(gen.resource_initializer, DefaultResourceInitializer)


def test_generator_with_custom_resource_initializer(minimal_config, mock_model) -> None:
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
    # Configure with list-based documents and instructions
    config_with_lists = minimal_config.copy()
    config_with_lists["task_docs"] = ["doc1.txt", "doc2.txt"]
    config_with_lists["instruction_docs"] = ["instruction1.txt", "instruction2.txt"]

    gen = Generator(config=config_with_lists, model=mock_model)

    # Mock the document loader to return lists
    with patch.object(gen, "_initialize_document_loader") as doc_loader_patch:
        doc_loader = Mock()
        doc_loader.load_task_document.side_effect = [
            "Document 1 content",
            "Document 2 content",
        ]
        doc_loader.load_instruction_document.side_effect = [
            "Instruction 1 content",
            "Instruction 2 content",
        ]
        doc_loader_patch.return_value = doc_loader

        # Mock other components to avoid actual processing
        with (
            patch.object(gen, "_initialize_task_generator") as task_gen_patch,
            patch.object(gen, "_initialize_best_practice_manager") as bpm_patch,
            patch.object(gen, "_initialize_reusable_task_manager") as rtm_patch,
            patch.object(gen, "_initialize_task_graph_formatter") as tgf_patch,
            patch("arklex.orchestrator.generator.core.generator.UI_AVAILABLE", False),
        ):
            task_gen = Mock()
            task_gen.add_provided_tasks.return_value = []
            task_gen.generate_tasks.return_value = []
            task_gen_patch.return_value = task_gen

            bpm = Mock()
            bpm.generate_best_practices.return_value = []
            bpm_patch.return_value = bpm

            rtm = Mock()
            rtm.generate_reusable_tasks.return_value = {}
            rtm_patch.return_value = rtm

            tgf = Mock()
            tgf.format_task_graph.return_value = {"nodes": [], "edges": []}
            tgf_patch.return_value = tgf

            # Call generate to trigger the type conversion
            gen.generate()

            # Verify that documents and instructions are now strings
            assert isinstance(gen.documents, str)
            assert isinstance(gen.instructions, str)

            # Verify the content contains both documents/instructions
            assert "Document 1 content" in gen.documents
            assert "Document 2 content" in gen.documents
            assert "Instruction 1 content" in gen.instructions
            assert "Instruction 2 content" in gen.instructions
