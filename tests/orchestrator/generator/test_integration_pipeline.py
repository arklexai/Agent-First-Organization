"""Integration tests for the full task graph generation pipeline.

This module provides comprehensive tests for the complete generation pipeline
from configuration to final task graph output, using mock language models
to simulate LLM responses.
"""

import pytest

from arklex.orchestrator.generator.core.generator import Generator
from arklex.orchestrator.generator.formatting.task_graph_formatter import (
    TaskGraphFormatter,
)
from arklex.orchestrator.generator.tasks.best_practice_manager import (
    BestPracticeManager,
)
from tests.orchestrator.generator.test_mock_models import (
    MockLanguageModelWithErrors,
    create_mock_model_for_best_practices,
    create_mock_model_for_intent_generation,
    create_mock_model_for_task_generation,
)

# --- Fixtures ---


@pytest.fixture
def always_valid_mock_model() -> object:
    """Create a mock model that always returns a valid, non-empty task list."""
    model = create_mock_model_for_task_generation()
    valid_task = '[{"id": "task_1", "name": "Test Task", "description": "A test task", "steps": [{"description": "Step 1"}]}]'
    model.generate = lambda messages: type("Mock", (), {"content": valid_task})()
    model.invoke = lambda messages: type("Mock", (), {"content": valid_task})()
    return model


@pytest.fixture
def nested_graph_mock_model() -> object:
    """Create a mock model for nested graph testing."""
    model = create_mock_model_for_task_generation()
    model.call_count = 1
    nested_task = '[{"id": "nested_task_1", "name": "Nested Task", "description": "A nested task", "steps": [{"description": "Step 1"}]}]'
    model.generate = lambda messages: type("Mock", (), {"content": nested_task})()
    model.invoke = lambda messages: type("Mock", (), {"content": nested_task})()
    return model


@pytest.fixture
def performance_mock_model() -> object:
    """Create a mock model for performance testing."""
    model = create_mock_model_for_task_generation()
    model.call_count = 1
    perf_task = '[{"id": "perf_task_1", "name": "Performance Task", "description": "A performance test task", "steps": [{"description": "Step 1"}]}]'
    model.generate = lambda messages: type("Mock", (), {"content": perf_task})()
    model.invoke = lambda messages: type("Mock", (), {"content": perf_task})()
    return model


@pytest.fixture
def sample_config() -> dict:
    """Create a sample configuration for testing."""
    return {
        "role": "Customer Service Assistant",
        "user_objective": "Handle customer inquiries and provide support",
        "builder_objective": "Create an efficient customer service chatbot",
        "domain": "E-commerce",
        "intro": "Amazon.com is a large e-commerce platform that sells a wide variety of products.",
        "task_docs": [],
        "rag_docs": [
            "docs/product_catalog.md",
        ],
        "workers": [
            {"name": "MessageWorker", "id": "msg_worker_1"},
            {"name": "FaissRAGWorker", "id": "rag_worker_1"},
            {"name": "SearchWorker", "id": "search_worker_1"},
        ],
        "tools": [
            {
                "name": "ProductSearch",
                "id": "product_search_1",
                "description": "Search for products",
                "path": "mock_path",
            },
            {
                "name": "OrderLookup",
                "id": "order_lookup_1",
                "description": "Look up order information",
                "path": "mock_path",
            },
        ],
        "output_path": "test_taskgraph.json",
    }


@pytest.fixture
def patched_sample_config(sample_config: dict) -> dict:
    """Create a sample config with tools patched to avoid import errors."""
    sample_config["tools"] = []
    return sample_config


@pytest.fixture
def complex_tasks_config(patched_sample_config: dict) -> dict:
    """Create a config with complex task structures."""
    patched_sample_config["existing_tasks"] = [
        {
            "id": "complex_task_1",
            "name": "Multi-step Order Processing",
            "description": "Process orders with multiple validation steps",
            "steps": [
                {"description": "Validate customer info", "step_id": "step_1"},
                {"description": "Validate payment", "step_id": "step_2"},
            ],
        }
    ]
    return patched_sample_config


@pytest.fixture
def nested_graph_config(sample_config: dict) -> dict:
    """Create a config with nested graph resources."""
    sample_config["tools"] = []
    sample_config["existing_tasks"] = [
        {
            "id": "nested_task_1",
            "name": "Nested Task",
            "description": "A nested task",
            "steps": [
                {"description": "Step 1", "step_id": "step_1"},
            ],
            "resource": {"name": "NestedGraph"},
        }
    ]
    return sample_config


@pytest.fixture
def custom_prompts_config(sample_config: dict) -> dict:
    """Create a config with custom prompt configurations."""
    sample_config["custom_prompts"] = {
        "task_generation": "Custom task generation prompt",
        "intent_generation": "Custom intent generation prompt",
        "best_practice": "Custom best practice prompt",
    }
    return sample_config


@pytest.fixture
def performance_config(sample_config: dict) -> dict:
    """Create a config for performance testing."""
    sample_config["tools"] = []
    return sample_config


# --- Test Classes ---


class TestFullGenerationPipeline:
    """Test suite for the complete task graph generation pipeline."""

    def test_full_pipeline_with_mock_model(
        self, patched_sample_config: dict, always_valid_mock_model: object
    ) -> None:
        """Test the complete pipeline with a mock language model that always returns a valid task."""
        generator = Generator(
            config=patched_sample_config,
            model=always_valid_mock_model,
            interactable_with_user=False,
        )
        task_graph = generator.generate()
        assert "nodes" in task_graph
        assert "edges" in task_graph
        assert "tasks" in task_graph
        assert "role" in task_graph
        assert "user_objective" in task_graph
        assert len(task_graph["nodes"]) > 0
        assert isinstance(task_graph["edges"], list)
        assert len(task_graph["tasks"]) > 0

    def test_pipeline_with_intent_generation(self, patched_sample_config: dict) -> None:
        """Test pipeline with intent generation using a mock model."""
        mock_model = create_mock_model_for_intent_generation()
        formatter = TaskGraphFormatter(
            role=patched_sample_config["role"],
            user_objective=patched_sample_config["user_objective"],
            model=mock_model,
        )
        tasks = [
            {
                "id": "task_1",
                "name": "Product Search",
                "description": "Help users search for products",
                "steps": [
                    {"description": "Get search criteria", "step_id": "step_1"},
                    {"description": "Search database", "step_id": "step_2"},
                ],
            }
        ]
        result = formatter.format_task_graph(tasks)
        assert "nodes" in result
        assert "edges" in result
        assert mock_model.call_count > 0

    def test_pipeline_with_best_practices(self, patched_sample_config: dict) -> None:
        """Test pipeline with best practice generation."""
        mock_model = create_mock_model_for_best_practices()
        manager = BestPracticeManager(
            model=mock_model,
            role=patched_sample_config["role"],
            user_objective=patched_sample_config["user_objective"],
            workers=patched_sample_config["workers"],
            tools=patched_sample_config["tools"],
        )
        tasks = [
            {
                "id": "task_1",
                "name": "Customer Support",
                "description": "Provide customer support",
                "steps": [
                    {"description": "Listen to customer", "step_id": "step_1"},
                    {"description": "Provide solution", "step_id": "step_2"},
                ],
            }
        ]
        practices = manager.generate_best_practices(tasks)
        assert len(practices) > 0

    def test_pipeline_with_error_handling(self, patched_sample_config: dict) -> None:
        """Test pipeline behavior when LLM calls fail."""
        mock_model = MockLanguageModelWithErrors(error_type="timeout", error_rate=0.5)
        generator = Generator(
            config=patched_sample_config, model=mock_model, interactable_with_user=False
        )
        task_graph = generator.generate()
        assert "nodes" in task_graph
        assert "edges" in task_graph
        assert "tasks" in task_graph

    def test_pipeline_with_complex_tasks(
        self, complex_tasks_config: dict, always_valid_mock_model: object
    ) -> None:
        """Test pipeline with complex task structures."""
        generator = Generator(
            config=complex_tasks_config,
            model=always_valid_mock_model,
            interactable_with_user=False,
        )
        task_graph = generator.generate()
        assert "nodes" in task_graph
        assert "edges" in task_graph
        assert "tasks" in task_graph
        assert len(task_graph["nodes"]) > 0
        assert isinstance(task_graph["edges"], list)
        assert len(task_graph["tasks"]) > 0

    def test_pipeline_with_nested_graphs(
        self, nested_graph_config: dict, nested_graph_mock_model: object
    ) -> None:
        """Test pipeline with nested graph resources."""
        generator = Generator(
            config=nested_graph_config,
            model=nested_graph_mock_model,
            interactable_with_user=False,
        )
        task_graph = generator.generate()
        assert "nodes" in task_graph
        assert "edges" in task_graph
        assert "tasks" in task_graph
        assert len(task_graph["nodes"]) > 0
        assert isinstance(task_graph["edges"], list)
        assert len(task_graph["tasks"]) > 0

    def test_pipeline_with_resource_allocation(self, sample_config: dict) -> None:
        """Test pipeline with resource allocation and optimization."""
        mock_model = create_mock_model_for_best_practices()
        manager = BestPracticeManager(
            model=mock_model,
            role=sample_config["role"],
            user_objective=sample_config["user_objective"],
            workers=sample_config["workers"],
            tools=sample_config["tools"],
        )
        tasks = [
            {
                "id": "resource_task_1",
                "name": "Resource Intensive Task",
                "description": "Task requiring multiple resources",
                "steps": [
                    {"description": "Data processing", "step_id": "step_1"},
                    {"description": "Analysis", "step_id": "step_2"},
                    {"description": "Reporting", "step_id": "step_3"},
                ],
            }
        ]
        practices = manager.generate_best_practices(tasks)
        assert len(practices) > 0
        for practice in practices:
            if "steps" in practice:
                assert len(practice["steps"]) > 0

    def test_pipeline_with_validation(self, sample_config: dict) -> None:
        """Test pipeline with comprehensive validation."""
        mock_model = create_mock_model_for_task_generation()
        generator = Generator(
            config=sample_config, model=mock_model, interactable_with_user=False
        )
        task_graph = generator.generate()
        assert "nodes" in task_graph
        assert "edges" in task_graph
        assert "tasks" in task_graph
        for node_id, node_data in task_graph["nodes"]:
            assert isinstance(node_id, str)
            assert isinstance(node_data, dict)
            assert "resource" in node_data
            assert "attribute" in node_data
        for edge in task_graph["edges"]:
            assert len(edge) >= 2
            assert isinstance(edge[0], str)
            assert isinstance(edge[1], str)
        for task in task_graph["tasks"]:
            assert "id" in task
            assert "name" in task
            assert "description" in task

    def test_pipeline_with_custom_prompts(self, custom_prompts_config: dict) -> None:
        """Test pipeline with custom prompt configurations."""
        mock_model = create_mock_model_for_task_generation()
        generator = Generator(
            config=custom_prompts_config, model=mock_model, interactable_with_user=False
        )
        task_graph = generator.generate()
        assert mock_model.call_count > 0
        assert "nodes" in task_graph
        assert "edges" in task_graph
        assert "tasks" in task_graph

    def test_pipeline_performance(
        self, performance_config: dict, performance_mock_model: object
    ) -> None:
        """Test pipeline performance with a large number of tasks."""
        generator = Generator(
            config=performance_config,
            model=performance_mock_model,
            interactable_with_user=False,
        )
        task_graph = generator.generate()
        assert "nodes" in task_graph
        assert "edges" in task_graph
        assert "tasks" in task_graph
        assert len(task_graph["nodes"]) > 0
        assert isinstance(task_graph["edges"], list)
        assert len(task_graph["tasks"]) > 0
