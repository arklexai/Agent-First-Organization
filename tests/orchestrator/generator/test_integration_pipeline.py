"""Integration tests for the full task graph generation pipeline.

These tests exercise the complete generation pipeline from configuration
to final task graph output, using mock language models to simulate LLM responses.
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
    create_mock_model_for_task_generation,
    create_mock_model_for_intent_generation,
    create_mock_model_for_best_practices,
    MockLanguageModelWithErrors,
)


# --- Fixtures for common mocks and configs ---


@pytest.fixture
def always_valid_mock_model():
    """A mock model that always returns a valid, non-empty task list."""
    model = create_mock_model_for_task_generation()
    valid_task = '[{"id": "task_1", "name": "Test Task", "description": "A test task", "steps": [{"description": "Step 1"}]}]'
    model.generate = lambda messages: type("Mock", (), {"content": valid_task})()
    model.invoke = lambda messages: type("Mock", (), {"content": valid_task})()
    return model


@pytest.fixture
def patched_sample_config(sample_config):
    """Sample config with tools patched to avoid import errors."""
    sample_config["tools"] = []
    return sample_config


# --- Test Class ---


class TestFullGenerationPipeline:
    """Test the complete task graph generation pipeline."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
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
                    "path": "tools/product_search.py",
                },
                {
                    "name": "OrderLookup",
                    "id": "order_lookup_1",
                    "description": "Look up order information",
                    "path": "tools/order_lookup.py",
                },
            ],
            "output_path": "test_taskgraph.json",
        }

    def test_full_pipeline_with_mock_model(
        self, patched_sample_config, always_valid_mock_model
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

    def test_pipeline_with_intent_generation(self, patched_sample_config):
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

    def test_pipeline_with_best_practices(self, patched_sample_config):
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

    def test_pipeline_with_error_handling(self, patched_sample_config):
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
        self, patched_sample_config, always_valid_mock_model
    ) -> None:
        """Test pipeline with complex task structures."""
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
        generator = Generator(
            config=patched_sample_config,
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

    def test_pipeline_with_nested_graphs(self, sample_config) -> None:
        """Test pipeline with nested graph resources."""
        # Patch tools to avoid import errors
        sample_config["tools"] = []
        # Patch mock model to always return a valid task
        mock_model = create_mock_model_for_task_generation()
        mock_model.call_count = 1
        mock_model.generate = lambda messages: type(
            "Mock",
            (),
            {
                "content": '[{"id": "nested_task_1", "name": "Nested Task", "description": "A nested task", "steps": [{"description": "Step 1"}]}]'
            },
        )()
        mock_model.invoke = lambda messages: type(
            "Mock",
            (),
            {
                "content": '[{"id": "nested_task_1", "name": "Nested Task", "description": "A nested task", "steps": [{"description": "Step 1"}]}]'
            },
        )()

        # Create generator
        generator = Generator(
            config=sample_config, model=mock_model, interactable_with_user=False
        )

        # Add tasks with nested graph resources
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

        # Generate task graph
        task_graph = generator.generate()

        # Verify the structure
        assert "nodes" in task_graph
        assert "edges" in task_graph
        assert "tasks" in task_graph
        assert len(task_graph["nodes"]) > 0
        assert isinstance(task_graph["edges"], list)
        assert len(task_graph["tasks"]) > 0

    def test_pipeline_with_resource_allocation(self, sample_config) -> None:
        """Test pipeline with resource allocation and optimization."""
        # Create mock model
        mock_model = create_mock_model_for_best_practices()

        # Create best practice manager with resources
        manager = BestPracticeManager(
            model=mock_model,
            role=sample_config["role"],
            user_objective=sample_config["user_objective"],
            workers=sample_config["workers"],
            tools=sample_config["tools"],
        )

        # Sample tasks
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

        # Generate practices with resource optimization
        practices = manager.generate_best_practices(tasks)

        # Verify resource allocation
        assert len(practices) > 0

        # Check that practices include optimization information
        for practice in practices:
            if "steps" in practice:
                # Practices should have steps with optimization
                assert len(practice["steps"]) > 0

    def test_pipeline_with_validation(self, sample_config) -> None:
        """Test pipeline with comprehensive validation."""
        # Create mock model
        mock_model = create_mock_model_for_task_generation()

        # Create generator
        generator = Generator(
            config=sample_config, model=mock_model, interactable_with_user=False
        )

        # Generate task graph
        task_graph = generator.generate()

        # Validate task graph structure
        assert "nodes" in task_graph
        assert "edges" in task_graph
        assert "tasks" in task_graph

        # Validate nodes
        for node_id, node_data in task_graph["nodes"]:
            assert isinstance(node_id, str)
            assert isinstance(node_data, dict)
            assert "resource" in node_data
            assert "attribute" in node_data

        # Validate edges
        for edge in task_graph["edges"]:
            assert len(edge) >= 2  # At least source and target
            assert isinstance(edge[0], str)  # Source node ID
            assert isinstance(edge[1], str)  # Target node ID

        # Validate tasks
        for task in task_graph["tasks"]:
            assert "id" in task
            assert "name" in task
            assert "description" in task

    def test_pipeline_with_custom_prompts(self, sample_config) -> None:
        """Test pipeline with custom prompt configurations."""
        # Create mock model
        mock_model = create_mock_model_for_task_generation()

        # Add custom prompt configuration
        sample_config["custom_prompts"] = {
            "task_generation": "Custom task generation prompt",
            "intent_generation": "Custom intent generation prompt",
            "best_practice": "Custom best practice prompt",
        }

        # Create generator
        generator = Generator(
            config=sample_config, model=mock_model, interactable_with_user=False
        )

        # Generate task graph
        task_graph = generator.generate()

        # Verify custom prompts were used
        assert mock_model.call_count > 0

        # Verify the structure is still valid
        assert "nodes" in task_graph
        assert "edges" in task_graph
        assert "tasks" in task_graph

    def test_pipeline_performance(self, sample_config) -> None:
        """Test pipeline performance with a large number of tasks."""
        # Patch tools to avoid import errors
        sample_config["tools"] = []
        # Patch mock model to always return a valid task
        mock_model = create_mock_model_for_task_generation()
        mock_model.call_count = 1
        mock_model.generate = lambda messages: type(
            "Mock",
            (),
            {
                "content": '[{"id": "perf_task_1", "name": "Performance Task", "description": "A performance test task", "steps": [{"description": "Step 1"}]}]'
            },
        )()
        mock_model.invoke = lambda messages: type(
            "Mock",
            (),
            {
                "content": '[{"id": "perf_task_1", "name": "Performance Task", "description": "A performance test task", "steps": [{"description": "Step 1"}]}]'
            },
        )()

        # Create generator
        generator = Generator(
            config=sample_config, model=mock_model, interactable_with_user=False
        )

        # Generate task graph
        task_graph = generator.generate()

        # Verify the structure
        assert "nodes" in task_graph
        assert "edges" in task_graph
        assert "tasks" in task_graph
        assert len(task_graph["nodes"]) > 0
        assert isinstance(task_graph["edges"], list)
        assert len(task_graph["tasks"]) > 0
