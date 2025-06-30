"""Tests for the generator components.

This module provides comprehensive tests for the task generation, best practices,
and reusable task components of the Arklex framework.
"""

from unittest.mock import Mock, mock_open, patch

import pytest

from arklex.orchestrator.generator.core.generator import Generator
from arklex.orchestrator.generator.tasks.best_practice_manager import (
    BestPractice,
    BestPracticeManager,
)
from arklex.orchestrator.generator.tasks.reusable_task_manager import (
    ReusableTask,
    ReusableTaskManager,
)

# --- Mock Classes ---


class FakeResponse:
    """Mock response class for testing."""

    content = '[{"task": "Test", "intent": "Test intent", "description": "Test description", "steps": [{"task": "Step 1", "description": "Test step description"}]}]'
    text = content


class FakeGeneration:
    """Mock generation class for testing."""

    def __init__(self) -> None:
        self.text = '[{"task": "Test", "intent": "Test intent", "description": "Test description", "steps": [{"task": "Step 1", "description": "Test step description"}]}]'


class FakeGenerationResponse:
    """Mock generation response class for testing."""

    def __init__(self) -> None:
        self.generations = [[FakeGeneration()]]


class FakeResourceResponse:
    """Mock resource response class for testing."""

    content = '[{"step": 1, "task": "Step 1", "resource": "MessageWorker", "example_response": "Test response", "description": "Test step description"}]'
    text = content


# --- Fixtures ---


@pytest.fixture
def always_valid_mock_model() -> Mock:
    """Create a mock model that always returns valid responses."""
    mock_model = Mock()

    # Mock responses for different methods
    mock_response = Mock()
    mock_response.content = (
        '[{"task": "Step 1", "intent": "User inquires about step 1"}]'
    )
    mock_model.invoke.return_value = mock_response

    return mock_model


@pytest.fixture
def sample_objective() -> dict:
    """Create a sample objective for testing."""
    return {
        "description": "Create a new product in the store",
        "requirements": ["Product details", "Pricing information"],
        "constraints": ["Must follow store guidelines"],
    }


@pytest.fixture
def sample_tasks() -> list[dict]:
    """Create sample tasks for testing."""
    return [
        {
            "task_id": "task1",
            "name": "Gather product details",
            "description": "Collect all required product information",
            "steps": [
                {
                    "task": "Get product name",
                    "description": "Get the name of the product.",
                },
                {
                    "task": "Get product description",
                    "description": "Get the description of the product.",
                },
            ],
            "dependencies": [],
            "required_resources": ["Product form"],
            "estimated_duration": "30 minutes",
            "priority": 1,
        },
        {
            "task_id": "task2",
            "name": "Set product pricing",
            "description": "Determine product pricing strategy",
            "steps": [
                {
                    "task": "Research market prices",
                    "description": "Research market prices to determine the best pricing strategy.",
                },
                {
                    "task": "Set final price",
                    "description": "Set the final price based on the research.",
                },
            ],
            "dependencies": ["task1"],
            "required_resources": ["Pricing guide"],
            "estimated_duration": "45 minutes",
            "priority": 2,
        },
    ]


@pytest.fixture
def sample_practices() -> list[BestPractice]:
    """Create sample best practices for testing."""
    return [
        BestPractice(
            practice_id="practice1",
            name="Product validation",
            description="Validate product information before submission",
            steps=[
                {
                    "task": "Check required fields",
                    "description": "Check all required fields are filled out.",
                },
                {
                    "task": "Verify data format",
                    "description": "Verify the data format is correct.",
                },
            ],
            rationale="Ensure data quality",
            examples=["Product name validation", "Price format check"],
            priority=1,
            category="validation",
        )
    ]


@pytest.fixture
def task_generator(always_valid_mock_model: Mock) -> Generator:
    """Create a TaskGenerator instance for testing."""
    config = {
        "role": "product_manager",
        "user_objective": "Create a new product",
        "instruction_docs": [],
        "task_docs": [],
        "workers": [],
        "tools": [],
    }
    return Generator(
        config=config,
        model=always_valid_mock_model,
    )


@pytest.fixture
def best_practice_manager(always_valid_mock_model: Mock) -> BestPracticeManager:
    """Create a BestPracticeManager instance for testing."""
    return BestPracticeManager(
        model=always_valid_mock_model,
        role="product_manager",
        user_objective="Create a new product",
    )


@pytest.fixture
def reusable_task_manager(always_valid_mock_model: Mock) -> ReusableTaskManager:
    """Create a ReusableTaskManager instance for testing."""
    return ReusableTaskManager(
        model=always_valid_mock_model,
        role="product_manager",
        user_objective="Create a new product",
    )


@pytest.fixture
def reusable_template() -> ReusableTask:
    """Create a reusable task template for testing."""
    return ReusableTask(
        template_id="tmpl1",
        name="Reusable Template",
        description="Reusable template description",
        parameters={"param1": "string", "param2": "string"},
        steps=[
            {"task": "Step 1", "description": "Step 1 desc"},
            {"task": "Step 2", "description": "Step 2 desc"},
        ],
        examples=[],
        version="1.0",
        category="test",
    )


@pytest.fixture
def sample_tasks_dict() -> list[dict]:
    """Sample tasks as dicts for reusable task manager tests."""
    return [
        {
            "task_id": "task1",
            "name": "Gather product details",
            "description": "Collect all required product information",
            "steps": [
                {
                    "task": "Get product name",
                    "description": "Get the name of the product.",
                },
                {
                    "task": "Get product description",
                    "description": "Get the description of the product.",
                },
            ],
            "dependencies": [],
            "required_resources": ["Product form"],
            "estimated_duration": "30 minutes",
            "priority": 1,
        },
        {
            "task_id": "task2",
            "name": "Set product pricing",
            "description": "Determine product pricing strategy",
            "steps": [
                {
                    "task": "Research market prices",
                    "description": "Research market prices to determine the best pricing strategy.",
                },
                {
                    "task": "Set final price",
                    "description": "Set the final price based on the research.",
                },
            ],
            "dependencies": ["task1"],
            "required_resources": ["Pricing guide"],
            "estimated_duration": "45 minutes",
            "priority": 2,
        },
    ]


@pytest.fixture
def sample_practices_dict() -> list[dict]:
    """Sample best practices as dicts for best practice manager tests."""
    return [
        {
            "practice_id": "practice1",
            "name": "Product validation",
            "description": "Validate product information before submission",
            "steps": [
                {
                    "task": "Check required fields",
                    "description": "Check all required fields are filled out.",
                },
                {
                    "task": "Verify data format",
                    "description": "Verify the data format is correct.",
                },
            ],
            "rationale": "Ensure data quality",
            "examples": ["Product name validation", "Price format check"],
            "priority": 1,
            "category": "validation",
        }
    ]


# --- Test Classes ---


class TestGenerator:
    """Test cases for the main Generator class."""

    def test_save_task_graph_with_non_serializable_objects(self) -> None:
        """Test save_task_graph with non-serializable objects like functools.partial and callables."""
        import collections.abc
        import functools

        config = {
            "role": "test_role",
            "user_objective": "test_objective",
            "instruction_docs": "test_instructions",
            "task_docs": "test_documents",
            "workers": [],
            "tools": [],
        }
        generator = Generator(
            config=config,
            model=Mock(),
            output_dir="/tmp",  # Add output_dir to avoid None error
        )

        # Create a task graph with non-serializable objects
        task_graph = {
            "nodes": {"node1": {"data": "normal_data"}},
            "partial_func": functools.partial(lambda x: x * 2, 5),
            "callable_obj": lambda x: x + 1,
            "custom_callable": collections.abc.Callable,
            "normal_data": {"key": "value"},
        }

        with (
            patch("builtins.open", mock_open()) as mock_file,
            patch("json.dump") as mock_json_dump,
        ):
            generator.save_task_graph(task_graph)

            # Check that the file was opened and json.dump was called
            mock_file.assert_called_once()
            mock_json_dump.assert_called_once()

            # Check that the sanitized data was passed to json.dump
            call_args = mock_json_dump.call_args[0]
            sanitized_data = call_args[0]

            # Non-serializable objects should be converted to strings
            assert isinstance(sanitized_data["partial_func"], str)
            assert "functools.partial" in sanitized_data["partial_func"]
            assert isinstance(sanitized_data["callable_obj"], str)
            assert "<function" in sanitized_data["callable_obj"]
            assert isinstance(sanitized_data["custom_callable"], str)
            assert "collections.abc.Callable" in sanitized_data["custom_callable"]
            assert sanitized_data["normal_data"] == {"key": "value"}


def test_integration_generation_pipeline(always_valid_mock_model: Mock) -> None:
    """Integration test for the full generation pipeline."""
    config = {
        "role": "product_manager",
        "user_objective": "Create a new product",
        "instruction_docs": [],
        "task_docs": [],
        "workers": [],
        "tools": [],
    }
    task_gen = Generator(
        config=config,
        model=always_valid_mock_model,
    )

    # Test the full pipeline
    result = task_gen.generate()

    assert isinstance(result, dict)
    assert "tasks" in result
    assert "reusable_tasks" in result
    # The result contains the task graph directly, not under a "task_graph" key
    assert "nodes" in result or "edges" in result
