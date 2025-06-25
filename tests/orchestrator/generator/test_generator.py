"""Tests for the generator components.

This module provides comprehensive tests for the task generation, best practices,
and reusable task components of the Arklex framework.
"""

import dataclasses
from typing import Dict, List

import pytest
from unittest.mock import Mock

from arklex.orchestrator.generator.tasks.task_generator import (
    TaskGenerator,
    TaskDefinition,
)
from arklex.orchestrator.generator.tasks.best_practice_manager import (
    BestPracticeManager,
    BestPractice,
)
from arklex.orchestrator.generator.tasks.reusable_task_manager import (
    ReusableTaskManager,
    ReusableTask,
)


# --- Mock Classes ---


class FakeResponse:
    """Mock response class for testing."""

    content = '[{"task": "Test", "intent": "Test intent", "description": "Test description", "steps": [{"task": "Step 1", "description": "Test step description"}]}]'
    text = content


class FakeGeneration:
    """Mock generation class for testing."""

    def __init__(self):
        self.text = '[{"task": "Test", "intent": "Test intent", "description": "Test description", "steps": [{"task": "Step 1", "description": "Test step description"}]}]'


class FakeGenerationResponse:
    """Mock generation response class for testing."""

    def __init__(self):
        self.generations = [[FakeGeneration()]]


class FakeResourceResponse:
    """Mock resource response class for testing."""

    content = '[{"step": 1, "task": "Step 1", "resource": "MessageWorker", "example_response": "Test response", "description": "Test step description"}]'
    text = content


# --- Fixtures ---


@pytest.fixture
def always_valid_mock_model() -> Mock:
    """Create a mock language model that always returns valid responses."""
    mock_model = Mock()
    mock_model.generate.return_value = FakeGenerationResponse()
    mock_model.invoke.return_value = FakeResourceResponse()
    return mock_model


@pytest.fixture
def sample_objective() -> Dict:
    """Create a sample objective for testing."""
    return {
        "description": "Create a new product in the store",
        "requirements": ["Product details", "Pricing information"],
        "constraints": ["Must follow store guidelines"],
    }


@pytest.fixture
def sample_tasks() -> List[Dict]:
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
def sample_practices() -> List[BestPractice]:
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
def task_generator(always_valid_mock_model: Mock) -> TaskGenerator:
    """Create a TaskGenerator instance for testing."""
    return TaskGenerator(
        model=always_valid_mock_model,
        role="product_manager",
        user_objective="Create a new product",
        instructions=[],
        documents=[],
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
def sample_tasks_dict() -> List[Dict]:
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
def sample_practices_dict() -> List[Dict]:
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


class TestTaskGenerator:
    """Test TaskGenerator core logic and validation."""

    def test_generate_tasks(self, task_generator: TaskGenerator) -> None:
        """Should generate a list of tasks from objectives."""
        tasks = task_generator.generate_tasks(
            intro="Create a new product", existing_tasks=[]
        )
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        assert all(isinstance(task, dict) for task in tasks)
        assert all("id" in task for task in tasks)

    def test_add_provided_tasks(self, task_generator: TaskGenerator) -> None:
        """Should add user-provided tasks to the task list."""
        user_tasks = [
            {
                "name": "Custom task",
                "description": "User-defined task",
                "steps": [
                    {"task": "Custom step", "description": "Custom step description"}
                ],
            }
        ]
        tasks = task_generator.add_provided_tasks(
            user_tasks=user_tasks, intro="Add custom task"
        )
        assert isinstance(tasks, list)
        assert len(tasks) == 1
        assert tasks[0]["name"] == "Custom task"

    def test_validate_tasks(self, task_generator: TaskGenerator) -> None:
        """Should validate and convert TaskDefinition objects to dicts."""
        task_definitions = [
            TaskDefinition(
                task_id="test1",
                name="Test task",
                description="Test description",
                steps=[{"task": "Test step", "description": "Test step description"}],
                dependencies=[],
                required_resources=[],
                estimated_duration="1 hour",
                priority=1,
            )
        ]
        validated_tasks = task_generator._validate_tasks(task_definitions)
        assert isinstance(validated_tasks, list)
        assert len(validated_tasks) == 1
        assert validated_tasks[0]["id"] == "test1"

    def test_establish_relationships(
        self, task_generator: TaskGenerator, sample_tasks_dict: List[Dict]
    ) -> None:
        """Should establish dependencies between tasks."""
        tasks = [dict(t) for t in sample_tasks_dict]
        task_generator._establish_relationships(tasks)
        assert tasks[1]["dependencies"] == ["task1"]

    def test_build_hierarchy(
        self, task_generator: TaskGenerator, sample_tasks_dict: List[Dict]
    ) -> None:
        """Should build hierarchy levels for tasks."""
        tasks = [dict(t) for t in sample_tasks_dict]
        task_generator._build_hierarchy(tasks)
        assert all("level" in task for task in tasks)


class TestBestPracticeManager:
    """Test BestPracticeManager core logic and validation."""

    def test_generate_best_practices(
        self, best_practice_manager: BestPracticeManager, sample_tasks_dict: List[Dict]
    ) -> None:
        """Should generate best practices from tasks."""
        practices = best_practice_manager.generate_best_practices(sample_tasks_dict)
        assert isinstance(practices, list)
        assert len(practices) > 0
        assert all(isinstance(practice, dict) for practice in practices)
        assert all("practice_id" in practice for practice in practices)

    def test_finetune_best_practice(
        self, best_practice_manager: BestPracticeManager
    ) -> None:
        """Should refine a best practice with a new task."""
        practice = {
            "practice_id": "test1",
            "name": "Test Practice",
            "description": "Test Description",
            "steps": [
                {"task": "Original step", "description": "Original step description"}
            ],
            "rationale": "Test Rationale",
            "examples": [],
            "priority": 3,
            "category": "test",
        }
        task = {
            "name": "Test Task",
            "steps": [{"task": "New step", "description": "New step description"}],
        }
        refined_practice = best_practice_manager.finetune_best_practice(practice, task)
        assert isinstance(refined_practice, dict)
        assert "steps" in refined_practice
        assert len(refined_practice["steps"]) > 0

    def test_validate_practices(self, best_practice_manager, sample_practices) -> None:
        """Should validate and convert BestPractice objects to dicts."""
        validated = best_practice_manager._validate_practices(sample_practices)
        assert isinstance(validated, list)
        assert len(validated) == 1
        assert validated[0]["practice_id"] == "practice1"

    def test_categorize_practices(
        self, best_practice_manager, sample_practices_dict
    ) -> None:
        """Should categorize best practices by category."""
        dict_practices = [dict(p) for p in sample_practices_dict]
        best_practice_manager._categorize_practices(dict_practices)
        assert isinstance(best_practice_manager._practice_categories, dict)
        assert "validation" in best_practice_manager._practice_categories

    def test_optimize_practices(
        self, best_practice_manager, sample_practices_dict
    ) -> None:
        """Should optimize best practices."""
        dict_practices = [dict(p) for p in sample_practices_dict]
        optimized = best_practice_manager._optimize_practices(dict_practices)
        assert isinstance(optimized, list)
        assert len(optimized) == 1


class TestReusableTaskManager:
    """Test ReusableTaskManager core logic and validation."""

    def test_generate_reusable_tasks(
        self, reusable_task_manager, sample_tasks_dict
    ) -> None:
        """Should generate reusable task templates from tasks."""
        templates = reusable_task_manager.generate_reusable_tasks(sample_tasks_dict)
        assert isinstance(templates, dict)
        assert len(templates) > 0
        assert all(isinstance(template, dict) for template in templates.values())

    def test_instantiate_template(
        self, reusable_task_manager, reusable_template
    ) -> None:
        """Should instantiate a reusable task template."""
        reusable_task_manager._templates[reusable_template.template_id] = (
            reusable_template
        )
        params = {"param1": "value1", "param2": "value2"}
        instantiated = reusable_task_manager.instantiate_template(
            reusable_template.template_id, params
        )
        assert isinstance(instantiated, dict)
        assert instantiated["template_id"] == "tmpl1"

    def test_validate_templates(self, reusable_task_manager, reusable_template) -> None:
        """Should validate reusable task templates."""
        templates = {reusable_template.template_id: reusable_template}
        validated = reusable_task_manager._validate_templates(templates)
        assert isinstance(validated, dict)
        assert "tmpl1" in validated

    def test_validate_parameters(
        self, reusable_task_manager, reusable_template
    ) -> None:
        """Should validate parameters for reusable task templates."""
        params = {"param1": "value1", "param2": "value2"}
        valid = reusable_task_manager._validate_parameters(reusable_template, params)
        assert valid is True

    def test_categorize_templates(
        self, reusable_task_manager, reusable_template
    ) -> None:
        """Should categorize reusable task templates by category."""
        templates = {
            reusable_template.template_id: dataclasses.asdict(reusable_template)
        }
        reusable_task_manager._categorize_templates(templates)
        assert isinstance(reusable_task_manager._template_categories, dict)
        assert "test" in reusable_task_manager._template_categories


def test_integration_generation_pipeline(always_valid_mock_model: Mock) -> None:
    """Integration test for the full generation pipeline."""
    task_gen = TaskGenerator(
        model=always_valid_mock_model,
        role="product_manager",
        user_objective="Create a new product",
        instructions=[],
        documents=[],
    )
    tasks = task_gen.generate_tasks(intro="Create a new product", existing_tasks=[])

    best_mgr = BestPracticeManager(
        model=always_valid_mock_model,
        role="product_manager",
        user_objective="Create a new product",
    )
    practices = best_mgr.generate_best_practices(tasks)

    reusable_mgr = ReusableTaskManager(
        model=always_valid_mock_model,
        role="product_manager",
        user_objective="Create a new product",
    )
    reusable = reusable_mgr.generate_reusable_tasks(tasks)

    assert isinstance(tasks, list)
    assert isinstance(practices, list)
    assert isinstance(reusable, dict)
