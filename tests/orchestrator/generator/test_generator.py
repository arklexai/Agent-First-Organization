"""Test suite for the Arklex generator components.

This module contains comprehensive tests for the task generation, best practices,
and reusable task components of the Arklex framework. It includes unit tests
for individual components and integration tests for the complete generation
pipeline.
"""

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

# Sample test data
SAMPLE_OBJECTIVE = {
    "description": "Create a new product in the store",
    "requirements": ["Product details", "Pricing information"],
    "constraints": ["Must follow store guidelines"],
}

SAMPLE_TASKS = [
    {
        "task_id": "task1",
        "name": "Gather product details",
        "description": "Collect all required product information",
        "steps": [
            {"task": "Get product name", "description": "Get the name of the product."},
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

SAMPLE_PRACTICES = [
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


# Patch for test_generate_tasks and test_integration_generation_pipeline
class FakeResponse:
    content = '[{"task": "Test", "intent": "Test intent", "description": "Test description", "steps": [{"task": "Step 1", "description": "Test step description"}]}]'
    text = content


class FakeGenerationResponse:
    def __init__(self):
        self.generations = [[FakeGeneration()]]


class FakeGeneration:
    def __init__(self):
        self.text = '[{"task": "Test", "intent": "Test intent", "description": "Test description", "steps": [{"task": "Step 1", "description": "Test step description"}]}]'


class FakeResourceResponse:
    content = '[{"step": 1, "task": "Step 1", "resource": "MessageWorker", "example_response": "Test response", "description": "Test step description"}]'
    text = content


def get_mock_model():
    mock_model = Mock()
    mock_model.generate.return_value = FakeGenerationResponse()
    mock_model.invoke.return_value = FakeResourceResponse()
    return mock_model


@pytest.fixture
def mock_model():
    """Create a mock language model for testing."""
    return get_mock_model()


@pytest.fixture
def task_generator(mock_model):
    """Create a TaskGenerator instance for testing."""
    return TaskGenerator(
        model=mock_model,
        role="product_manager",
        user_objective="Create a new product",
        instructions=[],
        documents=[],
    )


@pytest.fixture
def best_practice_manager(mock_model):
    """Create a BestPracticeManager instance for testing."""
    return BestPracticeManager(
        model=mock_model, role="product_manager", user_objective="Create a new product"
    )


@pytest.fixture
def reusable_task_manager(mock_model):
    """Create a ReusableTaskManager instance for testing."""
    return ReusableTaskManager(
        model=mock_model, role="product_manager", user_objective="Create a new product"
    )


class TestTaskGenerator:
    """Test suite for the TaskGenerator class."""

    def test_generate_tasks(self, task_generator):
        """Test task generation from objectives."""
        tasks = task_generator.generate_tasks(
            intro="Create a new product", existing_tasks=[]
        )
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        assert all(isinstance(task, dict) for task in tasks)
        assert all("task_id" in task for task in tasks)

    def test_add_provided_tasks(self, task_generator):
        """Test adding user-provided tasks."""
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

    def test_validate_tasks(self, task_generator):
        """Test task validation."""
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
        assert validated_tasks[0]["task_id"] == "test1"

    def test_establish_relationships(self, task_generator):
        """Test establishing task relationships."""
        tasks = SAMPLE_TASKS.copy()
        task_generator._establish_relationships(tasks)
        assert tasks[1]["dependencies"] == ["task1"]

    def test_build_hierarchy(self, task_generator):
        """Test building task hierarchy."""
        tasks = SAMPLE_TASKS.copy()
        task_generator._build_hierarchy(tasks)
        assert all("level" in task for task in tasks)


class TestBestPracticeManager:
    """Test suite for the BestPracticeManager class."""

    def test_generate_best_practices(self, best_practice_manager):
        """Test best practice generation."""
        practices = best_practice_manager.generate_best_practices(SAMPLE_TASKS)
        assert isinstance(practices, list)
        assert len(practices) > 0
        assert all(isinstance(practice, dict) for practice in practices)
        assert all("practice_id" in practice for practice in practices)

    def test_finetune_best_practice(self, best_practice_manager):
        """Test best practice refinement."""
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

    def test_validate_practices(self, best_practice_manager):
        """Test practice validation."""
        practice_definitions = [
            BestPractice(
                practice_id="test1",
                name="Test practice",
                description="Test description",
                steps=[{"task": "Test step", "description": "Test step description"}],
                rationale="Test rationale",
                examples=[],
                priority=1,
                category="test",
            )
        ]
        validated_practices = best_practice_manager._validate_practices(
            practice_definitions
        )
        assert isinstance(validated_practices, list)
        assert len(validated_practices) == 1
        assert validated_practices[0]["practice_id"] == "test1"

    def test_categorize_practices(self, best_practice_manager):
        """Test practice categorization."""
        practices = SAMPLE_PRACTICES.copy()
        best_practice_manager._categorize_practices(practices)
        assert all("category" in practice for practice in practices)

    def test_optimize_practices(self, best_practice_manager):
        """Test practice optimization."""
        practices = SAMPLE_PRACTICES.copy()
        optimized_practices = best_practice_manager._optimize_practices(practices)
        assert isinstance(optimized_practices, list)
        assert len(optimized_practices) == len(practices)


class TestReusableTaskManager:
    """Test suite for the ReusableTaskManager class."""

    def test_generate_reusable_tasks(self, reusable_task_manager):
        """Test reusable task generation."""
        templates = reusable_task_manager.generate_reusable_tasks(SAMPLE_TASKS)
        assert isinstance(templates, dict)
        assert len(templates) > 0
        assert all(isinstance(template, dict) for template in templates.values())

    def test_instantiate_template(self, reusable_task_manager):
        """Test template instantiation."""
        template = ReusableTask(
            template_id="test1",
            name="Test template",
            description="Test description",
            steps=[{"task": "Test step", "description": "Test step description"}],
            parameters={"param1": "string"},
            examples=[],
            version="1.0",
            category="test",
        )
        reusable_task_manager._templates["test1"] = template
        instance = reusable_task_manager.instantiate_template(
            template_id="test1", parameters={"param1": "value"}
        )
        assert isinstance(instance, dict)
        assert "steps" in instance

    def test_validate_templates(self, reusable_task_manager):
        """Test template validation."""
        templates = {
            "test1": ReusableTask(
                template_id="test1",
                name="Test template",
                description="Test description",
                steps=[{"task": "Test step", "description": "Test step description"}],
                parameters={},
                examples=[],
                version="1.0",
                category="test",
            )
        }
        validated_templates = reusable_task_manager._validate_templates(templates)
        assert isinstance(validated_templates, dict)
        assert len(validated_templates) == 1
        assert "test1" in validated_templates

    def test_validate_parameters(self, reusable_task_manager):
        """Test parameter validation."""
        template = ReusableTask(
            template_id="test1",
            name="Test template",
            description="Test description",
            steps=[{"task": "Test step", "description": "Test step description"}],
            parameters={"param1": "string"},
            examples=[],
            version="1.0",
            category="test",
        )
        is_valid = reusable_task_manager._validate_parameters(
            template=template, parameters={"param1": "value"}
        )
        assert is_valid

    def test_categorize_templates(self, reusable_task_manager):
        """Test template categorization."""
        templates = {
            "test1": {"id": "test1", "name": "Test template", "category": "test"}
        }
        reusable_task_manager._categorize_templates(templates)
        assert "test" in reusable_task_manager._template_categories


def test_integration_generation_pipeline(mock_model):
    """Test the complete generation pipeline integration."""
    # Initialize components
    task_generator = TaskGenerator(
        model=mock_model,
        role="product_manager",
        user_objective="Create a new product",
        instructions=[],
        documents=[],
    )
    best_practice_manager = BestPracticeManager(
        model=mock_model, role="product_manager", user_objective="Create a new product"
    )
    reusable_task_manager = ReusableTaskManager(
        model=mock_model, role="product_manager", user_objective="Create a new product"
    )

    # Generate tasks
    tasks = task_generator.generate_tasks(
        intro="Create a new product", existing_tasks=[]
    )
    assert isinstance(tasks, list)
    assert len(tasks) > 0

    # Generate best practices
    practices = best_practice_manager.generate_best_practices(tasks)
    assert isinstance(practices, list)
    assert len(practices) > 0

    # Generate reusable templates
    templates = reusable_task_manager.generate_reusable_tasks(tasks)
    assert isinstance(templates, dict)
    assert len(templates) > 0

    # Verify integration
    assert all("task_id" in task for task in tasks)
    assert all("practice_id" in practice for practice in practices)
    assert all(isinstance(template, dict) for template in templates.values())
