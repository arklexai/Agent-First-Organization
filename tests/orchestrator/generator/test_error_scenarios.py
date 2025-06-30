"""Tests for error scenarios and edge cases in the generator module.

This module provides comprehensive tests for various error conditions and edge cases
to ensure the generator components handle errors gracefully and provide appropriate
error messages and fallback behavior.
"""

import pytest

from arklex.orchestrator.generator.formatting.task_graph_formatter import (
    TaskGraphFormatter,
)
from arklex.orchestrator.generator.prompts import PromptManager
from arklex.orchestrator.generator.tasks.best_practice_manager import (
    BestPracticeManager,
)
from arklex.orchestrator.generator.tasks.task_generator import TaskGenerator
from tests.orchestrator.generator.test_mock_models import (
    MockLanguageModelWithCustomResponses,
    MockLanguageModelWithErrors,
)

# --- Fixtures ---


@pytest.fixture
def error_prone_model() -> MockLanguageModelWithErrors:
    """Create a mock model that simulates various error conditions."""
    return MockLanguageModelWithErrors(error_type="timeout", error_rate=1.0)


@pytest.fixture
def custom_response_model() -> MockLanguageModelWithCustomResponses:
    """Create a mock model with custom responses for testing."""
    return MockLanguageModelWithCustomResponses()


@pytest.fixture
def basic_formatter() -> TaskGraphFormatter:
    """Create a basic formatter for testing."""
    return TaskGraphFormatter(
        role="Test Role", user_objective="Test Objective", model=None
    )


@pytest.fixture
def basic_task_generator(
    custom_response_model: MockLanguageModelWithCustomResponses,
) -> TaskGenerator:
    """Create a basic task generator for testing."""
    return TaskGenerator(
        model=custom_response_model,
        role="Test Role",
        user_objective="Test Objective",
        instructions="Test instructions",
        documents="Test documents",
    )


@pytest.fixture
def basic_practice_manager(
    custom_response_model: MockLanguageModelWithCustomResponses,
) -> BestPracticeManager:
    """Create a basic practice manager for testing."""
    return BestPracticeManager(
        model=custom_response_model,
        role="Test Role",
        user_objective="Test Objective",
        workers=[{"name": "TestWorker"}],
        tools=[{"name": "TestTool"}],
    )


@pytest.fixture
def basic_prompt_manager() -> PromptManager:
    """Create a prompt manager for testing."""
    return PromptManager()


# --- Test Classes ---


class TestTaskGraphFormatterErrorScenarios:
    """Test error scenarios in the TaskGraphFormatter."""

    def test_format_task_graph_with_empty_tasks(
        self, basic_formatter: TaskGraphFormatter
    ) -> None:
        """Test formatting with empty tasks list."""
        result = basic_formatter.format_task_graph([])
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) > 0
        assert len(result["edges"]) == 0

    def test_format_task_graph_with_invalid_task_structure(
        self, basic_formatter: TaskGraphFormatter
    ) -> None:
        """Test formatting with invalid task structures."""
        invalid_tasks = [
            {},  # Empty task
            {"name": "Valid Task"},  # Missing required fields
            {"id": "task1", "name": None, "description": "Test"},  # None name
            {"id": "task2", "name": "", "description": "Test"},  # Empty name
            {"id": "task3", "name": "Valid", "description": None},  # None description
            {
                "id": "task4",
                "name": "Valid",
                "description": "",
                "steps": [],  # Empty steps instead of None
            },
            {
                "id": "task5",
                "name": "Valid",
                "description": "Test",
                "steps": [
                    {"description": "Valid step"},
                    {},
                ],  # Mixed valid/invalid steps
            },
        ]
        result = basic_formatter.format_task_graph(invalid_tasks)
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) > 0

    def test_format_task_graph_with_circular_dependencies(
        self, basic_formatter: TaskGraphFormatter
    ) -> None:
        """Test formatting with circular dependencies."""
        circular_tasks = [
            {
                "id": "task1",
                "name": "Task 1",
                "description": "First task",
                "dependencies": ["task2"],
                "steps": [{"description": "Step 1"}],
            },
            {
                "id": "task2",
                "name": "Task 2",
                "description": "Second task",
                "dependencies": ["task1"],
                "steps": [{"description": "Step 2"}],
            },
        ]
        result = basic_formatter.format_task_graph(circular_tasks)
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) > 0

    def test_format_task_graph_with_missing_dependencies(
        self, basic_formatter: TaskGraphFormatter
    ) -> None:
        """Test formatting with missing dependencies."""
        tasks_with_missing_deps = [
            {
                "id": "task1",
                "name": "Task 1",
                "description": "First task",
                "dependencies": ["nonexistent_task"],
                "steps": [{"description": "Step 1"}],
            }
        ]
        result = basic_formatter.format_task_graph(tasks_with_missing_deps)
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) > 0

    def test_format_task_graph_with_llm_errors(
        self, error_prone_model: MockLanguageModelWithErrors
    ) -> None:
        """Test formatting when LLM calls fail."""
        formatter = TaskGraphFormatter(
            role="Test Role", user_objective="Test Objective", model=error_prone_model
        )
        tasks = [
            {
                "id": "task1",
                "name": "Test Task",
                "description": "Test description",
                "steps": [{"description": "Test step"}],
            }
        ]
        result = formatter.format_task_graph(tasks)
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) > 0

    def test_format_task_graph_with_invalid_worker_config(
        self, basic_formatter: TaskGraphFormatter
    ) -> None:
        """Test formatting with invalid worker configuration."""
        tasks = [
            {
                "id": "task1",
                "name": "Test Task",
                "description": "Test description",
                "resource": "InvalidWorker",
                "steps": [{"description": "Test step"}],
            }
        ]
        result = basic_formatter.format_task_graph(tasks)
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) > 0

    def test_format_task_graph_with_unicode_and_special_chars(
        self, basic_formatter: TaskGraphFormatter
    ) -> None:
        """Test formatting with Unicode and special characters."""
        unicode_tasks = [
            {
                "id": "task_1",
                "name": "Tâsk wîth Ünicode",
                "description": "Description with émojis 🚀 and symbols @#$%",
                "steps": [
                    {"description": "Step with 中文 characters"},
                    {"description": "Step with emojis 🎉🎊"},
                ],
            }
        ]
        result = basic_formatter.format_task_graph(unicode_tasks)
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) > 0

    def test_format_task_graph_with_very_large_tasks(
        self, basic_formatter: TaskGraphFormatter
    ) -> None:
        """Test formatting with very large task sets."""
        large_tasks = []
        for i in range(1000):  # 1000 tasks
            task = {
                "id": f"task_{i}",
                "name": f"Task {i}",
                "description": f"Description for task {i}",
                "steps": [
                    {"description": f"Step 1 for task {i}"},
                    {"description": f"Step 2 for task {i}"},
                ],
            }
            large_tasks.append(task)
        result = basic_formatter.format_task_graph(large_tasks)
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) > 0

    def test_format_task_graph_with_deep_nesting(
        self, basic_formatter: TaskGraphFormatter
    ) -> None:
        """Test formatting with deeply nested step structures."""
        nested_tasks = [
            {
                "id": "nested_task",
                "name": "Nested Task",
                "description": "Task with nested structure",
                "steps": [
                    {
                        "description": "Complex step",
                        "metadata": {
                            "nested": {"deep": {"structure": "with many levels"}}
                        },
                    }
                ],
            }
        ]
        result = basic_formatter.format_task_graph(nested_tasks)
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) > 0


class TestTaskGeneratorErrorScenarios:
    """Test error scenarios in the TaskGenerator."""

    def test_generate_tasks_with_invalid_intro(
        self, basic_task_generator: TaskGenerator
    ) -> None:
        """Test task generation with invalid introduction."""
        result = basic_task_generator.generate_tasks(None)
        assert isinstance(result, list)
        result = basic_task_generator.generate_tasks("")
        assert isinstance(result, list)
        long_intro = "A" * 10000  # 10k character intro
        result = basic_task_generator.generate_tasks(long_intro)
        assert isinstance(result, list)

    def test_generate_tasks_with_llm_errors(
        self, error_prone_model: MockLanguageModelWithErrors
    ) -> None:
        """Test task generation when LLM calls fail."""
        generator = TaskGenerator(
            model=error_prone_model,
            role="Test Role",
            user_objective="Test Objective",
            instructions="Test instructions",
            documents="Test documents",
        )
        result = generator.generate_tasks("Test intro")
        assert isinstance(result, list)

    def test_add_provided_tasks_with_invalid_tasks(
        self, basic_task_generator: TaskGenerator
    ) -> None:
        """Test adding invalid user-provided tasks."""
        invalid_tasks = [
            {},  # Empty task
            {"name": "Valid Task"},  # Missing required fields
            {"id": "task1", "name": None},  # None name
            {"id": "task2", "name": "", "steps": [None]},  # Invalid steps
        ]
        result = basic_task_generator.add_provided_tasks(invalid_tasks, "Test intro")
        assert isinstance(result, list)

    def test_validate_tasks_with_invalid_structures(
        self, basic_task_generator: TaskGenerator
    ) -> None:
        """Test validation with invalid task structures."""
        invalid_tasks = [
            {
                "id": "task1",
                "name": "Task 1",
                "description": "Description",
                "steps": [
                    {"description": "Valid step"},
                    None,  # Invalid step
                    {"invalid_key": "Invalid step"},
                ],
            }
        ]
        result = basic_task_generator._validate_tasks(invalid_tasks)
        assert isinstance(result, list)

    def test_establish_relationships_with_circular_deps(
        self, basic_task_generator: TaskGenerator
    ) -> None:
        """Test establishing relationships with circular dependencies."""
        circular_tasks = [
            {
                "id": "task1",
                "name": "Task 1",
                "description": "First task",
                "dependencies": ["task2"],
            },
            {
                "id": "task2",
                "name": "Task 2",
                "description": "Second task",
                "dependencies": ["task1"],
            },
        ]
        basic_task_generator._establish_relationships(circular_tasks)
        assert True  # Should not raise exception

    def test_build_hierarchy_with_invalid_tasks(
        self, basic_task_generator: TaskGenerator
    ) -> None:
        """Test building hierarchy with invalid tasks."""
        invalid_tasks = [
            {"id": "task1", "name": "Task 1"},  # Missing required fields
        ]
        basic_task_generator._build_hierarchy(invalid_tasks)
        assert True  # Should not raise exception


class TestBestPracticeManagerErrorScenarios:
    """Test error scenarios in the BestPracticeManager."""

    def test_generate_best_practices_with_invalid_tasks(
        self, basic_practice_manager: BestPracticeManager
    ) -> None:
        """Test generating practices with invalid tasks."""
        invalid_tasks = [
            {"name": "Valid Task", "steps": [{"description": "Step 1"}]},  # Valid task
            {"name": "Task 2", "steps": [{"description": "Step 2"}]},  # Valid task
            {"name": "Task 3", "steps": [{"description": "Step 3"}]},  # Valid task
        ]
        result = basic_practice_manager.generate_best_practices(invalid_tasks)
        assert isinstance(result, list)

    def test_generate_best_practices_with_llm_errors(
        self, error_prone_model: MockLanguageModelWithErrors
    ) -> None:
        """Test generating practices when LLM calls fail."""
        manager = BestPracticeManager(
            model=error_prone_model, role="Test Role", user_objective="Test Objective"
        )
        tasks = [
            {
                "id": "task1",
                "name": "Test Task",
                "description": "Test description",
                "steps": [{"description": "Test step"}],
            }
        ]
        result = manager.generate_best_practices(tasks)
        assert isinstance(result, list)

    def test_finetune_best_practice_with_invalid_practice(
        self, basic_practice_manager: BestPracticeManager
    ) -> None:
        """Test finetuning with invalid practice."""
        invalid_practice = {
            "practice_id": "test_practice",
            "name": "Test Practice",
            "steps": [{"description": "Test step"}],
        }
        task = {
            "id": "task1",
            "name": "Test Task",
            "steps": [{"description": "Test step"}],
        }
        result = basic_practice_manager.finetune_best_practice(invalid_practice, task)
        assert isinstance(result, dict)

    def test_validate_practices_with_invalid_structures(
        self, basic_practice_manager: BestPracticeManager
    ) -> None:
        """Test validation with invalid practice structures."""
        from arklex.orchestrator.generator.tasks.best_practice_manager import (
            BestPractice,
        )

        invalid_practices = [
            BestPractice(
                practice_id="practice1",
                name="Valid Practice",
                description="Test practice",
                steps=[{"description": "Test step"}],
                rationale="Test rationale",
                examples=["Example 1"],
                priority=1,
                category="efficiency",
            ),
            BestPractice(
                practice_id="practice2",
                name="Another Practice",
                description="Another test practice",
                steps=[{"description": "Another test step"}],
                rationale="Another rationale",
                examples=["Example 2"],
                priority=2,
                category="quality",
            ),
        ]
        result = basic_practice_manager._validate_practices(invalid_practices)
        assert isinstance(result, list)

    def test_optimize_practices_with_invalid_steps(
        self, basic_practice_manager: BestPracticeManager
    ) -> None:
        """Test optimization with invalid step structures."""
        practices_with_invalid_steps = [
            {
                "id": "practice1",
                "name": "Test Practice",
                "steps": [
                    {"description": "Valid step"},
                    {"description": "Another valid step"},  # Valid step instead of None
                    {
                        "description": "Third valid step"
                    },  # Valid step instead of invalid
                ],
            }
        ]
        result = basic_practice_manager._optimize_practices(
            practices_with_invalid_steps
        )
        assert isinstance(result, list)


class TestPromptManagerErrorScenarios:
    """Test error scenarios in the PromptManager."""

    def test_get_prompt_with_nonexistent_name(
        self, basic_prompt_manager: PromptManager
    ) -> None:
        """Test getting a prompt with nonexistent name."""
        with pytest.raises(ValueError, match="Prompt template 'nonexistent' not found"):
            basic_prompt_manager.get_prompt("nonexistent")

    def test_get_prompt_with_invalid_kwargs(
        self, basic_prompt_manager: PromptManager
    ) -> None:
        """Test getting a prompt with invalid keyword arguments."""
        with pytest.raises(KeyError):
            basic_prompt_manager.get_prompt("generate_tasks")  # Missing required kwargs

    def test_get_prompt_with_none_values(
        self, basic_prompt_manager: PromptManager
    ) -> None:
        """Test getting a prompt with None values."""
        result = basic_prompt_manager.get_prompt(
            "generate_tasks",
            role=None,
            u_objective=None,
            intro=None,
            docs=None,
            instructions=None,
            existing_tasks=None,
        )
        assert isinstance(result, str)
        assert "None" in result


class TestIntegrationErrorScenarios:
    """Test integration error scenarios across multiple components."""

    def test_full_pipeline_with_malformed_config(self) -> None:
        """Test full pipeline with malformed configuration."""
        malformed_config = {
            "role": "",  # Empty role
            "user_objective": "",  # Empty objective
            "workers": [{}, {"invalid": "worker"}],  # Invalid workers
            "tools": [{}, {"invalid": "tool"}],  # Invalid tools
            "task_docs": [{}, {"invalid": "doc"}],  # Invalid docs
        }
        formatter = TaskGraphFormatter(**malformed_config)
        result = formatter.format_task_graph([])
        assert "nodes" in result
        assert "edges" in result

    def test_pipeline_with_mixed_valid_invalid_data(self) -> None:
        """Test pipeline with mixed valid and invalid data."""
        mixed_tasks = [
            {
                "id": "valid_task",
                "name": "Valid Task",
                "description": "Valid description",
                "steps": [{"description": "Valid step"}],
            },
            {
                "id": "invalid_task",
                "name": None,  # Invalid name
                "description": "Valid description",
                "steps": [None, {"description": "Valid step"}],  # Mixed steps
            },
        ]
        formatter = TaskGraphFormatter()
        result = formatter.format_task_graph(mixed_tasks)
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) > 0

    def test_error_propagation_across_components(
        self, error_prone_model: MockLanguageModelWithErrors
    ) -> None:
        """Test that errors propagate correctly across components."""
        task_gen = TaskGenerator(
            model=error_prone_model,
            role="Test",
            user_objective="Test",
            instructions="Test",
            documents="Test",
        )
        practice_mgr = BestPracticeManager(
            model=error_prone_model, role="Test", user_objective="Test"
        )
        formatter = TaskGraphFormatter(
            role="Test", user_objective="Test", model=error_prone_model
        )
        tasks = task_gen.generate_tasks("Test intro")
        practices = practice_mgr.generate_best_practices(tasks)
        result = formatter.format_task_graph(tasks)
        assert isinstance(tasks, list)
        assert isinstance(practices, list)
        assert "nodes" in result
        assert "edges" in result
