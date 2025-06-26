"""Comprehensive tests for TaskGenerator.

This module provides extensive test coverage for the TaskGenerator class,
including all methods, edge cases, error conditions, and the TaskDefinition dataclass.
"""

import pytest
from unittest.mock import Mock, patch

from arklex.orchestrator.generator.tasks.task_generator import (
    TaskGenerator,
    TaskDefinition,
)


# =============================================================================
# FIXTURES - Core Test Data
# =============================================================================


@pytest.fixture
def always_valid_mock_model():
    """Create a mock model that always returns valid responses."""
    mock = Mock()
    mock.generate.return_value = {"text": '[{"task": "test"}]'}
    mock.invoke.return_value.content = '[{"task": "test"}]'
    return mock


@pytest.fixture
def task_generator(always_valid_mock_model):
    """Create a TaskGenerator instance for testing."""
    return TaskGenerator(
        always_valid_mock_model,
        "test_role",
        "test_objective",
        "test_instructions",
        "test_docs",
    )


@pytest.fixture
def sample_task_definition():
    """Sample TaskDefinition for testing."""
    return TaskDefinition(
        task_id="task1",
        name="Test Task",
        description="Test description",
        steps=[{"task": "1"}],
        dependencies=["task2"],
        required_resources=["resource1"],
        estimated_duration="1 hour",
        priority=3,
    )


@pytest.fixture
def sample_task_definition_minimal():
    """Sample TaskDefinition with minimal fields for testing."""
    return TaskDefinition(
        task_id="task1",
        name="Test Task",
        description="Test description",
        steps=[],
        dependencies=[],
        required_resources=[],
        estimated_duration=None,
        priority=1,
    )


@pytest.fixture
def sample_tasks_with_steps():
    """Sample tasks with steps for testing."""
    return [
        {
            "task": "task1",
            "intent": "intent1",
            "steps": [{"task": "step1", "description": "Execute step1"}],
            "dependencies": ["task2"],
            "required_resources": ["resource1"],
            "estimated_duration": "1 hour",
            "priority": 3,
        }
    ]


@pytest.fixture
def sample_tasks():
    """Sample tasks for testing."""
    return [
        {"name": "task1", "description": "intent1", "steps": [{"task": "step1"}]},
        {"name": "task2", "description": "intent2", "steps": [{"task": "step2"}]},
    ]


@pytest.fixture
def sample_existing_tasks():
    """Sample existing tasks for testing."""
    return [{"task": "existing_task", "intent": "existing_intent"}]


# =============================================================================
# FIXTURES - Patched Configurations
# =============================================================================


@pytest.fixture
def patched_sample_config(task_generator):
    """Create a TaskGenerator with all common methods patched for testing."""
    with (
        patch.object(task_generator, "_generate_high_level_tasks") as mock_generate,
        patch.object(task_generator, "_check_task_breakdown_original") as mock_check,
        patch.object(task_generator, "_validate_tasks") as mock_validate,
        patch.object(task_generator, "_convert_to_task_definitions") as mock_convert,
        patch.object(task_generator, "_build_hierarchy") as mock_hierarchy,
    ):
        mock_generate.return_value = [{"task": "new_task", "intent": "new_intent"}]
        mock_check.return_value = False
        mock_validate.return_value = [
            {
                "id": "task_1",
                "name": "new_task",
                "description": "new_intent",
                "steps": [{"task": "Execute new_task"}],
                "dependencies": [],
                "required_resources": [],
                "estimated_duration": "1 hour",
                "priority": 3,
            }
        ]
        mock_convert.return_value = []

        yield {
            "generator": task_generator,
            "mock_generate": mock_generate,
            "mock_check": mock_check,
            "mock_validate": mock_validate,
            "mock_convert": mock_convert,
            "mock_hierarchy": mock_hierarchy,
        }


@pytest.fixture
def patched_model_generate(task_generator):
    """Create a TaskGenerator with patched model.generate method."""
    with patch.object(task_generator.model, "generate") as mock_generate:
        yield {"generator": task_generator, "mock_generate": mock_generate}


@pytest.fixture
def patched_model_invoke(task_generator):
    """Create a TaskGenerator with patched model.invoke method."""
    with patch.object(task_generator.model, "invoke") as mock_invoke:
        yield {"generator": task_generator, "mock_invoke": mock_invoke}


@pytest.fixture
def patched_breakdown_methods(task_generator):
    """Create a TaskGenerator with patched breakdown-related methods."""
    with (
        patch.object(task_generator, "_check_task_breakdown_original") as mock_check,
        patch.object(task_generator, "_generate_task_steps_original") as mock_steps,
        patch.object(task_generator, "_validate_task_definition") as mock_validate,
    ):
        yield {
            "generator": task_generator,
            "mock_check": mock_check,
            "mock_steps": mock_steps,
            "mock_validate": mock_validate,
        }


@pytest.fixture
def patched_import_error():
    """Patch to simulate ImportError for langchain_core."""
    with patch(
        "builtins.__import__",
        side_effect=ImportError("No module named 'langchain_core'"),
    ):
        yield


# =============================================================================
# TEST CLASSES
# =============================================================================


class TestTaskDefinition:
    """Test the TaskDefinition dataclass."""

    def test_task_definition_initialization(self, sample_task_definition) -> None:
        """Test TaskDefinition initialization with all fields."""
        task_def = sample_task_definition
        assert task_def.task_id == "task1"
        assert task_def.name == "Test Task"
        assert task_def.description == "Test description"
        assert len(task_def.steps) == 1
        assert len(task_def.dependencies) == 1
        assert len(task_def.required_resources) == 1
        assert task_def.estimated_duration == "1 hour"
        assert task_def.priority == 3

    def test_task_definition_with_optional_fields(
        self, sample_task_definition_minimal
    ) -> None:
        """Test TaskDefinition with optional fields set to defaults."""
        task_def = sample_task_definition_minimal
        assert task_def.estimated_duration is None
        assert task_def.priority == 1


class TestTaskGenerator:
    """Test the TaskGenerator class."""

    def test_task_generator_initialization(self, always_valid_mock_model) -> None:
        """Test TaskGenerator initialization with all required parameters."""
        role = "test_role"
        user_objective = "test_objective"
        instructions = "test_instructions"
        documents = "test_documents"

        generator = TaskGenerator(
            always_valid_mock_model, role, user_objective, instructions, documents
        )
        assert generator.model == always_valid_mock_model
        assert generator.role == role
        assert generator.user_objective == user_objective
        assert generator.instructions == instructions
        assert generator.documents == documents

    def test_generate_tasks_with_existing_tasks(
        self, patched_sample_config, sample_existing_tasks
    ) -> None:
        """Test generate_tasks with existing tasks provided."""
        config = patched_sample_config
        result = config["generator"].generate_tasks("intro", sample_existing_tasks)

        config["mock_generate"].assert_called_once()
        config["mock_check"].assert_called_once()
        config["mock_validate"].assert_called_once()

    def test_generate_tasks_without_existing_tasks(self, patched_sample_config) -> None:
        """Test generate_tasks without existing tasks."""
        config = patched_sample_config
        result = config["generator"].generate_tasks("intro")

        config["mock_generate"].assert_called_once()
        config["mock_check"].assert_called_once()
        config["mock_validate"].assert_called_once()

    def test_generate_tasks_no_breakdown_needed(self, task_generator) -> None:
        """Test generate_tasks when tasks don't need breakdown."""
        with (
            patch.object(task_generator, "_generate_high_level_tasks") as mock_generate,
            patch.object(
                task_generator, "_check_task_breakdown_original"
            ) as mock_check,
            patch.object(
                task_generator, "_convert_to_task_definitions"
            ) as mock_convert,
            patch.object(task_generator, "_validate_tasks") as mock_validate,
            patch.object(task_generator, "_build_hierarchy"),
        ):
            mock_generate.return_value = [
                {"task": "Simple Task", "intent": "Simple intent"}
            ]
            mock_check.return_value = False
            mock_convert.return_value = []
            mock_validate.return_value = []

            result = task_generator.generate_tasks("intro")

            called_tasks = mock_convert.call_args[0][0]
            assert len(called_tasks) == 1
            assert called_tasks[0]["steps"] == [{"task": "Execute Simple Task"}]

    def test_process_objective_with_existing_tasks(
        self, patched_model_generate, sample_existing_tasks
    ) -> None:
        """Test _process_objective with existing tasks."""
        config = patched_model_generate
        mock_response = Mock()
        mock_generation = Mock()
        mock_generation.text = '[{"task": "test"}]'
        mock_response.generations = [[mock_generation]]
        config["mock_generate"].return_value = mock_response

        result = config["generator"]._process_objective(
            "obj", "intro", "docs", sample_existing_tasks
        )

        assert "tasks" in result
        assert len(result["tasks"]) == 1

    def test_process_objective_without_existing_tasks(
        self, patched_model_generate
    ) -> None:
        """Test _process_objective without existing tasks."""
        config = patched_model_generate
        config["mock_generate"].return_value = {"text": '[{"task": "test"}]'}

        result = config["generator"]._process_objective("obj", "intro", "docs")

        assert "tasks" in result
        assert len(result["tasks"]) == 1

    def test_process_objective_import_error(
        self, task_generator, patched_import_error
    ) -> None:
        """Test _process_objective with ImportError handling."""
        with patch.object(task_generator.model, "generate") as mock_generate:
            mock_generate.return_value = {"text": '[{"task": "test"}]'}

            result = task_generator._process_objective("obj", "intro", "docs")

            assert "tasks" in result
            assert len(result["tasks"]) == 1

    def test_process_objective_response_text_extraction(
        self, patched_model_generate
    ) -> None:
        """Test _process_objective with response text extraction."""
        config = patched_model_generate
        mock_response = Mock()
        mock_generation = Mock()
        mock_generation.text = '[{"task": "test"}]'
        mock_response.generations = [[mock_generation]]
        config["mock_generate"].return_value = mock_response

        result = config["generator"]._process_objective("obj", "intro", "docs")

        assert "tasks" in result
        assert len(result["tasks"]) == 1

    def test_generate_high_level_tasks_with_existing_tasks(
        self, patched_model_invoke, sample_existing_tasks
    ) -> None:
        """Test _generate_high_level_tasks with existing tasks."""
        config = patched_model_invoke
        config["mock_invoke"].return_value.content = '[{"task": "test"}]'

        result = config["generator"]._generate_high_level_tasks(
            "intro", sample_existing_tasks
        )

        assert len(result) == 1
        assert result[0]["task"] == "test"

    def test_generate_high_level_tasks_without_existing_tasks(
        self, patched_model_invoke
    ) -> None:
        """Test _generate_high_level_tasks without existing tasks."""
        config = patched_model_invoke
        config["mock_invoke"].return_value.content = '[{"task": "test"}]'

        result = config["generator"]._generate_high_level_tasks("intro")

        assert len(result) == 1
        assert result[0]["task"] == "test"

    def test_generate_high_level_tasks_with_string_response(
        self, patched_model_invoke
    ) -> None:
        """Test _generate_high_level_tasks with string response."""
        config = patched_model_invoke
        config["mock_invoke"].return_value = "string response"

        result = config["generator"]._generate_high_level_tasks("intro")

        assert result == []

    def test_generate_high_level_tasks_with_invalid_json(
        self, patched_model_invoke
    ) -> None:
        """Test _generate_high_level_tasks with invalid JSON."""
        config = patched_model_invoke
        config["mock_invoke"].return_value.content = "invalid json"

        result = config["generator"]._generate_high_level_tasks("intro")

        assert result == []

    def test_generate_high_level_tasks_with_exception(
        self, patched_model_invoke
    ) -> None:
        """Test _generate_high_level_tasks with exception."""
        config = patched_model_invoke
        config["mock_invoke"].side_effect = Exception("Test error")

        result = config["generator"]._generate_high_level_tasks("intro")

        assert result == []

    def test_check_task_breakdown_original_yes(self, patched_model_invoke) -> None:
        """Test _check_task_breakdown_original returns True."""
        config = patched_model_invoke
        config["mock_invoke"].return_value.content = "yes"

        result = config["generator"]._check_task_breakdown_original("task", "intent")

        assert result is True

    def test_check_task_breakdown_original_no(self, patched_model_invoke) -> None:
        """Test _check_task_breakdown_original returns False."""
        config = patched_model_invoke
        config["mock_invoke"].return_value.content = '{"answer": "no"}'

        result = config["generator"]._check_task_breakdown_original("task", "intent")

        assert result is False

    def test_check_task_breakdown_original_with_string_response(
        self, patched_model_invoke
    ) -> None:
        """Test _check_task_breakdown_original with string response."""
        config = patched_model_invoke
        config["mock_invoke"].return_value = "string response"

        result = config["generator"]._check_task_breakdown_original("task", "intent")

        assert result is True  # Default to breakdown if we can't parse

    def test_check_task_breakdown_original_with_invalid_json(
        self, patched_model_invoke
    ) -> None:
        """Test _check_task_breakdown_original with invalid JSON."""
        config = patched_model_invoke
        config["mock_invoke"].return_value.content = "invalid json"

        result = config["generator"]._check_task_breakdown_original("task", "intent")

        assert result is True  # Default to breakdown if we can't parse

    def test_check_task_breakdown_original_with_exception(
        self, patched_model_invoke
    ) -> None:
        """Test _check_task_breakdown_original with exception."""
        config = patched_model_invoke
        config["mock_invoke"].side_effect = Exception("Test error")

        result = config["generator"]._check_task_breakdown_original("task", "intent")

        assert result is True  # Default to breakdown on error

    def test_generate_task_steps_original(self, patched_model_invoke) -> None:
        """Test _generate_task_steps_original with valid response."""
        config = patched_model_invoke
        config["mock_invoke"].return_value.content = '[{"task": "step1"}]'

        result = config["generator"]._generate_task_steps_original("task", "intent")

        assert len(result) == 1
        assert result[0]["task"] == "step1"

    def test_generate_task_steps_original_with_string_response(
        self, patched_model_invoke
    ) -> None:
        """Test _generate_task_steps_original with string response."""
        config = patched_model_invoke
        config["mock_invoke"].return_value = "string response"

        result = config["generator"]._generate_task_steps_original("task", "intent")

        assert len(result) == 1
        assert result[0]["task"] == "Execute task"

    def test_generate_task_steps_original_with_invalid_json(
        self, patched_model_invoke
    ) -> None:
        """Test _generate_task_steps_original with invalid JSON."""
        config = patched_model_invoke
        config["mock_invoke"].return_value.content = "invalid json"

        result = config["generator"]._generate_task_steps_original("task", "intent")

        assert len(result) == 1
        assert result[0]["task"] == "Execute task"

    def test_generate_task_steps_original_with_exception(
        self, patched_model_invoke
    ) -> None:
        """Test _generate_task_steps_original with exception."""
        config = patched_model_invoke
        config["mock_invoke"].side_effect = Exception("Test error")

        result = config["generator"]._generate_task_steps_original("task", "intent")

        assert len(result) == 1
        assert result[0]["task"] == "Execute task"

    def test_generate_task_steps_different_formats(self, patched_model_invoke) -> None:
        """Test _generate_task_steps_original with different step formats."""
        config = patched_model_invoke
        config[
            "mock_invoke"
        ].return_value.content = '[{"step": "Step 1", "task": "Task 1"}, "String Step"]'

        result = config["generator"]._generate_task_steps_original(
            "Test Task", "Test Intent"
        )

        assert len(result) == 2
        assert result[0]["task"] == "Task 1"
        assert result[1]["task"] == "String Step"

    def test_convert_to_task_definitions(
        self, task_generator, sample_tasks_with_steps
    ) -> None:
        """Test _convert_to_task_definitions with valid input."""
        result = task_generator._convert_to_task_definitions(sample_tasks_with_steps)

        assert len(result) == 1
        assert result[0].name == "task1"
        assert result[0].description == "intent1"

    def test_convert_to_task_definitions_string_steps(self, task_generator) -> None:
        """Test _convert_to_task_definitions with string steps."""
        tasks_with_steps = [{"task": "Test Task", "steps": ["Step 1", "Step 2"]}]

        result = task_generator._convert_to_task_definitions(tasks_with_steps)

        assert len(result) == 1
        assert len(result[0].steps) == 2
        assert result[0].steps[0]["task"] == "Step 1"
        assert result[0].steps[0]["description"] == "Execute step: Step 1"

    def test_validate_tasks(self, task_generator, sample_tasks) -> None:
        """Test _validate_tasks with valid input."""
        result = task_generator._validate_tasks(sample_tasks)

        assert len(result) == 2
        assert result[0]["name"] == "task1"
        assert result[1]["name"] == "task2"

    def test_validate_tasks_with_invalid_task(self, task_generator) -> None:
        """Test _validate_tasks with invalid task."""
        tasks = [{"name": "test"}]  # Missing required fields

        result = task_generator._validate_tasks(tasks)

        assert len(result) == 0

    def test_validate_tasks_priority_validation(self, task_generator) -> None:
        """Test _validate_tasks with priority validation."""
        tasks = [
            {
                "name": "test",
                "description": "test",
                "steps": [{"task": "step"}],
                "priority": 6,
            }
        ]

        result = task_generator._validate_tasks(tasks)

        assert len(result) == 1
        assert result[0]["priority"] == 3

    def test_validate_tasks_non_list_dependencies(self, task_generator) -> None:
        """Test _validate_tasks with non-list dependencies."""
        tasks = [
            {
                "name": "test",
                "description": "test",
                "steps": [{"task": "step"}],
                "dependencies": "not_a_list",
            }
        ]

        result = task_generator._validate_tasks(tasks)

        assert len(result) == 1
        assert result[0]["dependencies"] == []

    def test_validate_tasks_non_list_resources(self, task_generator) -> None:
        """Test _validate_tasks with non-list required_resources."""
        tasks = [
            {
                "name": "test",
                "description": "test",
                "steps": [{"task": "step"}],
                "required_resources": "not_a_list",
            }
        ]

        result = task_generator._validate_tasks(tasks)

        assert len(result) == 1
        assert result[0]["required_resources"] == []

    def test_validate_tasks_non_string_duration(self, task_generator) -> None:
        """Test _validate_tasks with non-string estimated_duration."""
        tasks = [
            {
                "name": "test",
                "description": "test",
                "steps": [{"task": "step"}],
                "estimated_duration": 123,
            }
        ]

        result = task_generator._validate_tasks(tasks)

        assert len(result) == 1
        assert result[0]["estimated_duration"] == "1 hour"

    def test_validate_tasks_priority_out_of_range(self, task_generator) -> None:
        """Test _validate_tasks with priority out of range."""
        tasks = [
            {
                "name": "test",
                "description": "test",
                "steps": [{"task": "step"}],
                "priority": 0,
            }
        ]

        result = task_generator._validate_tasks(tasks)

        assert len(result) == 1
        assert result[0]["priority"] == 3

    def test_validate_tasks_priority_float(self, task_generator) -> None:
        """Test _validate_tasks with float priority."""
        tasks = [
            {
                "name": "test",
                "description": "test",
                "steps": [{"task": "step"}],
                "priority": 4.5,
            }
        ]

        result = task_generator._validate_tasks(tasks)

        assert len(result) == 1
        assert result[0]["priority"] == 4.5

    def test_validate_tasks_empty_step_description(self, task_generator) -> None:
        """Test _validate_tasks with empty step description."""
        tasks = [
            {
                "name": "test",
                "description": "test",
                "steps": [{"task": "step", "description": ""}],
            }
        ]

        result = task_generator._validate_tasks(tasks)

        assert len(result) == 1
        assert result[0]["steps"][0]["description"] == "Execute: step"

    def test_validate_tasks_whitespace_step_description(self, task_generator) -> None:
        """Test _validate_tasks with whitespace-only step description."""
        tasks = [
            {
                "name": "test",
                "description": "test",
                "steps": [{"task": "step", "description": "   "}],
            }
        ]

        result = task_generator._validate_tasks(tasks)

        assert len(result) == 1
        assert result[0]["steps"][0]["description"] == "Execute: step"

    def test_validate_tasks_missing_step_task(self, task_generator) -> None:
        """Test _validate_tasks with step missing task key."""
        tasks = [
            {"name": "test", "description": "test", "steps": [{"description": "step"}]}
        ]

        result = task_generator._validate_tasks(tasks)

        assert len(result) == 0

    def test_validate_tasks_non_dict_step(self, task_generator) -> None:
        """Test _validate_tasks with non-dict step."""
        tasks = [{"name": "test", "description": "test", "steps": ["not_a_dict"]}]

        result = task_generator._validate_tasks(tasks)

        assert len(result) == 1
        assert result[0]["steps"][0]["task"] == "not_a_dict"
        assert result[0]["steps"][0]["description"] == "Execute step: not_a_dict"

    def test_validate_tasks_non_list_steps(self, task_generator) -> None:
        """Test _validate_tasks with non-list steps."""
        tasks = [{"name": "test", "description": "test", "steps": "not_a_list"}]

        result = task_generator._validate_tasks(tasks)

        assert len(result) == 0

    def test_validate_tasks_missing_required_fields(self, task_generator) -> None:
        """Test _validate_tasks with missing required fields."""
        tasks = [{"name": "test"}]  # Missing description and steps

        result = task_generator._validate_tasks(tasks)

        assert len(result) == 0

    def test_validate_task_definition_valid(
        self, task_generator, sample_task_definition
    ) -> None:
        """Test _validate_task_definition with valid input."""
        result = task_generator._validate_task_definition(sample_task_definition)

        assert result is True

    def test_validate_task_definition_invalid(self, task_generator) -> None:
        """Test _validate_task_definition with invalid input."""
        task_def = TaskDefinition(
            task_id="test",
            name="",
            description="test",
            steps=[],
            dependencies=[],
            required_resources=[],
            estimated_duration=None,
            priority=3,
        )

        result = task_generator._validate_task_definition(task_def)

        assert result is False

    def test_validate_task_definition_invalid_priority(self, task_generator) -> None:
        """Test _validate_task_definition with invalid priority."""
        task_def = TaskDefinition(
            task_id="test",
            name="test",
            description="test",
            steps=[{"task": "step"}],
            dependencies=[],
            required_resources=[],
            estimated_duration=None,
            priority=6,
        )

        result = task_generator._validate_task_definition(task_def)

        assert result is False

    def test_validate_task_definition_invalid_step(self, task_generator) -> None:
        """Test _validate_task_definition with invalid step."""
        task_def = TaskDefinition(
            task_id="test",
            name="test",
            description="test",
            steps=["not_a_dict"],
            dependencies=[],
            required_resources=[],
            estimated_duration=None,
            priority=3,
        )

        result = task_generator._validate_task_definition(task_def)

        assert result is False

    def test_validate_task_definition_missing_step_task(self, task_generator) -> None:
        """Test _validate_task_definition with step missing task key."""
        task_def = TaskDefinition(
            task_id="test",
            name="test",
            description="test",
            steps=[{"description": "step"}],
            dependencies=[],
            required_resources=[],
            estimated_duration=None,
            priority=3,
        )

        result = task_generator._validate_task_definition(task_def)

        assert result is False

    def test_validate_task_definition_empty_fields(self, task_generator) -> None:
        """Test _validate_task_definition with empty required fields."""
        task_def = TaskDefinition(
            task_id="test",
            name="",
            description="test",
            steps=[{"task": "step"}],
            dependencies=[],
            required_resources=[],
            estimated_duration=None,
            priority=3,
        )

        result = task_generator._validate_task_definition(task_def)

        assert result is False

    def test_establish_relationships(self, task_generator) -> None:
        """Test _establish_relationships method."""
        tasks = [{"name": "test", "description": "test", "steps": [{"task": "step"}]}]

        task_generator._establish_relationships(tasks)

        # Method is currently a no-op, so just verify it doesn't raise an exception
        assert len(tasks) == 1

    def test_build_hierarchy(self, task_generator) -> None:
        """Test _build_hierarchy method."""
        tasks = [
            {
                "id": "task1",
                "name": "test1",
                "description": "test1",
                "steps": [{"task": "step1"}],
                "dependencies": [],
            },
            {
                "id": "task2",
                "name": "test2",
                "description": "test2",
                "steps": [{"task": "step2"}],
                "dependencies": ["task1"],
            },
        ]

        task_generator._build_hierarchy(tasks)

        assert tasks[0]["level"] == 0
        assert tasks[1]["level"] == 1

    def test_build_hierarchy_missing_task_id(self, task_generator) -> None:
        """Test _build_hierarchy with missing task_id."""
        tasks = [{"name": "test", "description": "test", "steps": [{"task": "step"}]}]

        task_generator._build_hierarchy(tasks)

        assert tasks[0]["level"] == 0

    def test_convert_to_dict(self, task_generator, sample_task_definition) -> None:
        """Test _convert_to_dict method."""
        result = task_generator._convert_to_dict(sample_task_definition)

        assert result["id"] == "task1"
        assert result["name"] == "Test Task"
        assert result["description"] == "Test description"

    def test_convert_to_task_dict(self, task_generator) -> None:
        """Test _convert_to_task_dict method."""
        task_definitions = [
            {"task": "Task 1", "intent": "Intent 1", "steps": [{"task": "step"}]},
            {"task": "Task 2", "intent": "Intent 2", "steps": [{"task": "step"}]},
        ]

        result = task_generator._convert_to_task_dict(task_definitions)

        assert len(result) == 2
        assert "task_1" in result
        assert "task_2" in result
        assert result["task_1"]["name"] == "Task 1"
        assert result["task_2"]["name"] == "Task 2"

    def test_add_provided_tasks_with_breakdown(self, task_generator) -> None:
        """Test add_provided_tasks with task breakdown logic."""
        user_tasks = [{"task": "Complex Task", "intent": "Complex intent"}]

        with (
            patch.object(
                task_generator, "_check_task_breakdown_original"
            ) as mock_check,
            patch.object(task_generator, "_generate_task_steps_original") as mock_steps,
            patch.object(task_generator, "_validate_task_definition") as mock_validate,
        ):
            mock_check.return_value = True
            mock_steps.return_value = [{"task": "Step 1"}, {"task": "Step 2"}]
            mock_validate.return_value = True

            result = task_generator.add_provided_tasks(user_tasks, "intro")

            assert len(result) == 1
            assert result[0]["steps"] == [{"task": "Step 1"}, {"task": "Step 2"}]

    def test_add_provided_tasks_add_name_field(self, task_generator) -> None:
        """Test add_provided_tasks with name field addition."""
        user_tasks = [{"task": "Test Task", "intent": "Test intent"}]

        with (
            patch.object(
                task_generator, "_check_task_breakdown_original"
            ) as mock_check,
            patch.object(task_generator, "_validate_task_definition") as mock_validate,
        ):
            mock_check.return_value = False
            mock_validate.return_value = True

            result = task_generator.add_provided_tasks(user_tasks, "intro")

            assert len(result) == 1
            assert result[0]["name"] == "Test Task"

    def test_add_provided_tasks_empty(self, task_generator) -> None:
        """Test add_provided_tasks with empty input."""
        result = task_generator.add_provided_tasks([], "intro")

        assert result == []

    def test_add_provided_tasks_valid_all_fields(self, task_generator) -> None:
        """Test add_provided_tasks with valid task containing all fields."""
        user_tasks = [
            {
                "id": "task1",
                "name": "Test Task",
                "description": "Test description",
                "steps": [{"task": "step1"}],
                "dependencies": ["task2"],
                "required_resources": ["resource1"],
                "estimated_duration": "2 hours",
                "priority": 4,
            }
        ]

        with patch.object(task_generator, "_validate_task_definition") as mock_validate:
            mock_validate.return_value = True

            result = task_generator.add_provided_tasks(user_tasks, "intro")

            assert len(result) == 1
            assert result[0]["id"] == "task1"
            assert result[0]["name"] == "Test Task"

    def test_add_provided_tasks_missing_optional_fields(self, task_generator) -> None:
        """Test add_provided_tasks with missing optional fields."""
        user_tasks = [{"task": "Test Task", "intent": "Test intent"}]

        with (
            patch.object(
                task_generator, "_check_task_breakdown_original"
            ) as mock_check,
            patch.object(task_generator, "_validate_task_definition") as mock_validate,
        ):
            mock_check.return_value = False
            mock_validate.return_value = True

            result = task_generator.add_provided_tasks(user_tasks, "intro")

            assert len(result) == 1
            # The priority is set in the TaskDefinition, not in the returned dict
            # Check that the task was processed successfully
            assert result[0]["task"] == "Test Task"

    def test_add_provided_tasks_invalid_and_exception(self, task_generator) -> None:
        """Test add_provided_tasks with invalid task and exception handling."""
        user_tasks = [{"invalid": "task"}]

        with patch.object(task_generator, "_validate_task_definition") as mock_validate:
            mock_validate.side_effect = Exception("Validation error")

            result = task_generator.add_provided_tasks(user_tasks, "intro")

            assert result == []

    def test_process_objective_handles_dict_content(
        self, patched_model_generate
    ) -> None:
        gen = patched_model_generate["generator"]
        patched_model_generate["mock_generate"].return_value = {
            "content": '[{"task": "t"}]'
        }
        result = gen._process_objective("obj", "intro", "docs")
        assert "tasks" in result

    def test_process_objective_handles_non_json_response(
        self, patched_model_generate
    ) -> None:
        gen = patched_model_generate["generator"]
        patched_model_generate["mock_generate"].return_value = {"text": "no json here"}
        result = gen._process_objective("obj", "intro", "docs")
        assert result["tasks"] == []

    def test_process_objective_handles_json_decode_error(
        self, patched_model_generate
    ) -> None:
        gen = patched_model_generate["generator"]
        patched_model_generate["mock_generate"].return_value = {
            "text": "[not valid json]"
        }
        result = gen._process_objective("obj", "intro", "docs")
        assert result["tasks"] == []

    def test_generate_high_level_tasks_handles_string_response(
        self, patched_model_invoke
    ) -> None:
        gen = patched_model_invoke["generator"]
        patched_model_invoke["mock_invoke"].return_value = '[{"task": "t"}]'
        result = gen._generate_high_level_tasks("intro")
        assert isinstance(result, list)

    def test_generate_high_level_tasks_handles_invalid_json(
        self, patched_model_invoke
    ) -> None:
        gen = patched_model_invoke["generator"]
        patched_model_invoke["mock_invoke"].return_value = "not a json list"
        result = gen._generate_high_level_tasks("intro")
        assert result == []

    def test_generate_high_level_tasks_handles_exception(
        self, patched_model_invoke
    ) -> None:
        gen = patched_model_invoke["generator"]
        patched_model_invoke["mock_invoke"].side_effect = Exception("fail")
        result = gen._generate_high_level_tasks("intro")
        assert result == []

    def test_generate_task_steps_original_handles_invalid_json(
        self, patched_model_invoke
    ) -> None:
        gen = patched_model_invoke["generator"]
        patched_model_invoke["mock_invoke"].return_value = "not a json list"
        result = gen._generate_task_steps_original("task", "intent")
        assert isinstance(result, list)
        assert "task" in result[0]

    def test_generate_task_steps_original_handles_exception(
        self, patched_model_invoke
    ) -> None:
        gen = patched_model_invoke["generator"]
        patched_model_invoke["mock_invoke"].side_effect = Exception("fail")
        result = gen._generate_task_steps_original("task", "intent")
        assert isinstance(result, list)
        assert "task" in result[0]

    def test_validate_tasks_handles_non_dict_step(self, task_generator) -> None:
        tasks = [
            {
                "name": "task1",
                "description": "desc",
                "steps": [123],
            }
        ]
        result = task_generator._validate_tasks(tasks)
        assert isinstance(result, list)

    def test_convert_to_task_definitions_handles_empty(self, task_generator) -> None:
        result = task_generator._convert_to_task_definitions([])
        assert result == []
