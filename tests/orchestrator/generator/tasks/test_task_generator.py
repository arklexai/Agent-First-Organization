"""Comprehensive tests for TaskGenerator.

This module provides extensive test coverage for the TaskGenerator class,
including all methods, edge cases, error conditions, and the TaskDefinition dataclass.
"""

import logging
import pytest
from unittest.mock import Mock, patch

from arklex.orchestrator.generator.tasks.task_generator import (
    TaskGenerator,
    TaskDefinition,
)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    return Mock()


@pytest.fixture
def task_generator(mock_model):
    """Create a TaskGenerator instance for testing."""
    return TaskGenerator(
        mock_model, "test_role", "test_objective", "test_instructions", "test_docs"
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


@pytest.fixture
def patched_generate_tasks(task_generator):
    """Create a TaskGenerator with patched generate_tasks methods."""
    with (
        patch.object(task_generator, "_generate_high_level_tasks") as mock_generate,
        patch.object(task_generator, "_check_task_breakdown_original") as mock_check,
        patch.object(task_generator, "_validate_tasks") as mock_validate,
    ):
        mock_generate.return_value = [{"task": "new_task", "intent": "new_intent"}]
        mock_check.return_value = False  # Task doesn't need breakdown
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
        yield {
            "generator": task_generator,
            "mock_generate": mock_generate,
            "mock_check": mock_check,
            "mock_validate": mock_validate,
        }


@pytest.fixture
def patched_process_objective(task_generator):
    """Create a TaskGenerator with patched _process_objective method."""
    with patch.object(task_generator, "_process_objective") as mock_process:
        mock_process.side_effect = Exception("Processing error")
        yield {"generator": task_generator, "mock_process": mock_process}


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


class TestTaskDefinition:
    """Test the TaskDefinition dataclass."""

    def test_task_definition_initialization(self, sample_task_definition) -> None:
        """Test TaskDefinition initialization."""
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
        """Test TaskDefinition with optional fields."""
        task_def = sample_task_definition_minimal
        assert task_def.estimated_duration is None
        assert task_def.priority == 1


class TestTaskGenerator:
    """Test the TaskGenerator class."""

    def test_task_generator_initialization(self, mock_model) -> None:
        """Test TaskGenerator initialization."""
        role = "test_role"
        user_objective = "test_objective"
        instructions = "test_instructions"
        documents = "test_documents"

        generator = TaskGenerator(
            mock_model, role, user_objective, instructions, documents
        )
        assert generator.model == mock_model
        assert generator.role == role
        assert generator.user_objective == user_objective
        assert generator.instructions == instructions
        assert generator.documents == documents

    def test_generate_tasks_with_existing_tasks(
        self, patched_generate_tasks, sample_existing_tasks
    ) -> None:
        """Test generate_tasks with existing tasks."""
        generator = patched_generate_tasks["generator"]
        mock_generate = patched_generate_tasks["mock_generate"]
        mock_check = patched_generate_tasks["mock_check"]
        mock_validate = patched_generate_tasks["mock_validate"]

        result = generator.generate_tasks("intro", sample_existing_tasks)

        mock_generate.assert_called_once()
        mock_check.assert_called_once()
        mock_validate.assert_called_once()
        assert len(result) == 1
        assert result[0]["name"] == "new_task"

    def test_generate_tasks_without_existing_tasks(
        self, patched_generate_tasks
    ) -> None:
        """Test generate_tasks without existing tasks."""
        generator = patched_generate_tasks["generator"]
        mock_generate = patched_generate_tasks["mock_generate"]
        mock_check = patched_generate_tasks["mock_check"]
        mock_validate = patched_generate_tasks["mock_validate"]

        result = generator.generate_tasks("intro")

        mock_generate.assert_called_once()
        mock_check.assert_called_once()
        mock_validate.assert_called_once()
        assert len(result) == 1
        assert result[0]["name"] == "new_task"

    def test_generate_tasks_with_exception(self, patched_process_objective) -> None:
        """Test generate_tasks with exception handling."""
        generator = patched_process_objective["generator"]

        result = generator.generate_tasks("intro")
        assert result == []

    def test_process_objective_with_existing_tasks(
        self, patched_model_generate, sample_existing_tasks
    ) -> None:
        """Test _process_objective with existing tasks."""
        generator = patched_model_generate["generator"]
        mock_generate = patched_model_generate["mock_generate"]

        mock_response = Mock()
        mock_response.generations = [
            [Mock(text='[{"task": "new_task", "intent": "new_intent"}]')]
        ]
        mock_generate.return_value = mock_response

        result = generator._process_objective(
            "objective", "intro", "docs", sample_existing_tasks
        )
        assert "tasks" in result
        assert len(result["tasks"]) == 1

    def test_process_objective_without_existing_tasks(
        self, patched_model_generate
    ) -> None:
        """Test _process_objective without existing tasks."""
        generator = patched_model_generate["generator"]
        mock_generate = patched_model_generate["mock_generate"]

        mock_response = Mock()
        mock_response.generations = [
            [Mock(text='[{"task": "new_task", "intent": "new_intent"}]')]
        ]
        mock_generate.return_value = mock_response

        result = generator._process_objective("objective", "intro", "docs")
        assert "tasks" in result
        assert len(result["tasks"]) == 1

    def test_process_objective_with_message_content(
        self, patched_model_generate
    ) -> None:
        """Test _process_objective with message content response."""
        generator = patched_model_generate["generator"]
        mock_generate = patched_model_generate["mock_generate"]

        logging.disable(logging.CRITICAL)
        try:
            mock_response = Mock()
            mock_message = Mock()
            mock_message.content = '[{"task": "new_task", "intent": "new_intent"}]'
            mock_generation = Mock()
            mock_generation.message = mock_message
            mock_generation.text = (
                None  # Ensure text is None so it uses message.content
            )
            mock_response.generations = [[mock_generation]]
            mock_generate.return_value = mock_response

            # Mock the hasattr calls to return the expected values
            with patch("builtins.hasattr") as mock_hasattr:
                mock_hasattr.side_effect = lambda obj, attr: {
                    (mock_response, "generations"): True,
                    (mock_generation, "text"): False,
                    (mock_generation, "message"): True,
                    (mock_message, "content"): True,
                }.get((obj, attr), False)

                result = generator._process_objective("objective", "intro", "docs")
                assert "tasks" in result
                assert len(result["tasks"]) == 1
        finally:
            logging.disable(logging.NOTSET)

    def test_process_objective_with_dict_response(self, patched_model_generate) -> None:
        """Test _process_objective with dictionary response."""
        generator = patched_model_generate["generator"]
        mock_generate = patched_model_generate["mock_generate"]

        mock_response = {"text": '[{"task": "new_task", "intent": "new_intent"}]'}
        mock_generate.return_value = mock_response

        result = generator._process_objective("objective", "intro", "docs")
        assert "tasks" in result
        assert len(result["tasks"]) == 1

    def test_process_objective_with_content_dict_response(
        self, patched_model_generate
    ) -> None:
        """Test _process_objective with content dictionary response."""
        generator = patched_model_generate["generator"]
        mock_generate = patched_model_generate["mock_generate"]

        mock_response = {"content": '[{"task": "new_task", "intent": "new_intent"}]'}
        mock_generate.return_value = mock_response

        result = generator._process_objective("objective", "intro", "docs")
        assert "tasks" in result
        assert len(result["tasks"]) == 1

    def test_process_objective_with_string_response(
        self, patched_model_generate
    ) -> None:
        """Test _process_objective with string response."""
        generator = patched_model_generate["generator"]
        mock_generate = patched_model_generate["mock_generate"]

        mock_response = '[{"task": "new_task", "intent": "new_intent"}]'
        mock_generate.return_value = mock_response

        result = generator._process_objective("objective", "intro", "docs")
        assert "tasks" in result
        assert len(result["tasks"]) == 1

    def test_process_objective_with_invalid_json(self, patched_model_generate) -> None:
        """Test _process_objective with invalid JSON response."""
        generator = patched_model_generate["generator"]
        mock_generate = patched_model_generate["mock_generate"]

        mock_response = Mock()
        mock_response.generations = [[Mock(text="invalid json")]]
        mock_generate.return_value = mock_response

        result = generator._process_objective("objective", "intro", "docs")
        assert "tasks" in result
        assert result["tasks"] == []

    def test_process_objective_with_no_json_array(self, patched_model_generate) -> None:
        """Test _process_objective with no JSON array in response."""
        generator = patched_model_generate["generator"]
        mock_generate = patched_model_generate["mock_generate"]

        mock_response = Mock()
        mock_response.generations = [[Mock(text='{"not": "an array"}')]]
        mock_generate.return_value = mock_response

        result = generator._process_objective("objective", "intro", "docs")
        assert "tasks" in result
        assert result["tasks"] == []

    def test_generate_high_level_tasks_with_existing_tasks(
        self, patched_model_invoke, sample_existing_tasks
    ) -> None:
        """Test _generate_high_level_tasks with existing tasks."""
        generator = patched_model_invoke["generator"]
        mock_invoke = patched_model_invoke["mock_invoke"]

        mock_response = Mock()
        mock_response.content = '[{"task": "new_task", "intent": "new_intent"}]'
        mock_invoke.return_value = mock_response

        result = generator._generate_high_level_tasks("intro", sample_existing_tasks)
        assert len(result) == 1
        assert result[0]["task"] == "new_task"

    def test_generate_high_level_tasks_without_existing_tasks(
        self, patched_model_invoke
    ) -> None:
        """Test _generate_high_level_tasks without existing tasks."""
        generator = patched_model_invoke["generator"]
        mock_invoke = patched_model_invoke["mock_invoke"]

        mock_response = Mock()
        mock_response.content = '[{"task": "new_task", "intent": "new_intent"}]'
        mock_invoke.return_value = mock_response

        result = generator._generate_high_level_tasks("intro")
        assert len(result) == 1
        assert result[0]["task"] == "new_task"

    def test_generate_high_level_tasks_with_string_response(
        self, patched_model_invoke
    ) -> None:
        """Test _generate_high_level_tasks with string response."""
        generator = patched_model_invoke["generator"]
        mock_invoke = patched_model_invoke["mock_invoke"]

        mock_response = '[{"task": "new_task", "intent": "new_intent"}]'
        mock_invoke.return_value = mock_response

        result = generator._generate_high_level_tasks("intro")
        assert len(result) == 1
        assert result[0]["task"] == "new_task"

    def test_generate_high_level_tasks_with_invalid_json(
        self, patched_model_invoke
    ) -> None:
        """Test _generate_high_level_tasks with invalid JSON."""
        generator = patched_model_invoke["generator"]
        mock_invoke = patched_model_invoke["mock_invoke"]

        mock_response = Mock()
        mock_response.content = "invalid json"
        mock_invoke.return_value = mock_response

        result = generator._generate_high_level_tasks("intro")
        assert result == []

    def test_generate_high_level_tasks_with_exception(
        self, patched_model_invoke
    ) -> None:
        """Test _generate_high_level_tasks with exception."""
        generator = patched_model_invoke["generator"]
        mock_invoke = patched_model_invoke["mock_invoke"]

        mock_invoke.side_effect = Exception("API error")

        result = generator._generate_high_level_tasks("intro")
        assert result == []

    def test_check_task_breakdown_original_yes(self, patched_model_invoke) -> None:
        """Test _check_task_breakdown_original with 'yes' response."""
        generator = patched_model_invoke["generator"]
        mock_invoke = patched_model_invoke["mock_invoke"]

        mock_response = Mock()
        mock_response.content = '{"answer": "yes"}'
        mock_invoke.return_value = mock_response

        result = generator._check_task_breakdown_original("task_name", "task_intent")
        assert result is True

    def test_check_task_breakdown_original_no(self, patched_model_invoke) -> None:
        """Test _check_task_breakdown_original with 'no' response."""
        generator = patched_model_invoke["generator"]
        mock_invoke = patched_model_invoke["mock_invoke"]

        mock_response = Mock()
        mock_response.content = '{"answer": "no"}'
        mock_invoke.return_value = mock_response

        result = generator._check_task_breakdown_original("task_name", "task_intent")
        assert result is False

    def test_check_task_breakdown_original_with_string_response(
        self, patched_model_invoke
    ) -> None:
        """Test _check_task_breakdown_original with string response."""
        generator = patched_model_invoke["generator"]
        mock_invoke = patched_model_invoke["mock_invoke"]

        mock_response = '{"answer": "yes"}'
        mock_invoke.return_value = mock_response

        result = generator._check_task_breakdown_original("task_name", "task_intent")
        assert result is True

    def test_check_task_breakdown_original_with_invalid_json(
        self, patched_model_invoke
    ) -> None:
        """Test _check_task_breakdown_original with invalid JSON."""
        generator = patched_model_invoke["generator"]
        mock_invoke = patched_model_invoke["mock_invoke"]

        mock_response = Mock()
        mock_response.content = "invalid json"
        mock_invoke.return_value = mock_response

        result = generator._check_task_breakdown_original("task_name", "task_intent")
        assert result is True  # Default to breakdown

    def test_check_task_breakdown_original_with_exception(
        self, patched_model_invoke
    ) -> None:
        """Test _check_task_breakdown_original with exception."""
        generator = patched_model_invoke["generator"]
        mock_invoke = patched_model_invoke["mock_invoke"]

        mock_invoke.side_effect = Exception("API error")

        result = generator._check_task_breakdown_original("task_name", "task_intent")
        assert result is True  # Default to breakdown

    def test_generate_task_steps_original(self, patched_model_invoke) -> None:
        """Test _generate_task_steps_original."""
        generator = patched_model_invoke["generator"]
        mock_invoke = patched_model_invoke["mock_invoke"]

        mock_response = Mock()
        mock_response.content = '[{"task": "step1", "description": "Execute step1"}, {"task": "step2", "description": "Execute step2"}]'
        mock_invoke.return_value = mock_response

        result = generator._generate_task_steps_original("task_name", "task_intent")
        assert len(result) == 2
        assert result[0]["task"] == "step1"
        assert result[0]["description"] == "Execute step1"

    def test_generate_task_steps_original_with_string_response(
        self, patched_model_invoke
    ) -> None:
        """Test _generate_task_steps_original with string response."""
        generator = patched_model_invoke["generator"]
        mock_invoke = patched_model_invoke["mock_invoke"]

        mock_response = '[{"task": "step1", "description": "Execute step1"}]'
        mock_invoke.return_value = mock_response

        result = generator._generate_task_steps_original("task_name", "task_intent")
        assert len(result) == 1
        assert result[0]["task"] == "step1"
        assert result[0]["description"] == "Execute step1"

    def test_generate_task_steps_original_with_invalid_json(
        self, patched_model_invoke
    ) -> None:
        """Test _generate_task_steps_original with invalid JSON."""
        generator = patched_model_invoke["generator"]
        mock_invoke = patched_model_invoke["mock_invoke"]

        mock_response = Mock()
        mock_response.content = "invalid json"
        mock_invoke.return_value = mock_response

        result = generator._generate_task_steps_original("task_name", "task_intent")
        assert len(result) == 1
        assert result[0]["task"] == "Execute task_name"
        assert result[0]["description"] == "Execute the task: task_name"

    def test_generate_task_steps_original_with_exception(
        self, patched_model_invoke
    ) -> None:
        """Test _generate_task_steps_original with exception."""
        generator = patched_model_invoke["generator"]
        mock_invoke = patched_model_invoke["mock_invoke"]

        mock_invoke.side_effect = Exception("API error")

        result = generator._generate_task_steps_original("task_name", "task_intent")
        assert len(result) == 1
        assert result[0]["task"] == "Execute task_name"
        assert result[0]["description"] == "Execute the task: task_name"

    def test_convert_to_task_definitions(
        self, task_generator, sample_tasks_with_steps
    ) -> None:
        """Test _convert_to_task_definitions."""
        result = task_generator._convert_to_task_definitions(sample_tasks_with_steps)
        assert len(result) == 1
        assert isinstance(result[0], TaskDefinition)
        assert result[0].task_id == "task_1"
        assert result[0].name == "task1"

    def test_validate_tasks(self, task_generator, sample_tasks) -> None:
        """Test _validate_tasks."""
        result = task_generator._validate_tasks(sample_tasks)
        assert len(result) == 2
        assert result[0]["name"] == "task1"
        assert result[1]["name"] == "task2"

    def test_validate_tasks_with_invalid_task(self, task_generator) -> None:
        """Test _validate_tasks with invalid task."""
        tasks = [
            {"name": "task1", "description": "intent1", "steps": [{"task": "step1"}]},
            {"invalid": "task"},  # Missing required fields
        ]

        result = task_generator._validate_tasks(tasks)
        assert len(result) == 1  # Only valid task should remain
        assert result[0]["name"] == "task1"

    def test_validate_task_definition_valid(
        self, task_generator, sample_task_definition
    ) -> None:
        """Test _validate_task_definition with valid task."""
        result = task_generator._validate_task_definition(sample_task_definition)
        assert result is True

    def test_validate_task_definition_invalid(self, task_generator) -> None:
        """Test _validate_task_definition with invalid task."""
        task_def = TaskDefinition(
            task_id="",
            name="",
            description="",
            steps=[],
            dependencies=[],
            required_resources=[],
            estimated_duration=None,
            priority=0,
        )

        result = task_generator._validate_task_definition(task_def)
        assert result is False

    def test_establish_relationships(self, task_generator) -> None:
        """Test _establish_relationships."""
        tasks = [
            {"task": "task1", "intent": "intent1"},
            {"task": "task2", "intent": "intent2"},
        ]

        task_generator._establish_relationships(tasks)
        # Should not raise an error

    def test_build_hierarchy(self, task_generator) -> None:
        """Test _build_hierarchy."""
        tasks = [
            {"task": "task1", "intent": "intent1"},
            {"task": "task2", "intent": "intent2"},
        ]

        task_generator._build_hierarchy(tasks)
        # Should not raise an error

    def test_convert_to_dict(self, task_generator, sample_task_definition) -> None:
        """Test _convert_to_dict."""
        result = task_generator._convert_to_dict(sample_task_definition)
        assert result["id"] == "task1"
        assert result["name"] == "Test Task"
        assert result["description"] == "Test description"
        assert len(result["steps"]) == 1
        assert result["estimated_duration"] == "1 hour"
        assert result["priority"] == 3

    def test_convert_to_task_dict(self, task_generator) -> None:
        """Test _convert_to_task_dict."""
        task_definitions = [
            {
                "task": "task1",
                "intent": "Test description",
                "steps": [{"task": "step1"}],
                "dependencies": [],
                "required_resources": [],
                "estimated_duration": "1 hour",
                "priority": 3,
            }
        ]

        result = task_generator._convert_to_task_dict(task_definitions)
        assert "task_1" in result
        assert result["task_1"]["name"] == "task1"
        assert result["task_1"]["description"] == "Test description"

    def test_add_provided_tasks_invalid_and_exception(self, task_generator) -> None:
        """Test add_provided_tasks with invalid tasks and exceptions."""
        # Invalid: missing required fields
        user_tasks = [
            {"id": "1", "name": "", "description": "", "steps": []},
            {
                "id": "2",
                "name": "Valid",
                "description": "desc",
                "steps": [{"task": "t"}],
                "priority": 10,
            },
        ]
        # Exception: steps is not a list
        user_tasks.append(
            {"id": "3", "name": "Bad", "description": "desc", "steps": "notalist"}
        )
        # Exception: steps contains invalid type
        user_tasks.append(
            {"id": "4", "name": "Bad2", "description": "desc", "steps": [123]}
        )
        # All are invalid, so result should be empty
        result = task_generator.add_provided_tasks(user_tasks, "intro")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_establish_relationships_noop(self, task_generator) -> None:
        """Test _establish_relationships with no-op cases."""
        # Should not raise for empty or simple input
        task_generator._establish_relationships([])
        task_generator._establish_relationships([{"id": "1", "dependencies": []}])

    def test_build_hierarchy_edge_cases(self, task_generator) -> None:
        """Test _build_hierarchy with edge cases."""
        # Missing id
        tasks = [{"name": "A"}]
        task_generator._build_hierarchy(tasks)
        assert tasks[0]["level"] == 0
        # Duplicate id
        tasks = [{"id": "1", "dependencies": []}, {"id": "1", "dependencies": []}]
        task_generator._build_hierarchy(tasks)
        assert all("level" in t for t in tasks)
        # Circular dependency
        tasks = [
            {"id": "1", "dependencies": ["2"]},
            {"id": "2", "dependencies": ["1"]},
        ]
        task_generator._build_hierarchy(tasks)
        assert all("level" in t for t in tasks)
        # Empty dependencies
        tasks = [{"id": "1"}]
        task_generator._build_hierarchy(tasks)
        assert tasks[0]["level"] == 0

    def test_convert_to_task_dict_various_inputs(self, task_generator) -> None:
        """Test _convert_to_task_dict with various inputs."""
        # Normal input
        task_defs = [
            {"task": "A", "intent": "desc", "steps": ["step1"]},
            {"task": "B", "intent": "desc2", "steps": []},
        ]
        result = task_generator._convert_to_task_dict(task_defs)
        assert isinstance(result, dict)
        assert "task_1" in result and "task_2" in result
        # Missing fields: should raise KeyError if 'intent' is missing
        task_defs = [{"task": "A"}]
        try:
            task_generator._convert_to_task_dict(task_defs)
            assert False, "Expected KeyError for missing 'intent'"
        except KeyError:
            pass
        # Empty input
        result = task_generator._convert_to_task_dict([])
        assert result == {}

    def test_add_provided_tasks_empty(self, task_generator) -> None:
        """Test add_provided_tasks with empty input."""
        result = task_generator.add_provided_tasks([], "intro")
        assert result == []

    def test_add_provided_tasks_valid_all_fields(self, task_generator) -> None:
        """Test add_provided_tasks with valid tasks containing all fields."""
        user_tasks = [
            {
                "id": "1",
                "name": "Task",
                "description": "desc",
                "steps": [{"task": "t"}],
                "dependencies": ["2"],
                "required_resources": ["r"],
                "estimated_duration": "2h",
                "priority": 2,
            }
        ]
        result = task_generator.add_provided_tasks(user_tasks, "intro")
        assert len(result) == 1
        assert result[0]["id"] == "1"

    def test_add_provided_tasks_missing_optional_fields(self, task_generator) -> None:
        """Test add_provided_tasks with tasks missing optional fields."""
        user_tasks = [
            {"id": "1", "name": "Task", "description": "desc", "steps": [{"task": "t"}]}
        ]
        result = task_generator.add_provided_tasks(user_tasks, "intro")
        assert len(result) == 1
        assert result[0]["id"] == "1"

    def test_validate_tasks_various_invalid(self, task_generator) -> None:
        """Test _validate_tasks with various invalid inputs."""
        # Missing required fields
        tasks = [{"name": "A"}]
        assert task_generator._validate_tasks(tasks) == []
        # Steps not a list
        tasks = [{"name": "A", "description": "desc", "steps": "notalist"}]
        assert task_generator._validate_tasks(tasks) == []
        # Steps as list of strings
        tasks = [{"name": "A", "description": "desc", "steps": ["s1", "s2"]}]
        out = task_generator._validate_tasks(tasks)
        assert isinstance(out[0]["steps"][0], dict)
        # Steps with missing 'task' key
        tasks = [{"name": "A", "description": "desc", "steps": [{"not_task": "x"}]}]
        assert task_generator._validate_tasks(tasks) == []
        # Bad dependencies/resources/priority
        tasks = [
            {
                "name": "A",
                "description": "desc",
                "steps": [{"task": "t"}],
                "dependencies": "notalist",
                "required_resources": "notalist",
                "priority": "bad",
            }
        ]
        out = task_generator._validate_tasks(tasks)
        assert out[0]["dependencies"] == []
        assert out[0]["required_resources"] == []
        assert out[0]["priority"] == 3
        # Priority out of range
        tasks = [
            {
                "name": "A",
                "description": "desc",
                "steps": [{"task": "t"}],
                "priority": 10,
            }
        ]
        out = task_generator._validate_tasks(tasks)
        assert out[0]["priority"] == 3

    def test_validate_task_definition_cases(self, task_generator) -> None:
        """Test _validate_task_definition with various cases."""
        # Valid
        td = TaskDefinition("id", "n", "d", [{"task": "t"}], [], [], "1h", 2)
        assert task_generator._validate_task_definition(td)
        # Missing name
        td = TaskDefinition("id", "", "d", [{"task": "t"}], [], [], "1h", 2)
        assert not task_generator._validate_task_definition(td)
        # Missing description
        td = TaskDefinition("id", "n", "", [{"task": "t"}], [], [], "1h", 2)
        assert not task_generator._validate_task_definition(td)
        # Missing steps
        td = TaskDefinition("id", "n", "d", [], [], [], "1h", 2)
        assert not task_generator._validate_task_definition(td)
        # Steps missing 'task'
        td = TaskDefinition("id", "n", "d", [{"not_task": "x"}], [], [], "1h", 2)
        assert not task_generator._validate_task_definition(td)
        # Priority out of range
        td = TaskDefinition("id", "n", "d", [{"task": "t"}], [], [], "1h", 10)
        assert not task_generator._validate_task_definition(td)

    def test_convert_to_dict_variants(self, task_generator) -> None:
        """Test _convert_to_dict with various variants."""
        td = TaskDefinition("id", "n", "d", [{"task": "t"}], ["d"], ["r"], "1h", 2)
        d = task_generator._convert_to_dict(td)
        assert d["id"] == "id"
        # Minimal
        td = TaskDefinition("id", "n", "d", [{"task": "t"}], [], [], None, 1)
        d = task_generator._convert_to_dict(td)
        assert d["estimated_duration"] is None

    def test_build_hierarchy_nonexistent_dependency(self, task_generator) -> None:
        """Test _build_hierarchy with nonexistent dependencies."""
        tasks = [
            {"id": "1", "dependencies": ["2"]},
            {"id": "2", "dependencies": ["3"]},
        ]
        task_generator._build_hierarchy(tasks)
        assert all("level" in t for t in tasks)

    def test_convert_to_task_definitions_variants(self, task_generator) -> None:
        """Test _convert_to_task_definitions with various variants."""
        # Steps as strings
        tasks = [{"task": "A", "steps": ["s1"]}]
        out = task_generator._convert_to_task_definitions(tasks)
        assert isinstance(out[0], TaskDefinition)
        # Steps as dicts
        tasks = [{"task": "A", "steps": [{"task": "s1"}]}]
        out = task_generator._convert_to_task_definitions(tasks)
        assert isinstance(out[0], TaskDefinition)
        # Missing steps
        tasks = [{"task": "A"}]
        out = task_generator._convert_to_task_definitions(tasks)
        assert isinstance(out[0], TaskDefinition)

    def test_generate_high_level_tasks_error_handling(self, task_generator) -> None:
        """Test _generate_high_level_tasks error handling."""
        task_generator.model.invoke = Mock(side_effect=Exception("fail"))
        out = task_generator._generate_high_level_tasks("intro")
        assert out == []

    def test_check_task_breakdown_original_error_handling(self, task_generator) -> None:
        """Test _check_task_breakdown_original error handling."""
        task_generator.model.invoke = Mock(side_effect=Exception("fail"))
        out = task_generator._check_task_breakdown_original("t", "i")
        assert out is True

    def test_generate_task_steps_original_error_handling(self, task_generator) -> None:
        """Test _generate_task_steps_original error handling."""
        task_generator.model.invoke = Mock(side_effect=Exception("fail"))
        out = task_generator._generate_task_steps_original("t", "i")
        assert isinstance(out, list) and out[0]["task"].startswith("Execute")
