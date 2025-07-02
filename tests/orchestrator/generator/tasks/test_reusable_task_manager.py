"""Comprehensive tests for ReusableTaskManager.

This module provides extensive test coverage for the ReusableTaskManager class,
including all methods, edge cases, error conditions, and the ReusableTask dataclass.
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from arklex.orchestrator.generator.tasks.reusable_task_manager import (
    ReusableTask,
    ReusableTaskManager,
)


@pytest.fixture
def mock_model() -> Mock:
    """Create a mock model for testing."""
    model = Mock()
    return model


@pytest.fixture
def sample_tasks() -> list[dict[str, Any]]:
    """Sample tasks for testing."""
    return [
        {
            "name": "Task 1",
            "description": "Description 1",
            "steps": [
                {"task": "Step 1", "description": "Step 1 desc"},
                {"task": "Step 2", "description": "Step 2 desc"},
            ],
        },
        {
            "name": "Task 2",
            "description": "Description 2",
            "steps": [
                {"task": "Step 1", "description": "Step 1 desc"},
            ],
        },
    ]


@pytest.fixture
def sample_template() -> ReusableTask:
    """Sample template for testing."""
    return ReusableTask(
        template_id="tid1",
        name="Template 1",
        description="Desc",
        steps=[{"task": "Step", "description": "Desc"}],
        parameters={"param1": "string"},
        examples=[],
        version="1.0",
        category="cat",
    )


@pytest.fixture
def sample_template_with_multiple_params() -> ReusableTask:
    """Sample template with multiple parameters for testing."""
    return ReusableTask(
        template_id="tid4",
        name="Template 4",
        description="Desc",
        steps=[{"task": "Step", "description": "Desc"}],
        parameters={"param1": "string", "param2": "int"},
        examples=[],
        version="1.0",
        category="cat",
    )


@pytest.fixture
def sample_template_string_param() -> ReusableTask:
    """Sample template with string parameter for testing."""
    return ReusableTask(
        template_id="tid_string",
        name="Template String",
        description="Desc",
        steps=[{"task": "Step"}],
        parameters={"param1": "string"},
        examples=[],
        version="1.0",
        category="cat",
    )


@pytest.fixture
def sample_template_number_param() -> ReusableTask:
    """Sample template with number parameter for testing."""
    return ReusableTask(
        template_id="tid_number",
        name="Template Number",
        description="Desc",
        steps=[{"task": "Step"}],
        parameters={"param1": "number"},
        examples=[],
        version="1.0",
        category="cat",
    )


@pytest.fixture
def sample_template_boolean_param() -> ReusableTask:
    """Sample template with boolean parameter for testing."""
    return ReusableTask(
        template_id="tid_boolean",
        name="Template Boolean",
        description="Desc",
        steps=[{"task": "Step"}],
        parameters={"param1": "boolean"},
        examples=[],
        version="1.0",
        category="cat",
    )


@pytest.fixture
def invalid_template() -> ReusableTask:
    """Invalid template for testing validation."""
    return ReusableTask(
        template_id="",
        name="",
        description="",
        steps=[],
        parameters={},
        examples=[],
        version="",
        category="",
    )


@pytest.fixture
def reusable_task_manager(mock_model: Mock) -> ReusableTaskManager:
    """Create a ReusableTaskManager instance for testing."""
    return ReusableTaskManager(
        model=mock_model, role="test_role", user_objective="test objective"
    )


@pytest.fixture
def patched_validate_parameters(
    reusable_task_manager: ReusableTaskManager,
) -> dict[str, Any]:
    """Create a ReusableTaskManager with patched _validate_parameters method."""
    with patch.object(reusable_task_manager, "_validate_parameters") as mock_validate:
        yield {"manager": reusable_task_manager, "mock_validate": mock_validate}


class TestReusableTaskManager:
    """Test the ReusableTaskManager class."""

    def test_generate_reusable_tasks(
        self,
        reusable_task_manager: ReusableTaskManager,
        sample_tasks: list[dict[str, Any]],
    ) -> None:
        """Test successful reusable task generation."""
        templates = reusable_task_manager.generate_reusable_tasks(sample_tasks)
        assert isinstance(templates, dict)
        assert len(templates) > 0
        for t in templates.values():
            assert isinstance(t, dict)
            assert "name" in t
            assert "description" in t
            assert "steps" in t

    def test_generate_reusable_tasks_empty(
        self, reusable_task_manager: ReusableTaskManager
    ) -> None:
        """Test reusable task generation with empty tasks."""
        templates = reusable_task_manager.generate_reusable_tasks([])
        assert isinstance(templates, dict)
        assert len(templates) == 0

    def test_generate_reusable_tasks_with_patterns_fallback(
        self, reusable_task_manager: ReusableTaskManager
    ) -> None:
        """Test the fallback case when no patterns are identified but tasks exist."""
        # Create a task that will result in no patterns but tasks exist
        tasks = [{"name": "Task 1", "description": "Description 1", "steps": []}]
        templates = reusable_task_manager.generate_reusable_tasks(tasks)
        assert isinstance(templates, dict)
        # The template will fail validation due to missing steps, so expect zero templates
        assert len(templates) == 0

    def test_instantiate_template_success(
        self, reusable_task_manager: ReusableTaskManager, sample_template: ReusableTask
    ) -> None:
        """Test successful template instantiation."""
        reusable_task_manager._templates["tid1"] = sample_template
        params = {"param1": "value"}
        instance = reusable_task_manager.instantiate_template("tid1", params)
        assert isinstance(instance, dict)
        assert "steps" in instance

    def test_instantiate_template_not_found(
        self, reusable_task_manager: ReusableTaskManager
    ) -> None:
        """Test template instantiation with non-existent template."""
        with pytest.raises(ValueError, match="Template not found"):
            reusable_task_manager.instantiate_template("not_exist", {})

    def test_instantiate_template_invalid_params(
        self, reusable_task_manager: ReusableTaskManager, sample_template: ReusableTask
    ) -> None:
        """Test template instantiation with invalid parameters."""
        reusable_task_manager._templates["tid1"] = sample_template
        with pytest.raises(ValueError, match="Invalid parameters"):
            reusable_task_manager.instantiate_template("tid1", {"param1": 123})

    def test_instantiate_template_exception(
        self, patched_validate_parameters: dict[str, Any], sample_template: ReusableTask
    ) -> None:
        """Test template instantiation with validation exception."""
        manager = patched_validate_parameters["manager"]
        mock_validate = patched_validate_parameters["mock_validate"]
        manager._templates["tid3"] = sample_template
        mock_validate.return_value = False
        with pytest.raises(ValueError, match="Invalid parameters"):
            manager.instantiate_template("tid3", {"param1": "v"})

    def test_identify_patterns(
        self,
        reusable_task_manager: ReusableTaskManager,
        sample_tasks: list[dict[str, Any]],
    ) -> None:
        """Test pattern identification from tasks."""
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        assert isinstance(patterns, list)
        assert len(patterns) == len(sample_tasks)
        for p in patterns:
            assert "name" in p
            assert "description" in p
            assert "steps" in p
            assert "parameters" in p

    def test_extract_components(
        self,
        reusable_task_manager: ReusableTaskManager,
        sample_tasks: list[dict[str, Any]],
    ) -> None:
        """Test component extraction from patterns."""
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        components = reusable_task_manager._extract_components(patterns)
        assert isinstance(components, list)
        assert len(components) > 0
        for c in components:
            assert "name" in c
            assert "description" in c
            assert "steps" in c
            assert "parameters" in c

    def test_extract_components_with_required_fields(
        self, reusable_task_manager: ReusableTaskManager
    ) -> None:
        """Test extracting components with required_fields in steps."""
        patterns = [
            {
                "name": "Task with required fields",
                "description": "Description",
                "steps": [
                    {
                        "task": "Step 1",
                        "description": "Step 1 desc",
                        "required_fields": ["field1"],
                    }
                ],
                "parameters": {},
            }
        ]
        components = reusable_task_manager._extract_components(patterns)
        assert len(components) == 1
        assert "parameters" in components[0]
        assert "field1" in components[0]["parameters"]
        assert components[0]["parameters"]["field1"] == "string"

    def test_extract_components_with_duplicate_fields(
        self, reusable_task_manager: ReusableTaskManager
    ) -> None:
        """Test that duplicate required fields are handled correctly."""
        patterns = [
            {
                "name": "Task 1",
                "description": "Description 1",
                "steps": [
                    {
                        "task": "Step 1",
                        "description": "Step 1 desc",
                        "required_fields": ["field1", "field2"],
                    }
                ],
                "parameters": {},
            },
            {
                "name": "Task 2",
                "description": "Description 2",
                "steps": [
                    {
                        "task": "Step 1",
                        "description": "Step 1 desc",
                        "required_fields": ["field1", "field3"],
                    }
                ],
                "parameters": {},
            },
        ]
        components = reusable_task_manager._extract_components(patterns)
        assert len(components) == 2
        # First component should have field1 and field2
        assert "parameters" in components[0]
        assert "field1" in components[0]["parameters"]
        assert "field2" in components[0]["parameters"]
        # Second component should have field1 and field3
        assert "parameters" in components[1]
        assert "field1" in components[1]["parameters"]
        assert "field3" in components[1]["parameters"]

    def test_create_templates(
        self,
        reusable_task_manager: ReusableTaskManager,
        sample_tasks: list[dict[str, Any]],
    ) -> None:
        """Test template creation from components."""
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        components = reusable_task_manager._extract_components(patterns)
        templates = reusable_task_manager._create_templates(components)
        assert isinstance(templates, dict)
        assert len(templates) > 0
        for t in templates.values():
            assert isinstance(t, ReusableTask)

    def test_validate_templates(
        self,
        reusable_task_manager: ReusableTaskManager,
        sample_tasks: list[dict[str, Any]],
    ) -> None:
        """Test template validation."""
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        components = reusable_task_manager._extract_components(patterns)
        templates = reusable_task_manager._create_templates(components)
        validated = reusable_task_manager._validate_templates(templates)
        assert isinstance(validated, dict)
        assert len(validated) > 0

    def test_validate_template(
        self,
        reusable_task_manager: ReusableTaskManager,
        sample_tasks: list[dict[str, Any]],
        invalid_template: ReusableTask,
    ) -> None:
        """Test individual template validation."""
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        components = reusable_task_manager._extract_components(patterns)
        templates = reusable_task_manager._create_templates(components)
        # Test valid template
        valid_template = list(templates.values())[0]
        assert reusable_task_manager._validate_template(valid_template)
        # Test invalid template
        assert not reusable_task_manager._validate_template(invalid_template)

    def test_validate_template_missing_template_id(
        self, reusable_task_manager: ReusableTaskManager
    ) -> None:
        """Test template validation with missing template_id."""
        template = ReusableTask(
            template_id="",  # Empty template_id
            name="Test Template",
            description="Test Description",
            steps=[{"task": "Step 1"}],
            parameters={},
            examples=[],
            version="1.0",
            category="test",
        )
        assert not reusable_task_manager._validate_template(template)

    def test_validate_template_missing_name(
        self, reusable_task_manager: ReusableTaskManager
    ) -> None:
        """Test template validation with missing name."""
        template = ReusableTask(
            template_id="test_id",
            name="",  # Empty name
            description="Test Description",
            steps=[{"task": "Step 1"}],
            parameters={},
            examples=[],
            version="1.0",
            category="test",
        )
        assert not reusable_task_manager._validate_template(template)

    def test_validate_template_missing_description(
        self, reusable_task_manager: ReusableTaskManager
    ) -> None:
        """Test template validation with missing description."""
        template = ReusableTask(
            template_id="test_id",
            name="Test Template",
            description="",  # Empty description
            steps=[{"task": "Step 1"}],
            parameters={},
            examples=[],
            version="1.0",
            category="test",
        )
        assert not reusable_task_manager._validate_template(template)

    def test_validate_template_missing_steps(
        self, reusable_task_manager: ReusableTaskManager
    ) -> None:
        """Test template validation with missing steps."""
        template = ReusableTask(
            template_id="test_id",
            name="Test Template",
            description="Test Description",
            steps=[],  # Empty steps
            parameters={},
            examples=[],
            version="1.0",
            category="test",
        )
        assert not reusable_task_manager._validate_template(template)

    def test_validate_template_steps_not_list(
        self, reusable_task_manager: ReusableTaskManager
    ) -> None:
        """Test template validation with steps not being a list."""
        template = ReusableTask(
            template_id="test_id",
            name="Test Template",
            description="Test Description",
            steps="not a list",  # Wrong type
            parameters={},
            examples=[],
            version="1.0",
            category="test",
        )
        assert not reusable_task_manager._validate_template(template)

    def test_validate_template_parameters_not_dict(
        self, reusable_task_manager: ReusableTaskManager
    ) -> None:
        """Test template validation with parameters not being a dict."""
        template = ReusableTask(
            template_id="test_id",
            name="Test Template",
            description="Test Description",
            steps=[{"task": "Step 1"}],
            parameters="not a dict",  # Wrong type
            examples=[],
            version="1.0",
            category="test",
        )
        assert not reusable_task_manager._validate_template(template)

    def test_validate_template_examples_not_list(
        self, reusable_task_manager: ReusableTaskManager
    ) -> None:
        """Test template validation with examples not being a list."""
        template = ReusableTask(
            template_id="test_id",
            name="Test Template",
            description="Test Description",
            steps=[{"task": "Step 1"}],
            parameters={},
            examples="not a list",  # Wrong type
            version="1.0",
            category="test",
        )
        assert not reusable_task_manager._validate_template(template)

    def test_validate_template_missing_version(
        self, reusable_task_manager: ReusableTaskManager
    ) -> None:
        """Test template validation with missing version."""
        template = ReusableTask(
            template_id="test_id",
            name="Test Template",
            description="Test Description",
            steps=[{"task": "Step 1"}],
            parameters={},
            examples=[],
            version="",  # Empty version
            category="test",
        )
        assert not reusable_task_manager._validate_template(template)

    def test_validate_template_missing_category(
        self, reusable_task_manager: ReusableTaskManager
    ) -> None:
        """Test template validation with missing category."""
        template = ReusableTask(
            template_id="test_id",
            name="Test Template",
            description="Test Description",
            steps=[{"task": "Step 1"}],
            parameters={},
            examples=[],
            version="1.0",
            category="",  # Empty category
        )
        assert not reusable_task_manager._validate_template(template)

    def test_validate_parameters(
        self,
        reusable_task_manager: ReusableTaskManager,
        sample_template_with_multiple_params: ReusableTask,
    ) -> None:
        """Test parameter validation with multiple parameters."""
        template = sample_template_with_multiple_params
        # Valid parameters
        assert reusable_task_manager._validate_parameters(
            template, {"param1": "value", "param2": 42}
        )
        # Missing required parameter
        assert not reusable_task_manager._validate_parameters(
            template, {"param1": "value"}
        )
        # Extra parameter (should be allowed)
        assert reusable_task_manager._validate_parameters(
            template, {"param1": "value", "param2": 42, "extra": "value"}
        )

    def test_validate_parameters_string_type(
        self,
        reusable_task_manager: ReusableTaskManager,
        sample_template_string_param: ReusableTask,
    ) -> None:
        """Test parameter validation with string type."""
        template = sample_template_string_param
        # Valid string
        assert reusable_task_manager._validate_parameters(
            template, {"param1": "valid string"}
        )
        # Invalid type
        assert not reusable_task_manager._validate_parameters(template, {"param1": 123})

    def test_validate_parameters_number_type(
        self,
        reusable_task_manager: ReusableTaskManager,
        sample_template_number_param: ReusableTask,
    ) -> None:
        """Test parameter validation with number type."""
        template = sample_template_number_param
        # Valid numbers
        assert reusable_task_manager._validate_parameters(template, {"param1": 42})
        assert reusable_task_manager._validate_parameters(template, {"param1": 3.14})
        # Invalid type
        assert not reusable_task_manager._validate_parameters(
            template, {"param1": "not a number"}
        )

    def test_validate_parameters_boolean_type(
        self,
        reusable_task_manager: ReusableTaskManager,
        sample_template_boolean_param: ReusableTask,
    ) -> None:
        """Test parameter validation with boolean type."""
        template = sample_template_boolean_param
        # Valid booleans
        assert reusable_task_manager._validate_parameters(template, {"param1": True})
        assert reusable_task_manager._validate_parameters(template, {"param1": False})
        # Invalid type
        assert not reusable_task_manager._validate_parameters(
            template, {"param1": "not a boolean"}
        )

    def test_create_instance(
        self, reusable_task_manager: ReusableTaskManager, sample_template: ReusableTask
    ) -> None:
        """Test instance creation from template."""
        params = {"param1": "value"}
        instance = reusable_task_manager._create_instance(sample_template, params)
        assert isinstance(instance, dict)
        assert "steps" in instance

    def test_categorize_templates(
        self,
        reusable_task_manager: ReusableTaskManager,
        sample_tasks: list[dict[str, Any]],
    ) -> None:
        """Test template categorization."""
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        components = reusable_task_manager._extract_components(patterns)
        templates = reusable_task_manager._create_templates(components)
        validated = reusable_task_manager._validate_templates(templates)
        reusable_task_manager._categorize_templates(validated)
        assert "general" in reusable_task_manager._template_categories
        assert len(reusable_task_manager._template_categories["general"]) > 0

    def test_categorize_templates_with_different_categories(
        self, reusable_task_manager: ReusableTaskManager
    ) -> None:
        """Test categorization with different template categories."""
        # Create templates with different categories
        template1 = ReusableTask(
            template_id="tid1",
            name="Template 1",
            description="Desc",
            steps=[{"task": "Step"}],
            parameters={},
            examples=[],
            version="1.0",
            category="utility",
        )
        template2 = ReusableTask(
            template_id="tid2",
            name="Template 2",
            description="Desc",
            steps=[{"task": "Step"}],
            parameters={},
            examples=[],
            version="1.0",
            category="workflow",
        )
        templates = {"tid1": template1, "tid2": template2}
        validated = reusable_task_manager._validate_templates(templates)
        reusable_task_manager._categorize_templates(validated)
        assert "utility" in reusable_task_manager._template_categories
        assert "workflow" in reusable_task_manager._template_categories
        assert len(reusable_task_manager._template_categories["utility"]) == 1
        assert len(reusable_task_manager._template_categories["workflow"]) == 1

    def test_convert_to_dict(
        self,
        reusable_task_manager: ReusableTaskManager,
        sample_tasks: list[dict[str, Any]],
    ) -> None:
        """Test template conversion to dictionary."""
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        components = reusable_task_manager._extract_components(patterns)
        templates = reusable_task_manager._create_templates(components)
        for template in templates.values():
            d = reusable_task_manager._convert_to_dict(template)
            assert isinstance(d, dict)
            assert "template_id" in d
            assert "name" in d
            assert "description" in d
            assert "steps" in d
            assert "parameters" in d
            assert "examples" in d
            assert "version" in d
            assert "category" in d

    def test_generate_reusable_tasks_log_info_line(
        self, reusable_task_manager: ReusableTaskManager
    ) -> None:
        """Test reusable task generation with logging coverage."""
        tasks = [
            {
                "name": "Task 1",
                "description": "Description 1",
                "steps": [
                    {"task": "Step 1", "description": "Step 1 desc"},
                ],
            }
        ]
        templates = reusable_task_manager.generate_reusable_tasks(tasks)
        assert isinstance(templates, dict)
        assert len(templates) > 0

    def test_generate_reusable_tasks_with_no_patterns_but_tasks_exist(self) -> None:
        """Test generate_reusable_tasks when no patterns are identified but tasks exist."""
        from arklex.orchestrator.generator.tasks.reusable_task_manager import (
            ReusableTaskManager,
        )

        # Create a manager
        manager = ReusableTaskManager(
            model=Mock(), role="test_role", user_objective="test objective"
        )

        # Create tasks that will result in no patterns but tasks exist
        tasks = [
            {
                "name": "Task 1",
                "description": "Description 1",
                "steps": [{"task": "Step 1", "description": "Step 1 desc"}],
            }
        ]

        # Mock _identify_patterns to return empty list
        with patch.object(manager, "_identify_patterns", return_value=[]):
            templates = manager.generate_reusable_tasks(tasks)

            # Should create a fallback pattern from the first task
            assert isinstance(templates, dict)
            # The template should be created from the first task
            assert len(templates) > 0
