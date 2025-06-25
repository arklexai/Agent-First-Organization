"""Comprehensive tests for ReusableTaskManager.

This module provides extensive test coverage for the ReusableTaskManager class,
including all methods, edge cases, error conditions, and the ReusableTask dataclass.
"""

import pytest
from unittest.mock import Mock, patch

from arklex.orchestrator.generator.tasks.reusable_task_manager import (
    ReusableTask,
    ReusableTaskManager,
)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    return model


@pytest.fixture
def sample_tasks():
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
def sample_template():
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
def sample_template_with_multiple_params():
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
def sample_template_string_param():
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
def sample_template_number_param():
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
def sample_template_boolean_param():
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
def invalid_template():
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
def reusable_task_manager(mock_model):
    """Create a ReusableTaskManager instance for testing."""
    return ReusableTaskManager(
        model=mock_model, role="test_role", user_objective="test objective"
    )


@pytest.fixture
def patched_validate_parameters(reusable_task_manager):
    """Create a ReusableTaskManager with patched _validate_parameters method."""
    with patch.object(reusable_task_manager, "_validate_parameters") as mock_validate:
        yield {"manager": reusable_task_manager, "mock_validate": mock_validate}


class TestReusableTaskManager:
    """Test the ReusableTaskManager class."""

    def test_generate_reusable_tasks(self, reusable_task_manager, sample_tasks) -> None:
        """Test successful reusable task generation."""
        templates = reusable_task_manager.generate_reusable_tasks(sample_tasks)
        assert isinstance(templates, dict)
        assert len(templates) > 0
        for t in templates.values():
            assert isinstance(t, dict)
            assert "name" in t
            assert "description" in t
            assert "steps" in t

    def test_generate_reusable_tasks_empty(self, reusable_task_manager) -> None:
        """Test reusable task generation with empty tasks."""
        templates = reusable_task_manager.generate_reusable_tasks([])
        assert isinstance(templates, dict)
        assert len(templates) == 0

    def test_generate_reusable_tasks_with_patterns_fallback(
        self, reusable_task_manager
    ) -> None:
        """Test the fallback case when no patterns are identified but tasks exist."""
        # Create a task that will result in no patterns but tasks exist
        tasks = [{"name": "Task 1", "description": "Description 1", "steps": []}]
        templates = reusable_task_manager.generate_reusable_tasks(tasks)
        assert isinstance(templates, dict)
        # The template will fail validation due to missing steps, so expect zero templates
        assert len(templates) == 0

    def test_instantiate_template_success(
        self, reusable_task_manager, sample_template
    ) -> None:
        """Test successful template instantiation."""
        reusable_task_manager._templates["tid1"] = sample_template
        params = {"param1": "value"}
        instance = reusable_task_manager.instantiate_template("tid1", params)
        assert isinstance(instance, dict)
        assert "steps" in instance

    def test_instantiate_template_not_found(self, reusable_task_manager) -> None:
        """Test template instantiation with non-existent template."""
        with pytest.raises(ValueError, match="Template not found"):
            reusable_task_manager.instantiate_template("not_exist", {})

    def test_instantiate_template_invalid_params(
        self, reusable_task_manager, sample_template
    ) -> None:
        """Test template instantiation with invalid parameters."""
        reusable_task_manager._templates["tid2"] = sample_template
        with pytest.raises(ValueError, match="Invalid parameters"):
            reusable_task_manager.instantiate_template("tid2", {"wrong": 1})

    def test_instantiate_template_exception(
        self, patched_validate_parameters, sample_template
    ) -> None:
        """Test template instantiation with validation exception."""
        manager = patched_validate_parameters["manager"]
        mock_validate = patched_validate_parameters["mock_validate"]
        mock_validate.side_effect = Exception("fail")

        manager._templates["tid3"] = sample_template
        with pytest.raises(Exception, match="fail"):
            manager.instantiate_template("tid3", {"param1": "v"})

    def test_identify_patterns(self, reusable_task_manager, sample_tasks) -> None:
        """Test pattern identification from tasks."""
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        assert isinstance(patterns, list)
        assert len(patterns) == len(sample_tasks)
        for p in patterns:
            assert "name" in p
            assert "description" in p
            assert "steps" in p
            assert "parameters" in p

    def test_extract_components(self, reusable_task_manager, sample_tasks) -> None:
        """Test component extraction from patterns."""
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        components = reusable_task_manager._extract_components(patterns)
        assert isinstance(components, list)
        assert len(components) == len(patterns)

    def test_extract_components_with_required_fields(
        self, reusable_task_manager
    ) -> None:
        """Test extracting components with required_fields in steps."""
        patterns = [
            {
                "name": "Task with fields",
                "description": "Description",
                "steps": [
                    {"task": "Step 1", "required_fields": ["field1", "field2"]},
                    {"task": "Step 2", "required_fields": ["field1", "field3"]},
                ],
            }
        ]
        components = reusable_task_manager._extract_components(patterns)
        assert len(components) == 1
        assert "field1" in components[0]["parameters"]
        assert "field2" in components[0]["parameters"]
        assert "field3" in components[0]["parameters"]
        assert components[0]["parameters"]["field1"] == "string"
        assert components[0]["parameters"]["field2"] == "string"
        assert components[0]["parameters"]["field3"] == "string"

    def test_extract_components_with_duplicate_fields(
        self, reusable_task_manager
    ) -> None:
        """Test that duplicate required fields are handled correctly."""
        patterns = [
            {
                "name": "Task with duplicate fields",
                "description": "Description",
                "steps": [
                    {"task": "Step 1", "required_fields": ["field1"]},
                    {"task": "Step 2", "required_fields": ["field1"]},  # Duplicate
                ],
            }
        ]
        components = reusable_task_manager._extract_components(patterns)
        assert len(components) == 1
        assert "field1" in components[0]["parameters"]
        assert components[0]["parameters"]["field1"] == "string"

    def test_create_templates(self, reusable_task_manager, sample_tasks) -> None:
        """Test template creation from components."""
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        components = reusable_task_manager._extract_components(patterns)
        templates = reusable_task_manager._create_templates(components)
        assert isinstance(templates, dict)
        assert len(templates) == len(components)
        for t in templates.values():
            assert isinstance(t, ReusableTask)

    def test_validate_templates(self, reusable_task_manager, sample_tasks) -> None:
        """Test template validation."""
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        components = reusable_task_manager._extract_components(patterns)
        templates = reusable_task_manager._create_templates(components)
        validated = reusable_task_manager._validate_templates(templates)
        assert isinstance(validated, dict)
        assert len(validated) == len(templates)
        for v in validated.values():
            assert isinstance(v, dict)

    def test_validate_template(
        self, reusable_task_manager, sample_tasks, invalid_template
    ) -> None:
        """Test individual template validation."""
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        components = reusable_task_manager._extract_components(patterns)
        templates = reusable_task_manager._create_templates(components)
        for t in templates.values():
            assert reusable_task_manager._validate_template(t)
        # Test with missing fields
        assert not reusable_task_manager._validate_template(invalid_template)

    def test_validate_template_missing_template_id(self, reusable_task_manager) -> None:
        """Test template validation with missing template_id."""
        template = ReusableTask(
            template_id="",
            name="Valid Name",
            description="Valid Description",
            steps=[{"task": "Step"}],
            parameters={},
            examples=[],
            version="1.0",
            category="general",
        )
        assert not reusable_task_manager._validate_template(template)

    def test_validate_template_missing_name(self, reusable_task_manager) -> None:
        """Test template validation with missing name."""
        template = ReusableTask(
            template_id="valid_id",
            name="",
            description="Valid Description",
            steps=[{"task": "Step"}],
            parameters={},
            examples=[],
            version="1.0",
            category="general",
        )
        assert not reusable_task_manager._validate_template(template)

    def test_validate_template_missing_description(self, reusable_task_manager) -> None:
        """Test template validation with missing description."""
        template = ReusableTask(
            template_id="valid_id",
            name="Valid Name",
            description="",
            steps=[{"task": "Step"}],
            parameters={},
            examples=[],
            version="1.0",
            category="general",
        )
        assert not reusable_task_manager._validate_template(template)

    def test_validate_template_missing_steps(self, reusable_task_manager) -> None:
        """Test template validation with missing steps."""
        template = ReusableTask(
            template_id="valid_id",
            name="Valid Name",
            description="Valid Description",
            steps=[],
            parameters={},
            examples=[],
            version="1.0",
            category="general",
        )
        assert not reusable_task_manager._validate_template(template)

    def test_validate_template_steps_not_list(self, reusable_task_manager) -> None:
        """Test template validation with steps not being a list."""
        template = ReusableTask(
            template_id="valid_id",
            name="Valid Name",
            description="Valid Description",
            steps="not a list",  # type: ignore
            parameters={},
            examples=[],
            version="1.0",
            category="general",
        )
        assert not reusable_task_manager._validate_template(template)

    def test_validate_template_parameters_not_dict(self, reusable_task_manager) -> None:
        """Test template validation with parameters not being a dict."""
        template = ReusableTask(
            template_id="valid_id",
            name="Valid Name",
            description="Valid Description",
            steps=[{"task": "Step"}],
            parameters="not a dict",  # type: ignore
            examples=[],
            version="1.0",
            category="general",
        )
        assert not reusable_task_manager._validate_template(template)

    def test_validate_template_examples_not_list(self, reusable_task_manager) -> None:
        """Test template validation with examples not being a list."""
        template = ReusableTask(
            template_id="valid_id",
            name="Valid Name",
            description="Valid Description",
            steps=[{"task": "Step"}],
            parameters={},
            examples="not a list",  # type: ignore
            version="1.0",
            category="general",
        )
        assert not reusable_task_manager._validate_template(template)

    def test_validate_template_missing_version(self, reusable_task_manager) -> None:
        """Test template validation with missing version."""
        template = ReusableTask(
            template_id="valid_id",
            name="Valid Name",
            description="Valid Description",
            steps=[{"task": "Step"}],
            parameters={},
            examples=[],
            version="",
            category="general",
        )
        assert not reusable_task_manager._validate_template(template)

    def test_validate_template_missing_category(self, reusable_task_manager) -> None:
        """Test template validation with missing category."""
        template = ReusableTask(
            template_id="valid_id",
            name="Valid Name",
            description="Valid Description",
            steps=[{"task": "Step"}],
            parameters={},
            examples=[],
            version="1.0",
            category="",
        )
        assert not reusable_task_manager._validate_template(template)

    def test_validate_parameters(
        self, reusable_task_manager, sample_template_with_multiple_params
    ) -> None:
        """Test parameter validation with multiple parameters."""
        template = sample_template_with_multiple_params
        assert reusable_task_manager._validate_parameters(
            template, {"param1": "v", "param2": 1}
        )
        assert not reusable_task_manager._validate_parameters(template, {"param1": "v"})
        assert not reusable_task_manager._validate_parameters(template, {"param2": 1})
        assert not reusable_task_manager._validate_parameters(template, {})

    def test_validate_parameters_string_type(
        self, reusable_task_manager, sample_template_string_param
    ) -> None:
        """Test parameter validation with string type."""
        template = sample_template_string_param
        # Valid string
        assert reusable_task_manager._validate_parameters(
            template, {"param1": "valid string"}
        )
        # Invalid: not a string
        assert not reusable_task_manager._validate_parameters(template, {"param1": 123})
        assert not reusable_task_manager._validate_parameters(
            template, {"param1": True}
        )

    def test_validate_parameters_number_type(
        self, reusable_task_manager, sample_template_number_param
    ) -> None:
        """Test parameter validation with number type."""
        template = sample_template_number_param
        # Valid numbers
        assert reusable_task_manager._validate_parameters(template, {"param1": 123})
        assert reusable_task_manager._validate_parameters(template, {"param1": 123.45})
        # Invalid: not a number
        assert not reusable_task_manager._validate_parameters(
            template, {"param1": "not a number"}
        )
        # Python treats bool as int, so True is accepted as a number. Accept this in the test.
        assert reusable_task_manager._validate_parameters(template, {"param1": True})

    def test_validate_parameters_boolean_type(
        self, reusable_task_manager, sample_template_boolean_param
    ) -> None:
        """Test parameter validation with boolean type."""
        template = sample_template_boolean_param
        # Valid booleans
        assert reusable_task_manager._validate_parameters(template, {"param1": True})
        assert reusable_task_manager._validate_parameters(template, {"param1": False})
        # Invalid: not a boolean
        assert not reusable_task_manager._validate_parameters(
            template, {"param1": "not a boolean"}
        )
        assert not reusable_task_manager._validate_parameters(template, {"param1": 123})

    def test_create_instance(self, reusable_task_manager, sample_template) -> None:
        """Test instance creation from template."""
        params = {"param1": "value"}
        instance = reusable_task_manager._create_instance(sample_template, params)
        assert isinstance(instance, dict)
        assert "steps" in instance

    def test_categorize_templates(self, reusable_task_manager, sample_tasks) -> None:
        """Test template categorization."""
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        components = reusable_task_manager._extract_components(patterns)
        templates = reusable_task_manager._create_templates(components)
        validated = reusable_task_manager._validate_templates(templates)
        reusable_task_manager._categorize_templates(validated)
        assert isinstance(reusable_task_manager._template_categories, dict)
        for cat, ids in reusable_task_manager._template_categories.items():
            assert isinstance(cat, str)
            assert isinstance(ids, list)

    def test_categorize_templates_with_different_categories(
        self, reusable_task_manager
    ) -> None:
        """Test categorization with different template categories."""
        templates = {
            "template1": {"category": "utility", "name": "Template 1"},
            "template2": {"category": "workflow", "name": "Template 2"},
            "template3": {"category": "utility", "name": "Template 3"},
            "template4": {
                "name": "Template 4"
            },  # No category, should default to "general"
        }
        reusable_task_manager._categorize_templates(templates)

        assert "utility" in reusable_task_manager._template_categories
        assert "workflow" in reusable_task_manager._template_categories
        assert "general" in reusable_task_manager._template_categories

        assert len(reusable_task_manager._template_categories["utility"]) == 2
        assert len(reusable_task_manager._template_categories["workflow"]) == 1
        assert len(reusable_task_manager._template_categories["general"]) == 1

    def test_convert_to_dict(self, reusable_task_manager, sample_tasks) -> None:
        """Test template conversion to dictionary."""
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        components = reusable_task_manager._extract_components(patterns)
        templates = reusable_task_manager._create_templates(components)
        for t in templates.values():
            d = reusable_task_manager._convert_to_dict(t)
            assert isinstance(d, dict)
            assert "template_id" in d
            assert "name" in d
            assert "description" in d
            assert "steps" in d
            assert "parameters" in d
            assert "examples" in d
            assert "version" in d
            assert "category" in d

    def test_generate_reusable_tasks_log_info_line(self, reusable_task_manager) -> None:
        """Test reusable task generation with logging coverage."""
        tasks = [
            {
                "name": "Log Test Task",
                "description": "Covers log line",
                "steps": [{"task": "Step 1", "description": "desc"}],
            }
        ]
        result = reusable_task_manager.generate_reusable_tasks(tasks)
        assert isinstance(result, dict)
