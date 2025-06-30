"""Tests for the prompts module.

This module provides comprehensive tests for the PromptManager class
and prompt template functionality.
"""

import re

import pytest

from arklex.orchestrator.generator import prompts

# --- Fixtures ---


@pytest.fixture
def all_prompt_names() -> list:
    """All expected prompt names for testing."""
    return [
        "generate_tasks",
        "generate_reusable_tasks",
        "check_best_practice",
        "generate_intents",
        "generate_best_practice",
        "embed_resources",
        "embed_builder_obj",
    ]


@pytest.fixture
def prompt_manager() -> prompts.PromptManager:
    """Create a PromptManager instance for testing."""
    return prompts.PromptManager()


@pytest.fixture
def sample_prompt_fields() -> set[str]:
    """Get sample fields from a prompt template for testing."""
    template = prompts.generate_tasks_sys_prompt
    return set(re.findall(r"{([a-zA-Z0-9_]+)}", template))


@pytest.fixture
def valid_prompt_kwargs() -> dict:
    """Create valid kwargs for prompt testing."""
    return {
        "role": "test_role",
        "u_objective": "test_objective",
        "intro": "test_intro",
        "docs": "test_docs",
        "instructions": "test_instructions",
        "existing_tasks": "[]",
        "task_name": "test_task",
        "task_description": "test_description",
        "practice_name": "test_practice",
        "practice_description": "test_practice_description",
        "resource_name": "test_resource",
        "resource_description": "test_resource_description",
        "builder_obj": "test_builder_obj",
    }


@pytest.fixture
def invalid_prompt_name() -> str:
    """Create an invalid prompt name for testing."""
    return "not_a_prompt"


# --- Test Classes ---


class TestPromptTemplates:
    """Test prompt template structure and content."""

    def test_prompt_templates_are_strings(self) -> None:
        """Should have all top-level prompt templates as strings."""
        assert isinstance(prompts.generate_tasks_sys_prompt, str)
        assert isinstance(prompts.generate_reusable_tasks_sys_prompt, str)
        assert isinstance(prompts.check_best_practice_sys_prompt, str)
        assert isinstance(prompts.generate_intents_sys_prompt, str)
        assert isinstance(prompts.generate_best_practice_sys_prompt, str)
        assert isinstance(prompts.embed_builder_obj_sys_prompt, str)
        assert isinstance(prompts.embed_resources_sys_prompt, str)
        assert isinstance(prompts.task_intents_prediction_prompt, str)


class TestPromptManager:
    """Test PromptManager initialization and functionality."""

    def test_prompt_manager_init(
        self, prompt_manager: prompts.PromptManager, all_prompt_names: list
    ) -> None:
        """Should initialize with all expected prompts and attributes."""
        # All expected keys should be present in .prompts
        for key in all_prompt_names:
            assert key in prompt_manager.prompts
            assert isinstance(prompt_manager.prompts[key], str)

        # All attributes should be set
        assert hasattr(prompt_manager, "generate_tasks_sys_prompt")
        assert hasattr(prompt_manager, "generate_reusable_tasks_sys_prompt")
        assert hasattr(prompt_manager, "check_best_practice_sys_prompt")
        assert hasattr(prompt_manager, "generate_best_practice_sys_prompt")
        assert hasattr(prompt_manager, "embed_resources_sys_prompt")
        assert hasattr(prompt_manager, "embed_builder_obj_sys_prompt")
        assert hasattr(prompt_manager, "task_intents_prediction_prompt")

    def test_get_prompt_valid(
        self, prompt_manager: prompts.PromptManager, all_prompt_names: list
    ) -> None:
        """Should format prompts with valid names and required kwargs."""
        # Test all prompt names with minimal required kwargs
        for name in all_prompt_names:
            # Find all curly-brace fields in the template
            template = prompt_manager.prompts[name]
            fields = set(re.findall(r"{([a-zA-Z0-9_]+)}", template))
            kwargs = {field: f"val_{field}" for field in fields}
            result = prompt_manager.get_prompt(name, **kwargs)

            assert isinstance(result, str)
            # All fields should be replaced
            for field in fields:
                assert f"val_{field}" in result or f"{{{field}}}" not in result

    def test_get_prompt_invalid_name(
        self, prompt_manager: prompts.PromptManager, invalid_prompt_name: str
    ) -> None:
        """Should raise ValueError for invalid prompt names."""
        with pytest.raises(ValueError) as exc_info:
            prompt_manager.get_prompt(invalid_prompt_name)
        assert "Prompt template 'not_a_prompt' not found" in str(exc_info.value)

    def test_get_prompt_missing_kwargs(
        self, prompt_manager: prompts.PromptManager
    ) -> None:
        """Should raise KeyError when required kwargs are missing."""
        with pytest.raises(KeyError):
            prompt_manager.get_prompt(
                "generate_tasks", role="foo"
            )  # missing required fields

    def test_get_prompt_extra_kwargs(
        self, prompt_manager: prompts.PromptManager, valid_prompt_kwargs: dict
    ) -> None:
        """Should ignore extra kwargs when formatting prompts."""
        # Add extra kwargs to valid ones
        test_kwargs = valid_prompt_kwargs.copy()
        test_kwargs.update({"extra1": "ignoreme", "extra2": 123})

        # Extra kwargs should be ignored by str.format
        result = prompt_manager.get_prompt("generate_tasks", **test_kwargs)
        assert "test_role" in result and "test_objective" in result

    def test_prompt_manager_repr(self, prompt_manager: prompts.PromptManager) -> None:
        """Should have proper string representation."""
        assert "PromptManager" in str(prompt_manager)
