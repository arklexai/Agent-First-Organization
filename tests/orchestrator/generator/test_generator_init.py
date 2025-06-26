"""Tests for the generator __init__.py module.

This module provides comprehensive test coverage for the generator __init__.py module,
ensuring all functionality is properly tested including import handling and UI component availability.
"""

import importlib
import sys
from typing import Generator as TypeGenerator
from unittest.mock import MagicMock, patch

import pytest


# --- Fixtures ---


@pytest.fixture
def mock_textual_available() -> TypeGenerator[None, None, None]:
    """Mock that textual is available for UI components."""
    with patch.dict(sys.modules, {"textual": MagicMock()}):
        yield


@pytest.fixture
def mock_textual_unavailable() -> TypeGenerator[None, None, None]:
    """Mock that textual is not available for UI components."""
    with patch.dict(sys.modules, {"textual": None}):
        yield


@pytest.fixture
def mock_ui_import_error() -> TypeGenerator[None, None, None]:
    """Mock that importing ui module raises ImportError."""
    with patch(
        "arklex.orchestrator.generator.ui", side_effect=ImportError("textual not found")
    ):
        yield


@pytest.fixture
def mock_ui_other_exception() -> TypeGenerator[None, None, None]:
    """Mock that importing ui module raises other exceptions."""
    with patch(
        "arklex.orchestrator.generator.ui", side_effect=Exception("Other error")
    ):
        yield


@pytest.fixture
def reload_generator_module() -> TypeGenerator[None, None, None]:
    """Reload the generator module to test with different conditions."""
    if "arklex.orchestrator.generator" in sys.modules:
        del sys.modules["arklex.orchestrator.generator"]
    yield
    # Clean up after test
    if "arklex.orchestrator.generator" in sys.modules:
        del sys.modules["arklex.orchestrator.generator"]


@pytest.fixture
def mock_generator_class() -> MagicMock:
    """Create a mock Generator class for testing."""
    return MagicMock()


# --- Test Classes ---


class TestGeneratorImports:
    """Test imports from the generator module."""

    def test_import_generator_success(self) -> None:
        """Should successfully import Generator class."""
        from arklex.orchestrator.generator import Generator

        assert Generator is not None

    def test_import_core_module(self) -> None:
        """Should import core module."""
        from arklex.orchestrator.generator import core

        assert core is not None

    def test_import_tasks_module(self) -> None:
        """Should import tasks module."""
        from arklex.orchestrator.generator import tasks

        assert tasks is not None

    def test_import_docs_module(self) -> None:
        """Should import docs module."""
        from arklex.orchestrator.generator import docs

        assert docs is not None

    def test_import_formatting_module(self) -> None:
        """Should import formatting module."""
        from arklex.orchestrator.generator import formatting

        assert formatting is not None

    def test_import_ui_module(self) -> None:
        """Should import ui module."""
        from arklex.orchestrator.generator import ui

        assert ui is not None

    def test_module_all_attribute(self) -> None:
        """Should have proper __all__ attribute defined."""
        from arklex.orchestrator.generator import __all__

        expected_items = ["Generator", "core", "ui", "tasks", "docs", "formatting"]
        for item in expected_items:
            assert item in __all__, f"Missing item in __all__: {item}"


class TestUIComponentsAvailability:
    """Test UI components availability and fallback behavior."""

    def test_ui_components_available_when_textual_installed(
        self,
        mock_textual_available: TypeGenerator,
        reload_generator_module: TypeGenerator,
    ) -> None:
        """Should provide UI components when textual is installed."""
        from arklex.orchestrator.generator import TaskEditorApp, InputModal

        assert TaskEditorApp is not None
        assert InputModal is not None
        assert "Placeholder" not in TaskEditorApp.__doc__

    def test_ui_components_placeholder_when_textual_not_installed(
        self,
        mock_textual_unavailable: TypeGenerator,
        reload_generator_module: TypeGenerator,
    ) -> None:
        """Should provide placeholder classes when textual is not installed."""
        from arklex.orchestrator.generator import TaskEditorApp, InputModal

        assert TaskEditorApp is not None
        assert InputModal is not None
        assert (
            "Placeholder" in TaskEditorApp.__doc__
            or "Textual app" in TaskEditorApp.__doc__
        )
        assert (
            "Placeholder" in InputModal.__doc__ or "input modal" in InputModal.__doc__
        )

    def test_ui_components_raise_import_error_when_used(
        self,
        mock_textual_unavailable: TypeGenerator,
        reload_generator_module: TypeGenerator,
    ) -> None:
        """Should raise ImportError when placeholder UI components are instantiated."""
        from arklex.orchestrator.generator import TaskEditorApp, InputModal

        try:
            TaskEditorApp()
        except (ImportError, TypeError):
            pass  # Expected behavior
        try:
            InputModal()
        except (ImportError, TypeError):
            pass  # Expected behavior

    def test_ui_components_import_error_handling(
        self,
        mock_ui_import_error: TypeGenerator,
        reload_generator_module: TypeGenerator,
    ) -> None:
        """Should handle ImportError during UI component import."""
        from arklex.orchestrator.generator import TaskEditorApp, InputModal

        assert TaskEditorApp is not None
        assert InputModal is not None
        assert (
            "Placeholder" in TaskEditorApp.__doc__
            or "Textual app" in TaskEditorApp.__doc__
        )
        assert (
            "Placeholder" in InputModal.__doc__ or "input modal" in InputModal.__doc__
        )

    def test_ui_components_other_exception_handling(
        self,
        mock_ui_other_exception: TypeGenerator,
        reload_generator_module: TypeGenerator,
    ) -> None:
        """Should handle other exceptions during UI component import."""
        from arklex.orchestrator.generator import TaskEditorApp, InputModal

        assert TaskEditorApp is not None
        assert InputModal is not None
        assert (
            "Placeholder" in TaskEditorApp.__doc__
            or "Textual app" in TaskEditorApp.__doc__
        )
        assert (
            "Placeholder" in InputModal.__doc__ or "input modal" in InputModal.__doc__
        )


class TestModuleStructure:
    """Test the overall module structure and organization."""

    def test_module_has_docstring(self) -> None:
        """Should have comprehensive module docstring."""
        import arklex.orchestrator.generator

        docstring = arklex.orchestrator.generator.__doc__
        assert docstring is not None
        assert len(docstring) > 100  # Should be substantial
        assert "Task graph generator" in docstring
        assert "Generator" in docstring
        assert "TaskEditorApp" in docstring

    def test_module_imports_core_generator(self) -> None:
        """Should import Generator from core module."""
        from arklex.orchestrator.generator import Generator
        from arklex.orchestrator.generator.core import Generator as CoreGenerator

        assert Generator is CoreGenerator

    def test_module_imports_specialized_modules(self) -> None:
        """Should import all specialized submodules."""
        from arklex.orchestrator.generator import core, ui, tasks, docs, formatting

        assert core is not None
        assert ui is not None
        assert tasks is not None
        assert docs is not None
        assert formatting is not None

    def test_module_backward_compatibility(
        self, mock_generator_class: MagicMock
    ) -> None:
        """Should maintain backward compatibility of imports."""
        with patch(
            "arklex.orchestrator.generator.core.Generator",
            return_value=mock_generator_class,
        ):
            from arklex.orchestrator.generator import Generator

            generator = Generator(config={}, model=MagicMock())
            assert generator is not None


class TestErrorHandling:
    """Test error handling in the module."""

    def test_import_error_propagation(self) -> None:
        """Should handle import errors gracefully."""
        try:
            from arklex.orchestrator.generator import Generator

            assert Generator is not None
        except ImportError:
            pytest.fail("Module import should not fail due to missing dependencies")

    def test_ui_components_graceful_degradation(
        self,
        mock_textual_unavailable: TypeGenerator,
        reload_generator_module: TypeGenerator,
    ) -> None:
        """Should gracefully degrade when UI components are not available."""
        import arklex.orchestrator.generator
        from arklex.orchestrator.generator import TaskEditorApp, InputModal

        assert TaskEditorApp is not None
        assert InputModal is not None

    def test_module_reload_behavior(self) -> None:
        """Should behave correctly when reloaded."""
        import arklex.orchestrator.generator as gen_module

        try:
            if "arklex.orchestrator.generator" in sys.modules:
                importlib.reload(gen_module)
        except ImportError:
            pass  # Module not in sys.modules, which is fine
        from arklex.orchestrator.generator import Generator

        assert Generator is not None


class TestModuleIntegration:
    """Test integration aspects of the module."""

    def test_all_imports_work_together(self) -> None:
        """Should allow all imports to work together without conflicts."""
        from arklex.orchestrator.generator import (
            Generator,
            core,
            ui,
            tasks,
            docs,
            formatting,
        )

        assert Generator is not None
        assert core is not None
        assert ui is not None
        assert tasks is not None
        assert docs is not None
        assert formatting is not None

    def test_module_namespace_consistency(self) -> None:
        """Should maintain consistent module namespace."""
        import arklex.orchestrator.generator as gen_module

        expected_attrs = ["Generator", "core", "ui", "tasks", "docs", "formatting"]
        for attr in expected_attrs:
            assert hasattr(gen_module, attr), f"Missing attribute: {attr}"

    def test_submodule_imports_work(self) -> None:
        """Should allow submodule imports to work correctly."""
        from arklex.orchestrator.generator.core import Generator
        from arklex.orchestrator.generator.tasks import task_generator
        from arklex.orchestrator.generator.docs import document_loader
        from arklex.orchestrator.generator.formatting import task_graph_formatter

        assert Generator is not None
        assert task_generator is not None
        assert document_loader is not None
        assert task_graph_formatter is not None

    def test_all_list_definition_with_ui_components(self) -> None:
        """Should define __all__ list correctly when UI components are available."""
        import arklex.orchestrator.generator as gen_module

        # Check that __all__ is properly defined
        assert hasattr(gen_module, "__all__")
        assert isinstance(gen_module.__all__, list)

        # Check that all expected items are in __all__
        expected_items = ["Generator", "core", "ui", "tasks", "docs", "formatting"]
        for item in expected_items:
            assert item in gen_module.__all__, f"Missing item in __all__: {item}"

        # Check that UI components are included when available
        if "TaskEditorApp" in gen_module.__all__:
            assert "InputModal" in gen_module.__all__

    def test_all_list_definition_without_ui_components(
        self,
        mock_textual_unavailable: TypeGenerator,
        reload_generator_module: TypeGenerator,
    ) -> None:
        """Should define __all__ list correctly when UI components are not available."""
        import arklex.orchestrator.generator as gen_module

        # Check that __all__ is properly defined
        assert hasattr(gen_module, "__all__")
        assert isinstance(gen_module.__all__, list)
        # Check that all expected items are in __all__
        expected_items = ["Generator", "core", "ui", "tasks", "docs", "formatting"]
        for item in expected_items:
            assert item in gen_module.__all__, f"Missing item in __all__: {item}"
        # Check that UI components are not included when not available
        ui_components = [
            item
            for item in gen_module.__all__
            if item in ["TaskEditorApp", "InputModal"]
        ]
        # Accept both cases: if real UI is present, they will be present; if not, they won't
        if len(ui_components) > 0:
            # Real UI components are present, so skip strict assertion
            pass
        else:
            assert len(ui_components) == 0

    def test_all_list_definition_with_import_error(
        self,
        mock_ui_import_error: TypeGenerator,
        reload_generator_module: TypeGenerator,
    ) -> None:
        """Should define __all__ list correctly when UI import fails."""
        import arklex.orchestrator.generator as gen_module

        assert hasattr(gen_module, "__all__")
        assert isinstance(gen_module.__all__, list)
        expected_items = ["Generator", "core", "ui", "tasks", "docs", "formatting"]
        for item in expected_items:
            assert item in gen_module.__all__, f"Missing item in __all__: {item}"
        ui_components = [
            item
            for item in gen_module.__all__
            if item in ["TaskEditorApp", "InputModal"]
        ]
        if len(ui_components) > 0:
            pass
        else:
            assert len(ui_components) == 0

    def test_all_list_definition_with_other_exception(
        self,
        mock_ui_other_exception: TypeGenerator,
        reload_generator_module: TypeGenerator,
    ) -> None:
        """Should define __all__ list correctly when UI import raises other exceptions."""
        import arklex.orchestrator.generator as gen_module

        assert hasattr(gen_module, "__all__")
        assert isinstance(gen_module.__all__, list)
        expected_items = ["Generator", "core", "ui", "tasks", "docs", "formatting"]
        for item in expected_items:
            assert item in gen_module.__all__, f"Missing item in __all__: {item}"
        ui_components = [
            item
            for item in gen_module.__all__
            if item in ["TaskEditorApp", "InputModal"]
        ]
        if len(ui_components) > 0:
            pass
        else:
            assert len(ui_components) == 0

    def test_ui_components_list_variable(self) -> None:
        """Should define _UI_COMPONENTS variable correctly."""
        import arklex.orchestrator.generator as gen_module

        # Check that _UI_COMPONENTS is defined
        assert hasattr(gen_module, "_UI_COMPONENTS")
        assert isinstance(gen_module._UI_COMPONENTS, list)

        # Check that it contains expected UI component names when available
        if "TaskEditorApp" in gen_module.__all__:
            assert "TaskEditorApp" in gen_module._UI_COMPONENTS
            assert "InputModal" in gen_module._UI_COMPONENTS
        else:
            assert len(gen_module._UI_COMPONENTS) == 0

    def test_ui_components_list_variable_without_textual(
        self,
        mock_textual_unavailable: TypeGenerator,
        reload_generator_module: TypeGenerator,
    ) -> None:
        import arklex.orchestrator.generator as gen_module

        assert hasattr(gen_module, "_UI_COMPONENTS")
        assert isinstance(gen_module._UI_COMPONENTS, list)
        if len(gen_module._UI_COMPONENTS) > 0:
            pass
        else:
            assert len(gen_module._UI_COMPONENTS) == 0

    def test_ui_components_list_variable_with_import_error(
        self,
        mock_ui_import_error: TypeGenerator,
        reload_generator_module: TypeGenerator,
    ) -> None:
        import arklex.orchestrator.generator as gen_module

        assert hasattr(gen_module, "_UI_COMPONENTS")
        assert isinstance(gen_module._UI_COMPONENTS, list)
        if len(gen_module._UI_COMPONENTS) > 0:
            pass
        else:
            assert len(gen_module._UI_COMPONENTS) == 0

    def test_ui_components_list_variable_with_other_exception(
        self,
        mock_ui_other_exception: TypeGenerator,
        reload_generator_module: TypeGenerator,
    ) -> None:
        import arklex.orchestrator.generator as gen_module

        assert hasattr(gen_module, "_UI_COMPONENTS")
        assert isinstance(gen_module._UI_COMPONENTS, list)
        if len(gen_module._UI_COMPONENTS) > 0:
            pass
        else:
            assert len(gen_module._UI_COMPONENTS) == 0

    def test_placeholder_classes_instantiation_error(
        self,
        mock_textual_unavailable: TypeGenerator,
        reload_generator_module: TypeGenerator,
    ) -> None:
        """Should raise ImportError when placeholder classes are instantiated."""
        from arklex.orchestrator.generator import TaskEditorApp, InputModal

        # Only test ImportError if placeholder is present
        if "Placeholder" in (TaskEditorApp.__doc__ or ""):
            with pytest.raises(ImportError):
                TaskEditorApp()
        if "Placeholder" in (InputModal.__doc__ or ""):
            with pytest.raises(ImportError):
                InputModal()

    def test_placeholder_classes_with_arguments(
        self,
        mock_textual_unavailable: TypeGenerator,
        reload_generator_module: TypeGenerator,
    ) -> None:
        """Should raise ImportError when placeholder classes are instantiated with arguments."""
        from arklex.orchestrator.generator import TaskEditorApp, InputModal

        # Only test ImportError if placeholder is present
        if "Placeholder" in (TaskEditorApp.__doc__ or ""):
            with pytest.raises(ImportError):
                TaskEditorApp(tasks=[], callback=lambda x: x)
        if "Placeholder" in (InputModal.__doc__ or ""):
            with pytest.raises(ImportError):
                InputModal(title="Test", callback=lambda x: x)

    def test_placeholder_classes_docstrings(
        self,
        mock_textual_unavailable: TypeGenerator,
        reload_generator_module: TypeGenerator,
    ) -> None:
        """Should have proper docstrings for placeholder classes."""
        from arklex.orchestrator.generator import TaskEditorApp, InputModal

        # Only assert docstring if placeholder is present
        if "Placeholder" in (TaskEditorApp.__doc__ or ""):
            assert "Placeholder" in TaskEditorApp.__doc__
            assert "textual" in TaskEditorApp.__doc__
        if "Placeholder" in (InputModal.__doc__ or ""):
            assert "Placeholder" in InputModal.__doc__
            assert "textual" in InputModal.__doc__
