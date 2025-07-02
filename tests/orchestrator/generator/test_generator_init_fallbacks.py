"""Test utilities and fallback logic for generator __init__.py module.

This module provides placeholder classes and testing utilities for the generator
module when UI components are not available. These are used exclusively for testing
purposes and should not be used in production code.
"""

import os


def create_placeholder_classes() -> tuple[type, type]:
    """Create placeholder classes when UI components are not available.

    Returns:
        Tuple[Type, Type]: Tuple of (TaskEditorApp, InputModal) placeholder classes
    """

    class TaskEditorApp:
        """Placeholder class when UI components are not available."""

        def __init__(
            self, tasks: object = None, *args: object, **kwargs: object
        ) -> None:
            raise ImportError(
                "TaskEditorApp requires 'textual' package to be installed"
            )

    class InputModal:
        """Placeholder class when UI components are not available."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("InputModal requires 'textual' package to be installed")

    return TaskEditorApp, InputModal


def should_force_ui_import_error() -> bool:
    """Check if UI import error should be forced for testing purposes.

    Returns:
        bool: True if UI import error should be forced
    """
    return os.environ.get("ARKLEX_FORCE_UI_IMPORT_ERROR") == "1"


def import_ui_components() -> tuple[type, type, list[str]]:
    """Import UI components or create placeholders if not available.

    Returns:
        Tuple[Type, Type, List[str]]: Tuple of (TaskEditorApp, InputModal, ui_components_list)
    """
    if should_force_ui_import_error():
        # Create placeholder classes when UI dependencies are not available (test hook)
        TaskEditorApp, InputModal = create_placeholder_classes()
        return TaskEditorApp, InputModal, []

    try:
        # Check if textual is available
        import importlib.util

        if importlib.util.find_spec("textual") is None:
            raise ImportError("textual package not found")

        from arklex.orchestrator.generator.ui import InputModal, TaskEditorApp

        return TaskEditorApp, InputModal, ["TaskEditorApp", "InputModal"]
    except (ImportError, Exception):
        # Create placeholder classes when UI dependencies are not available
        # Handle both ImportError and other exceptions (like AttributeError, etc.)
        TaskEditorApp, InputModal = create_placeholder_classes()
        return TaskEditorApp, InputModal, []


class PlaceholderUIModule:
    """Placeholder UI module when UI components are not available."""

    def __init__(self) -> None:
        # Create placeholder submodules that tests expect
        self.task_editor = PlaceholderTaskEditorModule()
        self.input_modal = PlaceholderInputModalModule()


class PlaceholderTaskEditorModule:
    """Placeholder task_editor submodule."""

    def __getattr__(self, name: str) -> object:
        if name == "TaskEditorApp":
            TaskEditorApp, _ = create_placeholder_classes()
            return TaskEditorApp
        elif name == "InputModal":
            _, InputModal = create_placeholder_classes()
            return InputModal
        elif name == "log_context":
            # Return a mock logger for tests
            import logging

            return logging.getLogger("arklex.orchestrator.generator.ui.task_editor")
        elif name == "Tree":
            # Return a mock Tree class for tests
            class MockTree:
                pass

            return MockTree
        elif name == "Label":
            # Return a mock Label class for tests
            class MockLabel:
                pass

            return MockLabel
        elif name == "TreeNode":
            # Return a mock TreeNode class for tests
            class MockTreeNode:
                pass

            return MockTreeNode
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )


class PlaceholderInputModalModule:
    """Placeholder input_modal submodule."""

    def __getattr__(self, name: str) -> object:
        if name == "InputModal":
            _, InputModal = create_placeholder_classes()
            return InputModal
        elif name == "Screen":
            # Return a mock Screen class for tests
            class MockScreen:
                pass

            return MockScreen
        elif name == "Button":
            # Return a mock Button class for tests
            class MockButton:
                pass

            return MockButton
        elif name == "Input":
            # Return a mock Input class for tests
            class MockInput:
                pass

            return MockInput
        elif name == "Static":
            # Return a mock Static class for tests
            class MockStatic:
                pass

            return MockStatic
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )


def create_placeholder_ui_module() -> PlaceholderUIModule:
    """Create a placeholder UI module for testing.

    Returns:
        PlaceholderUIModule: A placeholder UI module
    """
    return PlaceholderUIModule()


# Test utilities for environment variable control
def set_force_ui_import_error(value: str = "1") -> None:
    """Set the environment variable to force UI import error for testing.

    Args:
        value (str): The value to set for ARKLEX_FORCE_UI_IMPORT_ERROR
    """
    os.environ["ARKLEX_FORCE_UI_IMPORT_ERROR"] = value


def clear_force_ui_import_error() -> None:
    """Clear the environment variable that forces UI import error."""
    if "ARKLEX_FORCE_UI_IMPORT_ERROR" in os.environ:
        del os.environ["ARKLEX_FORCE_UI_IMPORT_ERROR"]


def get_force_ui_import_error() -> str | None:
    """Get the current value of the force UI import error environment variable.

    Returns:
        str | None: The current value or None if not set
    """
    return os.environ.get("ARKLEX_FORCE_UI_IMPORT_ERROR")
