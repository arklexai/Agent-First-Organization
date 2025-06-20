"""Task graph generator module for the Arklex framework.

This module provides a comprehensive framework for generating task graphs and managing
task hierarchies. It includes classes for task editing, task generation, and handling
reusable tasks.

The module has been refactored into specialized components:
- Core: Main orchestration logic (Generator class)
- UI: Interactive components for task editing
- Tasks: Task generation, best practices, and reusable tasks
- Docs: Document loading and processing
- Formatting: Task graph structure formatting

Key Components:
- Generator: Main class for creating task graphs based on user objectives and documentation
- TaskEditorApp: Text-based UI for editing tasks and their steps
- InputModal: Modal dialog for editing task and step descriptions

Features:
- Natural language task generation
- Interactive task editing
- Reusable task management
- Best practice integration
- Documentation processing
- Resource initialization
- Task graph formatting
- Configuration management

Usage:
    from arklex.orchestrator.generator import Generator
    from arklex.env.env import DefaultResourceInitializer

    # Initialize generator
    config = {
        "role": "customer_service",
        "user_objective": "Handle customer inquiries",
        "builder_objective": "Create efficient response system",
        "instructions": [...],
        "tasks": [...],
        "workers": [...],
        "tools": [...]
    }

    generator = Generator(
        config=config,
        model=language_model,
        output_dir="output",
        resource_inizializer=DefaultResourceInitializer()
    )

    # Generate task graph
    task_graph = generator.generate()

    # Save task graph
    output_path = generator.save_task_graph(task_graph)
"""

# Import main classes for backward compatibility
from .core import Generator

# Make UI components optional to avoid dependency issues
try:
    from .ui import TaskEditorApp, InputModal

    _UI_COMPONENTS = ["TaskEditorApp", "InputModal"]
except ImportError:
    # Create placeholder classes when UI dependencies are not available
    class TaskEditorApp:
        """Placeholder class when UI components are not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TaskEditorApp requires 'textual' package to be installed"
            )

    class InputModal:
        """Placeholder class when UI components are not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError("InputModal requires 'textual' package to be installed")

    _UI_COMPONENTS = []

# Import specialized modules for advanced usage
from . import core
from . import ui
from . import tasks
from . import docs
from . import formatting

__all__ = [
    "Generator",
    *_UI_COMPONENTS,
    "core",
    "ui",
    "tasks",
    "docs",
    "formatting",
]
