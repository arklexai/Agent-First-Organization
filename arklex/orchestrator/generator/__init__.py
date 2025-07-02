"""Task graph generator module for the Arklex framework.

This module provides a comprehensive framework for generating task graphs and managing
task hierarchies. It includes classes for task editing, task generation, and handling
reusable tasks.

The module has been refactored into specialized components:
- Core: Main orchestration logic (Generator class)
- UI: Interactive components for task editing (TaskEditorApp, InputModal)
- Tasks: Task generation, best practices, and reusable tasks
- Docs: Document loading and processing
- Formatting: Task graph structure formatting

Key Components:
- Generator: Main class for creating task graphs based on user objectives and documentation
- TaskEditorApp: Text-based UI for editing tasks and their steps in a tree structure
- InputModal: Modal dialog for editing task and step descriptions with callback support

Features:
- Natural language task generation from user objectives
- Interactive task editing with keyboard shortcuts
- Reusable task management and best practice integration
- Documentation processing and resource initialization
- Task graph formatting and configuration management
- Graceful fallback when UI components are unavailable

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
        resource_initializer=DefaultResourceInitializer()
    )

    # Generate task graph
    task_graph = generator.generate()

    # Save task graph
    output_path = generator.save_task_graph(task_graph)
"""

# Import the generator module for backward compatibility
from . import core, docs, formatting, generator, tasks
from .core import Generator

# Import UI components if available
try:
    from .ui import InputModal, TaskEditorApp

    _UI_COMPONENTS = ["TaskEditorApp", "InputModal"]
    from . import ui

    # Set the ui attribute on the module and expose the components
    globals()["ui"] = ui
    # Also expose the components directly on the ui module for backward compatibility
    ui.InputModal = InputModal
    ui.TaskEditorApp = TaskEditorApp
    _UI_AVAILABLE = True
except ImportError:
    # UI components not available (e.g., textual not installed)
    _UI_COMPONENTS = []
    # Create a dummy ui module to prevent import errors
    import types

    ui = types.ModuleType("ui")
    globals()["ui"] = ui
    _UI_AVAILABLE = False

__all__ = [
    "Generator",
    *_UI_COMPONENTS,
    "core",
    "tasks",
    "docs",
    "formatting",
    "generator",
]

# Only add "ui" to __all__ if UI components are available
if _UI_AVAILABLE:
    __all__.append("ui")
