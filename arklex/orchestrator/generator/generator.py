"""Task graph generator compatibility layer for the Arklex framework.

This module serves as a compatibility layer, re-exporting the main Generator class and optional UI components
from the new modular structure for backward compatibility. All core logic has been refactored into modular
subcomponents for maintainability and clarity.

Modular structure includes:
- core/generator.py: Main Generator class with orchestration logic
- ui/: Interactive components (TaskEditorApp, InputModal)
- tasks/: Task generation, best practices, and reusable tasks
- docs/: Document loading and processing
- formatting/: Task graph structure formatting

Usage:
    from arklex.orchestrator.generator import Generator
    # or for direct access to components:
    from arklex.orchestrator.generator.core import Generator
    from arklex.orchestrator.generator.ui import TaskEditorApp, InputModal
    from arklex.orchestrator.generator.tasks import TaskGenerator, BestPracticeManager
"""

# Import the main classes from the new modular structure
from .core import Generator

# Make UI components optional to avoid dependency issues
try:
    from .ui import TaskEditorApp, InputModal

    _UI_AVAILABLE = True
    _UI_EXPORTS = ["TaskEditorApp", "InputModal"]
except ImportError:
    _UI_AVAILABLE = False
    _UI_EXPORTS = []

# Export the main classes for backward compatibility
__all__ = ["Generator", *_UI_EXPORTS]

# The original classes have been refactored into modular components.
# All functionality is preserved in the new structure:
# - Generator class is now in core/generator.py
# - TaskEditorApp and InputModal are in ui/
# - Task generation logic is in tasks/
# - Document processing is in docs/
# - Graph formatting is in formatting/
