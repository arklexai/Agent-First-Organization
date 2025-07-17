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

import argparse
import json
import logging
import sys
from typing import Any

from langchain_openai import ChatOpenAI

from arklex.orchestrator.generator.core.generator import (
    Generator as CoreGenerator,
)
from arklex.utils.logging_utils import LogContext
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP

# Import the main classes from the new modular structure
from .core import Generator

# Configure basic logging for debugging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")

log_context = LogContext(__name__)

# Make UI components optional to avoid dependency issues
try:
    _UI_EXPORTS = ["TaskEditorApp"]
    _UI_AVAILABLE = True
except ImportError:
    _UI_EXPORTS = []
    _UI_AVAILABLE = False

# Export the main classes for backward compatibility
__all__ = ["Generator", "ChatOpenAI", *_UI_EXPORTS]

# The original classes have been refactored into modular components.
# All functionality is preserved in the new structure:
# - Generator class is now in core/generator.py
# - TaskEditorApp and InputModal are in ui/
# - Task generation logic is in tasks/
# - Document processing is in docs/
# - Graph formatting is in formatting/


def load_config(file_path: str) -> dict[str, Any]:
    """Load configuration from a JSON file."""
    try:
        with open(file_path) as f:
            return json.load(f)
    except FileNotFoundError:
        log_context.error(f"Configuration file not found at {file_path}")
        raise
    except json.JSONDecodeError:
        log_context.error(f"Invalid JSON in configuration file: {file_path}")
        raise


def main() -> None:
    """Main function to run the task graph generator."""
    try:
        parser = argparse.ArgumentParser(
            description="Generate a task graph from a configuration file."
        )
        parser.add_argument(
            "--file_path",
            type=str,
            required=True,
            help="Path to the configuration JSON file.",
        )
        args = parser.parse_args()

        log_context.info(f"Loading configuration from {args.file_path}")
        config = load_config(args.file_path)

        log_context.info("Initializing language model...")
        provider = MODEL.get("llm_provider")
        if not provider:
            raise ValueError(
                "llm_provider must be explicitly specified in MODEL configuration"
            )

        model_class = PROVIDER_MAP.get(provider)
        if not model_class:
            raise ValueError(f"Unsupported provider: {provider}")

        model = model_class(
            model=MODEL.get("model_type_or_path", "gpt-4"), timeout=30000
        )

        log_context.info("Initializing task graph generator...")
        generator = CoreGenerator(config=config, model=model)

        log_context.info("Generating task graph...")
        task_graph = generator.generate()

        output_path = config.get("output_path", "taskgraph.json")
        log_context.info(f"Saving task graph to {output_path}")

        with open(output_path, "w") as f:
            json.dump(task_graph, f, indent=4)

        log_context.info("✅ Task graph generation complete.")
    except Exception as e:
        log_context.error(f"Error during task graph generation: {e}")
        # Exit with error code only if this is the main module
        if __name__ == "__main__":
            sys.exit(1)


if __name__ == "__main__":
    main()
