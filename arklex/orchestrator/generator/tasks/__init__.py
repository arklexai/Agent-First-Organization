"""Task generation and management components for the Arklex framework.

This module provides specialized components for task generation, best practices,
and reusable task management. Each component is designed to handle a specific
aspect of task processing:

1. TaskGenerator
   - Generates tasks from objectives and documentation
   - Manages task hierarchy and relationships
   - Handles task validation and refinement

2. BestPracticeManager
   - Generates best practices for task execution
   - Refines practices based on feedback
   - Ensures quality and consistency

3. ReusableTaskManager
   - Creates reusable task components
   - Manages task templates and patterns
   - Promotes code reuse and maintainability

The components work together to create a comprehensive task management system
that ensures high-quality, maintainable, and reusable task definitions.

Usage:
    from arklex.orchestrator.generator.tasks import (
        TaskGenerator,
        BestPracticeManager,
        ReusableTaskManager,
    )

    # Initialize components
    task_generator = TaskGenerator(
        model=language_model,
        role="customer_service",
        u_objective="Handle customer inquiries",
        instructions=instruction_docs,
        documents=task_docs,
    )

    # Generate tasks
    tasks = task_generator.generate_tasks(intro_text, existing_tasks)

    # Generate best practices
    best_practice_manager = BestPracticeManager(
        model=language_model,
        role="customer_service",
        u_objective="Handle customer inquiries",
    )
    best_practices = best_practice_manager.generate_best_practices(tasks)

    # Create reusable tasks
    reusable_task_manager = ReusableTaskManager(
        model=language_model,
        role="customer_service",
        u_objective="Handle customer inquiries",
    )
    reusable_tasks = reusable_task_manager.generate_reusable_tasks(tasks)
"""

from .best_practice_manager import BestPracticeManager
from .reusable_task_manager import ReusableTaskManager
from .task_generator import TaskGenerator

__all__ = ["TaskGenerator", "BestPracticeManager", "ReusableTaskManager"]
