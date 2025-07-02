"""Task graph formatting components for the Arklex framework.

This module provides specialized components for formatting and structuring task
graphs. The components ensure consistent output format and handle various
aspects of task graph representation.

Components:

1. TaskGraphFormatter
   - Formats task graphs into a consistent structure
   - Handles node and edge formatting
   - Manages task hierarchy representation
   - Ensures output compatibility

2. NodeFormatter
   - Formats individual task nodes
   - Handles node attributes and metadata
   - Manages node type-specific formatting
   - Ensures node consistency

3. EdgeFormatter
   - Formats task graph edges
   - Handles edge attributes and conditions
   - Manages edge type-specific formatting
   - Ensures edge consistency

4. GraphValidator
   - Validates task graph structure
   - Ensures graph connectivity
   - Checks for required attributes
   - Provides detailed error messages

The components work together to create well-structured, consistent, and
valid task graphs that can be used by the orchestrator.

Usage:
    from arklex.orchestrator.generator.formatting import TaskGraphFormatter

    # Initialize formatter
    formatter = TaskGraphFormatter(
        model=language_model,
        role="customer_service",
        u_objective="Handle customer inquiries",
        product_kwargs=config,
        workers=workers,
        tools=tools,
        reusable_tasks=reusable_tasks,
    )

    # Format task graph
    task_graph = formatter.format_task_graph(
        best_practices=best_practices,
        tasks=tasks
    )

    # Validate graph
    if formatter.validate_graph(task_graph):
        # Use the formatted graph
        pass
    else:
        # Handle validation errors
        pass
"""

from .edge_formatter import EdgeFormatter
from .graph_validator import GraphValidator
from .node_formatter import NodeFormatter
from .task_graph_formatter import TaskGraphFormatter

__all__ = ["TaskGraphFormatter", "EdgeFormatter", "NodeFormatter", "GraphValidator"]
