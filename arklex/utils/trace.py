"""Trace run name definitions for the Arklex framework.

This module defines the different types of trace runs that can be performed in the system,
including task graph execution, NLU processing, and slot filling operations.
These trace names are used to identify and categorize different types of operations
in the system's execution flow.

Key Components:
1. TraceRunName: Enumeration of different trace run types
   - TaskGraph: Task graph execution traces
   - ExecutionResult: Execution result traces
   - OrchestResponse: Orchestrator response traces
   - NLU: Natural Language Understanding traces
   - SlotFilling: Slot filling operation traces

Usage:
    from arklex.utils.trace import TraceRunName

    # Use in tracing
    trace_name = TraceRunName.TaskGraph
    if trace_name == TraceRunName.NLU:
        process_nlu_trace()
"""

from enum import Enum


class TraceRunName(str, Enum):
    """Enumeration of trace run types in the system.

    This enum defines the different types of trace runs that can be performed
    in the system. Each type represents a specific category of operation that
    can be traced for monitoring and debugging purposes.

    Values:
        TaskGraph: Traces for task graph execution
        ExecutionResult: Traces for execution results
        OrchestResponse: Traces for orchestrator responses
        NLU: Traces for Natural Language Understanding operations
        SlotFilling: Traces for slot filling operations

    Example:
        trace_name = TraceRunName.TaskGraph
        if trace_name == TraceRunName.NLU:
            process_nlu_trace()
    """

    TaskGraph = "TaskGraph"  # Task graph execution traces
    ExecutionResult = "ExecutionResult"  # Execution result traces
    OrchestResponse = "OrchestResponse"  # Orchestrator response traces
    NLU = "NLU"  # Natural Language Understanding traces
    SlotFilling = "SlotFilling"  # Slot filling operation traces
