"""Trace run name definitions for the Arklex framework.

This module defines the different types of trace runs that can be performed in the system,
including task graph execution, NLU processing, and slot filling operations.
These trace names are used to identify and categorize different types of operations
in the system's execution flow.
"""

from enum import Enum


class TraceRunName(str, Enum):
    TaskGraph = "TaskGraph"
    ExecutionResult = "ExecutionResult"
    OrchestResponse = "OrchestResponse"
    NLU = "NLU"
    SlotFilling = "SlotFilling"
