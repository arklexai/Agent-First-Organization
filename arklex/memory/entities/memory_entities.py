"""Entities for memory management.

This module defines the entities used in the memory management system,
including ResourceRecord and Memory.

Key Components:
- ResourceRecord: Represents a record of resource usage and processing.
- Memory: Represents the memory of the conversation system.
"""

from typing import Any

from pydantic import BaseModel, Field


class ResourceRecord(BaseModel):
    """Record of resource usage and processing.

    This class maintains a record of how resources are used during task execution,
    including input/output data, processing steps, and intent information.

    The class provides:
    1. Resource usage tracking
    2. Input/output data management
    3. Processing step recording
    4. Intent tracking
    5. Type-safe resource management

    Attributes:
        info (Dict[str, Any]): General information about the resource.
        intent (str): The intent associated with the resource.
        input (List[Any]): Input data for the resource.
        output (str): Output data from the resource.
        steps (List[Any]): Processing steps taken.
        personalized_intent (str): User-specific intent for the resource.
    """

    info: dict[str, Any]
    intent: str = Field(default="")
    input: list[Any] = Field(default_factory=list)
    output: str = Field(default="")
    steps: list[Any] = Field(default_factory=list)
    personalized_intent: str = Field(default="")


class Memory(BaseModel):
    """Memory management for conversation and processing.

    This class maintains the memory of the conversation system, including
    processing trajectories and function call history.

    The class provides:
    1. Trajectory tracking
    2. Function call history
    3. Type-safe memory management

    Attributes:
        trajectory (List[List[ResourceRecord]]): Processing trajectory history.
        function_calling_trajectory (List[Dict[str, Any]]): Function call history.
    """

    trajectory: list[list[ResourceRecord]] = Field(default_factory=list)
    function_calling_trajectory: list[dict[str, Any]] = Field(default_factory=list)
