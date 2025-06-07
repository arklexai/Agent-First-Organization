"""Tracing utilities for the Arklex framework.

This module provides utilities for tracing operations in the Arklex framework,
including intent detection, slot filling, and tool execution.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TraceType(Enum):
    """Types of traces in the Arklex framework.

    Attributes:
        IntentDetection: Traces for intent detection operations
        SlotFiller: Traces for slot filling operations
        ToolExecution: Traces for tool execution operations
    """

    IntentDetection = "IntentDetection"  # Intent detection operation traces
    SlotFiller = "SlotFiller"  # Slot filling operation traces
    ToolExecution = "ToolExecution"  # Tool execution operation traces


class Trace:
    """Trace for an operation in the Arklex framework.

    This class represents a trace of an operation, including
    the operation type, input, output, and metadata.

    Attributes:
        type (TraceType): Type of the operation
        input (Any): Input to the operation
        output (Any): Output from the operation
        metadata (Dict[str, Any]): Additional metadata
    """

    def __init__(
        self,
        type: TraceType,
        input: Any,
        output: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the trace.

        Args:
            type: Type of the operation
            input: Input to the operation
            output: Output from the operation
            metadata: Additional metadata
        """
        self.type = type
        self.input = input
        self.output = output
        self.metadata = metadata or {}
