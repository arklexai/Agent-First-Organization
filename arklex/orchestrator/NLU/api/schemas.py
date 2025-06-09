"""Pydantic models for NLU API requests and responses.

This module defines the data models used for API requests and responses,
ensuring proper validation and serialization of data. It provides
type-safe models for intent detection, slot filling, and slot verification
operations.

The module includes:
- Request models for intent detection and slot operations
- Response models for API results
- Field validation and documentation
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class IntentRequest(BaseModel):
    """Request model for intent prediction.

    This model defines the structure and validation rules for intent
    detection requests, including input text, available intents,
    chat history, and model configuration.

    Attributes:
        text: Input text to analyze for intent detection
        intents: Dictionary mapping intent names to their definitions
        chat_history_str: Formatted chat history providing context
        model: Configuration parameters for the language model
    """

    text: str = Field(..., description="Input text to analyze for intent detection")
    intents: Dict[str, List[Dict[str, Any]]] = Field(
        ..., description="Dictionary mapping intent names to their definitions"
    )
    chat_history_str: str = Field(
        ..., description="Formatted chat history providing context"
    )
    model: Dict[str, Any] = Field(
        ..., description="Configuration parameters for the language model"
    )


class IntentResponse(BaseModel):
    """Response model for intent prediction.

    This model defines the structure and validation rules for intent
    detection responses, containing the predicted intent.

    Attributes:
        intent: The predicted intent name
    """

    intent: str = Field(..., description="The predicted intent name")


class SlotRequest(BaseModel):
    """Request model for slot filling.

    This model defines the structure and validation rules for slot
    filling requests, including slots to fill, input context,
    and model configuration.

    Attributes:
        slots: List of slots to fill with their definitions
        context: Input context to extract values from
        model: Configuration parameters for the language model
        type: Type of slot filling operation (default: "chat")
    """

    slots: List[Dict[str, Any]] = Field(
        ..., description="List of slots to fill with their definitions"
    )
    context: str = Field(..., description="Input context to extract values from")
    model: Dict[str, Any] = Field(
        ..., description="Configuration parameters for the language model"
    )
    type: str = Field(default="chat", description="Type of slot filling operation")


class SlotVerificationRequest(BaseModel):
    """Request model for slot verification.

    This model defines the structure and validation rules for slot
    verification requests, including the slot to verify, chat history,
    and model configuration.

    Attributes:
        slot: The slot to verify with its current value
        chat_history_str: Formatted chat history providing context
        model: Configuration parameters for the language model
    """

    slot: Dict[str, Any] = Field(
        ..., description="The slot to verify with its current value"
    )
    chat_history_str: str = Field(
        ..., description="Formatted chat history providing context"
    )
    model: Dict[str, Any] = Field(
        ..., description="Configuration parameters for the language model"
    )


class VerificationResponse(BaseModel):
    """Response model for slot verification.

    This model defines the structure and validation rules for slot
    verification responses, containing the verification decision
    and reasoning.

    Attributes:
        verification_needed: Whether the slot value needs verification
        thought: Reasoning for the verification decision
    """

    verification_needed: bool = Field(
        ..., description="Whether the slot value needs verification"
    )
    thought: str = Field(..., description="Reasoning for the verification decision")
