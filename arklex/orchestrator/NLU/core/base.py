"""Base classes and interfaces for NLU functionality.

This module provides the foundational classes and interfaces that define the core
Natural Language Understanding (NLU) functionality. It includes abstract base
classes for intent detection and slot filling operations, establishing the
contract that all NLU implementations must follow.

The module defines two main abstract base classes:
- BaseNLU: Interface for intent detection operations
- BaseSlotFilling: Interface for slot filling and verification operations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseNLU(ABC):
    """Base class for Natural Language Understanding operations.

    This abstract base class defines the interface for NLU operations,
    establishing the contract that all NLU implementations must follow.
    It focuses on intent detection, which is the process of determining
    the user's intended action from their input text.

    Key responsibilities:
    - Intent detection from user input
    - Integration with language models
    - Handling of chat history context

    Note:
        All concrete implementations must provide an implementation of
        the predict_intent method.
    """

    @abstractmethod
    def predict_intent(
        self,
        text: str,
        intents: Dict[str, List[Dict[str, Any]]],
        chat_history_str: str,
        model_config: Dict[str, Any],
    ) -> str:
        """Predict intent from input text.

        Analyzes the input text to determine the most likely intent based on
        the available intent definitions and chat history context.

        Args:
            text: The input text to analyze for intent detection
            intents: Dictionary mapping intent names to their definitions and attributes
            chat_history_str: Formatted chat history providing conversation context
            model_config: Configuration parameters for the language model

        Returns:
            The predicted intent name as a string

        Raises:
            NotImplementedError: If the concrete class does not implement this method
        """
        pass


class BaseSlotFilling(ABC):
    """Base class for Slot Filling operations.

    This abstract base class defines the interface for slot filling operations,
    establishing the contract that all slot filling implementations must follow.
    It handles both slot value extraction and verification of extracted values.

    Key responsibilities:
    - Slot value extraction from context
    - Slot value verification and confirmation
    - Integration with language models

    Note:
        All concrete implementations must provide implementations of
        the verify_slot and fill_slots methods.
    """

    @abstractmethod
    def verify_slot(
        self, slot: Dict[str, Any], chat_history_str: str, model_config: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Verify if a slot value needs confirmation.

        Determines whether a slot value requires user confirmation based on
        confidence level, ambiguity, or other verification criteria.

        Args:
            slot: The slot to verify, containing extracted value and metadata
            chat_history_str: Formatted chat history providing conversation context
            model_config: Configuration parameters for the language model

        Returns:
            A tuple containing:
                - bool: Whether verification is needed
                - str: Reasoning for the verification decision

        Raises:
            NotImplementedError: If the concrete class does not implement this method
        """
        pass

    @abstractmethod
    def fill_slots(
        self,
        slots: List[Dict[str, Any]],
        context: str,
        model_config: Dict[str, Any],
        type: str = "chat",
    ) -> List[Dict[str, Any]]:
        """Extract slot values from context.

        Analyzes the input context to extract values for the specified slots,
        using the provided language model configuration.

        Args:
            slots: List of slots to fill, each containing slot definition and metadata
            context: Input context to extract values from
            model_config: Configuration parameters for the language model
            type: Type of slot filling operation (default: "chat")

        Returns:
            List of filled slots, each containing extracted values and metadata

        Raises:
            NotImplementedError: If the concrete class does not implement this method
        """
        pass
