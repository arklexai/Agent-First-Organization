"""Slot filling implementation for NLU.

This module provides the core implementation for slot filling functionality,
supporting both local model-based and remote API-based approaches. It implements
the BaseSlotFilling interface to provide a unified way of extracting and
verifying slot values from input text.

The module includes:
- SlotFiller: Main class for slot filling and verification
- Support for both local and remote slot filling
- Integration with language models and APIs
- Slot value verification and validation
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from arklex.orchestrator.NLU.core.base import BaseSlotFilling
from arklex.orchestrator.NLU.services.model_service import ModelService
from arklex.orchestrator.NLU.services.api_service import APIClientService
from arklex.utils.slot import Slot, Verification

logger = logging.getLogger(__name__)


class SlotFiller(BaseSlotFilling):
    """Slot filling implementation.

    This class provides functionality for extracting and verifying slot values,
    supporting both local model-based and remote API-based approaches. It
    implements the BaseSlotFilling interface and can be configured to use either
    a local language model or a remote API service.

    Key features:
    - Dual-mode operation (local/remote)
    - Slot value extraction and validation
    - Integration with language models
    - Support for chat history context
    - Verification of extracted values

    Attributes:
        model_service: Service for local model-based slot filling
        api_service: Optional service for remote API-based slot filling
    """

    def __init__(
        self, api_url: Optional[str] = None, model_config: Optional[dict] = None
    ) -> None:
        """Initialize the slot filler.

        Creates a new slot filler instance, optionally configuring it
        to use a remote API service for slot filling and verification.

        Args:
            api_url: Optional URL for remote API service. If provided,
                    the filler will use the remote API instead of
                    local model-based slot filling.
            model_config: Optional model configuration dictionary. If not provided,
                    a default config will be used.

        Note:
            If api_url is not provided, the filler will use local
            model-based slot filling exclusively.
        """
        if model_config is None:
            model_config = {
                "model_type_or_path": "gpt-3.5-turbo",
                "llm_provider": "openai",
            }
        self.model_service = ModelService(model_config)
        if api_url is not None and not isinstance(api_url, str):
            logger.error("api_url must be a string")
            self.api_service = None
        else:
            self.api_service = APIClientService(api_url) if api_url else None

    def _verify_slot_remote(
        self, slot: Dict[str, Any], chat_history_str: str, model_config: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Verify slot using remote API.

        Args:
            slot: The slot to verify
            chat_history_str: Formatted chat history
            model_config: Model configuration

        Returns:
            Tuple of (verification_needed, reasoning)

        Raises:
            ValueError: If API service is not configured
        """
        if not self.api_service:
            raise ValueError("API service not configured")

        logger.info("Using remote API for slot verification")
        return self.api_service.verify_slot(slot, chat_history_str, model_config)

    def _verify_slot_local(
        self, slot: Dict[str, Any], chat_history_str: str, model_config: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Verify slot using local model.

        Args:
            slot: The slot to verify
            chat_history_str: Formatted chat history
            model_config: Model configuration

        Returns:
            Tuple of (verification_needed, reasoning)
        """
        logger.info("Using local model for slot verification")
        prompt = self.model_service.format_verification_input(slot, chat_history_str)

        response = self.model_service.get_response(
            prompt, model_config, note="slot verification"
        )

        verification_needed, thought = self.model_service.process_verification_response(
            response
        )
        logger.info(f"Verification needed: {verification_needed}, Reason: {thought}")

        return verification_needed, thought

    def _fill_slots_remote(
        self,
        slots: List[Slot],
        context: str,
        model_config: Dict[str, Any],
        type: str = "chat",
    ) -> List[Slot]:
        """Fill slots using remote API.

        Args:
            slots: List of slots to fill
            context: Input context
            model_config: Model configuration
            type: Slot filling type

        Returns:
            List of filled slots

        Raises:
            ValueError: If API service is not configured
        """
        if not self.api_service:
            raise ValueError("API service not configured")

        logger.info("Using remote API for slot filling")
        return self.api_service.fill_slots(slots, context, model_config, type)

    def _fill_slots_local(
        self,
        slots: List[Slot],
        context: str,
        model_config: Dict[str, Any],
        type: str = "chat",
    ) -> List[Slot]:
        """Fill slots using local model.

        Args:
            slots: List of slots to fill
            context: Input context
            model_config: Model configuration
            type: Slot filling type

        Returns:
            List of filled slots
        """
        logger.info("Using local model for slot filling")
        user_prompt, system_prompt = self.model_service.format_slot_input(
            slots, context, type
        )

        response = self.model_service.get_response(
            user_prompt,
            model_config,
            system_prompt=system_prompt,
            response_format="json",
            note="slot filling",
        )

        filled_slots = self.model_service.process_slot_response(response, slots)
        logger.info(f"Filled slots: {filled_slots}")

        return filled_slots

    def verify_slot(
        self, slot: Dict[str, Any], chat_history_str: str, model_config: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Verify if a slot value needs confirmation.

        Determines whether a slot value requires user confirmation based on
        confidence level, ambiguity, or other verification criteria. Can operate
        in either local model-based or remote API-based mode.

        Args:
            slot: The slot to verify, containing extracted value and metadata
            chat_history_str: Formatted chat history providing conversation context
            model_config: Configuration parameters for the language model

        Returns:
            A tuple containing:
                - bool: Whether verification is needed
                - str: Reasoning for the verification decision

        Note:
            The method automatically chooses between local and remote verification
            based on whether an API service is configured.
        """
        try:
            if self.api_service:
                return self._verify_slot_remote(slot, chat_history_str, model_config)
            return self._verify_slot_local(slot, chat_history_str, model_config)
        except Exception as e:
            logger.error(f"Error in slot verification: {str(e)}")
            return False, "Error during verification"

    def fill_slots(
        self,
        slots: List[Slot],
        context: str,
        model_config: Dict[str, Any],
        type: str = "chat",
    ) -> List[Slot]:
        """Extract slot values from context.

        Analyzes the input context to extract values for the specified slots,
        using either a local language model or a remote API service.

        Args:
            slots: List of slots to fill, each containing slot definition and metadata
            context: Input context to extract values from
            model_config: Configuration parameters for the language model
            type: Type of slot filling operation (default: "chat")

        Returns:
            List of filled slots, each containing extracted values and metadata

        Note:
            The method automatically chooses between local and remote slot filling
            based on whether an API service is configured.
        """
        try:
            if self.api_service:
                return self._fill_slots_remote(slots, context, model_config, type)
            return self._fill_slots_local(slots, context, model_config, type)
        except Exception as e:
            logger.error(f"Error in slot filling: {str(e)}")
            return slots


def initialize_slotfillapi(slotsfillapi: str) -> SlotFiller:
    """Initialize the slot filling API.

    This function creates a new SlotFiller instance, configuring it to use either
    a remote API service or local model-based slot filling based on the provided
    API URL.

    Args:
        slotsfillapi: API endpoint for slot filling. If not a string or empty,
                     falls back to local model-based slot filling.

    Returns:
        SlotFiller: Initialized slot filler instance, either API-based or local model-based.

    Note:
        The function will automatically fall back to local model-based slot filling
        if the API URL is invalid or not provided.
    """
    if not isinstance(slotsfillapi, str) or not slotsfillapi:
        logger.warning("Using local model-based slot filling")
        return SlotFiller(None)
    logger.info(f"Initializing SlotFiller with API URL: {slotsfillapi}")
    return SlotFiller(slotsfillapi)
