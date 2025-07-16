"""Slot filling implementation for NLU.

This module provides the core implementation for slot filling functionality,
supporting both local model-based and remote API-based approaches. It implements
the BaseSlotFilling interface to provide a unified way of extracting and
verifying slot values from input text.

The module includes:
- SlotFiller: Main class for slot filling
- Support for both local and remote slot filling
- Integration with language models and APIs
"""

from typing import Any

from arklex.orchestrator.NLU.core.base import BaseSlotFilling
from arklex.orchestrator.NLU.entities.slot_entities import Slot
from arklex.orchestrator.NLU.services.api_service import APIClientService
from arklex.orchestrator.NLU.services.model_service import ModelService
from arklex.utils.exceptions import APIError, ModelError, ValidationError
from arklex.utils.logging_utils import LogContext, handle_exceptions

log_context = LogContext(__name__)


def create_slot_filler(
    model_service: ModelService,
    api_service: APIClientService | None = None,
) -> "SlotFiller":
    """Create a new SlotFiller instance.

    Args:
        model_service: Service for local model-based slot filling
        api_service: Optional service for remote API-based slot filling

    Returns:
        A new SlotFiller instance

    Raises:
        ValidationError: If model_service is not provided
    """
    return SlotFiller(model_service=model_service, api_service=api_service)


class SlotFiller(BaseSlotFilling):
    """Slot filling implementation.

    This class provides functionality for extracting and verifying slot values
    from user input, supporting both local model-based and remote API-based
    approaches. It implements the BaseSlotFilling interface and can be configured
    to use either a local language model or a remote API service.

    Key features:
    - Dual-mode operation (local/remote)
    - Integration with language models
    - Support for chat history context
    - Slot value extraction and verification

    Attributes:
        model_service: Service for local model-based slot filling
        api_service: Optional service for remote API-based slot filling
    """

    def __init__(
        self,
        model_service: ModelService,
        api_service: APIClientService | None = None,
    ) -> None:
        """Initialize the slot filler.

        Args:
            model_service: Service for local model-based slot filling
            api_service: Optional service for remote API-based slot filling

        Raises:
            ValidationError: If model_service is not provided
        """
        if not model_service:
            log_context.error(
                "Model service is required",
                extra={"operation": "initialization"},
            )
            raise ValidationError(
                "Model service is required",
                details={
                    "service": "SlotFiller",
                    "operation": "initialization",
                },
            )
        self.model_service = model_service
        self.api_service = api_service
        if not api_service:
            log_context.warning(
                "Using local model-based slot filling",
                extra={"operation": "initialization"},
            )
        log_context.info(
            "SlotFiller initialized successfully",
            extra={
                "mode": "remote" if api_service else "local",
                "operation": "initialization",
            },
        )

    @handle_exceptions()
    def _fill_slots_local(
        self,
        slots: list[Slot],
        context: str,
        model_config: dict[str, Any],
        type: str = "chat",
    ) -> list[Slot]:
        """Fill slots using local model.

        Args:
            slots: List of slots to fill
            context: Input context to extract values from
            model_config: Model configuration
            type: Type of slot filling operation (default: "chat")

        Returns:
            List of filled slots

        Raises:
            ModelError: If slot filling fails
            ValidationError: If input validation fails
        """
        log_context.info(
            "Using local model for slot filling",
            extra={
                "slots": [slot.name for slot in slots],
                "context_length": len(context),
                "type": type,
                "operation": "slot_filling_local",
            },
        )

        # Format input
        prompt, system_prompt = self.model_service.format_slot_input(
            slots, context, type
        )
        log_context.info(
            "Slot filling input prepared",
            extra={
                "prompt": prompt,
                "system_prompt": system_prompt,
                "operation": "slot_filling_local",
            },
        )

        # Get model response
        response = self.model_service.get_response(prompt, model_config, system_prompt)
        log_context.info(
            "Model response received",
            extra={
                "prompt": prompt,
                "system_prompt": system_prompt,
                "raw_response": response,
                "operation": "slot_filling_local",
            },
        )

        # Process response
        try:
            filled_slots = self.model_service.process_slot_response(response, slots)
            log_context.info(
                "Slot filling completed",
                extra={
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "raw_response": response,
                    "filled_slots": [slot.name for slot in filled_slots],
                    "operation": "slot_filling_local",
                },
            )
            return filled_slots
        except Exception as e:
            log_context.error(
                "Failed to process slot filling response",
                extra={
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "raw_response": response,
                    "error": str(e),
                    "operation": "slot_filling_local",
                },
            )
            raise ModelError(
                "Failed to process slot filling response",
                details={
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "raw_response": response,
                    "error": str(e),
                    "operation": "slot_filling_local",
                },
            ) from e

    @handle_exceptions()
    def _fill_slots_remote(
        self,
        slots: list[Slot],
        context: str,
        model_config: dict[str, Any],
        type: str = "chat",
    ) -> list[Slot]:
        """Fill slots using remote API.

        Args:
            slots: List of slots to fill
            context: Input context to extract values from
            model_config: Model configuration
            type: Type of slot filling operation (default: "chat")

        Returns:
            List of filled slots

        Raises:
            ModelError: If slot filling fails
            ValidationError: If input validation fails
            APIError: If API request fails
        """
        if not self.api_service:
            log_context.error(
                "API service not configured",
                extra={"operation": "slot_filling_remote"},
            )
            raise ValidationError(
                "API service not configured",
                details={"operation": "slot_filling_remote"},
            )

        log_context.info(
            "Using remote API for slot filling",
            extra={
                "slots": [slot.name for slot in slots],
                "context_length": len(context),
                "type": type,
                "operation": "slot_filling_remote",
            },
        )

        try:
            filled_slots = self.api_service.predict_slots(
                text=context,
                slots=slots,
                model_config=model_config,
            )
            log_context.info(
                "Slot filling completed",
                extra={
                    "filled_slots": [slot.name for slot in filled_slots],
                    "operation": "slot_filling_remote",
                },
            )
            return filled_slots
        except APIError as e:
            log_context.error(
                "Failed to fill slots via API",
                extra={
                    "error": str(e),
                    "slots": [slot.name for slot in slots],
                    "operation": "slot_filling_remote",
                },
            )
            raise APIError(
                "Failed to fill slots via API",
                details={
                    "error": str(e),
                    "slots": [slot.name for slot in slots],
                    "operation": "slot_filling_remote",
                },
            ) from e

    @handle_exceptions()
    def _verify_slot_local(
        self,
        slot: dict[str, Any],
        chat_history_str: str,
        model_config: dict[str, Any],
    ) -> tuple[bool, str]:
        """Verify slot value using local model.

        Args:
            slot: Slot to verify
            chat_history_str: Formatted chat history
            model_config: Model configuration

        Returns:
            Tuple of (is_valid, reason)

        Raises:
            ModelError: If slot verification fails
            ValidationError: If input validation fails
        """
        log_context.info(
            "Using local model for slot verification",
            extra={
                "slot": slot.get("name"),
                "operation": "slot_verification_local",
            },
        )

        # Format input
        prompt = self.model_service.format_verification_input(slot, chat_history_str)
        log_context.info(
            "Slot verification input prepared",
            extra={
                "prompt": prompt,
                "operation": "slot_verification_local",
            },
        )

        # Get model response
        response = self.model_service.get_response(prompt, model_config)
        log_context.info(
            "Model response received",
            extra={
                "response": response,
                "operation": "slot_verification_local",
            },
        )

        # Process response
        try:
            is_valid, reason = self.model_service.process_verification_response(
                response
            )
            log_context.info(
                "Slot verification completed",
                extra={
                    "is_valid": is_valid,
                    "reason": reason,
                    "operation": "slot_verification_local",
                },
            )
            return is_valid, reason
        except Exception as e:
            log_context.error(
                "Failed to process slot verification response",
                extra={
                    "error": str(e),
                    "response": response,
                    "operation": "slot_verification_local",
                },
            )
            raise ModelError(
                "Failed to process slot verification response",
                details={
                    "error": str(e),
                    "response": response,
                    "operation": "slot_verification_local",
                },
            ) from e

    @handle_exceptions()
    def _verify_slot_remote(
        self,
        slot: dict[str, Any],
        chat_history_str: str,
        model_config: dict[str, Any],
    ) -> tuple[bool, str]:
        """Verify slot value using remote API.

        Args:
            slot: Slot to verify
            chat_history_str: Formatted chat history
            model_config: Model configuration

        Returns:
            Tuple of (is_valid, reason)

        Raises:
            ModelError: If slot verification fails
            ValidationError: If input validation fails
            APIError: If API request fails
        """
        if not self.api_service:
            log_context.error(
                "API service not configured",
                extra={"operation": "slot_verification_remote"},
            )
            raise ValidationError(
                "API service not configured",
                details={"operation": "slot_verification_remote"},
            )

        log_context.info(
            "Using remote API for slot verification",
            extra={
                "slot": slot.get("name"),
                "operation": "slot_verification_remote",
            },
        )

        try:
            is_valid, reason = self.api_service.verify_slots(
                text=chat_history_str,
                slots=[Slot(**slot)],
                model_config=model_config,
            )
            log_context.info(
                "Slot verification completed",
                extra={
                    "is_valid": is_valid,
                    "reason": reason,
                    "operation": "slot_verification_remote",
                },
            )
            return is_valid, reason
        except APIError as e:
            log_context.error(
                "Failed to verify slot via API",
                extra={
                    "error": str(e),
                    "slot": slot.get("name"),
                    "operation": "slot_verification_remote",
                },
            )
            raise APIError(
                "Failed to verify slot via API",
                details={
                    "error": str(e),
                    "slot": slot.get("name"),
                    "operation": "slot_verification_remote",
                },
            ) from e

    @handle_exceptions()
    def verify_slot(
        self,
        slot: dict[str, Any],
        chat_history_str: str,
        model_config: dict[str, Any],
    ) -> tuple[bool, str]:
        """Verify slot value.

        Args:
            slot: Slot to verify
            chat_history_str: Formatted chat history
            model_config: Model configuration

        Returns:
            Tuple of (is_valid, reason)

        Raises:
            ModelError: If slot verification fails
            ValidationError: If input validation fails
            APIError: If API request fails
        """
        log_context.info(
            "Starting slot verification",
            extra={
                "slot": slot.get("name"),
                "mode": "remote" if self.api_service else "local",
                "operation": "slot_verification",
            },
        )

        try:
            if self.api_service:
                is_valid, reason = self._verify_slot_remote(
                    slot, chat_history_str, model_config
                )
            else:
                is_valid, reason = self._verify_slot_local(
                    slot, chat_history_str, model_config
                )

            log_context.info(
                "Slot verification completed",
                extra={
                    "is_valid": is_valid,
                    "reason": reason,
                    "operation": "slot_verification",
                },
            )
            return is_valid, reason
        except Exception as e:
            log_context.error(
                "Slot verification failed",
                extra={
                    "error": str(e),
                    "slot": slot.get("name"),
                    "operation": "slot_verification",
                },
            )
            raise

    @handle_exceptions()
    def fill_slots(
        self,
        slots: list[Slot],
        context: str,
        model_config: dict[str, Any],
        type: str = "chat",
    ) -> list[Slot]:
        """Fill slots from input context.

        Args:
            slots: List of slots to fill
            context: Input context to extract values from
            model_config: Model configuration
            type: Type of slot filling operation (default: "chat")

        Returns:
            List of filled slots

        Raises:
            ModelError: If slot filling fails
            ValidationError: If input validation fails
            APIError: If API request fails
        """
        log_context.info(
            "Starting slot filling",
            extra={
                "slots": [slot.name for slot in slots],
                "context_length": len(context),
                "mode": "remote" if self.api_service else "local",
                "operation": "slot_filling",
            },
        )

        try:
            if self.api_service:
                filled_slots = self._fill_slots_remote(
                    slots, context, model_config, type
                )
            else:
                filled_slots = self._fill_slots_local(
                    slots, context, model_config, type
                )

            log_context.info(
                "Slot filling completed",
                extra={
                    "filled_slots": [slot.name for slot in filled_slots],
                    "operation": "slot_filling",
                },
            )
            return filled_slots
        except Exception as e:
            log_context.error(
                "Slot filling failed",
                extra={
                    "error": str(e),
                    "slots": [slot.name for slot in slots],
                    "operation": "slot_filling",
                },
            )
            raise
