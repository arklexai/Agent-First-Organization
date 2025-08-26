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
from arklex.orchestrator.NLU.services.model_service import ModelService
from arklex.utils.exceptions import ModelError
from arklex.utils.logging_utils import LogContext, handle_exceptions

log_context = LogContext(__name__)


def create_slot_filler(
    model_service: ModelService,
) -> "SlotFiller":
    """Create a new SlotFiller instance.

    Args:
        model_service: Service for local model-based slot filling

    Returns:
        A new SlotFiller instance

    Raises:
        ValidationError: If model_service is not provided
    """
    return SlotFiller(model_service=model_service)


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
    ) -> None:
        """Initialize the slot filler.

        Args:
            model_service: Service for local model-based slot filling

        Raises:
            ValidationError: If model_service is not provided
        """
        self.model_service = model_service
        log_context.info(
            "SlotFiller initialized successfully",
            extra={
                "mode": "local",
                "operation": "initialization",
            },
        )

    def _slots_to_openai_schema(self, slots: list[Slot]) -> dict[str, Any]:
        """Convert list of Slot objects to OpenAI JSON schema format.

        Args:
            slots: List of Slot objects to convert

        Returns:
            OpenAI JSON schema dictionary
        """
        properties = {}
        required = []

        for slot in slots:
            # Use the to_openai_schema method from the Slot class
            slot_schema = slot.to_openai_schema()

            if slot_schema is None:
                continue  # Skip slots that return None (like fixed value slots)

            properties[slot.name] = slot_schema

            if getattr(slot, "required", False):
                required.append(slot.name)

        return {
            "title": "SlotFillingOutput",
            "description": "Structured output for slot filling",
            "type": "object",
            "properties": properties,
            "required": required,
        }

    @handle_exceptions()
    def _fill_slots(
        self,
        slots: list[Slot],
        context: str,
        model_config: dict[str, Any],
        type: str = "chat",
    ) -> list[Slot]:
        """Fill slots.

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

        # Generate OpenAI schema from slots
        schema = self._slots_to_openai_schema(slots)
        log_context.info(
            "OpenAI schema generated",
            extra={
                "schema": schema,
                "operation": "slot_filling_local",
            },
        )

        # Get model response
        response = self.model_service.get_response_with_structured_output(
            prompt, schema, system_prompt
        )
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
        response = self.model_service.get_response(prompt)
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
                "mode": "local",
                "operation": "slot_verification",
            },
        )

        try:
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
                "mode": "local",
                "operation": "slot_filling",
            },
        )

        try:
            filled_slots = self._fill_slots(slots, context, model_config, type)

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
