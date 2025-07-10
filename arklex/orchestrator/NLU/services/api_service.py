"""API client service for NLU operations.

This module provides services for interacting with remote NLU APIs,
handling API requests, and processing API responses.
It manages HTTP communication, request formatting, and response handling
for NLU operations including intent detection and slot filling.
"""

from typing import Any

import httpx

from arklex.orchestrator.NLU.entities.slot_entities import Slot
from arklex.orchestrator.NLU.utils.validators import validate_intent_response
from arklex.utils.exceptions import APIError, ValidationError
from arklex.utils.logging_utils import LogContext, handle_exceptions

log_context = LogContext(__name__)

# API configuration constants
DEFAULT_TIMEOUT: int = 30  # Default request timeout in seconds
HTTP_METHOD_POST: str = "POST"  # HTTP POST method identifier


class APIClientService:
    """Service for interacting with remote NLU APIs.

    This class manages communication with remote NLU APIs, handling
    request formatting, response processing, and error handling.

    Key responsibilities:
    - HTTP request handling and formatting
    - Response processing and validation
    - Error handling and logging
    - Resource management

    Attributes:
        base_url: Base URL for the API
        timeout: Request timeout in seconds
        client: HTTP client instance
    """

    def __init__(self, base_url: str, timeout: int = DEFAULT_TIMEOUT) -> None:
        """Initialize the API client service.

        Creates a new API client service instance with the specified
        configuration and initializes the HTTP client.

        Args:
            base_url: Base URL for the API
            timeout: Request timeout in seconds

        Raises:
            ValidationError: If base_url is empty
            APIError: If client initialization fails
        """
        if not base_url:
            log_context.error(
                "Empty base URL provided",
                extra={"operation": "initialization"},
            )
            raise ValidationError(
                "Base URL cannot be empty",
                details={"operation": "initialization"},
            )

        try:
            self.base_url = base_url.rstrip("/")
            self.timeout = timeout
            self.client = httpx.Client(timeout=timeout)
            log_context.info(
                "API client service initialized successfully",
                extra={
                    "base_url": self.base_url,
                    "timeout": self.timeout,
                    "operation": "initialization",
                },
            )
        except Exception as e:
            log_context.error(
                "Failed to initialize API client",
                extra={
                    "error": str(e),
                    "base_url": base_url,
                    "timeout": timeout,
                    "operation": "initialization",
                },
            )
            raise APIError(
                "Failed to initialize API client",
                details={
                    "error": str(e),
                    "base_url": base_url,
                    "timeout": timeout,
                    "operation": "initialization",
                },
            ) from e

    @handle_exceptions()
    def _make_request(
        self, endpoint: str, method: str, data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make HTTP request to the API.

        Sends an HTTP request to the specified endpoint and processes
        the response. Handles error cases and response validation.

        Args:
            endpoint: API endpoint path
            method: HTTP method to use
            data: Request data to send

        Returns:
            API response as dictionary

        Raises:
            APIError: If request fails or response is invalid
            ValidationError: If response validation fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        log_context.info(
            "Making API request",
            extra={
                "url": url,
                "method": method,
                "endpoint": endpoint,
                "operation": "api_request",
            },
        )

        try:
            response = self.client.request(method, url, json=data)
            response.raise_for_status()
            result = response.json()
            log_context.info(
                "API request successful",
                extra={
                    "url": url,
                    "method": method,
                    "status_code": response.status_code,
                    "operation": "api_request",
                },
            )
            return result
        except httpx.HTTPError as e:
            log_context.error(
                "HTTP error in API request",
                extra={
                    "error": str(e),
                    "url": url,
                    "method": method,
                    "status_code": getattr(e.response, "status_code", None),
                    "operation": "api_request",
                },
            )
            raise APIError(
                "API request failed",
                details={
                    "error": str(e),
                    "url": url,
                    "method": method,
                    "status_code": getattr(e.response, "status_code", None),
                    "operation": "api_request",
                },
            ) from e
        except Exception as e:
            log_context.error(
                "Unexpected error in API request",
                extra={
                    "error": str(e),
                    "url": url,
                    "method": method,
                    "operation": "api_request",
                },
            )
            raise APIError(
                "API request failed",
                details={
                    "error": str(e),
                    "url": url,
                    "method": method,
                    "operation": "api_request",
                },
            ) from e

    @handle_exceptions()
    def predict_intent(
        self,
        text: str,
        intents: dict[str, list[dict[str, Any]]],
        chat_history_str: str,
        model_config: dict[str, Any],
    ) -> str:
        """Predict intent from text.

        Sends a request to predict the intent of the given text using
        the provided intents and chat history.

        Args:
            text: Input text to analyze
            intents: Dictionary of intents containing:
                - intent_name: List of intent definitions
                - attribute: Intent attributes (definition, sample_utterances)
            chat_history_str: Formatted chat history
            model_config: Model configuration dictionary

        Returns:
            Predicted intent name

        Raises:
            APIError: If request fails
            ValidationError: If response validation fails
        """
        log_context.info(
            "Predicting intent",
            extra={
                "text": text,
                "intents": intents,
                "operation": "intent_prediction",
            },
        )

        data = {
            "text": text,
            "intents": intents,
            "chat_history_str": chat_history_str,
            "model_config": model_config,
        }
        response = self._make_request("/nlu/predict", HTTP_METHOD_POST, data)
        result = validate_intent_response(
            response.get("intent", ""), response.get("idx2intents_mapping", {})
        )
        log_context.info(
            "Intent prediction successful",
            extra={
                "intent": result,
                "operation": "intent_prediction",
            },
        )
        return result

    @handle_exceptions()
    def predict_slots(
        self, text: str, slots: list[Slot], model_config: dict[str, Any]
    ) -> list[Slot]:
        """Predict slots from text.

        Sends a request to fill slots in the given text using the
        provided slot definitions.

        Args:
            text: Input text to analyze
            slots: List of slots to fill
            model_config: Model configuration dictionary

        Returns:
            List of filled slots

        Raises:
            APIError: If request fails
            ValidationError: If response validation fails
        """
        log_context.info(
            "Predicting slots",
            extra={
                "text": text,
                "slots": [slot.model_dump() for slot in slots],
                "operation": "slot_prediction",
            },
        )

        data = {
            "text": text,
            "slots": [slot.model_dump() for slot in slots],
            "model_config": model_config,
        }
        response = self._make_request("/slotfill/predict", HTTP_METHOD_POST, data)
        result = [Slot(**slot) for slot in response.get("slots", [])]
        log_context.info(
            "Slot prediction successful",
            extra={
                "slots": [slot.model_dump() for slot in result],
                "operation": "slot_prediction",
            },
        )
        return result

    @handle_exceptions()
    def verify_slots(
        self, text: str, slots: list[Slot], model_config: dict[str, Any]
    ) -> tuple[bool, str]:
        """Verify slots from text.

        Sends a request to verify the filled slots in the given text.

        Args:
            text: Input text to analyze
            slots: List of slots to verify
            model_config: Model configuration dictionary

        Returns:
            Tuple containing:
                - verification_needed: Whether verification is needed
                - thought: Reasoning for verification

        Raises:
            APIError: If request fails
            ValidationError: If response validation fails
        """
        log_context.info(
            "Verifying slots",
            extra={
                "text": text,
                "slots": [slot.model_dump() for slot in slots],
                "operation": "slot_verification",
            },
        )

        data = {
            "text": text,
            "slots": [slot.model_dump() for slot in slots],
            "model_config": model_config,
        }
        response = self._make_request("/slotfill/verify", HTTP_METHOD_POST, data)
        result = (
            response.get("verification_needed", False),
            response.get("thought", "No need to verify"),
        )
        log_context.info(
            "Slot verification successful",
            extra={
                "verification_needed": result[0],
                "thought": result[1],
                "operation": "slot_verification",
            },
        )
        return result

    def close(self) -> None:
        """Close the HTTP client.

        This method should be called when the service is no longer needed
        to properly clean up resources and close the HTTP client connection.
        """
        try:
            self.client.close()
            log_context.info(
                "API client closed successfully",
                extra={"operation": "cleanup"},
            )
        except Exception as e:
            log_context.error(
                "Error closing API client",
                extra={
                    "error": str(e),
                    "operation": "cleanup",
                },
            )
