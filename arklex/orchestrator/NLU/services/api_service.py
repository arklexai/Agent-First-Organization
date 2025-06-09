"""API client service for NLU operations.

This module provides services for interacting with remote NLU APIs,
handling API requests, and processing API responses.
It manages HTTP communication, request formatting, and response handling
for NLU operations including intent detection and slot filling.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import httpx
from arklex.utils.slot import Slot
from arklex.orchestrator.NLU.utils.validators import validate_intent_response

logger = logging.getLogger(__name__)

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
            ValueError: If base_url is empty
        """
        if not base_url:
            raise ValueError("Base URL cannot be empty")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def _make_request(
        self, endpoint: str, method: str, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
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
            ValueError: If request fails or response is invalid
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = self.client.request(method, url, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"HTTP error: {str(e)}")
            raise ValueError(f"API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            raise ValueError(f"API request failed: {str(e)}")

    def predict_intent(
        self,
        text: str,
        intents: Dict[str, List[Dict[str, Any]]],
        chat_history_str: str,
        model_config: Dict[str, Any],
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
            ValueError: If request fails or response is invalid
        """
        data = {
            "text": text,
            "intents": intents,
            "chat_history_str": chat_history_str,
            "model_config": model_config,
        }
        response = self._make_request("/nlu/predict", HTTP_METHOD_POST, data)
        return validate_intent_response(
            response.get("intent", ""), response.get("idx2intents_mapping", {})
        )

    def predict_slots(
        self, text: str, slots: List[Slot], model_config: Dict[str, Any]
    ) -> List[Slot]:
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
            ValueError: If request fails or response is invalid
        """
        data = {
            "text": text,
            "slots": [slot.to_dict() for slot in slots],
            "model_config": model_config,
        }
        response = self._make_request("/slotfill/predict", HTTP_METHOD_POST, data)
        return [Slot(**slot) for slot in response.get("slots", [])]

    def verify_slots(
        self, text: str, slots: List[Slot], model_config: Dict[str, Any]
    ) -> Tuple[bool, str]:
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
            ValueError: If request fails or response is invalid
        """
        data = {
            "text": text,
            "slots": [slot.to_dict() for slot in slots],
            "model_config": model_config,
        }
        response = self._make_request("/slotfill/verify", HTTP_METHOD_POST, data)
        return response.get("verification_needed", False), response.get(
            "thought", "No need to verify"
        )

    def close(self) -> None:
        """Close the HTTP client.

        This method should be called when the service is no longer needed
        to properly clean up resources and close the HTTP client connection.
        """
        self.client.close()
