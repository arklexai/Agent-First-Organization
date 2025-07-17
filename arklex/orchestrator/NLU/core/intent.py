"""Intent detection implementation for NLU.

This module provides the core implementation for intent detection functionality,
supporting both local model-based and remote API-based approaches. It implements
the BaseNLU interface to provide a unified way of detecting user intents from
input text.

The module includes:
- IntentDetector: Main class for intent detection
- Support for both local and remote intent detection
- Integration with language models and APIs
"""

from typing import Any

from arklex.orchestrator.NLU.core.base import BaseNLU
from arklex.orchestrator.NLU.services.api_service import APIClientService
from arklex.orchestrator.NLU.services.model_service import ModelService
from arklex.utils.exceptions import APIError, ArklexError, ValidationError
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class IntentDetector(BaseNLU):
    """Intent detection implementation.

    This class provides functionality for detecting intents from user input,
    supporting both local model-based and remote API-based approaches. It
    implements the BaseNLU interface and can be configured to use either
    a local language model or a remote API service.

    Key features:
    - Dual-mode operation (local/remote)
    - Integration with language models
    - Support for chat history context
    - Intent mapping and validation

    Attributes:
        model_service: Service for local model-based intent detection
        api_service: Optional service for remote API-based intent detection
    """

    def __init__(
        self,
        model_service: ModelService,
        api_service: APIClientService | None = None,
    ) -> None:
        """Initialize the intent detector.

        Args:
            model_service: Service for local model-based intent detection
            api_service: Optional service for remote API-based intent detection

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
                    "service": "IntentDetector",
                    "operation": "initialization",
                },
            )
        self.model_service = model_service
        self.api_service = api_service
        if not api_service:
            log_context.warning(
                "Using local model-based intent detection",
                extra={"operation": "initialization"},
            )
        log_context.info(
            "IntentDetector initialized successfully",
            extra={
                "mode": "remote" if api_service else "local",
                "operation": "initialization",
            },
        )

    def _detect_intent_local(
        self,
        intents: dict[str, list[dict[str, Any]]],
        chat_history_str: str,
        model_config: dict[str, Any],
    ) -> str:
        log_context.info(
            "Entered _detect_intent_local",
            extra={"operation": "intent_detection_local"},
        )
        """Detect intent using local model.

        Args:
            intents: Dictionary of available intents
            chat_history_str: Formatted chat history
            model_config: Model configuration

        Returns:
            Predicted intent name

        Raises:
            ModelError: If intent detection fails
            ValidationError: If input validation fails
        """
        log_context.info(
            "Using local model for intent detection",
            extra={"operation": "intent_detection_local"},
        )

        # Format input and get mapping
        prompt, idx2intents_mapping = self.model_service.format_intent_input(
            intents, chat_history_str
        )
        log_context.info(
            f"Intent detection input prepared:\nPrompt: {prompt}\n\nMapping: {idx2intents_mapping}",
            extra={
                "prompt": prompt,
                "mapping": idx2intents_mapping,
                "operation": "intent_detection_local",
            },
        )
        log_context.info(
            "Calling get_response on model_service",
            extra={"operation": "intent_detection_local"},
        )
        # Get model response
        response = self.model_service.get_response(prompt)
        log_context.info(
            f"Model response received:\nResponse: {response}",
            extra={
                "prompt": prompt,
                "raw_response": response,
                "operation": "intent_detection_local",
            },
        )

        # Parse response
        try:
            pred_idx, pred_intent = [i.strip() for i in response.split(")", 1)]
        except ValueError as e:
            log_context.error(
                "Invalid response format",
                extra={
                    "prompt": prompt,
                    "raw_response": response,
                    "error": str(e),
                    "operation": "intent_detection_local",
                },
            )
            raise ValidationError(
                "Invalid response format",
                details={
                    "prompt": prompt,
                    "raw_response": response,
                    "error": str(e),
                    "operation": "intent_detection_local",
                },
            ) from e

        # Validate intent
        if pred_intent not in idx2intents_mapping.values():
            log_context.warning(
                f"Predicted intent not in mapping:\nPredicted intent: {pred_intent}\n\nAvailable intents: {list(idx2intents_mapping.values())}",
                extra={
                    "prompt": prompt,
                    "raw_response": response,
                    "predicted_intent": pred_intent,
                    "available_intents": list(idx2intents_mapping.values()),
                    "operation": "intent_detection_local",
                },
            )
            pred_intent = idx2intents_mapping.get(pred_idx, "others")

        log_context.info(
            "Intent detection completed",
            extra={
                "prompt": prompt,
                "raw_response": response,
                "final_predicted_intent": pred_intent,
                "operation": "intent_detection_local",
            },
        )
        return pred_intent

    def _detect_intent_remote(
        self,
        text: str,
        intents: dict[str, list[dict[str, Any]]],
        chat_history_str: str,
        model_config: dict[str, Any],
    ) -> str:
        """Detect intent using remote API.

        Args:
            text: Input text to analyze
            intents: Dictionary of available intents
            chat_history_str: Formatted chat history
            model_config: Model configuration

        Returns:
            Predicted intent name

        Raises:
            ModelError: If intent detection fails
            ValidationError: If input validation fails
            APIError: If API request fails
        """
        if not self.api_service:
            log_context.error(
                "API service not configured",
                extra={"operation": "intent_detection_remote"},
            )
            raise ValidationError(
                "API service not configured",
                details={"operation": "intent_detection_remote"},
            )

        log_context.info(
            "Using remote API for intent detection",
            extra={
                "text": text,
                "operation": "intent_detection_remote",
            },
        )

        try:
            response = self.api_service.predict_intent(
                text=text,
                intents=intents,
                chat_history_str=chat_history_str,
                model_config=model_config,
            )
            log_context.info(
                "Intent detection completed",
                extra={
                    "predicted_intent": response,
                    "operation": "intent_detection_remote",
                },
            )
            return response
        except APIError as e:
            log_context.error(
                "Failed to detect intent via API",
                extra={
                    "error": str(e),
                    "text": text,
                    "operation": "intent_detection_remote",
                },
            )
            raise APIError(
                "Failed to detect intent via API",
                details={
                    "error": str(e),
                    "text": text,
                    "operation": "intent_detection_remote",
                },
            ) from e

    def predict_intent(
        self,
        text: str,
        intents: dict[str, list[dict[str, Any]]],
        chat_history_str: str,
        model_config: dict[str, Any],
    ) -> str:
        """Predict intent from input text.

        Analyzes the input text to determine the most likely intent based on
        the available intent definitions and chat history context. Can operate
        in either local model-based or remote API-based mode.

        Args:
            text: Input text to analyze for intent detection
            intents: Dictionary mapping intent names to their definitions and attributes
            chat_history_str: Formatted chat history providing conversation context
            model_config: Configuration parameters for the language model

        Returns:
            The predicted intent name as a string

        Raises:
            ModelError: If intent detection fails
            ValidationError: If input validation fails
            APIError: If API request fails
        """
        log_context.info(
            "Starting intent prediction",
            extra={
                "text": text,
                "mode": "remote" if self.api_service else "local",
                "operation": "intent_prediction",
            },
        )

        try:
            log_context.info(
                "Calling intent detection method",
                extra={"operation": "intent_prediction"},
            )
            if self.api_service:
                intent = self._detect_intent_remote(
                    text, intents, chat_history_str, model_config
                )
            else:
                intent = self._detect_intent_local(
                    intents, chat_history_str, model_config
                )
            log_context.info(
                "Intent detection method returned",
                extra={"operation": "intent_prediction"},
            )

            log_context.info(
                "Intent prediction completed",
                extra={
                    "predicted_intent": intent,
                    "operation": "intent_prediction",
                },
            )
            return intent
        except Exception as e:
            log_context.error(
                "Intent prediction failed",
                extra={
                    "error": str(e),
                    "text": text,
                    "operation": "intent_prediction",
                },
            )
            raise ArklexError(
                f"Intent prediction failed: {str(e)}",
                details={
                    "original_error": str(e),
                    "error_type": type(e).__name__,
                    "text": text,
                    "operation": "intent_prediction",
                },
            ) from e

    def execute(
        self,
        text: str,
        intents: dict[str, list[dict[str, Any]]],
        chat_history_str: str,
        model_config: dict[str, Any],
    ) -> str:
        """Execute intent detection.

        This method is an alias for predict_intent, implementing the BaseNLU
        interface. It provides the same functionality as predict_intent.

        Args:
            text: Input text to analyze
            intents: Dictionary of available intents
            chat_history_str: Formatted chat history
            model_config: Model configuration

        Returns:
            Predicted intent name

        Raises:
            ModelError: If intent detection fails
            ValidationError: If input validation fails
            APIError: If API request fails
        """
        return self.predict_intent(text, intents, chat_history_str, model_config)
