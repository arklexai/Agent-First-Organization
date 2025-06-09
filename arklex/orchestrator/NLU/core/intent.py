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

import logging
from typing import Dict, List, Any, Optional
from arklex.orchestrator.NLU.core.base import BaseNLU
from arklex.orchestrator.NLU.services.model_service import ModelService
from arklex.orchestrator.NLU.services.api_service import APIClientService

logger = logging.getLogger(__name__)


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
        self, api_url: Optional[str] = None, model_config: Optional[dict] = None
    ) -> None:
        """Initialize the intent detector.

        Creates a new intent detector instance, optionally configuring it
        to use a remote API service for intent detection.

        Args:
            api_url: Optional URL for remote API service. If provided,
                    the detector will use the remote API instead of
                    local model-based detection.
            model_config: Optional model configuration dictionary. If not provided,
                    a default config will be used.

        Note:
            If api_url is not provided, the detector will use local
            model-based intent detection exclusively.
        """
        if model_config is None:
            model_config = {
                "model_type_or_path": "gpt-3.5-turbo",
                "llm_provider": "openai",
            }
        self.model_service = ModelService(model_config)
        self.api_service = APIClientService(api_url) if api_url else None

    def _detect_intent_remote(
        self,
        text: str,
        intents: Dict[str, List[Dict[str, Any]]],
        chat_history_str: str,
        model_config: Dict[str, Any],
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
            ValueError: If API service is not configured
        """
        if not self.api_service:
            raise ValueError("API service not configured")

        logger.info("Using remote API for intent detection")
        return self.api_service.predict_intent(
            text, intents, chat_history_str, model_config
        )

    def _detect_intent_local(
        self,
        intents: Dict[str, List[Dict[str, Any]]],
        chat_history_str: str,
        model_config: Dict[str, Any],
    ) -> str:
        """Detect intent using local model.

        Args:
            intents: Dictionary of available intents
            chat_history_str: Formatted chat history
            model_config: Model configuration

        Returns:
            Predicted intent name
        """
        logger.info("Using local model for intent detection")
        prompt, idx2intents_mapping = self.model_service.format_intent_input(
            intents, chat_history_str
        )

        response = self.model_service.get_response(prompt)
        response = response.split(")")[0].strip()

        pred_intent = idx2intents_mapping.get(response.strip(), "others")
        logger.info(f"Predicted intent: {pred_intent}")

        return pred_intent

    def predict_intent(
        self,
        text: str,
        intents: Dict[str, List[Dict[str, Any]]],
        chat_history_str: str,
        model_config: Dict[str, Any],
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

        Note:
            The method automatically chooses between local and remote detection
            based on whether an API service is configured.
        """
        try:
            if self.api_service:
                return self._detect_intent_remote(
                    text, intents, chat_history_str, model_config
                )
            return self._detect_intent_local(intents, chat_history_str, model_config)
        except Exception as e:
            logger.error(f"Error in intent detection: {str(e)}")
            return "others"

    def execute(
        self,
        text: str,
        intents: Dict[str, List[Dict[str, Any]]],
        chat_history_str: str,
        model_config: Dict[str, Any],
    ) -> str:
        """Alias for predict_intent to match expected interface in TaskGraph."""
        return self.predict_intent(text, intents, chat_history_str, model_config)
