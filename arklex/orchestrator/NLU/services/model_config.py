"""Model configuration handling for NLU operations.

This module provides utilities for handling model configurations,
including provider-specific settings and response format handling.
It manages model initialization parameters, provider selection,
and response format configuration for different language models.
"""

from typing import Dict, Any
from langchain_core.language_models import BaseChatModel
from arklex.utils.model_provider_config import PROVIDER_MAP

# Model configuration constants
DEFAULT_TEMPERATURE: float = 0.1  # Default temperature for model generation
RESPONSE_FORMAT_JSON: str = "json"  # JSON response format identifier
RESPONSE_FORMAT_TEXT: str = "text"  # Text response format identifier
PROVIDER_ANTHROPIC: str = "anthropic"  # Anthropic provider identifier
PROVIDER_OPENAI: str = "openai"  # OpenAI provider identifier


class ModelConfig:
    """Configuration handler for language models.

    This class manages the configuration and initialization of language models,
    handling provider-specific settings and response format requirements.

    Key responsibilities:
    - Model provider configuration and initialization
    - Response format handling and validation
    - Model parameter management and validation

    Attributes:
        DEFAULT_TEMPERATURE: Default temperature for model generation
        RESPONSE_FORMAT_JSON: JSON response format identifier
        RESPONSE_FORMAT_TEXT: Text response format identifier
    """

    @staticmethod
    def get_model_kwargs(model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get model initialization parameters.

        Constructs a dictionary of parameters required for model initialization
        based on the provided configuration.

        Args:
            model_config: Model configuration dictionary containing:
                - model_type_or_path: Model identifier or path
                - llm_provider: Provider name (e.g., 'openai', 'anthropic')

        Returns:
            Dictionary of model initialization parameters including:
                - model: Model identifier
                - temperature: Generation temperature (default: 0.1)
                - n: Number of responses (1 for non-Anthropic providers)

        Raises:
            KeyError: If required configuration keys are missing
        """
        kwargs: Dict[str, Any] = {
            "model": model_config["model_type_or_path"],
            "temperature": DEFAULT_TEMPERATURE,
        }

        if model_config["llm_provider"] != PROVIDER_ANTHROPIC:
            kwargs["n"] = 1

        return kwargs

    @staticmethod
    def get_model_instance(model_config: Dict[str, Any]) -> BaseChatModel:
        """Get model instance based on configuration.

        Initializes and returns a language model instance based on the
        specified provider and configuration parameters.

        Args:
            model_config: Model configuration dictionary containing:
                - llm_provider: Provider name
                - model_type_or_path: Model identifier

        Returns:
            Initialized model instance from the specified provider

        Raises:
            ValueError: If provider is not supported or initialization fails
        """
        kwargs = ModelConfig.get_model_kwargs(model_config)
        provider = model_config["llm_provider"]

        if provider not in PROVIDER_MAP:
            raise ValueError(f"Unsupported provider: {provider}")

        return PROVIDER_MAP[provider](**kwargs)

    @staticmethod
    def configure_response_format(
        model: BaseChatModel,
        model_config: Dict[str, Any],
        response_format: str = RESPONSE_FORMAT_TEXT,
    ) -> BaseChatModel:
        """Configure model response format.

        Configures the response format for the model based on the provider
        and desired format type. Currently supports JSON and text formats.

        Args:
            model: Model instance to configure
            model_config: Model configuration dictionary containing:
                - llm_provider: Provider name
            response_format: Desired response format ('json' or 'text')

        Returns:
            Configured model instance with response format binding

        Raises:
            ValueError: If response format is invalid or configuration fails
        """
        if response_format not in [RESPONSE_FORMAT_JSON, RESPONSE_FORMAT_TEXT]:
            raise ValueError(f"Invalid response format: {response_format}")

        if model_config["llm_provider"] == PROVIDER_OPENAI:
            return model.bind(
                response_format={"type": "json_object"}
                if response_format == RESPONSE_FORMAT_JSON
                else {"type": "text"}
            )
        return model
