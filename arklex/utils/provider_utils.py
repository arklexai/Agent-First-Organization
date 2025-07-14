"""Provider utility functions for the Arklex framework.

This module provides utility functions for handling different model providers,
including API key management and endpoint configuration. It centralizes
provider-specific logic to ensure consistency across the framework.

Key Components:
1. get_api_key_for_provider: Get the appropriate API key for a provider
2. get_endpoint_for_provider: Get the appropriate endpoint for a provider
3. validate_and_get_model_class: Validate LLM provider and get model class
4. Provider-specific configuration mappings

Usage:
    from arklex.utils.provider_utils import get_api_key_for_provider, get_endpoint_for_provider, validate_and_get_model_class

    # Get API key for a specific provider
    api_key = get_api_key_for_provider("openai")

    # Get endpoint for a specific provider
    endpoint = get_endpoint_for_provider("anthropic")

    # Validate provider and get model class
    model_class = validate_and_get_model_class(llm_config)
"""

import os
from typing import Any

from arklex.utils.model_provider_config import PROVIDER_MAP


def get_api_key_for_provider(provider: str) -> str:
    """Get the appropriate API key for the specified provider.

    Args:
        provider (str): The model provider name

    Returns:
        str: The API key for the provider

    Raises:
        ValueError: If the API key is missing or empty
    """
    if provider is None:
        raise TypeError("Provider cannot be None")

    provider_api_keys = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",  # Google API key is used for Google models
        "huggingface": "HUGGINGFACE_API_KEY",
    }

    env_key = provider_api_keys.get(provider, "OPENAI_API_KEY")
    api_key = os.getenv(env_key)

    if not api_key or not api_key.strip():
        raise ValueError(
            f"API key for provider '{provider}' is missing or empty. "
            f"Please set the {env_key} environment variable with a valid API key."
        )

    return api_key


def get_endpoint_for_provider(provider: str) -> str:
    """Get the appropriate endpoint for the specified provider.

    Args:
        provider (str): The model provider name

    Returns:
        str: The endpoint URL for the provider
    """
    if provider is None:
        raise TypeError("Provider cannot be None")

    provider_endpoints = {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com",
        "google": "https://generativelanguage.googleapis.com",
        "huggingface": "https://api-inference.huggingface.co",
    }

    return provider_endpoints.get(provider, "https://api.openai.com/v1")


def validate_api_key_presence(provider: str, api_key: str) -> None:
    """Validate that an API key is present for the specified provider.

    Args:
        provider (str): The model provider name
        api_key (str): The API key to validate

    Raises:
        ValueError: If the API key is missing or empty
    """
    if not api_key or not api_key.strip():
        provider_api_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
        }
        env_key = provider_api_keys.get(provider, "OPENAI_API_KEY")
        raise ValueError(
            f"API key for provider '{provider}' is missing or empty. "
            f"Please set the {env_key} environment variable with a valid API key."
        )


def get_provider_config(provider: str, model: str) -> dict[str, Any]:
    """Get complete configuration for a specific provider and model.

    Args:
        provider (str): The model provider name
        model (str): The model name/identifier

    Returns:
        Dict[str, Any]: Complete model configuration including API key and endpoint

    Raises:
        ValueError: If the API key is missing or empty
    """
    if provider is None:
        raise TypeError("Provider cannot be None")
    if model is None:
        raise TypeError("Model cannot be None")

    api_key = get_api_key_for_provider(provider)

    # Validate that API key is present
    validate_api_key_presence(provider, api_key)

    return {
        "model_name": model,
        "model_type_or_path": model,
        "llm_provider": provider,
        "api_key": api_key,
        "endpoint": get_endpoint_for_provider(provider),
    }


def validate_and_get_model_class(llm_config: object) -> type:
    """Validate LLM provider and get the corresponding model class.

    This function implements the common pattern of checking if an LLM provider is specified
    in the configuration and retrieving the appropriate model class from the provider mapping.

    Args:
        llm_config: Configuration object containing llm_provider attribute.

    Returns:
        type: The model class for the specified provider.

    Raises:
        ValueError: If llm_provider is not specified or if the provider is not supported.
    """
    if not llm_config.llm_provider:
        raise ValueError("llm_provider must be explicitly specified in llm_config")

    model_class = PROVIDER_MAP.get(llm_config.llm_provider)
    if not model_class:
        raise ValueError(f"Unsupported provider: {llm_config.llm_provider}")

    return model_class
