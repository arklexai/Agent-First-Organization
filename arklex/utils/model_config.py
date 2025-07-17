"""Model configuration settings for the Arklex framework.

This module defines model configuration settings used throughout the framework,
including model type, provider, context window size, and token limits. These settings
are used to configure language model interactions and ensure consistent behavior across
different components of the system.

Key Components:
1. MODEL: Dictionary containing model configuration settings
   - model_type_or_path: The model identifier or path
   - llm_provider: The language model provider
   - context: Maximum context window size
   - max_tokens: Maximum number of tokens for generation
   - tokenizer: Tokenizer configuration

Usage:
    from arklex.utils.model_config import MODEL

    # Access model settings
    model_type = MODEL["model_type_or_path"]
    context_size = MODEL["context"]

    # Use in configuration
    config = {
        "model": MODEL["model_type_or_path"],
        "provider": MODEL["llm_provider"],
        "max_tokens": MODEL["max_tokens"]
    }
"""

from typing import Any

# Model configuration settings - no default provider or model
# Users must explicitly specify model and provider
MODEL: dict[str, Any] = {
    "model_name": None,  # Must be specified by user
    "model_type_or_path": None,  # Must be specified by user
    "llm_provider": None,  # Must be specified by user
    "api_key": None,  # Will be validated by provider_utils
    "endpoint": None,  # Will be set by provider_utils
    "context": 16000,  # Maximum context window size
    "max_tokens": 4096,  # Maximum tokens for generation
    "tokenizer": "o200k_base",  # Tokenizer configuration
}
