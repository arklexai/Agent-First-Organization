"""Model configuration settings for the Arklex framework.

This module defines the default model configuration settings used throughout the framework,
including model type, provider, context window size, and token limits. These settings
are used to configure language model interactions and ensure consistent behavior across
different components of the system.
"""

from typing import Dict, Any

MODEL: Dict[str, Any] = {
    "model_type_or_path": "gpt-4o",
    "llm_provider": "openai",
    "context": 16000,
    "max_tokens": 4096,
    "tokenizer": "o200k_base",
}
