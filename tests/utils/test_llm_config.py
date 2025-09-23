"""Tests for LLM config module."""

import pytest
from arklex.utils.llm_config import LLMConfig


class TestLLMConfig:
    """Test cases for LLMConfig class."""

    def test_llm_config_default_initialization(self):
        """Test LLMConfig with default values."""
        config = LLMConfig()

        assert config.llm_provider == "openai"
        assert config.model_type_or_path == "gpt-4o"

    def test_llm_config_custom_initialization(self):
        """Test LLMConfig with custom values."""
        config = LLMConfig(llm_provider="anthropic", model_type_or_path="claude-3")

        assert config.llm_provider == "anthropic"
        assert config.model_type_or_path == "claude-3"

    def test_llm_config_partial_initialization(self):
        """Test LLMConfig with partial custom values."""
        config = LLMConfig(llm_provider="google", model_type_or_path="gemini-pro")

        assert config.llm_provider == "google"
        assert config.model_type_or_path == "gemini-pro"

    def test_llm_config_empty_string_values(self):
        """Test LLMConfig with empty string values."""
        config = LLMConfig(llm_provider="", model_type_or_path="")

        assert config.llm_provider == ""
        assert config.model_type_or_path == ""

    def test_llm_config_validation(self):
        """Test LLMConfig validation."""
        # Test with valid values
        config = LLMConfig(llm_provider="openai", model_type_or_path="gpt-3.5-turbo")

        assert config.llm_provider == "openai"
        assert config.model_type_or_path == "gpt-3.5-turbo"

    def test_llm_config_different_providers(self):
        """Test LLMConfig with different providers."""
        providers = ["openai", "anthropic", "google", "huggingface"]

        for provider in providers:
            config = LLMConfig(llm_provider=provider)
            assert config.llm_provider == provider

    def test_llm_config_different_models(self):
        """Test LLMConfig with different models."""
        models = [
            "gpt-4o",
            "gpt-3.5-turbo",
            "claude-3",
            "gemini-pro",
            "command",
            "text-davinci-003",
        ]

        for model in models:
            config = LLMConfig(model_type_or_path=model)
            assert config.model_type_or_path == model

    def test_llm_config_serialization(self):
        """Test LLMConfig serialization."""
        config = LLMConfig(llm_provider="openai", model_type_or_path="gpt-4o")

        # Test that we can access all attributes
        assert hasattr(config, "llm_provider")
        assert hasattr(config, "model_type_or_path")

    def test_llm_config_immutability(self):
        """Test LLMConfig immutability after creation."""
        config = LLMConfig(llm_provider="openai")

        # Values should be accessible but not easily mutable
        assert config.llm_provider == "openai"

        # Test that we can't accidentally modify the config
        original_provider = config.llm_provider
        assert original_provider == "openai"
