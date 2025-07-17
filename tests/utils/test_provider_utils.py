"""Tests for provider utility functions."""

import os
from unittest.mock import patch

import pytest

from arklex.utils.provider_utils import (
    get_api_key_for_provider,
    get_endpoint_for_provider,
    get_provider_config,
)


class TestGetApiKeyForProvider:
    """Test cases for get_api_key_for_provider function."""

    def test_get_api_key_openai(self) -> None:
        """Test getting OpenAI API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}):
            api_key = get_api_key_for_provider("openai")
            assert api_key == "test-openai-key"

    def test_get_api_key_anthropic(self) -> None:
        """Test getting Anthropic API key."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key"}):
            api_key = get_api_key_for_provider("anthropic")
            assert api_key == "test-anthropic-key"

    def test_get_api_key_gemini(self) -> None:
        """Test getting Gemini API key."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-google-key"}):
            api_key = get_api_key_for_provider("google")
            assert api_key == "test-google-key"

    def test_get_api_key_huggingface(self) -> None:
        """Test getting HuggingFace API key."""
        with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "test-hf-key"}):
            api_key = get_api_key_for_provider("huggingface")
            assert api_key == "test-hf-key"

    def test_get_api_key_unknown_provider(self) -> None:
        """Test getting API key for unknown provider (should default to OpenAI)."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}):
            api_key = get_api_key_for_provider("unknown-provider")
            assert api_key == "test-openai-key"

    def test_get_api_key_missing_environment_variable(self) -> None:
        """Test getting API key when environment variable is not set."""
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(
                ValueError, match="API key for provider 'openai' is missing or empty"
            ),
        ):
            get_api_key_for_provider("openai")

    def test_get_api_key_empty_environment_variable(self) -> None:
        """Test getting API key when environment variable is empty."""
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}),
            pytest.raises(
                ValueError, match="API key for provider 'openai' is missing or empty"
            ),
        ):
            get_api_key_for_provider("openai")

    def test_get_api_key_case_sensitivity(self) -> None:
        """Test that provider names are case-sensitive."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}):
            api_key = get_api_key_for_provider("OpenAI")
            assert (
                api_key == "test-openai-key"
            )  # Should still work due to default fallback

    def test_get_api_key_all_providers(self) -> None:
        """Test getting API keys for all supported providers."""
        test_env = {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GOOGLE_API_KEY": "test-google-key",
            "HUGGINGFACE_API_KEY": "test-hf-key",
        }

        with patch.dict(os.environ, test_env):
            assert get_api_key_for_provider("openai") == "test-openai-key"
            assert get_api_key_for_provider("anthropic") == "test-anthropic-key"
            assert get_api_key_for_provider("google") == "test-google-key"
            assert get_api_key_for_provider("huggingface") == "test-hf-key"


class TestGetEndpointForProvider:
    """Test cases for get_endpoint_for_provider function."""

    def test_get_endpoint_openai(self) -> None:
        """Test getting OpenAI endpoint."""
        endpoint = get_endpoint_for_provider("openai")
        assert endpoint == "https://api.openai.com/v1"

    def test_get_endpoint_anthropic(self) -> None:
        """Test getting Anthropic endpoint."""
        endpoint = get_endpoint_for_provider("anthropic")
        assert endpoint == "https://api.anthropic.com"

    def test_get_endpoint_gemini(self) -> None:
        """Test getting Gemini endpoint."""
        endpoint = get_endpoint_for_provider("google")
        assert endpoint == "https://generativelanguage.googleapis.com"

    def test_get_endpoint_huggingface(self) -> None:
        """Test getting HuggingFace endpoint."""
        endpoint = get_endpoint_for_provider("huggingface")
        assert endpoint == "https://api-inference.huggingface.co"

    def test_get_endpoint_unknown_provider(self) -> None:
        """Test getting endpoint for unknown provider (should default to OpenAI)."""
        endpoint = get_endpoint_for_provider("unknown-provider")
        assert endpoint == "https://api.openai.com/v1"

    def test_get_endpoint_case_sensitivity(self) -> None:
        """Test that provider names are case-sensitive."""
        endpoint = get_endpoint_for_provider("OpenAI")
        assert (
            endpoint == "https://api.openai.com/v1"
        )  # Should still work due to default fallback

    def test_get_endpoint_all_providers(self) -> None:
        """Test getting endpoints for all supported providers."""
        expected_endpoints = {
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com",
            "google": "https://generativelanguage.googleapis.com",
            "huggingface": "https://api-inference.huggingface.co",
        }

        for provider, expected_endpoint in expected_endpoints.items():
            endpoint = get_endpoint_for_provider(provider)
            assert endpoint == expected_endpoint

    def test_get_endpoint_consistency(self) -> None:
        """Test that endpoints are consistent across multiple calls."""
        provider = "openai"
        endpoint1 = get_endpoint_for_provider(provider)
        endpoint2 = get_endpoint_for_provider(provider)
        assert endpoint1 == endpoint2


class TestGetProviderConfig:
    """Test cases for get_provider_config function."""

    def test_get_provider_config_openai(self) -> None:
        """Test getting complete configuration for OpenAI."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}):
            config = get_provider_config("openai", "gpt-4")

            assert config["model_name"] == "gpt-4"
            assert config["model_type_or_path"] == "gpt-4"
            assert config["llm_provider"] == "openai"
            assert config["api_key"] == "test-openai-key"
            assert config["endpoint"] == "https://api.openai.com/v1"

    def test_get_provider_config_anthropic(self) -> None:
        """Test getting complete configuration for Anthropic."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key"}):
            config = get_provider_config("anthropic", "claude-3-sonnet")

            assert config["model_name"] == "claude-3-sonnet"
            assert config["model_type_or_path"] == "claude-3-sonnet"
            assert config["llm_provider"] == "anthropic"
            assert config["api_key"] == "test-anthropic-key"
            assert config["endpoint"] == "https://api.anthropic.com"

    def test_get_provider_config_gemini(self) -> None:
        """Test getting complete configuration for Gemini."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-google-key"}):
            config = get_provider_config("google", "gemini-pro")

            assert config["model_name"] == "gemini-pro"
            assert config["model_type_or_path"] == "gemini-pro"
            assert config["llm_provider"] == "google"
            assert config["api_key"] == "test-google-key"
            assert config["endpoint"] == "https://generativelanguage.googleapis.com"

    def test_get_provider_config_huggingface(self) -> None:
        """Test getting complete configuration for HuggingFace."""
        with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "test-hf-key"}):
            config = get_provider_config("huggingface", "microsoft/DialoGPT-medium")

            assert config["model_name"] == "microsoft/DialoGPT-medium"
            assert config["model_type_or_path"] == "microsoft/DialoGPT-medium"
            assert config["llm_provider"] == "huggingface"
            assert config["api_key"] == "test-hf-key"
            assert config["endpoint"] == "https://api-inference.huggingface.co"

    def test_get_provider_config_unknown_provider(self) -> None:
        """Test getting configuration for unknown provider."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}):
            config = get_provider_config("unknown-provider", "test-model")

            assert config["model_name"] == "test-model"
            assert config["model_type_or_path"] == "test-model"
            assert config["llm_provider"] == "unknown-provider"
            assert config["api_key"] == "test-openai-key"  # Defaults to OpenAI
            assert (
                config["endpoint"] == "https://api.openai.com/v1"
            )  # Defaults to OpenAI

    def test_get_provider_config_missing_api_key(self) -> None:
        """Test getting configuration when API key is not set."""
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(
                ValueError, match="API key for provider 'openai' is missing or empty"
            ),
        ):
            get_provider_config("openai", "gpt-4")

    def test_get_provider_config_empty_model_name(self) -> None:
        """Test getting configuration with empty model name."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}):
            config = get_provider_config("openai", "")

            assert config["model_name"] == ""
            assert config["model_type_or_path"] == ""
            assert config["llm_provider"] == "openai"
            assert config["api_key"] == "test-openai-key"
            assert config["endpoint"] == "https://api.openai.com/v1"

    def test_get_provider_config_all_providers(self) -> None:
        """Test getting configuration for all supported providers."""
        test_env = {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GOOGLE_API_KEY": "test-google-key",
            "HUGGINGFACE_API_KEY": "test-hf-key",
        }

        test_cases = [
            ("openai", "gpt-4"),
            ("anthropic", "claude-3-sonnet"),
            ("google", "gemini-pro"),
            ("huggingface", "microsoft/DialoGPT-medium"),
        ]

        with patch.dict(os.environ, test_env):
            for provider, model in test_cases:
                config = get_provider_config(provider, model)

                assert config["model_name"] == model
                assert config["model_type_or_path"] == model
                assert config["llm_provider"] == provider
                assert config["api_key"] != ""  # Should have API key
                assert config["endpoint"] != ""  # Should have endpoint

    def test_get_provider_config_structure(self) -> None:
        """Test that provider config has the correct structure."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            config = get_provider_config("openai", "gpt-4")

            # Check that all required keys are present
            required_keys = [
                "model_name",
                "model_type_or_path",
                "llm_provider",
                "api_key",
                "endpoint",
            ]
            for key in required_keys:
                assert key in config

            # Check that values are strings
            for _key, value in config.items():
                assert isinstance(value, str)

    def test_get_provider_config_immutability(self) -> None:
        """Test that provider config is not affected by subsequent calls."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-anthropic-key"},
        ):
            config1 = get_provider_config("openai", "gpt-4")
            config2 = get_provider_config("anthropic", "claude-3-sonnet")

            # Configs should be independent
            assert config1["llm_provider"] == "openai"
            assert config2["llm_provider"] == "anthropic"
            assert config1["model_name"] == "gpt-4"
            assert config2["model_name"] == "claude-3-sonnet"


class TestProviderUtilsIntegration:
    """Integration tests for provider utilities."""

    def test_full_provider_configuration_flow(self) -> None:
        """Test the complete flow from API key to endpoint to full config."""
        test_env = {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GOOGLE_API_KEY": "test-google-key",
            "HUGGINGFACE_API_KEY": "test-hf-key",
        }

        with patch.dict(os.environ, test_env):
            providers = ["openai", "anthropic", "gemini", "huggingface"]
            models = [
                "gpt-4",
                "claude-3-sonnet",
                "gemini-pro",
                "microsoft/DialoGPT-medium",
            ]

            for provider, model in zip(providers, models, strict=False):
                # Test individual functions
                api_key = get_api_key_for_provider(provider)
                endpoint = get_endpoint_for_provider(provider)

                # Test combined function
                config = get_provider_config(provider, model)

                # Verify consistency
                assert config["api_key"] == api_key
                assert config["endpoint"] == endpoint
                assert config["llm_provider"] == provider
                assert config["model_name"] == model

    def test_provider_utils_with_model_config_integration(self) -> None:
        """Test integration with ModelConfig class."""
        from arklex.orchestrator.NLU.services.model_config import ModelConfig

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # Get provider config
            config = get_provider_config("openai", "gpt-4")

            # Test that it works with ModelConfig
            kwargs = ModelConfig.get_model_kwargs(config)

            assert kwargs["model"] == "gpt-4"
            assert kwargs["temperature"] == 0.1
            assert kwargs["api_key"] == "test-key"
            assert kwargs["n"] == 1

    def test_provider_utils_error_handling(self) -> None:
        """Test error handling in provider utilities."""
        # Test with None provider
        with pytest.raises(TypeError):
            get_api_key_for_provider(None)

        with pytest.raises(TypeError):
            get_endpoint_for_provider(None)

        with pytest.raises(TypeError):
            get_provider_config(None, "test-model")

        # Test with None model
        with pytest.raises(TypeError):
            get_provider_config("openai", None)


class TestProviderUtilsEdgeCases:
    """Test edge cases for provider utilities."""

    def test_provider_utils_with_special_characters(self) -> None:
        """Test provider utilities with special characters in model names."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            special_models = [
                "gpt-4-turbo-preview",
                "claude-3-sonnet-20240229",
                "gemini-1.5-pro",
                "microsoft/DialoGPT-medium",
                "meta-llama/Llama-2-7b-chat-hf",
            ]

            for model in special_models:
                config = get_provider_config("openai", model)
                assert config["model_name"] == model
                assert config["model_type_or_path"] == model

    def test_provider_utils_with_very_long_model_names(self) -> None:
        """Test provider utilities with very long model names."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            long_model = "a" * 1000  # Very long model name
            config = get_provider_config("openai", long_model)
            assert config["model_name"] == long_model
            assert len(config["model_name"]) == 1000

    def test_provider_utils_with_unicode_model_names(self) -> None:
        """Test provider utilities with Unicode model names."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            unicode_models = [
                "gpt-4-中文",
                "claude-3-日本語",
                "gemini-pro-한국어",
                "huggingface/模型-中文",
            ]

            for model in unicode_models:
                config = get_provider_config("openai", model)
                assert config["model_name"] == model
                assert config["model_type_or_path"] == model

    def test_provider_utils_with_whitespace(self) -> None:
        """Test provider utilities with whitespace in model names."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            whitespace_models = [
                " gpt-4 ",
                "claude-3-sonnet\t",
                "\ngemini-pro\n",
                "  huggingface/model  ",
            ]

            for model in whitespace_models:
                config = get_provider_config("openai", model)
                assert config["model_name"] == model  # Should preserve whitespace
                assert config["model_type_or_path"] == model
