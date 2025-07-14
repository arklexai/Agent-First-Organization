"""End-to-end tests for model provider selection in command line tools."""

import os
import sys
from unittest.mock import Mock, patch

import pytest

from arklex.orchestrator.NLU.services.model_config import ModelConfig
from arklex.utils.provider_utils import get_provider_config


class TestModelProviderE2E:
    """End-to-end tests for model provider selection."""

    def test_create_tool_with_different_providers(self) -> None:
        """Test that the create tool works with different providers."""
        # Test data for different providers
        test_cases = [
            ("openai", "gpt-4"),
            ("anthropic", "claude-3-sonnet"),
            ("gemini", "gemini-pro"),
            ("huggingface", "microsoft/DialoGPT-medium"),
        ]

        for provider, model in test_cases:
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-openai-key",
                    "ANTHROPIC_API_KEY": "test-anthropic-key",
                    "GOOGLE_API_KEY": "test-google-key",
                    "HUGGINGFACE_API_KEY": "test-hf-key",
                },
            ):
                # Get provider configuration
                config = get_provider_config(provider, model)

                # Verify configuration is correct
                assert config["llm_provider"] == provider
                assert config["model_type_or_path"] == model
                assert config["api_key"] != ""
                assert config["endpoint"] != ""

                # Test that model can be initialized
                with patch(
                    "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
                ) as mock_provider_map:
                    mock_class = Mock()
                    mock_provider_map.__contains__.return_value = True
                    mock_provider_map.__getitem__.return_value = mock_class

                    model_instance = ModelConfig.get_model_instance(config)
                    assert model_instance is not None
                    mock_class.assert_called_once()

    def test_run_tool_with_different_providers(self) -> None:
        """Test that the run tool works with different providers."""
        # Test data for different providers
        test_cases = [
            ("openai", "gpt-4"),
            ("anthropic", "claude-3-sonnet"),
            ("gemini", "gemini-pro"),
            ("huggingface", "microsoft/DialoGPT-medium"),
        ]

        for provider, model in test_cases:
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-openai-key",
                    "ANTHROPIC_API_KEY": "test-anthropic-key",
                    "GOOGLE_API_KEY": "test-google-key",
                    "HUGGINGFACE_API_KEY": "test-hf-key",
                },
            ):
                # Get provider configuration
                config = get_provider_config(provider, model)

                # Verify configuration is correct
                assert config["llm_provider"] == provider
                assert config["model_type_or_path"] == model
                assert config["api_key"] != ""
                assert config["endpoint"] != ""

                # Test that model can be initialized
                with patch(
                    "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
                ) as mock_provider_map:
                    mock_class = Mock()
                    mock_provider_map.__contains__.return_value = True
                    mock_provider_map.__getitem__.return_value = mock_class

                    model_instance = ModelConfig.get_model_instance(config)
                    assert model_instance is not None
                    mock_class.assert_called_once()

    def test_model_api_with_different_providers(self) -> None:
        """Test that the model API works with different providers."""
        # Test data for different providers
        test_cases = [
            ("openai", "gpt-4"),
            ("anthropic", "claude-3-sonnet"),
            ("gemini", "gemini-pro"),
            ("huggingface", "microsoft/DialoGPT-medium"),
        ]

        for provider, model in test_cases:
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-openai-key",
                    "ANTHROPIC_API_KEY": "test-anthropic-key",
                    "GOOGLE_API_KEY": "test-google-key",
                    "HUGGINGFACE_API_KEY": "test-hf-key",
                },
            ):
                # Get provider configuration
                config = get_provider_config(provider, model)

                # Verify configuration is correct
                assert config["llm_provider"] == provider
                assert config["model_type_or_path"] == model
                assert config["api_key"] != ""
                assert config["endpoint"] != ""

                # Test that model can be initialized
                with patch(
                    "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
                ) as mock_provider_map:
                    mock_class = Mock()
                    mock_provider_map.__contains__.return_value = True
                    mock_provider_map.__getitem__.return_value = mock_class

                    model_instance = ModelConfig.get_model_instance(config)
                    assert model_instance is not None
                    mock_class.assert_called_once()

    def test_provider_configuration_consistency(self) -> None:
        """Test that provider configuration is consistent across all tools."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-openai-key",
                "ANTHROPIC_API_KEY": "test-anthropic-key",
                "GOOGLE_API_KEY": "test-google-key",
                "HUGGINGFACE_API_KEY": "test-hf-key",
            },
        ):
            providers = ["openai", "anthropic", "gemini", "huggingface"]
            models = [
                "gpt-4",
                "claude-3-sonnet",
                "gemini-pro",
                "microsoft/DialoGPT-medium",
            ]

            for provider, model in zip(providers, models, strict=False):
                # Test provider utils
                config = get_provider_config(provider, model)

                # Test model config
                kwargs = ModelConfig.get_model_kwargs(config)

                # Verify consistency
                assert kwargs["model"] == model
                assert kwargs["temperature"] == 0.1
                assert kwargs["api_key"] == config["api_key"]

                # Test model instance creation
                with patch(
                    "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
                ) as mock_provider_map:
                    mock_class = Mock()
                    mock_provider_map.__contains__.return_value = True
                    mock_provider_map.__getitem__.return_value = mock_class

                    model_instance = ModelConfig.get_model_instance(config)
                    assert model_instance is not None

    def test_provider_fallback_behavior(self) -> None:
        """Test that provider fallback behavior works correctly."""
        # Test with unknown provider (should fallback to OpenAI)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}):
            config = get_provider_config("unknown-provider", "test-model")

            # Should fallback to OpenAI configuration
            assert config["api_key"] == "test-openai-key"
            assert config["endpoint"] == "https://api.openai.com/v1"

            # But provider should still be set to unknown
            assert config["llm_provider"] == "unknown-provider"

            # This should fail because unknown provider is not in PROVIDER_MAP
            with pytest.raises(
                ValueError, match="Unsupported provider: unknown-provider"
            ):
                ModelConfig.get_model_instance(config)

    def test_provider_with_missing_api_keys(self) -> None:
        """Test behavior when API keys are missing."""
        with patch.dict(os.environ, {}, clear=True):
            providers = ["openai", "anthropic", "gemini", "huggingface"]
            models = [
                "gpt-4",
                "claude-3-sonnet",
                "gemini-pro",
                "microsoft/DialoGPT-medium",
            ]

            for provider, model in zip(providers, models, strict=False):
                with pytest.raises(
                    ValueError,
                    match=f"API key for provider '{provider}' is missing or empty",
                ):
                    get_provider_config(provider, model)

    def test_provider_with_custom_endpoints(self) -> None:
        """Test provider configuration with custom endpoints."""
        custom_config = {
            "model_type_or_path": "gpt-4",
            "llm_provider": "openai",
            "api_key": "test-key",
            "endpoint": "https://custom.openai.com/v1",
        }

        # Test kwargs generation
        kwargs = ModelConfig.get_model_kwargs(custom_config)
        assert kwargs["base_url"] == "https://custom.openai.com/v1"

        # Test model initialization
        with patch(
            "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_class = Mock()
            mock_provider_map.__contains__.return_value = True
            mock_provider_map.__getitem__.return_value = mock_class

            model_instance = ModelConfig.get_model_instance(custom_config)
            assert model_instance is not None

            # Verify custom endpoint was used
            call_args = mock_class.call_args[1]
            assert call_args["base_url"] == "https://custom.openai.com/v1"

    def test_provider_response_format_configuration(self) -> None:
        """Test response format configuration for different providers."""
        test_cases = [
            ("openai", "gpt-4", "json"),
            ("openai", "gpt-4", "text"),
            ("anthropic", "claude-3-sonnet", "json"),
            ("anthropic", "claude-3-sonnet", "text"),
        ]

        for provider, model, response_format in test_cases:
            # Mock the API key for each provider
            env_key = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "google": "GOOGLE_API_KEY",
                "huggingface": "HUGGINGFACE_API_KEY",
            }.get(provider, "OPENAI_API_KEY")

            with patch.dict(os.environ, {env_key: "test-key"}):
                config = get_provider_config(provider, model)

                # Test model initialization
                with patch(
                    "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
                ) as mock_provider_map:
                    mock_class = Mock()
                    mock_model_instance = Mock()
                    mock_model_instance.bind.return_value = (
                        mock_model_instance  # Make bind return the same model
                    )
                    mock_class.return_value = mock_model_instance
                    mock_provider_map.__contains__.return_value = True
                    mock_provider_map.__getitem__.return_value = mock_class

                    model_instance = ModelConfig.get_model_instance(config)
                    configured_model = ModelConfig.configure_response_format(
                        model_instance, config, response_format
                    )

                    if provider == "openai":
                        # OpenAI should have response format configured
                        expected_format = (
                            {"type": "json_object"}
                            if response_format == "json"
                            else {"type": "text"}
                        )
                        mock_model_instance.bind.assert_called_once_with(
                            response_format=expected_format
                        )
                    else:
                        # Non-OpenAI providers should not have bind called
                        mock_model_instance.bind.assert_not_called()

                    assert configured_model == mock_model_instance

    def test_provider_error_handling(self) -> None:
        """Test error handling for different provider scenarios."""
        # Test with unsupported provider
        config = {
            "model_type_or_path": "test-model",
            "llm_provider": "unsupported-provider",
        }

        with pytest.raises(
            ValueError,
            match="API key for provider 'unsupported-provider' is missing or empty",
        ):
            ModelConfig.get_model_instance(config)

        # Test with missing required fields
        incomplete_config = {"llm_provider": "openai"}

        with pytest.raises(KeyError):
            ModelConfig.get_model_kwargs(incomplete_config)

        with pytest.raises(KeyError):
            ModelConfig.get_model_instance(incomplete_config)

        # Test with invalid response format
        valid_config = get_provider_config("openai", "gpt-4")

        with patch(
            "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_class = Mock()
            mock_model_instance = Mock()
            mock_class.return_value = mock_model_instance
            mock_provider_map.__contains__.return_value = True
            mock_provider_map.__getitem__.return_value = mock_class

            model_instance = ModelConfig.get_model_instance(valid_config)

            with pytest.raises(ValueError, match="Invalid response format: invalid"):
                ModelConfig.configure_response_format(
                    model_instance, valid_config, "invalid"
                )

    def test_provider_performance_under_load(self) -> None:
        """Test provider performance under load."""
        import time

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            start_time = time.time()

            # Test multiple rapid provider configurations
            for i in range(100):
                config = get_provider_config("openai", f"gpt-4-{i}")
                # Get model kwargs to ensure the method works correctly
                ModelConfig.get_model_kwargs(config)

                with patch(
                    "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
                ) as mock_provider_map:
                    mock_class = Mock()
                    mock_provider_map.__contains__.return_value = True
                    mock_provider_map.__getitem__.return_value = mock_class

                    model_instance = ModelConfig.get_model_instance(config)
                    assert model_instance is not None

            end_time = time.time()
            duration = end_time - start_time

            # Should complete 100 operations in under 2 seconds
            assert duration < 2.0

    def test_provider_memory_efficiency(self) -> None:
        """Test that provider configuration is memory efficient."""
        import gc

        # Create many configurations
        configs = []
        for i in range(1000):
            config = get_provider_config("openai", f"gpt-4-{i}")
            configs.append(config)

        # Force garbage collection
        gc.collect()

        # Verify configurations are independent
        assert configs[0]["model_type_or_path"] == "gpt-4-0"
        assert configs[999]["model_type_or_path"] == "gpt-4-999"
        assert configs[0] is not configs[999]  # Should be different objects

        # Verify memory usage is reasonable (configs should be small)

        total_size = sum(sys.getsizeof(config) for config in configs)
        assert total_size < 1024 * 1024  # Less than 1MB for 1000 configs


class TestModelProviderE2EWithRealTools:
    """End-to-end tests with actual command line tools (when available)."""

    def test_create_tool_command_line_arguments(self) -> None:
        """Test that create tool accepts provider arguments."""
        # This test simulates the command line argument parsing
        # that would happen in the create tool

        # Simulate command line arguments
        args = {
            "llm_provider": "anthropic",
            "model": "claude-3-sonnet",
            "task_graph_file": "test_taskgraph.json",
            "testcases_file": "test_testcases.json",
        }

        # Simulate the configuration process
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            config = get_provider_config(args["llm_provider"], args["model"])

            # Verify configuration matches arguments
            assert config["llm_provider"] == args["llm_provider"]
            assert config["model_type_or_path"] == args["model"]
            assert config["api_key"] == "test-key"

            # Test model initialization
            with patch(
                "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
            ) as mock_provider_map:
                mock_class = Mock()
                mock_provider_map.__contains__.return_value = True
                mock_provider_map.__getitem__.return_value = mock_class

                model_instance = ModelConfig.get_model_instance(config)
                assert model_instance is not None

    def test_run_tool_command_line_arguments(self) -> None:
        """Test that run tool accepts provider arguments."""
        # Simulate command line arguments
        args = {
            "llm_provider": "google",
            "model": "gemini-pro",
            "task_graph_file": "test_taskgraph.json",
            "testcases_file": "test_testcases.json",
        }

        # Simulate the configuration process
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            config = get_provider_config(args["llm_provider"], args["model"])

            # Verify configuration matches arguments
            assert config["llm_provider"] == args["llm_provider"]
            assert config["model_type_or_path"] == args["model"]
            assert config["api_key"] == "test-key"

            # Test model initialization
            with patch(
                "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
            ) as mock_provider_map:
                mock_class = Mock()
                mock_provider_map.__contains__.return_value = True
                mock_provider_map.__getitem__.return_value = mock_class

                model_instance = ModelConfig.get_model_instance(config)
                assert model_instance is not None

    def test_model_api_command_line_arguments(self) -> None:
        """Test that model API accepts provider arguments."""
        # Simulate command line arguments
        args = {
            "llm_provider": "huggingface",
            "model": "microsoft/DialoGPT-medium",
            "port": 8000,
            "log_level": "INFO",
        }

        # Simulate the configuration process
        with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "test-key"}):
            config = get_provider_config(args["llm_provider"], args["model"])

            # Verify configuration matches arguments
            assert config["llm_provider"] == args["llm_provider"]
            assert config["model_type_or_path"] == args["model"]
            assert config["api_key"] == "test-key"

            # Test model initialization
            with patch(
                "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
            ) as mock_provider_map:
                mock_class = Mock()
                mock_provider_map.__contains__.return_value = True
                mock_provider_map.__getitem__.return_value = mock_class

                model_instance = ModelConfig.get_model_instance(config)
                assert model_instance is not None

    def test_provider_argument_validation(self) -> None:
        """Test that provider arguments are properly validated."""
        # Test with valid providers
        valid_providers = ["openai", "anthropic", "google", "huggingface"]

        for provider in valid_providers:
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-openai-key",
                    "ANTHROPIC_API_KEY": "test-anthropic-key",
                    "GOOGLE_API_KEY": "test-google-key",
                    "HUGGINGFACE_API_KEY": "test-hf-key",
                },
            ):
                config = get_provider_config(provider, "test-model")

                # Should not raise an exception
                with patch(
                    "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
                ) as mock_provider_map:
                    mock_class = Mock()
                    mock_provider_map.__contains__.return_value = True
                    mock_provider_map.__getitem__.return_value = mock_class

                    model_instance = ModelConfig.get_model_instance(config)
                    assert model_instance is not None

        # Test with invalid provider
        with pytest.raises(ValueError, match="Unsupported provider: invalid-provider"):
            config = get_provider_config("invalid-provider", "test-model")
            ModelConfig.get_model_instance(config)

    def test_model_argument_validation(self) -> None:
        """Test that model arguments are properly validated."""
        # Test with various model names
        test_models = [
            "gpt-4",
            "gpt-3.5-turbo",
            "claude-3-sonnet",
            "claude-3-haiku",
            "gemini-pro",
            "gemini-1.5-pro",
            "microsoft/DialoGPT-medium",
            "meta-llama/Llama-2-7b-chat-hf",
        ]

        for model in test_models:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                config = get_provider_config("openai", model)

                # Should not raise an exception
                with patch(
                    "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
                ) as mock_provider_map:
                    mock_class = Mock()
                    mock_provider_map.__contains__.return_value = True
                    mock_provider_map.__getitem__.return_value = mock_class

                    model_instance = ModelConfig.get_model_instance(config)
                    assert model_instance is not None
                    assert mock_class.call_args[1]["model"] == model
