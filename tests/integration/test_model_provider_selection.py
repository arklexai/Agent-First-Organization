"""Integration tests for model provider selection functionality."""

import os
from unittest.mock import Mock, patch

import pytest

from arklex.orchestrator.NLU.services.model_config import ModelConfig
from arklex.utils.provider_utils import get_provider_config


class TestModelProviderSelectionIntegration:
    """Integration tests for model provider selection."""

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GOOGLE_API_KEY": "test-google-key",
            "HUGGINGFACE_API_KEY": "test-hf-key",
        },
    )
    def test_openai_provider_selection(self) -> None:
        """Test complete OpenAI provider selection flow."""
        # Get provider configuration
        config = get_provider_config("openai", "gpt-4")

        # Verify configuration
        assert config["llm_provider"] == "openai"
        assert config["model_type_or_path"] == "gpt-4"
        assert config["api_key"] == "test-openai-key"
        assert config["endpoint"] == "https://api.openai.com/v1"

        # Test model initialization
        with patch(
            "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_openai_class = Mock()
            mock_instance = Mock()
            mock_openai_class.return_value = mock_instance
            mock_provider_map.__getitem__.return_value = mock_openai_class
            mock_provider_map.__contains__.return_value = True

            # Initialize model instance
            ModelConfig.get_model_instance(config)

            # Verify model was initialized correctly
            mock_openai_class.assert_called_once()
            call_args = mock_openai_class.call_args[1]
            assert call_args["model"] == "gpt-4"
            assert call_args["temperature"] == 0.1
            assert call_args["api_key"] == "test-openai-key"
            assert call_args["n"] == 1

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GOOGLE_API_KEY": "test-google-key",
            "HUGGINGFACE_API_KEY": "test-hf-key",
        },
    )
    def test_anthropic_provider_selection(self) -> None:
        """Test complete Anthropic provider selection flow."""
        # Get provider configuration
        config = get_provider_config("anthropic", "claude-3-sonnet")

        # Verify configuration
        assert config["llm_provider"] == "anthropic"
        assert config["model_type_or_path"] == "claude-3-sonnet"
        assert config["api_key"] == "test-anthropic-key"
        assert config["endpoint"] == "https://api.anthropic.com"

        # Test model initialization
        with patch(
            "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_anthropic_class = Mock()
            mock_instance = Mock()
            mock_anthropic_class.return_value = mock_instance
            mock_provider_map.__getitem__.return_value = mock_anthropic_class
            mock_provider_map.__contains__.return_value = True

            # Initialize model instance
            ModelConfig.get_model_instance(config)

            # Verify model was initialized correctly
            mock_anthropic_class.assert_called_once()
            call_args = mock_anthropic_class.call_args[1]
            assert call_args["model"] == "claude-3-sonnet"
            assert call_args["temperature"] == 0.1
            assert call_args["api_key"] == "test-anthropic-key"
            assert "n" not in call_args  # Anthropic doesn't use 'n' parameter

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GOOGLE_API_KEY": "test-google-key",
            "HUGGINGFACE_API_KEY": "test-hf-key",
        },
    )
    def test_gemini_provider_selection(self) -> None:
        """Test complete Gemini provider selection flow."""
        # Get provider configuration
        config = get_provider_config("google", "gemini-pro")

        # Verify configuration
        assert config["llm_provider"] == "google"
        assert config["model_type_or_path"] == "gemini-pro"
        assert config["api_key"] == "test-google-key"
        assert config["endpoint"] == "https://generativelanguage.googleapis.com"

        # Test model initialization
        with patch(
            "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_gemini_class = Mock()
            mock_instance = Mock()
            mock_gemini_class.return_value = mock_instance
            mock_provider_map.__getitem__.return_value = mock_gemini_class
            mock_provider_map.__contains__.return_value = True

            # Initialize model instance
            ModelConfig.get_model_instance(config)

            # Verify model was initialized correctly
            mock_gemini_class.assert_called_once()
            call_args = mock_gemini_class.call_args[1]
            assert call_args["model"] == "gemini-pro"
            assert call_args["temperature"] == 0.1
            assert call_args["api_key"] == "test-google-key"
            assert call_args["n"] == 1

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GOOGLE_API_KEY": "test-google-key",
            "HUGGINGFACE_API_KEY": "test-hf-key",
        },
    )
    def test_huggingface_provider_selection(self) -> None:
        """Test complete HuggingFace provider selection flow."""
        # Get provider configuration
        config = get_provider_config("huggingface", "microsoft/DialoGPT-medium")

        # Verify configuration
        assert config["llm_provider"] == "huggingface"
        assert config["model_type_or_path"] == "microsoft/DialoGPT-medium"
        assert config["api_key"] == "test-hf-key"
        assert config["endpoint"] == "https://api-inference.huggingface.co"

        # Test model initialization
        with patch(
            "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_hf_function = Mock()
            mock_instance = Mock()
            mock_hf_function.return_value = mock_instance
            mock_provider_map.__getitem__.return_value = mock_hf_function
            mock_provider_map.__contains__.return_value = True

            # Initialize model instance
            ModelConfig.get_model_instance(config)

            # Verify model was initialized correctly
            mock_hf_function.assert_called_once()
            call_args = mock_hf_function.call_args[1]
            assert call_args["model"] == "microsoft/DialoGPT-medium"
            assert call_args["temperature"] == 0.1
            assert call_args["api_key"] == "test-hf-key"
            assert call_args["n"] == 1

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GOOGLE_API_KEY": "test-google-key",
            "HUGGINGFACE_API_KEY": "test-hf-key",
        },
    )
    def test_all_providers_round_trip(self) -> None:
        """Test that all providers can be selected and configured correctly."""
        providers_and_models = [
            ("openai", "gpt-4"),
            ("anthropic", "claude-3-sonnet"),
            ("google", "gemini-pro"),
            ("huggingface", "microsoft/DialoGPT-medium"),
        ]

        for provider, model in providers_and_models:
            # Get configuration
            config = get_provider_config(provider, model)

            # Verify basic configuration
            assert config["llm_provider"] == provider
            assert config["model_type_or_path"] == model
            assert config["api_key"] != ""
            assert config["endpoint"] != ""

            # Test model kwargs generation
            kwargs = ModelConfig.get_model_kwargs(config)
            assert kwargs["model"] == model
            assert kwargs["temperature"] == 0.1

            # Test model instance creation (with mocking)
            with patch(
                "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
            ) as mock_provider_map:
                mock_class = Mock()
                mock_instance = Mock()
                mock_class.return_value = mock_instance
                mock_provider_map.__getitem__.return_value = mock_class
                mock_provider_map.__contains__.return_value = True

                model_instance = ModelConfig.get_model_instance(config)
                assert model_instance is not None
                mock_class.assert_called_once()

    def test_provider_selection_with_custom_endpoints(self) -> None:
        """Test provider selection with custom endpoints."""
        custom_config = {
            "model_type_or_path": "gpt-4",
            "llm_provider": "openai",
            "api_key": "test-key",
            "endpoint": "https://custom.openai.com/v1",
        }

        # Test kwargs generation with custom endpoint
        kwargs = ModelConfig.get_model_kwargs(custom_config)
        assert kwargs["base_url"] == "https://custom.openai.com/v1"

        # Test model initialization with custom endpoint
        with patch(
            "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_openai_class = Mock()
            mock_instance = Mock()
            mock_openai_class.return_value = mock_instance
            mock_provider_map.__getitem__.return_value = mock_openai_class
            mock_provider_map.__contains__.return_value = True

            # Initialize model instance
            ModelConfig.get_model_instance(custom_config)

            mock_openai_class.assert_called_once()
            call_args = mock_openai_class.call_args[1]
            assert call_args["base_url"] == "https://custom.openai.com/v1"

    def test_provider_selection_with_response_format_configuration(self) -> None:
        """Test provider selection with response format configuration."""
        config = {
            "model_type_or_path": "gpt-4",
            "llm_provider": "openai",
            "api_key": "test-key",
        }

        # Test with OpenAI provider and JSON format
        with patch(
            "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_openai_class = Mock()
            mock_model_instance = Mock()
            mock_model_instance.bind.return_value = mock_model_instance
            mock_openai_class.return_value = mock_model_instance
            mock_provider_map.__getitem__.return_value = mock_openai_class
            mock_provider_map.__contains__.return_value = True

            model = ModelConfig.get_model_instance(config)
            ModelConfig.configure_response_format(model, config, "json")

            # Verify response format was configured
            mock_model_instance.bind.assert_called_once_with(
                response_format={"type": "json_object"}
            )

    def test_provider_selection_error_handling(self) -> None:
        """Test error handling in provider selection."""
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

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GOOGLE_API_KEY": "test-google-key",
            "HUGGINGFACE_API_KEY": "test-hf-key",
        },
    )
    def test_provider_selection_with_different_model_types(self) -> None:
        """Test provider selection with different model types and names."""
        test_cases = [
            ("openai", "gpt-4-turbo"),
            ("openai", "gpt-3.5-turbo"),
            ("anthropic", "claude-3-haiku"),
            ("anthropic", "claude-3-opus"),
            ("gemini", "gemini-1.5-pro"),
            ("gemini", "gemini-1.5-flash"),
            ("huggingface", "meta-llama/Llama-2-7b-chat-hf"),
            ("huggingface", "microsoft/DialoGPT-large"),
        ]

        for provider, model in test_cases:
            config = get_provider_config(provider, model)

            # Verify configuration
            assert config["llm_provider"] == provider
            assert config["model_type_or_path"] == model

            # Test kwargs generation
            kwargs = ModelConfig.get_model_kwargs(config)
            assert kwargs["model"] == model

            # Test model instance creation (with mocking)
            with patch(
                "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
            ) as mock_provider_map:
                mock_class = Mock()
                mock_instance = Mock()
                mock_class.return_value = mock_instance
                mock_provider_map.__getitem__.return_value = mock_class
                mock_provider_map.__contains__.return_value = True

                model_instance = ModelConfig.get_model_instance(config)
                assert model_instance is not None

    def test_provider_selection_performance(self) -> None:
        """Test that provider selection is performant."""
        import time

        config = get_provider_config("openai", "gpt-4")

        # Test multiple rapid selections
        start_time = time.time()
        for _ in range(100):
            kwargs = ModelConfig.get_model_kwargs(config)
            assert kwargs["model"] == "gpt-4"

        end_time = time.time()
        duration = end_time - start_time

        # Should complete 100 operations in under 1 second
        assert duration < 1.0

    def test_provider_selection_memory_usage(self) -> None:
        """Test that provider selection doesn't leak memory."""
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


class TestModelProviderSelectionRealModels:
    """Integration tests with real model classes (when available)."""

    def test_real_openai_model_initialization(self) -> None:
        """Test initialization with real OpenAI model (mocked for testing)."""
        from langchain_openai import ChatOpenAI

        config = get_provider_config("openai", "gpt-4")

        # Mock the PROVIDER_MAP to return ChatOpenAI class
        with patch(
            "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_provider_map.__getitem__.return_value = ChatOpenAI
            mock_provider_map.__contains__.return_value = True

            model = ModelConfig.get_model_instance(config)

            # Verify that ChatOpenAI was called with correct parameters
            assert model is not None
            # Check that the model was created with the right config
            assert hasattr(model, "model_name") or hasattr(model, "model")

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GOOGLE_API_KEY": "test-google-key",
            "HUGGINGFACE_API_KEY": "test-hf-key",
        },
    )
    def test_real_anthropic_model_initialization(self) -> None:
        """Test initialization with real Anthropic model (mocked for testing)."""
        from langchain_anthropic import ChatAnthropic

        config = get_provider_config("anthropic", "claude-3-sonnet")

        # Mock the PROVIDER_MAP to return ChatAnthropic class
        with patch(
            "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_provider_map.__getitem__.return_value = ChatAnthropic
            mock_provider_map.__contains__.return_value = True

            model = ModelConfig.get_model_instance(config)

            # Verify that ChatAnthropic was called with correct parameters
            assert model is not None
            # Check that the model was created with the right config
            assert hasattr(model, "model_name") or hasattr(model, "model")

    def test_dummy_provider_initialization(self) -> None:
        """Test initialization with dummy provider (always available)."""
        from arklex.utils.model_provider_config import DummyLLM

        config = {
            "model_type_or_path": "dummy-model",
            "llm_provider": "dummy",
            "api_key": "test-key",
        }

        with patch(
            "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_provider_map.__getitem__.return_value = DummyLLM
            mock_provider_map.__contains__.return_value = True

            model = ModelConfig.get_model_instance(config)

            # Verify that DummyLLM was called with correct parameters
            assert model is not None
            assert hasattr(model, "model_name") or hasattr(model, "model")


class TestModelProviderSelectionCommandLine:
    """Integration tests simulating command line provider selection."""

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GOOGLE_API_KEY": "test-google-key",
            "HUGGINGFACE_API_KEY": "test-hf-key",
        },
    )
    def test_command_line_provider_selection_simulation(self) -> None:
        """Simulate command line provider selection."""
        # Simulate command line arguments
        args = {
            "llm_provider": "anthropic",
            "model": "claude-3-sonnet",
        }

        # Simulate the configuration process
        config = get_provider_config(args["llm_provider"], args["model"])

        # Verify the configuration matches command line args
        assert config["llm_provider"] == args["llm_provider"]
        assert config["model_type_or_path"] == args["model"]

        # Test model initialization
        with patch(
            "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_class = Mock()
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            mock_provider_map.__getitem__.return_value = mock_class
            mock_provider_map.__contains__.return_value = True

            model = ModelConfig.get_model_instance(config)

            # Verify model was created
            assert model is not None
            mock_class.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GOOGLE_API_KEY": "test-google-key",
            "HUGGINGFACE_API_KEY": "test-hf-key",
        },
    )
    def test_command_line_provider_selection_all_providers(self) -> None:
        """Test command line provider selection for all providers."""
        test_args = [
            {"llm_provider": "openai", "model": "gpt-4"},
            {"llm_provider": "anthropic", "model": "claude-3-sonnet"},
            {"llm_provider": "gemini", "model": "gemini-pro"},
            {"llm_provider": "huggingface", "model": "microsoft/DialoGPT-medium"},
        ]

        for args in test_args:
            config = get_provider_config(args["llm_provider"], args["model"])

            assert config["llm_provider"] == args["llm_provider"]
            assert config["model_type_or_path"] == args["model"]

            # Test that model can be initialized
            with patch(
                "arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP"
            ) as mock_provider_map:
                mock_class = Mock()
                mock_instance = Mock()
                mock_class.return_value = mock_instance
                mock_provider_map.__getitem__.return_value = mock_class
                mock_provider_map.__contains__.return_value = True

                model = ModelConfig.get_model_instance(config)

                # Verify model was created
                assert model is not None
                mock_class.assert_called_once()
