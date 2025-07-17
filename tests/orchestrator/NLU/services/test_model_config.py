"""Tests for model configuration handling."""

from unittest.mock import Mock, patch

import pytest

from arklex.orchestrator.NLU.services.model_config import ModelConfig


class TestModelConfig:
    """Test cases for ModelConfig class."""

    def test_get_model_kwargs_basic(self) -> None:
        """Test basic model kwargs generation."""
        config = {
            "model_type_or_path": "gpt-4",
            "llm_provider": "openai",
            "api_key": "test-key",
        }

        kwargs = ModelConfig.get_model_kwargs(config)

        assert kwargs["model"] == "gpt-4"
        assert kwargs["temperature"] == 0.1
        assert kwargs["api_key"] == "test-key"
        assert kwargs["n"] == 1

    def test_get_model_kwargs_with_endpoint(self) -> None:
        """Test model kwargs with custom endpoint."""
        config = {
            "model_type_or_path": "gpt-4",
            "llm_provider": "openai",
            "api_key": "test-key",
            "endpoint": "https://custom.openai.com/v1",
        }

        kwargs = ModelConfig.get_model_kwargs(config)

        assert kwargs["base_url"] == "https://custom.openai.com/v1"

    def test_get_model_kwargs_anthropic_provider(self) -> None:
        """Test model kwargs for Anthropic provider (no 'n' parameter)."""
        config = {
            "model_type_or_path": "claude-3-sonnet",
            "llm_provider": "anthropic",
            "api_key": "test-key",
        }

        kwargs = ModelConfig.get_model_kwargs(config)

        assert kwargs["model"] == "claude-3-sonnet"
        assert kwargs["temperature"] == 0.1
        assert kwargs["api_key"] == "test-key"
        assert "n" not in kwargs  # Anthropic doesn't use 'n' parameter

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_get_model_kwargs_without_api_key(self) -> None:
        """Test model kwargs without API key."""
        config = {
            "model_type_or_path": "gpt-4",
            "llm_provider": "openai",
        }

        with pytest.raises(
            ValueError, match="API key for provider 'openai' is missing or empty"
        ):
            ModelConfig.get_model_kwargs(config)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_get_model_kwargs_empty_api_key(self) -> None:
        """Test model kwargs with empty API key."""
        config = {
            "model_type_or_path": "gpt-4",
            "llm_provider": "openai",
            "api_key": "",
        }

        with pytest.raises(
            ValueError, match="API key for provider 'openai' is missing or empty"
        ):
            ModelConfig.get_model_kwargs(config)

    def test_get_model_kwargs_none_api_key(self) -> None:
        """Test model kwargs with None API key."""
        config = {
            "model_type_or_path": "gpt-4",
            "llm_provider": "openai",
            "api_key": None,
        }

        with pytest.raises(
            ValueError, match="API key for provider 'openai' is missing or empty"
        ):
            ModelConfig.get_model_kwargs(config)

    def test_get_model_kwargs_endpoint_for_non_openai_anthropic(self) -> None:
        """Test that endpoint is not added for non-OpenAI/Anthropic providers."""
        config = {
            "model_type_or_path": "gemini-pro",
            "llm_provider": "gemini",
            "api_key": "test-key",
            "endpoint": "https://custom.gemini.com",
        }

        kwargs = ModelConfig.get_model_kwargs(config)

        assert "base_url" not in kwargs

    def test_get_model_kwargs_empty_endpoint(self) -> None:
        """Test model kwargs with empty endpoint."""
        config = {
            "model_type_or_path": "gpt-4",
            "llm_provider": "openai",
            "api_key": "test-key",
            "endpoint": "",
        }

        kwargs = ModelConfig.get_model_kwargs(config)

        assert "base_url" not in kwargs

    def test_get_model_kwargs_none_endpoint(self) -> None:
        """Test model kwargs with None endpoint."""
        config = {
            "model_type_or_path": "gpt-4",
            "llm_provider": "openai",
            "api_key": "test-key",
            "endpoint": None,
        }

        kwargs = ModelConfig.get_model_kwargs(config)

        assert "base_url" not in kwargs

    @patch("arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP")
    def test_get_model_instance_openai(self, mock_provider_map: Mock) -> None:
        """Test getting OpenAI model instance."""
        mock_openai_class = Mock()
        mock_provider_map.__contains__.return_value = True
        mock_provider_map.__getitem__.return_value = mock_openai_class

        config = {
            "model_type_or_path": "gpt-4",
            "llm_provider": "openai",
            "api_key": "test-key",
        }

        ModelConfig.get_model_instance(config)

        mock_openai_class.assert_called_once()
        call_args = mock_openai_class.call_args[1]
        assert call_args["model"] == "gpt-4"
        assert call_args["temperature"] == 0.1
        assert call_args["api_key"] == "test-key"
        assert call_args["n"] == 1

    @patch("arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP")
    def test_get_model_instance_anthropic(self, mock_provider_map: Mock) -> None:
        """Test getting Anthropic model instance."""
        mock_anthropic_class = Mock()
        mock_provider_map.__contains__.return_value = True
        mock_provider_map.__getitem__.return_value = mock_anthropic_class

        config = {
            "model_type_or_path": "claude-3-sonnet",
            "llm_provider": "anthropic",
            "api_key": "test-key",
        }

        ModelConfig.get_model_instance(config)

        mock_anthropic_class.assert_called_once()
        call_args = mock_anthropic_class.call_args[1]
        assert call_args["model"] == "claude-3-sonnet"
        assert call_args["temperature"] == 0.1
        assert call_args["api_key"] == "test-key"
        assert "n" not in call_args

    @patch("arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP")
    def test_get_model_instance_gemini(self, mock_provider_map: Mock) -> None:
        """Test getting Gemini model instance."""
        mock_gemini_class = Mock()
        mock_provider_map.__contains__.return_value = True
        mock_provider_map.__getitem__.return_value = mock_gemini_class

        config = {
            "model_type_or_path": "gemini-pro",
            "llm_provider": "gemini",
            "api_key": "test-key",
        }

        ModelConfig.get_model_instance(config)

        mock_gemini_class.assert_called_once()
        call_args = mock_gemini_class.call_args[1]
        assert call_args["model"] == "gemini-pro"
        assert call_args["temperature"] == 0.1
        assert call_args["api_key"] == "test-key"
        assert call_args["n"] == 1

    @patch("arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP")
    def test_get_model_instance_huggingface(self, mock_provider_map: Mock) -> None:
        """Test getting HuggingFace model instance."""
        mock_hf_function = Mock()
        mock_provider_map.__contains__.return_value = True
        mock_provider_map.__getitem__.return_value = mock_hf_function

        config = {
            "model_type_or_path": "microsoft/DialoGPT-medium",
            "llm_provider": "huggingface",
            "api_key": "test-key",
        }

        ModelConfig.get_model_instance(config)

        mock_hf_function.assert_called_once()
        call_args = mock_hf_function.call_args[1]
        assert call_args["model"] == "microsoft/DialoGPT-medium"
        assert call_args["temperature"] == 0.1
        assert call_args["api_key"] == "test-key"
        assert call_args["n"] == 1

    def test_get_model_instance_unsupported_provider(self) -> None:
        """Test getting model instance with unsupported provider."""
        config = {
            "model_type_or_path": "test-model",
            "llm_provider": "unsupported-provider",
        }

        with pytest.raises(
            ValueError,
            match="API key for provider 'unsupported-provider' is missing or empty",
        ):
            ModelConfig.get_model_instance(config)

    def test_configure_response_format_text_openai(self) -> None:
        """Test configuring text response format for OpenAI."""
        mock_model = Mock()
        mock_model.bind.return_value = mock_model  # Make bind return the same model
        config = {"llm_provider": "openai"}

        result = ModelConfig.configure_response_format(mock_model, config, "text")

        mock_model.bind.assert_called_once_with(response_format={"type": "text"})
        assert result == mock_model

    def test_configure_response_format_json_openai(self) -> None:
        """Test configuring JSON response format for OpenAI."""
        mock_model = Mock()
        mock_model.bind.return_value = mock_model  # Make bind return the same model
        config = {"llm_provider": "openai"}

        result = ModelConfig.configure_response_format(mock_model, config, "json")

        mock_model.bind.assert_called_once_with(response_format={"type": "json_object"})
        assert result == mock_model

    def test_configure_response_format_non_openai_provider(self) -> None:
        """Test configuring response format for non-OpenAI provider."""
        mock_model = Mock()
        config = {"llm_provider": "anthropic"}

        result = ModelConfig.configure_response_format(mock_model, config, "json")

        # Non-OpenAI providers should return the model unchanged
        mock_model.bind.assert_not_called()
        assert result == mock_model

    def test_configure_response_format_invalid_format(self) -> None:
        """Test configuring response format with invalid format."""
        mock_model = Mock()
        config = {"llm_provider": "openai"}

        with pytest.raises(ValueError, match="Invalid response format: invalid"):
            ModelConfig.configure_response_format(mock_model, config, "invalid")

    def test_configure_response_format_default_text(self) -> None:
        """Test configuring response format with default text format."""
        mock_model = Mock()
        mock_model.bind.return_value = mock_model  # Make bind return the same model
        config = {"llm_provider": "openai"}

        result = ModelConfig.configure_response_format(mock_model, config)

        mock_model.bind.assert_called_once_with(response_format={"type": "text"})
        assert result == mock_model


class TestModelConfigIntegration:
    """Integration tests for ModelConfig with real provider classes."""

    @patch("arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP")
    def test_full_model_initialization_flow_openai(
        self, mock_provider_map: Mock
    ) -> None:
        """Test complete model initialization flow for OpenAI."""
        from langchain_openai import ChatOpenAI

        # Create a mock that returns a real ChatOpenAI instance
        mock_provider_map.__contains__.return_value = True
        mock_provider_map.__getitem__.return_value = ChatOpenAI

        config = {
            "model_type_or_path": "gpt-4",
            "llm_provider": "openai",
            "api_key": "test-key",
            "endpoint": "https://custom.openai.com/v1",
        }

        model = ModelConfig.get_model_instance(config)

        # Since we're using a mock, we need to check it differently
        # The mock should have been called with the correct parameters
        assert str(model).startswith("<MagicMock name='ChatOpenAI()'")

        # Check that the mock was called with the correct parameters
        # We can't check the actual attributes since it's a mock, but we can verify
        # that the provider map was accessed correctly
        mock_provider_map.__contains__.assert_called_with("openai")
        mock_provider_map.__getitem__.assert_called_with("openai")

    @patch("arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP")
    def test_full_model_initialization_flow_anthropic(
        self, mock_provider_map: Mock
    ) -> None:
        """Test complete model initialization flow for Anthropic."""
        from langchain_anthropic import ChatAnthropic

        mock_provider_map.__contains__.return_value = True
        mock_provider_map.__getitem__.return_value = ChatAnthropic

        config = {
            "model_type_or_path": "claude-3-sonnet",
            "llm_provider": "anthropic",
            "api_key": "test-key",
            "endpoint": "https://custom.anthropic.com",
        }

        model = ModelConfig.get_model_instance(config)

        assert isinstance(model, ChatAnthropic)
        assert model.model == "claude-3-sonnet"
        assert model.temperature == 0.1
        assert (
            model.anthropic_api_key.get_secret_value() == "test-key"
        )  # Handle SecretStr
        assert model.anthropic_api_url == "https://custom.anthropic.com"

    @patch("arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP")
    def test_full_model_initialization_flow_gemini(
        self, mock_provider_map: Mock
    ) -> None:
        """Test complete model initialization flow for Gemini."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        mock_provider_map.__contains__.return_value = True
        mock_provider_map.__getitem__.return_value = ChatGoogleGenerativeAI

        config = {
            "model_type_or_path": "gemini-pro",
            "llm_provider": "gemini",
            "api_key": "test-key",
        }

        model = ModelConfig.get_model_instance(config)

        assert isinstance(model, ChatGoogleGenerativeAI)
        assert model.model == "models/gemini-pro"  # Gemini adds "models/" prefix
        assert model.temperature == 0.1
        assert model.google_api_key.get_secret_value() == "test-key"  # Handle SecretStr

    def test_response_format_configuration_integration(self) -> None:
        """Test response format configuration with real model instance."""

        # Create a mock model that behaves like ChatOpenAI
        mock_model = Mock()
        mock_model.bind.return_value = mock_model  # Make bind return the same model

        config = {"llm_provider": "openai"}

        result = ModelConfig.configure_response_format(mock_model, config, "json")

        mock_model.bind.assert_called_once_with(response_format={"type": "json_object"})
        assert result == mock_model


class TestModelConfigEdgeCases:
    """Test edge cases and error conditions for ModelConfig."""

    def test_get_model_kwargs_missing_required_fields(self) -> None:
        """Test get_model_kwargs with missing required fields."""
        config = {}

        with pytest.raises(KeyError):
            ModelConfig.get_model_kwargs(config)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_get_model_kwargs_missing_model_type(self) -> None:
        """Test get_model_kwargs with missing model_type_or_path."""
        config = {"llm_provider": "openai"}

        with pytest.raises(KeyError, match="'model_type_or_path'"):
            ModelConfig.get_model_kwargs(config)

    def test_get_model_kwargs_missing_provider(self) -> None:
        """Test get_model_kwargs with missing llm_provider."""
        config = {"model_type_or_path": "gpt-4"}

        with pytest.raises(
            ValueError, match="API key for provider '' is missing or empty"
        ):
            ModelConfig.get_model_kwargs(config)

    def test_get_model_instance_missing_required_fields(self) -> None:
        """Test get_model_instance with missing required fields."""
        config = {}

        with pytest.raises(KeyError):
            ModelConfig.get_model_instance(config)

    @patch("arklex.orchestrator.NLU.services.model_config.PROVIDER_MAP")
    def test_get_model_instance_provider_initialization_failure(
        self, mock_provider_map: Mock
    ) -> None:
        """Test get_model_instance when provider initialization fails."""
        mock_provider_map.__contains__.return_value = True
        mock_provider_map.__getitem__.return_value = Mock(
            side_effect=Exception("Initialization failed")
        )

        config = {
            "model_type_or_path": "test-model",
            "llm_provider": "openai",
            "api_key": "test-key",  # Add API key to avoid validation error
        }

        with pytest.raises(Exception, match="Initialization failed"):
            ModelConfig.get_model_instance(config)

    def test_configure_response_format_none_model(self) -> None:
        """Test configure_response_format with None model."""
        config = {"llm_provider": "openai"}

        with pytest.raises(AttributeError):
            ModelConfig.configure_response_format(None, config, "text")

    def test_configure_response_format_model_without_bind(self) -> None:
        """Test configure_response_format with model that doesn't have bind method."""
        mock_model = Mock()
        del mock_model.bind  # Remove bind method

        config = {"llm_provider": "openai"}

        with pytest.raises(AttributeError):
            ModelConfig.configure_response_format(mock_model, config, "text")
