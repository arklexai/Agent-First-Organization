"""Tests for model provider configuration module."""

from unittest.mock import Mock, patch

from arklex.utils.model_provider_config import (
    LLM_PROVIDERS,
    PROVIDER_EMBEDDING_MODELS,
    PROVIDER_EMBEDDINGS,
    PROVIDER_MAP,
    DummyLLM,
    get_huggingface_llm,
)


class TestModelProviderConfig:
    """Test cases for model provider configuration."""

    def test_get_huggingface_llm(self) -> None:
        """Test get_huggingface_llm function."""
        with patch(
            "arklex.utils.model_provider_config.HuggingFaceEndpoint"
        ) as mock_endpoint:
            mock_endpoint_instance = Mock()
            mock_endpoint.return_value = mock_endpoint_instance

            with patch(
                "arklex.utils.model_provider_config.ChatHuggingFace"
            ) as mock_chat:
                mock_chat_instance = Mock()
                mock_chat.return_value = mock_chat_instance

                result = get_huggingface_llm("test-model", temperature=0.7)

                # Verify HuggingFaceEndpoint was called correctly
                mock_endpoint.assert_called_once_with(
                    repo_id="test-model", task="text-generation", temperature=0.7
                )

                # Verify ChatHuggingFace was called with the endpoint
                mock_chat.assert_called_once_with(llm=mock_endpoint_instance)

                assert result == mock_chat_instance

    def test_dummy_llm_initialization(self) -> None:
        """Test DummyLLM initialization."""
        dummy_llm = DummyLLM(model_name="test", temperature=0.7)
        assert dummy_llm is not None

    def test_dummy_llm_invoke(self) -> None:
        """Test DummyLLM invoke method."""
        dummy_llm = DummyLLM()
        response = dummy_llm.invoke("test message")

        assert response is not None
        assert hasattr(response, "content")
        assert response.content == "dummy response"

    def test_llm_providers_list(self) -> None:
        """Test LLM_PROVIDERS list contains expected providers."""
        expected_providers = ["openai", "google", "anthropic", "huggingface"]

        for provider in expected_providers:
            assert provider in LLM_PROVIDERS

    def test_provider_map_contains_all_providers(self) -> None:
        """Test PROVIDER_MAP contains all expected providers."""
        expected_providers = ["anthropic", "google", "openai", "huggingface", "dummy"]

        for provider in expected_providers:
            assert provider in PROVIDER_MAP

    def test_provider_embeddings_contains_all_providers(self) -> None:
        """Test PROVIDER_EMBEDDINGS contains all expected providers."""
        expected_providers = ["anthropic", "google", "openai", "huggingface"]

        for provider in expected_providers:
            assert provider in PROVIDER_EMBEDDINGS

    def test_provider_embedding_models_contains_all_providers(self) -> None:
        """Test PROVIDER_EMBEDDING_MODELS contains all expected providers."""
        expected_providers = ["anthropic", "google", "openai", "huggingface"]

        for provider in expected_providers:
            assert provider in PROVIDER_EMBEDDING_MODELS

    def test_provider_map_values_are_callable_or_class(self) -> None:
        """Test that all values in PROVIDER_MAP are callable or classes."""
        for provider, value in PROVIDER_MAP.items():
            if provider == "huggingface":
                # huggingface is a function
                assert callable(value)
            else:
                # Others should be classes
                assert isinstance(value, type)

    def test_provider_embeddings_values_are_classes(self) -> None:
        """Test that all values in PROVIDER_EMBEDDINGS are classes."""
        for _provider, value in PROVIDER_EMBEDDINGS.items():
            assert isinstance(value, type)

    def test_provider_embedding_models_values_are_strings(self) -> None:
        """Test that all values in PROVIDER_EMBEDDING_MODELS are strings."""
        for _provider, value in PROVIDER_EMBEDDING_MODELS.items():
            assert isinstance(value, str)
            assert len(value) > 0
