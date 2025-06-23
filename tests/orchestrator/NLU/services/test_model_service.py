"""Tests for the model service module."""

import pytest
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from arklex.utils.exceptions import ValidationError
from arklex.orchestrator.NLU.services.model_service import ModelService


@pytest.fixture
def model_config() -> Dict[str, Any]:
    """Create a test model configuration.

    Returns:
        Dict[str, Any]: A dictionary containing model configuration.
    """
    return {
        "model_name": "test-model",
        "api_key": "test-key",
        "endpoint": "https://api.test.com/v1",
        "model_type_or_path": "gpt-3.5-turbo",
        "llm_provider": "openai",
        "temperature": 0.1,
        "max_tokens": 1000,
        "response_format": "json",
    }


@pytest.fixture
def model_service(model_config: Dict[str, Any]) -> ModelService:
    """Create a test model service instance.

    Args:
        model_config: The model configuration to use.

    Returns:
        ModelService: A configured model service instance.
    """
    return ModelService(model_config)


def test_model_service_initialization(model_config: Dict[str, Any]) -> None:
    """Test model service initialization.

    Args:
        model_config: The model configuration to use.
    """
    service = ModelService(model_config)
    assert service.model_config == model_config


def test_model_service_initialization_missing_config() -> None:
    """Test model service initialization with missing configuration."""
    with pytest.raises(ValidationError) as exc_info:
        ModelService({})
    assert "Missing required field" in str(exc_info.value)


@pytest.mark.asyncio
async def test_process_text_success(model_service: ModelService) -> None:
    """Test successful text processing.

    Args:
        model_service: The model service instance to test.
    """
    text = "Test input text"
    context = {"user_id": "123"}

    with patch.object(
        model_service, "_make_model_request", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = {"result": "Test response"}

        response = await model_service.process_text(text, context)

        assert response == {"result": "Test response"}
        mock_request.assert_called_once_with(
            {
                "text": text,
                "context": context,
                "model": model_service.model_config["model_name"],
            }
        )


@pytest.mark.asyncio
async def test_process_text_empty_input(model_service: ModelService) -> None:
    """Test text processing with empty input.

    Args:
        model_service: The model service instance to test.
    """
    with pytest.raises(ValidationError) as exc_info:
        await model_service.process_text("")
    assert "Text cannot be empty" in str(exc_info.value)


@pytest.mark.asyncio
async def test_process_text_request_failure(model_service: ModelService) -> None:
    """Test text processing when request fails.

    Args:
        model_service: The model service instance to test.
    """
    text = "Test input text"

    with patch.object(
        model_service, "_make_model_request", new_callable=AsyncMock
    ) as mock_request:
        mock_request.side_effect = Exception("API Error")

        with pytest.raises(Exception) as exc_info:
            await model_service.process_text(text)
        assert "API Error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_make_model_request(model_service: ModelService) -> None:
    """Test making a model request.

    Args:
        model_service: The model service instance to test.
    """
    request_data = {
        "text": "Test input",
        "context": {"user_id": "123"},
        "model": "test-model",
    }

    response = await model_service._make_model_request(request_data)
    assert isinstance(response, dict)
    assert "result" in response


@pytest.fixture
def dummy_config():
    return {
        "model_name": "dummy",
        "api_key": "dummy",
        "endpoint": "http://dummy",
        "model_type_or_path": "dummy-path",
        "llm_provider": "dummy",
    }


def test_model_service_initialization(dummy_config) -> None:
    service = ModelService(dummy_config)
    assert service.model_config["model_name"] == "dummy"


def test_model_service_get_response_success(dummy_config) -> None:
    service = ModelService(dummy_config)
    mock_response = MagicMock()
    mock_response.content = "0) greeting"
    with patch.object(service.model, "invoke", return_value=mock_response):
        result = service.get_response("User: hello")
        assert result == "0) greeting"


def test_model_service_get_response_empty(dummy_config) -> None:
    service = ModelService(dummy_config)
    mock_response = MagicMock()
    mock_response.content = ""
    with patch.object(service.model, "invoke", return_value=mock_response):
        with pytest.raises(ValueError, match="Empty response from model"):
            service.get_response("User: hello")


def test_model_service_get_response_model_error(dummy_config) -> None:
    service = ModelService(dummy_config)
    with patch.object(service.model, "invoke", side_effect=Exception("Model error")):
        with pytest.raises(
            ValueError, match="Failed to get model response: Model error"
        ):
            service.get_response("User: hello")


def test_model_service_missing_model_name() -> None:
    config = {"api_key": "key", "endpoint": "url"}
    with pytest.raises(ValidationError, match="Missing required field"):
        ModelService(config)


def test_model_service_missing_api_key() -> None:
    config = {"model_name": "name", "endpoint": "url"}
    with pytest.raises(ValidationError, match="Missing required field"):
        ModelService(config)


def test_model_service_missing_endpoint() -> None:
    config = {"model_name": "name", "api_key": "key"}
    with pytest.raises(ValidationError, match="Missing required field"):
        ModelService(config)
