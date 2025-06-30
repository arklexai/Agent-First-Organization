"""Tests for the IntentDetector class.

This module contains comprehensive tests for the IntentDetector class,
covering both local and remote intent detection functionality, as well
as error handling cases.
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from arklex.orchestrator.NLU.core.intent import IntentDetector
from arklex.orchestrator.NLU.services.api_service import APIClientService
from arklex.orchestrator.NLU.services.model_service import ModelService
from arklex.utils.exceptions import APIError, ModelError, ValidationError


@pytest.fixture
def mock_model_service() -> ModelService:
    """Create a mock ModelService for testing."""
    service = Mock(spec=ModelService)
    service.format_intent_input.return_value = (
        "Test prompt",
        {"1": "greeting", "2": "farewell"},
    )
    service.get_response.return_value = "1) greeting"
    return service


@pytest.fixture
def mock_api_service() -> APIClientService:
    """Create a mock APIClientService for testing."""
    service = Mock(spec=APIClientService)
    service.predict_intent.return_value = "greeting"
    return service


@pytest.fixture
def sample_intents() -> dict[str, list[dict[str, Any]]]:
    """Create sample intents for testing."""
    return {
        "greeting": [
            {
                "attribute": {
                    "definition": "A greeting",
                    "sample_utterances": ["hello", "hi"],
                }
            }
        ],
        "farewell": [
            {
                "attribute": {
                    "definition": "A farewell",
                    "sample_utterances": ["goodbye", "bye"],
                }
            }
        ],
    }


def test_intent_detector_initialization(mock_model_service: ModelService) -> None:
    """Test IntentDetector initialization."""
    # Test successful initialization with model service
    detector = IntentDetector(model_service=mock_model_service)
    assert detector.model_service == mock_model_service
    assert detector.api_service is None

    # Test initialization with both services
    api_service = Mock(spec=APIClientService)
    detector = IntentDetector(model_service=mock_model_service, api_service=api_service)
    assert detector.model_service == mock_model_service
    assert detector.api_service == api_service

    # Test initialization without model service
    with pytest.raises(ValidationError, match="Model service is required"):
        IntentDetector(model_service=None)


def test_detect_intent_local(
    mock_model_service: ModelService,
    sample_intents: dict[str, list[dict[str, Any]]],
) -> None:
    """Test local intent detection."""
    detector = IntentDetector(model_service=mock_model_service)

    # Test successful intent detection
    result = detector._detect_intent_local(
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config={"temperature": 0.7},
    )
    assert result == "greeting"
    mock_model_service.format_intent_input.assert_called_once()
    mock_model_service.get_response.assert_called_once()

    # Test invalid response format
    mock_model_service.get_response.return_value = "invalid_format"
    with pytest.raises(ValidationError, match="Invalid response format"):
        detector._detect_intent_local(
            intents=sample_intents,
            chat_history_str="User: hello",
            model_config={"temperature": 0.7},
        )

    # Test intent not in mapping
    mock_model_service.get_response.return_value = "3) unknown_intent"
    result = detector._detect_intent_local(
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config={"temperature": 0.7},
    )
    assert result == "others"


def test_detect_intent_remote(
    mock_model_service: ModelService,
    mock_api_service: APIClientService,
    sample_intents: dict[str, list[dict[str, Any]]],
) -> None:
    """Test remote intent detection."""
    detector = IntentDetector(
        model_service=mock_model_service, api_service=mock_api_service
    )

    # Test successful remote intent detection
    result = detector._detect_intent_remote(
        text="hello",
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config={"temperature": 0.7},
    )
    assert result == "greeting"
    mock_api_service.predict_intent.assert_called_once()

    # Test API error
    mock_api_service.predict_intent.side_effect = APIError("API error")
    with pytest.raises(APIError, match="Failed to detect intent via API"):
        detector._detect_intent_remote(
            text="hello",
            intents=sample_intents,
            chat_history_str="User: hello",
            model_config={"temperature": 0.7},
        )

    # Test without API service
    detector = IntentDetector(model_service=mock_model_service)
    with pytest.raises(ValidationError, match="API service not configured"):
        detector._detect_intent_remote(
            text="hello",
            intents=sample_intents,
            chat_history_str="User: hello",
            model_config={"temperature": 0.7},
        )


def test_predict_intent(
    mock_model_service: ModelService,
    mock_api_service: APIClientService,
    sample_intents: dict[str, list[dict[str, Any]]],
) -> None:
    """Test the main predict_intent method."""
    # Test local prediction
    detector = IntentDetector(model_service=mock_model_service)
    result = detector.predict_intent(
        text="hello",
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config={"temperature": 0.7},
    )
    assert result == "greeting"

    # Test remote prediction
    detector = IntentDetector(
        model_service=mock_model_service, api_service=mock_api_service
    )
    result = detector.predict_intent(
        text="hello",
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config={"temperature": 0.7},
    )
    assert result == "greeting"

    # Test local error handling
    detector = IntentDetector(model_service=mock_model_service)
    mock_model_service.get_response.side_effect = ModelError("Model error")
    with pytest.raises(ModelError):
        detector.predict_intent(
            text="hello",
            intents=sample_intents,
            chat_history_str="User: hello",
            model_config={"temperature": 0.7},
        )

    # Test remote error handling
    detector = IntentDetector(
        model_service=mock_model_service, api_service=mock_api_service
    )
    mock_api_service.predict_intent.side_effect = APIError("API error")
    with pytest.raises(APIError):
        detector.predict_intent(
            text="hello",
            intents=sample_intents,
            chat_history_str="User: hello",
            model_config={"temperature": 0.7},
        )


def test_execute(
    mock_model_service: ModelService,
    sample_intents: dict[str, list[dict[str, Any]]],
) -> None:
    """Test the execute method (alias for predict_intent)."""
    detector = IntentDetector(model_service=mock_model_service)

    # Test successful execution
    result = detector.execute(
        text="hello",
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config={"temperature": 0.7},
    )
    assert result == "greeting"

    # Test that execute calls predict_intent
    with patch.object(detector, "predict_intent") as mock_predict:
        mock_predict.return_value = "farewell"
        result = detector.execute(
            text="goodbye",
            intents=sample_intents,
            chat_history_str="User: goodbye",
            model_config={"temperature": 0.7},
        )
        assert result == "farewell"
        mock_predict.assert_called_once()
